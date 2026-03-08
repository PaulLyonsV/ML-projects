# This work is licensed under Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International. 
# To view a copy of this license, visit
# https://creativecommons.org/licenses/by-nc-sa/4.0/

from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Callable, Generator, Iterable, Sequence
import datasets
import gymnasium as gym
import math
import matplotlib.pyplot as plt
from moviepy import ImageSequenceClip
import os
import scipy
import torch
import torch.nn.functional as F
import tqdm
from typing import Optional


class Linear(torch.nn.Module):
    """
    Ensemble-ready affine transformation `y = x^T W + b`.

    Arguments
    ---------
    config : `dict`
        Configuration dictionary. Required key-value pairs:
        `"device"` : `str`
            The device to store parameters on.
        `"ensemble_shape"` : `tuple[int]`
            The shape of the ensemble of affine transformations
            the model represents.
    in_features : `int`
        The number of input features
    out_features : `int`
        The number of output features.
    bias : `bool`, optional
        Whether the model should include bias. Default: `True`.
    init_multiplier : `float`, optional
        The weight parameter values are initialized following
        a normal distribution with center 0 and std
        `in_features ** -.5` times this value. Default: `1.`

    Calling
    -------
    Instance calls require one positional argument:
    features : `torch.Tensor`
        The input tensor. It is required to be one of the following shapes:
        1. `ensemble_shape + batch_shape + (in_features,)`
        2. `batch_shape + (in_features,)

        Upon a call, the model thinks we're in the first case
        if the first `len(ensemble_shape)` many entries of the
        shape of the input tensor is `ensemble_shape`.
    """
    def __init__(
        self,
        config: dict,
        in_features: int,
        out_features: int,
        bias=True,
        init_multiplier=1.
    ):
        super().__init__()

        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(
                config["ensemble_shape"] + (out_features,),
                device=config["device"],
                dtype=torch.float32
            ))
        else:
            self.bias = None

        self.weight = torch.nn.Parameter(torch.empty(
            config["ensemble_shape"] + (in_features, out_features),
            device=config["device"],
            dtype=torch.float32
        ).normal_(std=out_features ** -.5) * init_multiplier)


    def forward(
        self,
        features: torch.Tensor
    ) -> torch.Tensor:
        ensemble_shape = self.weight.shape[:-2]
        ensemble_dim = len(ensemble_shape)
        ensemble_input = features.shape[:ensemble_dim] == ensemble_shape
        batch_dim = len(features.shape) - 1 - ensemble_dim * ensemble_input
        
        # (*e, *b, i) @ (*e, *b[:-1], i, o)
        weight = self.weight.reshape(
            ensemble_shape
          + (1,) * (batch_dim - 1)
          + self.weight.shape[-2:]
        )
        features = features @ weight

        if self.bias is None:
            return features
        
        # (*e, *b, o) + (*e, *b, o)
        bias = self.bias.reshape(
            ensemble_shape
          + (1,) * batch_dim
          + self.bias.shape[-1:]
        )
        features = features + bias

        return features
    

class Optimizer(ABC):
    """
    Optimizer base class.
    Can optimize model ensembles
    with training defined by hyperparameter ensembles.

    Arguments
    ---------
    parameters : `Iterable[torch.nn.Parameter]`
        An iterable of `torch.nn.Parameter` to track.
        In a simple case of optimizing a single `model: torch.nn.Module`,
        this can be `model.parameters()`.
    config : `dict`, optional
        If given, the `update_config` method is called on it
        to initialize hyperparameters. Default: `None`.

    Class attributes
    ----------------
    keys : `tuple[str]`
        The collection of the hyperparameter keys to track
        in the configuration dictionary.

        We expect the hyperparameter values to be either
        `float` or `torch.Tensor`. In the latter case,
        we expect the shape to be a prefix of the shape of the parameters.
        The hyperparameter shapes are regarded as ensemble shapes.

        Required keys:
        `"learning_rate"`
        `"weight_decay"`

    Instance attributes
    -------------------
    config : `dict`
        The hyperparameter dictionary.
    parameters : `list[torch.nn.Parameter]`
        The list of tracked parameters.
    step_id : `int`
        Train step counter.
    """
    keys=(
        "learning_rate",
        "weight_decay"
    )
    def __init__(
        self,
        parameters: Iterable[torch.nn.Parameter],
        config=None
    ):
        self.config = dict()
        self.parameters = list(parameters)
        self.step_id = 0

        if config is not None:
            self.update_config(config)
    

    def get_parameters(self) -> Iterable[torch.Tensor]:
        """
        Get an iterable over tracked parameters
        and optimizer state tensors.
        """
        return iter(self.parameters)


    def get_hyperparameter(
        self,
        key: str,
        parameter: torch.Tensor
    ) -> torch.Tensor:
        """
        Take the hyperparameter with name `key`,
        transform it to `torch.Tensor` with the same
        `device` and `dtype` as `parameter`
        and reshape it to be broadcastable
        to `parameter` by postfixing to its shape
        an appropriate number of dimensions of 1.
        """        
        hyperparameter = torch.asarray(
            self.config[key],
            device=parameter.device,
            dtype=parameter.dtype
        )

        return hyperparameter.reshape(
            hyperparameter.shape
            + (
                len(parameter.shape)
                - len(hyperparameter.shape)
            )
            * (1,)
        )


    def step(self):
        """
        Update optimizer state, then apply parameter updates in-place.
        Assumes that backpropagation has already occurred by
        a call to the `backward` method of the loss tensor.
        """
        self.step_id += 1
        with torch.no_grad():
            for i, parameter in enumerate(self.parameters):
                self._update_parameter(parameter, i)


    def update_config(self, config: dict):
        """
        Update hyperparameters by the values in `config: dict`.
        """
        for key in self.keys:
            self.config[key] = config[key]


    def zero_grad(self):
        """
        Make the `grad` attribute of each tracked parameter `None`.
        """
        for parameter in self.parameters:
            parameter.grad = None


    def _apply_parameter_update(
        self,
        parameter: torch.nn.Parameter,
        parameter_update: torch.Tensor
    ):
        parameter += parameter_update


    @abstractmethod
    def _get_parameter_update(
        self,
        parameter: torch.nn.Parameter,
        parameter_id: int
    ) -> torch.Tensor:
        if self.config["weight_decay"] is None:
            return torch.zeros_like(parameter)
        
        return -(
            self.get_hyperparameter("learning_rate", parameter)
          * self.get_hyperparameter("weight_decay", parameter)
          * parameter
        )


    def _update_state(
        self,
        parameter: torch.nn.Parameter,
        parameter_id: int
    ):
        pass


    def _update_parameter(
        self,
        parameter: torch.nn.Parameter,
        parameter_id: int
    ):
        self._update_state(parameter, parameter_id)
        parameter_update = self._get_parameter_update(
            parameter,
            parameter_id
        )
        self._apply_parameter_update(
            parameter,
            parameter_update
        )
    

class AdamW(Optimizer):
    """
    Adam optimizer with optionally weight decay.
    Can optimize model ensembles
    with training defined by hyperparameter ensembles.

    Arguments
    ---------
    parameters : `Iterable[torch.nn.Parameter]`
        An iterable of `torch.nn.Parameter` to track.
        In a simple case of optimizing a single `model: torch.nn.Module`,
        this can be `model.parameters()`.
    config : `dict`, optional
        If given, the `update_config` method is called on it
        to initialize hyperparameters. Default: `None`.

    Class attributes
    ----------------
    keys : `tuple[str]`
        The collection of the hyperparameter keys to track
        in the configuration dictionary.

        We expect the hyperparameter values to be either
        `float` or `torch.Tensor`. In the latter case,
        we expect the shape to be a prefix of the shape of the parameters.
        The hyperparameter shapes are regarded as ensemble shapes.

        Required keys:
        `"epsilon"`,
        `"first_moment_decay"`,
        `"learning_rate"`
        `"second_moment_decay"`,
        `"weight_decay"`
    """
    keys = (
        "epsilon",
        "first_moment_decay",
        "learning_rate",
        "second_moment_decay",
        "weight_decay"
    )
    def __init__(
        self,
        parameters: Iterable[torch.nn.Parameter],
        config=None
    ):
        super().__init__(parameters, config)
        self.first_moments = [
            torch.zeros_like(parameter)
            for parameter in self.parameters
        ]
        self.second_moments = [
            torch.zeros_like(parameter)
            for parameter in self.parameters
        ]


    def get_parameters(self) -> Iterable[torch.Tensor]:
        yield from self.parameters
        yield from self.first_moments
        yield from self.second_moments


    def _get_parameter_update(
        self,
        parameter: torch.nn.Parameter,
        parameter_id: int
    ) -> torch.Tensor:
        parameter_update = super()._get_parameter_update(
            parameter,
            parameter_id
        )

        epsilon = self.get_hyperparameter(
            "epsilon",
            parameter
        )
        first_moment = self.first_moments[parameter_id]
        first_moment_decay = self.get_hyperparameter(
            "first_moment_decay",
            parameter
        )
        learning_rate = self.get_hyperparameter(
            "learning_rate",
            parameter
        )
        second_moment = self.second_moments[parameter_id]
        second_moment_decay = self.get_hyperparameter(
            "second_moment_decay",
            parameter
        )

        first_moment_debiased = (
            first_moment
          / (1 - first_moment_decay ** self.step_id)
        )
        second_moment_debiased = (
            second_moment
          / (1 - second_moment_decay ** self.step_id)
        )        

        parameter_update -= (
            learning_rate
          * first_moment_debiased
          / (
                second_moment_debiased.sqrt()
              + epsilon
            )
        )

        return parameter_update


    def _update_state(
        self,
        parameter: torch.nn.Parameter,
        parameter_id: int
    ):
        first_moment = self.first_moments[parameter_id]
        first_moment_decay = self.get_hyperparameter(
            "first_moment_decay",
            parameter
        )
        second_moment = self.second_moments[parameter_id]
        second_moment_decay = self.get_hyperparameter(
            "second_moment_decay",
            parameter
        )

        first_moment[:] = (
            first_moment_decay
          * first_moment
          + (1 - first_moment_decay)
          * parameter.grad
        )
        second_moment[:] = (
            second_moment_decay
          * second_moment
          + (1 - second_moment_decay)
          * parameter.grad.square()
        )


def evaluate_model(
    config: dict,
    features: torch.Tensor,
    get_metric: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    model: torch.nn.Module,
    values: torch.Tensor
) -> torch.Tensor:
    """
    Evaluate a model on a supervised dataset.

    Parameters
    ----------
    config : `dict`
        Configuration dictionary. Required key-value pair:
        `"minibatch_size_eval"` : `int`
            Size of consecutive minibatches to take from the dataset.
            To be set according to RAM or GPU memory capacity.
    features : `torch.Tensor`
        Feature tensor.
    get_metric : `Callable[[torch.Tensor, torch.Tensor], torch.Tensor]`
        Function to get the metric from a pair of
        predicted and target value tensors.
    model : `torch.nn.Module`
        The model to evaluate.
    values : `torch.Tensor`
        Target value tensor.
    """
    dataset_size = len(features)
    minibatch_num = math.ceil(dataset_size / config["minibatch_size_eval"])
    metric = 0
    with torch.no_grad():
        for i in range(minibatch_num):
            minibatch_features, minibatch_values = (
                t[
                    i * config["minibatch_size_eval"]
                   :(i + 1) * config["minibatch_size_eval"]
                ]
                for t in (features, values)
            )
            minibatch_predict = model(minibatch_features)
            minibatch_metric = get_metric(
                minibatch_predict,
                minibatch_values
            )
            minibatch_size = len(minibatch_features)

            metric += minibatch_metric * minibatch_size

    return metric / dataset_size


def get_accuracy(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """
    Given logits output by a classification model, calculate the accuracy.
    Supports model ensembles of arbitrary ensemble shape.

    Parameters
    ----------
    logits : torch.Tensor
        Logit tensor of shape
        `ensemble_shape + (dataset_size, label_num)`.
    labels : torch.Tensor
        Label tensor of shape 
        `(dataset_size,)` or
        `ensemble_shape + (dataset_size,)`.

    Returns
    -------
    The tensor of accuracies of shape `ensemble_shape`.
    """
    labels_predict = logits.argmax(dim=-1)
    accuracy = (labels == labels_predict).to(torch.float32).mean(dim=-1)

    return accuracy


def get_binary_accuracy(
    logits: torch.Tensor,
    labels: torch.Tensor
) -> torch.Tensor:
    """
    Get the binary accuracy between a label and a logit tensor.
    It can handle arbitrary ensemble shapes.

    Parameters
    ----------
    logits : torch.Tensor
        The logit tensor. We assume it has shape
        `ensemble_shape + (dataset_size, 1)`.
    labels : torch.Tensor
        The tensor of true labels. We assume it has shape
        `(dataset_size,)` or `ensemble_shape + (dataset_size,)`.

    Returns
    -------
    The tensor of binary accuracies per ensemble member
    of shape `ensemble_shape`.
    """
    predict_positives = logits[..., 0] > 0
    true_positives = labels.to(torch.bool)

    return (
        predict_positives == true_positives
    ).to(torch.float32).mean(dim=-1)


def get_binary_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor
) -> torch.Tensor:
    """
    Get the binary cross-entropy between a label and a logit tensor.
    It can handle arbitrary ensemble shapes.

    Parameters
    ----------
    logits : torch.Tensor
        The logit tensor. We assume it has shape
        `ensemble_shape + (dataset_size,)`.
    labels : torch.Tensor
        The tensor of true labels. We assume it has shape
        `(dataset_size,)` or `ensemble_shape + (dataset_size, 1)`.

    Returns
    -------
    The tensor of binary cross-entropies per ensemble member
    of shape `ensemble_shape`.
    """

    return F.binary_cross_entropy_with_logits(
        logits[..., 0],
        labels.broadcast_to(logits.shape[:-1]),
        reduction="none"
    ).mean(dim=-1)


def get_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """
    Given logits output by a classification model, 
    calculate the cross-entropy.
    Supports model ensembles of arbitrary ensemble shape.

    Parameters
    ----------
    logits : torch.Tensor
        Logit tensor of shape
        `ensemble_shape + (dataset_size, label_num)`.
    labels : torch.Tensor
        Label tensor of shape 
        `(dataset_size,)` or
        `ensemble_shape + (dataset_size,)`.

    Returns
    -------
    The tensor of accuracies of shape `ensemble_shape`.
    """
    return F.cross_entropy(
        logits.movedim((-2, -1), (0, 1)),
        labels.broadcast_to(logits.shape[:-1]).movedim(-1, 0),
        reduction="none"
    ).mean(dim=0)


def get_dataloader_random_reshuffle(
    config: dict,
    features: torch.Tensor,
    labels: torch.Tensor
) -> Generator[tuple[torch.Tensor, torch.Tensor]]:
    """
    Given a feature and a label tensor,
    creates a random reshuffling (without replacement) dataloader
    that yields pairs `minibatch_features, minibatch_labels` indefinitely.
    Support arbitrary ensemble shapes.

    Parameters
    ----------
    config : dict
        Configuration dictionary. Required keys:
        ensemble_shape : tuple[int]
            The required ensemble shapes of the outputs.
        minibatch_size : int
            The size of the minibatches.
    features : torch.Tensor
        Tensor of dataset features.
        We assume that the first dimension is the batch dimension
    labels : torch.Tensor
        Tensor of dataset labels.

    Returns
    -------
    A generator of tuples `minibatch_features, minibatch_labels`.
    """
    for indices in get_random_reshuffler(
        len(labels),
        config["minibatch_size"],
        ensemble_shape=config["ensemble_shape"]
    ):
        yield features[indices], labels[indices]


def get_mlp(
    config: dict,
    in_features: int,
    out_features: int,
    hidden_layer_num: Optional[int] = None,
    hidden_layer_size: Optional[int] = None,
    hidden_layer_sizes: Optional[Iterable[int]] = None,
) -> torch.nn.Sequential:
    """
    Creates an MLP with ReLU activation functions.
    Can create a model ensemble.

    config : `dict`
        Configuration dictionary. Required key-value pairs:
        `"device"` : `str`
            The device to store parameters on.
        `"ensemble_shape"` : `tuple[int]`
            The shape of the ensemble of affine transformations
            the model represents.
    in_features : `int`
        The number of input features
    out_features : `int`
        The number of output features.
    hidden_layer_num : `int`, optional
        If `hidden_layer_sizes` is not given, we create an MLP with
        `hidden_layer_num` hidden layers of
        `hidden_layer_size` dimensions.
    hidden_layer_size : `int`, optional
        If `hidden_layer_sizes` is not given, we create an MLP with
        `hidden_layer_num` hidden layers of
        `hidden_layer_size` dimensions.
    hidden_layer_sizes: `Iterable[int]`, optional
        If given, each entry gives a hidden layer with the given size.
    """
    if hidden_layer_sizes is None:
        hidden_layer_sizes = (hidden_layer_size,) * hidden_layer_num

    layers = []
    layer_in_size = in_features
    for layer_out_size in hidden_layer_sizes:
        layers.extend([
            Linear(
                config,
                layer_in_size,
                layer_out_size,
                init_multiplier=2 ** .5
            ),
            torch.nn.ReLU()
        ])
        layer_in_size = layer_out_size
    
    layers.append(Linear(
        config,
        layer_in_size,
        out_features
    ))

    return torch.nn.Sequential(*layers)


def get_seed(
    upper=1 << 31
) -> int:
    """
    Generates a random integer by the `torch` PRNG,
    to be used as seed in a stochastic function.

    Parameters
    ----------
    upper : int, optional
        Exclusive upper bound of the interval to generate integers from.
        Default: 1 << 31.

    Returns
    -------
    A random integer.
    """
    return int(torch.randint(upper, size=()))


def get_random_reshuffler(
    dataset_size: int,
    minibatch_size: int,
    device="cpu",
    ensemble_shape=()
) -> Generator[torch.Tensor]:
    """
    Generate minibatch indices for a random shuffling dataloader.
    Supports arbitrary ensemble shapes.

    Parameters
    ----------
    dataset_size : int
        The size of the dataset to yield batches of minibatch indices for.
    minibatch_size : int
        The minibatch size.
    device : int | str | torch.device, optional
        The device to store the index tensors on. Default: "cpu"
    ensemble_shape : tuple[int], optional
        The ensemble shape of the minibatch indices. Default: ()
    """
    q, r = divmod(dataset_size, minibatch_size)
    minibatch_num = q + min(1, r)
    minibatch_index = minibatch_num
    while True:
        if minibatch_index == minibatch_num:
            minibatch_index = 0
            shuffled_indices = get_shuffled_indices(
                dataset_size,
                device=device,
                ensemble_shape=ensemble_shape
            )

        yield shuffled_indices[
            ...,
            minibatch_index * minibatch_size
        :(minibatch_index + 1) * minibatch_size
        ]

        minibatch_index += 1


def get_shuffled_indices(
    dataset_size: int,
    device="cpu",
    ensemble_shape=(),
) -> torch.Tensor:
    """
    Get a tensor of a batch of shuffles of indices `0,...,dataset_size - 1`.

    Parameters
    ----------
    dataset_size : int
        The size of the dataset the indices of which to shuffle
    device : int | str | torch.device, optional
        The device to store the resulting tensor on. Default: "cpu"
    ensemble_shape : tuple[int], optional
        The batch shape of the shuffled index tensors. Default: ()
    """
    total_shape = ensemble_shape + (dataset_size,)
    uniform = torch.rand(
        total_shape,
        device=device
    )
    indices = uniform.argsort(dim=-1)

    return indices


def line_plot_confidence_band(
    x: Sequence,
    y: torch.Tensor,
    color=None,
    confidence_level=.95,
    label="",
    opacity=.2
):
    """
    Plot training curves from an ensemble with a pointwise confidence band.

    Parameters
    ----------
    x : Sequence
        The sequence of time indicators (eg. number of train steps)
        when the measurements took place.
    y : torch.Tensor
        The tensor of measurements of shape `(len(x), ensemble_num)`.
    color : str | tuple[float] | None, optional
        The color of the plot. Default: `None`
    confidence_level : float, optional
        The confidence level of the confidence band. Default: 0.95
    label : str, optional
        The label of the plot. Default: ""
    opacity : float, optional
        The opacity of the confidence band, to be set via the
        `alpha` keyword argument of `plt.fill_between`. Default: 0.2
    """
    sample_size = y.shape[1]
    student_coefficient = -scipy.stats.t(sample_size - 1).ppf(
        (1 - confidence_level) / 2
    )
    y_mean = y.mean(dim=-1)
    y_std = y.std(dim=-1)
    
    interval_half_length = student_coefficient * y_std / sample_size ** .5
    y_low = y_mean - interval_half_length
    y_high = y_mean + interval_half_length

    plt.fill_between(x, y_low, y_high, alpha=opacity, color=color)
    plt.plot(x, y_mean, color=color, label=label)


def load_preprocessed_dataset(
    config: dict
) -> tuple[
    tuple[torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor]
]:
    """
    Loads a dataset that was saved with `torch.save`.
    We expect that the object that was saved is a dictionary with keys
    `train_features`, `train_labels`
    `valid_features`, `valid_labels`,
    `test_features`, `test_labels`
    storing the appropriate data in tensors.

    Parameters
    ----------
    config : dict
        Configuration dictionary. Required keys:  
        dataset_preprocessed_path : str
            The path where the preprocessed dataset was saved to.
        device : torch.device | int | str
            The device to map the tensors to.

    Returns
    -------
    The triple of pairs
    `(train_features, train_labels),
    (valid_feautres, valid_labels),
    (test_features, test_labels)`
    """
    loaded = torch.load(
        config["dataset_preprocessed_path"],
        weights_only=True
    )
    (
        train_features,
        train_labels,
        valid_features,
        valid_labels,
        test_features,
        test_labels
    ) = (
        loaded[key].to(config["device"])
        for key in [
            "train_features",
            "train_labels",
            "valid_features",
            "valid_labels",
            "test_features",
            "test_labels"
        ]
    )

    return (
        (train_features, train_labels),
        (valid_features, valid_labels),
        (test_features, test_labels)
    )


def normalize_features(
    train_features: torch.Tensor,
    additional_features=(),
    verbose=False
):
    """
    Normalize feature tensors by
    1. subtracting the total mean of the training features, then
    2. dividing by the total std of the offset training features.

    Optionally, apply the same transformation to additional feature tensors,
    eg. validation and test feature tensors.

    Parameters
    ----------
    train_features : `torch.Tensor`
        Training feature tensor.
    additional_features : `Iterable[torch.Tensor]`, optional
        Iterable of additional features to apply the transformation to.
        Default: `()`.
    verbose : `bool`, optional
        Whether to print the total mean and std
        gotten for the transformation.
    """
    sample_mean = train_features.mean()
    train_features -= sample_mean
    for features in additional_features:
        features -= sample_mean

    sample_std = train_features.std()
    train_features /= sample_std
    for features in additional_features:
        features /= sample_std

    if verbose:
        print(
            "Training feature tensor statistics before normalization:",
            f"mean {sample_mean.cpu().item():.4f}",
            f"std {sample_std.cpu().item():.4f}",
            flush=True
        )


def run_episode(
    config: dict,
    env: gym.Env,
    gif_name: Optional[str] = None,
    policy: Optional[Callable[[int], int]]=None,
) -> float:
    """
    Run an episode in a `gym.Env`
    with discrete observation and action spaces,
    following a policy.

    Make a gif video of the gameplay.

    Parameters
    ----------
    config : dict
        Configuration dictionary. Required key-value pairs:
        gif_fps : int
            Frames per second in the gif.
        video_directory : str
            If `gif_name` is given, the created movie will be saved
            to this directory.
    env : gym.Env
        The environment to get an episode in.
    gif_name : str, optional
        If given, a gif movie is saved to this filename
        in `config['videos_directory]`.
    policy : Callable[[int], int], optional
        The policy to get an episode with. Default: random policy.

    Returns
    -------
    The discounted return of the episode.
    """
    if policy is None:
        policy = lambda observation: env.action_space.sample()

    episode_return = 0
    frames = []
    step_id = 0
    observation, _ = env.reset(seed=get_seed())
    if gif_name is not None:
        os.makedirs(config["videos_directory"], exist_ok=True)
        frames.append(env.render())

    while True:
        action = policy(observation)
        observation, reward, _, terminated, _ = env.step(action)
        episode_return += reward * config["discount"] ** step_id
        if gif_name is not None:
            frames.append(env.render())

        if terminated:
            break

        step_id += 1

    if gif_name is not None:
        # https://stackoverflow.com/a/64796174
        clip = ImageSequenceClip(frames, fps=config["gif_fps"])
        gif_path = os.path.join(config["videos_directory"], gif_name)
        clip.write_gif(gif_path, fps=config["gif_fps"])

    return episode_return


def train_logistic_regression(
    config: dict,
    get_loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    get_metric: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    out_features: int,
    train_dataloader: Generator[tuple[torch.Tensor, torch.Tensor]],
    valid_features: torch.Tensor,
    valid_labels: torch.Tensor,
    loss_name="loss",
    metric_name="metric",
    use_bias=True
) -> dict:
    """
    Train a logistic regression model on a classification task.
    Support model ensembles of arbitrary shape.

    Parameters
    ----------
    config : dict
        Configuration dictionary. Required keys:
        ensemble_shape : tuple[int]
            The shape of the model ensemble.
        improvement_threshold : float
            Making the best validation score this much better
            counts as an improvement.
        learning_rate : float | torch.Tensor
            The learning rate of the SGD optimization.
            If a tensor, then it should have shape
            broadcastable to `ensemble_shape`.
            In that case, the members of the ensemble are trained with
            different learning rates.
        steps_num : int
            The maximum number of training steps to take.
        steps_without_improvement : int
            The maximum number of training steps without improvement to take.
        valid_interval : int
            The frequency of evaluations,
            measured in the number of train steps.
    out_features : int
        The number of output features.
        When training a binary logistic regression model, this should be 1.
        Otherwise, this should be
        the number of distinct labels in the classification task.
    train_dataloader : Generator[tuple[torch.Tensor, torch.Tensor]]
        A training minibatch dataloader, that yields pairs of
        feature and label tensors indefinitely.
        We assume that these have shape
        `ensemble_shape + (minibatch_size, feature_dim)`
        and `ensemble_shape + (minibatch_size,)`
        respectively.
    valid_features : torch.Tensor
        Validation feature matrix.
    valid_labels : torch.Tensor
        Validation label vector.
    loss_name : str, optional
        The name of the loss values in the output dictionary.
        Default: "loss"
    metric_name : str, optional
        The name of the metric values in the output dictionary.
        Default: "metric"
    use_bias : bool, optional
        Whether to use a bias vector in the logistic regression model.
        Default: `True`

    Returns
    -------
    An output dictionary with the following keys:
        best scores : torch.Tensor
            The best validation accuracy per each ensemble member
        best weights : torch.Tensor
            The logistic regression weights
            that were the best per each ensemble member.
        training {metric_name} : torch.Tensor
            The tensor of training metric values, of shape
            `(evaluation_num,) + ensemble_shape`.
        training {loss_name} : torch.Tensor
            The tensor of training loss values, of shape
            `(evaluation_num,) + ensemble_shape`.
        training steps : list[int]
            The list of the number of training steps at each evaluation.
        validation {metric_name} : torch.Tensor
            The tensor of validation metric values, of shape
            `(evaluation_num,) + ensemble_shape`.
        validation {loss_name} : torch.Tensor
            The tensor of validation loss values, of shape
            `(evaluation_num,) + ensemble_shape`.
        best bias : torch.Tensor, optional
            The logistic regression biases
            that were the best per each ensemble member, if used.
    """
    device = valid_features.device
    features_dtype = valid_features.dtype
    output = defaultdict(list)

    best_scores = torch.zeros(
        config["ensemble_shape"],
        device=device,
        dtype=features_dtype
    ).log()
    steps_without_improvement = 0

    if isinstance(config["learning_rate"], torch.Tensor):
        learning_rate = config["learning_rate"][..., None, None]
    else:
        learning_rate = config["learning_rate"]

    train_accuracies_step = torch.zeros(
        config["ensemble_shape"],
        device=device,
        dtype=features_dtype
    )
    train_entries = 0
    train_losses_step = torch.zeros(
        config["ensemble_shape"],
        device=device,
        dtype=features_dtype
    )

    progress_bar = tqdm.trange(config["steps_num"])
    step_id = 0
    weights = torch.zeros(
        config["ensemble_shape"] + (valid_features.shape[1], out_features),
        device=device,
        dtype=features_dtype,
        requires_grad=True
    )

    best_weights = torch.empty_like(weights, requires_grad=False)

    if use_bias:
        bias = torch.zeros_like(weights[..., 0:1, :], requires_grad=True)
        best_bias = torch.empty_like(bias, requires_grad=False)

    for minibatch_features, minibatch_labels in train_dataloader:
        minibatch_size = minibatch_labels.shape[-1]
        weights.grad = None
        if use_bias:
            bias.grad = None

        logits = minibatch_features @ weights
        if use_bias:
            logits = logits + bias

        train_accuracies_step += get_metric(
            logits.detach(),
            minibatch_labels
        ) * minibatch_size
        loss = get_loss(
            logits,
            minibatch_labels
        )
        loss.sum().backward()
        with torch.no_grad():
            weights -= learning_rate * weights.grad
            if use_bias:
                bias -= learning_rate * bias.grad

        train_losses_step += loss.detach() * minibatch_size
        train_entries += minibatch_size

        progress_bar.update()
        step_id += 1
        if step_id % config["valid_interval"] == 0:
            with torch.no_grad():
                logits = valid_features @ weights
                if use_bias:
                    logits = logits + bias

            valid_accuracy = get_metric(
                logits,
                valid_labels
            )

            valid_loss = get_loss(
                logits,
                valid_labels
            )

            output[f"training {metric_name}"].append(
                (train_accuracies_step / train_entries)
            )
            output[f"training {loss_name}"].append(
                (train_losses_step / train_entries)
            )
            output["training steps"].append(step_id)
            output[f"validation {metric_name}"].append(valid_accuracy)
            output[f"validation {loss_name}"].append(valid_loss)

            train_accuracies_step.zero_()
            train_entries = 0
            train_losses_step.zero_()

            improvement = valid_accuracy - best_scores
            improvement_mask = improvement > config["improvement_threshold"]

            if improvement_mask.any():
                best_scores[improvement_mask] \
                    = valid_accuracy[improvement_mask]
                best_weights[improvement_mask] = weights[improvement_mask]
                steps_without_improvement = 0
            else:
                steps_without_improvement += config["valid_interval"]

            if (
                step_id >= config["steps_num"]
             or (
                    steps_without_improvement
                 >= config["steps_without_improvement"]
                )  
            ):
                for key in (
                    f"training {metric_name}",
                    f"training {loss_name}",
                    f"validation {metric_name}",
                    f"validation {loss_name}"
                ):
                    output[key] = torch.stack(output[key]).cpu()

                output["best scores"] = best_scores
                output["best weights"] = best_weights
                if use_bias:
                    output["best_bias"] = best_bias
                progress_bar.close()

                return output
            

def welch_one_sided(
    source: torch.Tensor,
    target: torch.Tensor,
    confidence_level=.95
) -> torch.Tensor:
    """
    Performs Welch's t-test with null hypothesis: the expected value
    of the random variable the target tensor collects samples of
    is larger then the expected value
    of the random variable the source tensor collects samples of.

    In the tensors, dimensions after the first 
    are considered batch dimensions.

    Parameters
    ----------
    source : `torch.Tensor`
        Source sample, of shape `(sample_size,) + batch_shape`.
    target : `torch.Tensor`
        Target sample, of shape `(sample_size,) + batch_shape`.
    confidence_level : `float`, optional
        Confidence level of the test. Default: `.95`.
    Returns
    -------
    A Boolean tensor of shape `batch_shape` that is `False`
    where the null hypothesis is rejected.
    """
    sample_num = len(source)
    source_sample_mean, target_sample_mean = (
        t.mean(dim=0)
        for t in (source, target)
    )
    source_sample_var, target_sample_var = (
        t.var(dim=0)
        for t in (source, target)
    )
    var_sum = source_sample_var + target_sample_var

    t = (
        (target_sample_mean - source_sample_mean)
      * (sample_num / var_sum).sqrt()
    )

    nu = (
        var_sum.square()
      * (sample_num - 1)
      / (source_sample_var ** 2 + target_sample_var ** 2)
    )

    p = scipy.stats.t(
        nu.cpu().numpy()
    ).cdf(
        t.cpu().numpy()
    )

    return torch.asarray(
        p > confidence_level,
        device=source.device
    )