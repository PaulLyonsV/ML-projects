# This work is licensed under Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International. 
# To view a copy of this license, visit
# https://creativecommons.org/licenses/by-nc-sa/4.0/

from collections import defaultdict
from collections.abc import Callable, Generator, Sequence
import gymnasium as gym
import matplotlib.pyplot as plt
from moviepy import ImageSequenceClip
import os
import scipy
import torch
import torch.nn.functional as F
import tqdm
from typing import Optional


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