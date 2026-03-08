from collections import defaultdict
from collections.abc import Callable
import math
import matplotlib.pyplot as plt
import scipy
import torch
import torch.nn.functional as F
import tqdm
import os

from util_1022 import (
    get_accuracy,
    get_cross_entropy,
    get_dataloader_random_reshuffle,
    get_mlp,
    line_plot_confidence_band,
    load_preprocessed_dataset
)

device = "mps" if torch.backends.mps.is_available() else "cpu"

config = {
    "dataset_preprocessed_path": "data/mnist.pt",
    "device": device,
    "ensemble_shape": (16,),
    "hyperparameter_raw_init_distributions": {
        "learning_rate": torch.distributions.Uniform(
            low=torch.tensor(-5, device=device, dtype=torch.float32),
            high = torch.tensor(0, device=device, dtype=torch.float32)
            ),
        "weight_decay": torch.distributions.Uniform(
            low = torch.tensor(-5, device=device, dtype=torch.float32),
            high = torch.tensor(0, device=device, dtype=torch.float32)
            )
        },
    "hyperparameter_raw_perturb": {
        "learning_rate": torch.distributions.Normal(
            loc=torch.tensor(0, device = device, dtype=torch.float32),
            scale=torch.tensor(2, device = device, dtype=torch.float32)
            ),
        "weight_decay": torch.distributions.Normal(
            loc=torch.tensor(0, device = device, dtype=torch.float32),
            scale=torch.tensor(2, device = device, dtype=torch.float32)
            )
        },
    "hyperparameter_transforms": {
        "learning_rate": lambda log10: 10 ** log10,
        "weight_decay": lambda log10: 10 ** log10
        },
    "improvement_threshold": 1e-4,
    "minibatch_size": 128,
    "minibatch_size_eval": 1<<14,
    "pbt": True,
    "seed": 1,
    "steps_num": 100_000,
    "steps_without_improvement": 1000,
    "valid_interval": 100,
    "welch_confidence_level": .8,
    "welch_sample_size": 10
    }
    
torch.manual_seed(config["seed"])


ensemble_shape = config["ensemble_shape"]
key = "learning_rate"
transform = config["hyperparameter_transforms"][key]

learning_rate_raw_initial = config[
    "hyperparameter_raw_init_distributions"
    ][key].sample(ensemble_shape)

noise = config["hyperparameter_raw_perturb"][key].sample(ensemble_shape)

learning_rate_initial = transform(learning_rate_raw_initial)
learning_rate_raw_perturbed = learning_rate_raw_initial + noise
learning_rate_perturbed = transform(learning_rate_raw_perturbed)

(
    (train_features, train_labels),
    (valid_features, valid_labels),
    (test_features, test_labels),
) = load_preprocessed_dataset(config)

def normalize_features(
    train_features: torch.Tensor,
    additional_features = (),
    verbose = False
):
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

normalize_features(
    train_features,
    (valid_features, test_features),
    verbose = True
)


def evaluate_model(
    config: dict,
    features: torch.Tensor,
    get_metric: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    model: torch.nn.Module,
    values: torch.Tensor
) -> torch.Tensor:
    
    dataset_size = len(features)
    minibatch_num = math.ceil(dataset_size / config["minibatch_size_eval"])
    metric = 0
    with torch.no_grad():
        for i in range(minibatch_num):
            minibatch_features, minibatch_values = (
                t[
                    i * config["minibatch_size_eval"]
                    :(i+1) * config["minibatch_size_eval"]
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

model = get_mlp(config, train_features.shape[1], 10, 3, 128)

with torch.no_grad():
    metric = evaluate_model(
        config,
        train_features,
        get_accuracy,
        model,
        train_labels
    )

print("Training accuracies at initialization:", metric.cpu().numpy())

#We measure whether a given source model is underperforming 
# a target model with an 80% confidence level using 
# Welch's t-test to control for variance between models.
def welch_one_sided(
    source: torch.Tensor,
    target: torch.Tensor,
    confidence_level = .8,
) -> torch.Tensor:
    
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
        * (sample_num -1)
        / (source_sample_var ** 2 + target_sample_var **2 )
        )
    
    p = scipy.stats.t(
        nu.cpu().numpy()
        ).cdf(
            t.cpu().numpy()
        )
    
    return torch.tensor(
        p > confidence_level,
        device = source.device
    )
    
def pbt(
    config: dict,
    get_loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    get_metric: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    model: torch.nn.Module,
    train_features: torch.Tensor,
    train_values: torch.Tensor,
    valid_features: torch.Tensor,
    valid_values: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    ensemble_shape = config["ensemble_shape"]
    if len(ensemble_shape) != 1:
        raise ValueError(f"The number of dimensions in the ensemble shape should be 1 for the  population size, but it is {len(ensemble_shape)}")

    population_size = ensemble_shape[0]
    config_local = dict(config)
    output = defaultdict(list)

    for name, distribution in config[
        "hyperparameter_raw_init_distributions"
    ].items():
        value_raw = distribution.sample(ensemble_shape)
        config_local[name + "_raw"] = value_raw
        value = config[
            "hyperparameter_transforms"
        ][name](value_raw)
        config_local[name] = value
        output[name].append(value)

    best_valid_metric = -torch.inf
    progress_bar = tqdm.trange(config["steps_num"])
    steps_without_improvement = 0
    train_dataloader = get_dataloader_random_reshuffle(
        config,
        train_features,
        train_values
    )

    for step_id in progress_bar:
        minibatch_features, minibatch_values = next(train_dataloader)
        for parameter in model.parameters():
            parameter.grad = None

        predict = model(minibatch_features)
        loss = get_loss(predict, minibatch_values).sum()
        loss.backward()
        with torch.no_grad():
            for parameter in model.parameters():
                lr = config_local["learning_rate"].reshape(
                    config_local["learning_rate"].shape
                  + (
                        len(parameter.shape)
                      - len(config_local["learning_rate"].shape)
                    )
                  * (1,)
                ) 
                
                wd = config_local["weight_decay"].reshape(
                    config_local["weight_decay"].shape
                  + (
                        len(parameter.shape)
                      - len(config_local["weight_decay"].shape)
                    )
                  * (1,)
                )
                
                parameter -= lr * (parameter.grad + wd * parameter)
        
        if (step_id + 1) % config["valid_interval"] == 0:
            with torch.no_grad():
                for features, values, split_name in (
                    (train_features, train_values, "training"),
                    (valid_features, valid_values, "validation")
                ):
                    loss, metric = (
                        evaluate_model(
                            config,
                            features,
                            f,
                            model,
                            values
                        )
                        for f in (get_loss, get_metric)
                    )
                    output[f"{split_name} loss"].append(loss)
                    output[f"{split_name} metric"].append(metric)

                best_last_metric = output["validation metric"][-1].max()
                if (
                    best_valid_metric + config["improvement_threshold"]
                ) < best_last_metric:
                    best_valid_metric = best_last_metric
                    steps_without_improvement = 0
                else:
                    steps_without_improvement += config["valid_interval"]
                    if steps_without_improvement > config[
                        "steps_without_improvement"
                    ]:
                        break

                if config["pbt"] and (len(output["validation metric"]) >= config[
                    "welch_sample_size"
                ]):
                    validation_metrics = torch.stack(
                        output["validation metric"][
                            -config["welch_sample_size"]:
                        ]
                    )
                    target_indices = torch.randint(
                        device=validation_metrics.device,
                        high=population_size,
                        size=(population_size,)
                    )
                    source_mask = welch_one_sided(
                        validation_metrics,
                        validation_metrics[:, target_indices],
                        confidence_level=config["welch_confidence_level"]
                    )
                    output["source mask"].append(source_mask)
                    output["target indices"].append(target_indices)

                    if source_mask.any():
                        #Overwrite underperforming models with better performing models.
                        for parameter in model.parameters():
                            parameter[source_mask] = parameter[
                                target_indices[source_mask]
                            ]

                        for name, transform in config[
                            "hyperparameter_transforms"
                        ].items():
                            value_raw: torch.Tensor = config_local[
                                name + "_raw"
                            ]

                            #add noise to cloned model weights.
                            additive_noise = config[
                                "hyperparameter_raw_perturb"
                            ][name].sample(
                                (source_mask.sum(),)
                            )
                            perturbed_values = value_raw[
                                target_indices
                            ][source_mask] + additive_noise
                            value_raw[source_mask] = perturbed_values
                            value = transform(value_raw)
                            config_local[name] = value
                            output[name].append(value)


    progress_bar.close()
    
    for key, value in output.items():
        if isinstance(value, list):
            output[key] = torch.stack(value)

    return output

output = pbt(
    config,
    get_cross_entropy,
    get_accuracy,
    model,
    train_features,
    train_labels,
    valid_features,
    valid_labels
)


learning_rates = output["learning_rate"].log10().cpu()
weight_decays = output["weight_decay"].log10().cpu()

line_plot_confidence_band(
    torch.arange(0, len(learning_rates) * config["valid_interval"], config["valid_interval"]),
    learning_rates,
)

line_plot_confidence_band(
torch.arange(0, len(weight_decays) * config["valid_interval"], config["valid_interval"]),
    weight_decays
)

plt.show()
plt.close()

model = get_mlp(config, train_features.shape[-1], 10, 3, 128)
output = pbt(config | {"steps_num": 10_001, "pbt": False}, get_cross_entropy, get_accuracy, model, train_features, train_labels, valid_features, valid_labels)

print(f"best validation accuracy", output["validation metric"].max().cpu().item())
