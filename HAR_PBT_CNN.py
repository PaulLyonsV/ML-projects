from collections import defaultdict
from collections.abc import (
    Callable
)
import datasets
import math
import torch
import torch.nn.functional as F
import tqdm
from typing import Optional
import os
import requests
import zipfile
import numpy as np


from util_1114 import (
    AdamW,
    evaluate_model,
    get_accuracy,
    get_cross_entropy,
    get_dataloader_random_reshuffle,
    Linear,
    normalize_features,
    Optimizer,
    pbt_init,
    pbt_update,
)

config = {
    "dataset_path": "UCI HAR Dataset",
    "device": "cuda",
    "ensemble_shape": (16,),
    "hyperparameter_raw_init_distributions": {
        "epsilon": torch.distributions.Uniform(
            torch.tensor(-10, device="cuda", dtype=torch.float32),
            torch.tensor(-5, device="cuda", dtype=torch.float32)
        ),
        "first_moment_decay": torch.distributions.Uniform(
            torch.tensor(-3, device="cuda", dtype=torch.float32),
            torch.tensor(0, device="cuda", dtype=torch.float32)
        ),
        "learning_rate": torch.distributions.Uniform(
            torch.tensor(-5, device="cuda", dtype=torch.float32),
            torch.tensor(-1, device="cuda", dtype=torch.float32)
        ),
        "second_moment_decay": torch.distributions.Uniform(
            torch.tensor(-5, device="cuda", dtype=torch.float32),
            torch.tensor(-1, device="cuda", dtype=torch.float32)
        ),
        "weight_decay": torch.distributions.Uniform(
            torch.tensor(-5, device="cuda", dtype=torch.float32),
            torch.tensor(-1, device="cuda", dtype=torch.float32)
        )
    },
    "hyperparameter_raw_perturb": {
        "epsilon": torch.distributions.Normal(
            torch.tensor(0, device="cuda", dtype=torch.float32),
            torch.tensor(1, device="cuda", dtype=torch.float32)
        ),
        "first_moment_decay": torch.distributions.Normal(
            torch.tensor(0, device="cuda", dtype=torch.float32),
            torch.tensor(1, device="cuda", dtype=torch.float32)
        ),
        "learning_rate": torch.distributions.Normal(
            torch.tensor(0, device="cuda", dtype=torch.float32),
            torch.tensor(1, device="cuda", dtype=torch.float32)
        ),
        "second_moment_decay": torch.distributions.Normal(
            torch.tensor(0, device="cuda", dtype=torch.float32),
            torch.tensor(1, device="cuda", dtype=torch.float32)
        ),
        "weight_decay": torch.distributions.Normal(
            torch.tensor(0, device="cuda", dtype=torch.float32),
            torch.tensor(1, device="cuda", dtype=torch.float32)
        ),
    },
    "hyperparameter_transforms": {
        "epsilon": lambda log10: 10 ** log10,
        "first_moment_decay": lambda x: (1 - 10 ** x).clamp(0, 1),
        "learning_rate": lambda log10: 10 ** log10,
        "second_moment_decay": lambda x: (1 - 10 ** x).clamp(0, 1),
        "weight_decay": lambda log10: 10 ** log10,
    },
    "improvement_threshold": 1e-4,
    "minibatch_size": 64,
    "minibatch_size_eval": 1 << 7,
    "pbt": True,
    "seed": 0,
    "steps_num": 2001,
    "steps_without_improvement": 10_000,
    "valid_interval": 1000,
    "welch_confidence_level": .8,
    "welch_sample_size": 10,
}

torch.manual_seed(config["seed"])


r = requests.get("https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip")
with open("dataset.zip", "wb") as f:
    f.write(r.content)
with zipfile.ZipFile("dataset.zip", "r") as z:
    z.extractall(".")

y_train_path = "UCI HAR Dataset/train/y_train.txt"
labels = []
with open(y_train_path) as f:
    for line in f:
        labels.append(int(line.strip()) - 1)
full_labels = torch.tensor(labels, device=config["device"], dtype=torch.long)

feature_files = [
    "body_acc_x_train.txt", "body_acc_y_train.txt", "body_acc_z_train.txt",
    "body_gyro_x_train.txt", "body_gyro_y_train.txt", "body_gyro_z_train.txt"
]
base_path = "UCI HAR Dataset/train/Inertial Signals/"

feature_list = []
for file_name in feature_files:
    dim_data = []
    with open(base_path + file_name) as f:
        for line in f:
            dim_data.append([float(x) for x in line.strip().split()])
    feature_list.append(dim_data)

full_features = torch.tensor(feature_list, device=config["device"], dtype=torch.float32)
full_features = full_features.permute(1, 0, 2) # (7352, 6, 128)

print(f"Dataset Shape: {full_features.shape}, Labels Shape: {full_labels.shape}")

n_samples = len(full_features)
n_train = int(n_samples * 0.8)
indices = torch.randperm(n_samples, device=config["device"])

train_idx, valid_idx = indices[:n_train], indices[n_train:]
train_features = full_features[train_idx]
train_labels = full_labels[train_idx]
valid_features = full_features[valid_idx]
valid_labels = full_labels[valid_idx]

normalize_features(train_features, (valid_features,), verbose=True)

class Conv1D(torch.nn.Module):
    def __init__(
        self,
        config: dict,
        in_channels: int,
        kernel_shape: int,
        out_channels: int,
        bias=True,
        dilation=1,
        init_multiplier=1.,
        padding="same",
        stride=1
    ):
        super().__init__()
        self.dilation = dilation
        self.ensemble_shape = config["ensemble_shape"]
        self.in_channels = in_channels
        self.kernel_shape = kernel_shape
        self.out_channels = out_channels
        self.padding = padding
        self.stride = stride

        height = kernel_shape
        self.weight = torch.nn.Parameter(torch.empty(
            self.ensemble_shape
          + (
                out_channels,
                in_channels,
                height,
            ),
            device=config["device"],
            dtype=torch.float32
        ).normal_(std=out_channels ** -.5) * init_multiplier)

        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(
                self.ensemble_shape + (out_channels,),
                device=config["device"],
                dtype=torch.float32
            ))
        else:
            self.bias = None


    def forward(self, features: torch.Tensor) -> torch.Tensor:
        ensemble_dim = len(self.ensemble_shape)

        if features.ndim == 3:
            features = features.expand(
                self.ensemble_shape + features.shape
            )

        features = (
            features
           .movedim(ensemble_dim, 0)
           .flatten(1, ensemble_dim + 1)
        )

        #We use groups to perform 16 convolutions in parallel.
        features = F.conv1d(
            features,
            self.weight.flatten(end_dim=ensemble_dim),
            bias=self.bias.flatten(end_dim=ensemble_dim),
            dilation=self.dilation,
            groups=max(1, sum(self.ensemble_shape)),
            padding=self.padding,
            stride=self.stride
        )

        features = (
            features
           .unflatten(1, self.ensemble_shape + (self.out_channels,))
           .movedim(0, ensemble_dim)
        )
        return features

class Pool1D(torch.nn.Module):
    def __init__(
        self,
        config: dict,
        kernel_shape: Optional[tuple[int]] = None,
        padding=0,
        stride=1
    ):
        super().__init__()
        self.ensemble_shape = config["ensemble_shape"]
        self.kernel_shape = kernel_shape
        self.padding = padding
        self.stride = stride


    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim == 3:
            features = features.expand(
                self.ensemble_shape + features.shape
            )
        
        if self.kernel_shape is None:
            return features.mean(dim=(-1))

        channels = features.shape[-2]
        ensemble_dim = len(self.ensemble_shape)
        
        #Flatten batch and ensemble dimensions 
        # before taking average across population.
        features = (
            features
           .movedim(ensemble_dim, 0)
           .flatten(1, ensemble_dim + 1)
        ) 
        features = F.avg_pool1d(
            features,
            self.kernel_shape,
            padding=self.padding,
            stride=self.stride
        ) 
        features = (
            features
           .unflatten(1, self.ensemble_shape + (channels,))
           .movedim(0, ensemble_dim)
        )
        return features

flat_dim = 6 * 128
train_flat = train_features.flatten(start_dim=1)
valid_flat = valid_features.flatten(start_dim=1)

model_mlp = torch.nn.Sequential(
    Linear(config, flat_dim, 64),
    torch.nn.ReLU(),
    Linear(config, 64, 6)
).to(config["device"])
optimizer_mlp = AdamW(model_mlp.parameters())

model = torch.nn.Sequential(
    Conv1D(
        config,
        6,
        5,
        16,
        init_multiplier=2 ** .5
    ),
    torch.nn.ReLU(),
    Conv1D(
        config,
        16,
        3,
        32,
    ),
    Pool1D(
        config,
        stride=2
    ),
    Linear(
        config,
        32,
        6,
        init_multiplier=2 ** .5
    )
)

def train_supervised(
    config: dict,
    get_loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    get_metric: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    model: torch.nn.Module,
    optimizer: Optimizer,
    train_features: torch.Tensor,
    train_values: torch.Tensor,
    valid_features: torch.Tensor,
    valid_values: torch.Tensor,
) -> dict:
    ensemble_shape = config["ensemble_shape"]
    if len(ensemble_shape) != 1:
        raise ValueError(f"The number of dimensions in the ensemble shape should be 1 for the  population size, but it is {len(ensemble_shape)}")

    population_size = ensemble_shape[0]
    config_local = dict(config)
    log = defaultdict(list)
    pbt_init(config_local, log)
    optimizer.update_config(config_local)

    best_valid_metric = -torch.inf
    progress_bar = tqdm.trange(config["steps_num"])
    steps_without_improvement = 0
    train_dataloader = get_dataloader_random_reshuffle(
        config,
        train_features,
        train_values
    )

    for step_id in progress_bar:        
        if step_id % config["valid_interval"] == 0:
            with torch.no_grad():
                validation_metric = evaluate_model(
                    config,
                    valid_features,
                    get_metric,
                    model,
                    valid_values
                ).nan_to_num(-torch.inf)
                log["validation metric"].append(validation_metric)
                print(
                    f"validation metric {validation_metric.max().cpu().item():.4f}"
                )

                best_last_metric, best_last_metric_id \
                    = log["validation metric"][-1].max(dim=-1)
                print(
                    f"Best last metric {best_last_metric.cpu().item():.2f}",
                    flush=True
                )
                if (
                    best_valid_metric + config["improvement_threshold"]
                ) < best_last_metric:
                    print(
                        f"New best metric",
                        flush=True
                    )
                    best_valid_metric = best_last_metric
                    steps_without_improvement = 0

                    #Save only best parameters.
                    log["best parameters"] = {
                        key: value[best_last_metric_id].clone()
                        for key, value in model.state_dict().items()
                    }
                else:
                    print(
                        f"Best metric {best_valid_metric.cpu().item():.2f}",
                        flush=True
                    )
                    steps_without_improvement += config["valid_interval"]
                    if steps_without_improvement > config[
                        "steps_without_improvement"
                    ]:
                        break

                if config["pbt"] and (len(log["validation metric"]) >= config[
                    "welch_sample_size"
                ]):
                    evaluations = torch.stack(
                        log["validation metric"][-config["welch_sample_size"]:]
                    )
                    pbt_update(
                        config_local, evaluations, log, optimizer.get_parameters()
                    )

                    optimizer.update_config(config_local)

        minibatch_features, minibatch_labels = next(train_dataloader)
        optimizer.zero_grad()

        predict = model(minibatch_features)

        loss = get_loss(predict, minibatch_labels).sum()
        loss.backward()
        optimizer.step()


    progress_bar.close()
    for key, value in log.items():
        if isinstance(value, list):
            log[key] = torch.stack(value)

    return log

optimizer = AdamW(model.parameters())

log_mlp = train_supervised(
    config, get_cross_entropy, get_accuracy, model_mlp, optimizer_mlp,
    train_flat, train_labels, valid_flat, valid_labels)

log_cnn = train_supervised(
    config,
    get_cross_entropy,
    get_accuracy,
    model,
    optimizer,
    train_features,
    train_labels,
    valid_features,
    valid_labels
)