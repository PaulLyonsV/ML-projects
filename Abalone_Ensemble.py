import matplotlib.pyplot as plt
import torch
import datasets
import numpy as np
import os
os.makedirs("data", exist_ok=True) 

config = {
    "seed": 1,
    "train_size": .85
    }

torch.manual_seed(config["seed"])

path = "abalone.pt"
dataset = torch.load(path, weights_only=True)  

[dataset[key] for key in ["length", "diameter", "height", "whole_weight", "shucked_weight", "viscera_weight", "shell_weight",]]

features = torch.stack(
    [value for key, value in dataset.items() if key not in ["number_of_rings", "sex_id"]],
    dim=1
    )

torch.eye(3)[[0, 2, 0, 1, 1, 2]]

#one-hot encodes to use categorical sex data.
sex_id_one_hot = torch.eye(3)[dataset["sex_id"]]


# features_extended shape is (N, 10)
# 7 original feature vectors and the 3-d one-hot encoded sex features.
features_extended = torch.cat(
[features, sex_id_one_hot],
 dim=1
 )
features = features_extended
targets = dataset["number_of_rings"].to(torch.float32)

indices = torch.randperm(len(features))
train_size = int(len(features) * config["train_size"])
test_size = len(features) - train_size

test_indices = indices[train_size:]
train_indices = indices[:train_size]
    
(features_train, targets_train), (features_test, targets_test) = tuple(
    (features[split_indices], targets[split_indices])
    for split_indices in (train_indices, test_indices)
) 

torch.save(
{
"features_train": features_train,
"targets_train": targets_train,
"features_test": features_test,
"targets_test": targets_test,
},
"data/abalone_preprocessed.pt")

#solves for least weights using the method of least squares
weights = torch.linalg.lstsq(
features_train,
targets_train,
)[0]

predict_train = features_train @ weights
train_mse = torch.mean((targets_train - predict_train) ** 2)
predict_test = features_test @ weights
test_mse = torch.mean((targets_test - predict_test) ** 2)
train_argsort = torch.argsort(targets_train)

def plot_regression_values(
    values_predict: torch.Tensor,
    values_true: torch.Tensor
    ):

    values_argsort = torch.argsort(values_true)
dataset = datasets.load_dataset(
    "parquet",
    data_files="hf://datasets/mstz/abalone@refs/convert/parquet/abalone/train/*.parquet",
    split="train",
    )

sex_unique = set(dataset["sex"])
sex2id = {s: i for i, s in enumerate(sex_unique)}
print(sex2id)

def add_sex_id(row: dict) -> dict:
    row["sex_id"] = sex2id[row["sex"]]
    return row

dataset = dataset.map(
add_sex_id,
remove_columns="sex"
)

dataset = dataset.with_format("torch")

torch.save(
dataset[:],
"data/abalone.pt")

#Trained m different models on 3000 subsets each. 
#We use the average of the ensemble.
n_models_list = range(2, 101)
ensemble_mse_list = []
subset_size = 3000

for m in n_models_list:
    subset_indices = torch.randint(0, len(features_train), (m, subset_size))
    X = features_train[subset_indices]
    y = targets_train[subset_indices]
    #W shape: (m, 10, 1)
    W = torch.linalg.lstsq(X, y.unsqueeze(-1)).solution
    
    #z_test shape: (m, test_size)
    z_test = (features_test @ W.squeeze(-1).T).T
    #z_enstest shape: (test_size,)
    z_enstest = z_test.mean(dim=0)

    mse = torch.mean((targets_test - z_enstest) ** 2).item()
    ensemble_mse_list.append(mse)

plt.plot(n_models_list, ensemble_mse_list, label="Ensemble Test MSE")
plt.hlines(test_mse.item(), 2, 100, colors="red", linestyles="dashed", label="Single Model Test MSE")
plt.xlabel("Number of Models in Ensemble")
plt.ylabel("Test MSE")
plt.legend()
plt.show()
