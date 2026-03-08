import datasets
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch
import numpy as np
import os

#We compare two error metrics, Brier and Cross Entropy, in a 
# supervised learning task on the Cifar image dataset. For 
# practice, we updated weights manually rather than using an optimizer.

config = {
    "dataset_path": "uoft-cs/cifar10",
    "device": "cpu",       
    "learning_rates": [10 ** i for i in [-2, -1.5, -1, -0.5, 0, 0.5, 1]],
    "steps_num": 100,
    "seed": 0,
    "save_dir": "./cifar_tensors"
    }

torch.manual_seed(config["seed"])

cifar = datasets.load_dataset(
    config["dataset_path"]
).with_format(
    "torch",
    device=config["device"]
)

train_full, test = cifar["train"], cifar["test"]

train_valid = train_full.train_test_split(
    seed=config["seed"],
    test_size= len(test)
)

train, valid = train_valid["train"], train_valid["test"]

def flatten_images(images, dtype=torch.float32, scale=1/255.0):
    N, C, H, W = images.shape
    return (images.reshape(N, C*H*W).to(dtype) * scale)

def to_tensors(dataset, device="cpu"):
    dataset = dataset.with_format("torch", device=device)
    features = flatten_images(dataset["img"][:])
    labels_int = dataset["label"][:].to(torch.long)  
    num_classes = 10
    labels_onehot = F.one_hot(labels_int, num_classes=num_classes).float()
    return features, labels_int, labels_onehot

train_features, train_labels_int, train_labels_onehot = to_tensors(train, config["device"])
valid_features, valid_labels_int, valid_labels_onehot = to_tensors(valid, config["device"])
test_features, test_labels_int, test_labels_onehot = to_tensors(test, config["device"])

os.makedirs(config["save_dir"], exist_ok=True)

config["save_dir"] = "./cifar_tensors"

torch.save(
    {
        "train": (train_features, train_labels_int, train_labels_onehot),
        "valid": (valid_features, valid_labels_int, valid_labels_onehot),
        "test": (test_features, test_labels_int, test_labels_onehot),
    }, f"{config['save_dir']}/cifar_tensors.pt"
)

save_path = os.path.join(config["save_dir"], "cifar_tensors.pt")

if os.path.exists(save_path):
    data = torch.load(save_path)
    train_features, train_labels_int, train_labels_onehot = data["train"]
    valid_features, valid_labels_int, valid_labels_onehot = data["valid"]
    test_features,  test_labels_int,  test_labels_onehot  = data["test"]

else:
    train_features, train_labels_int, train_labels_onehot = to_tensors(train, config["device"])
    valid_features, valid_labels_int, valid_labels_onehot = to_tensors(valid, config["device"])
    test_features,  test_labels_int,  test_labels_onehot  = to_tensors(test, config["device"])

    os.makedirs(config["save_dir"], exist_ok=True)
    torch.save({
        "train": (train_features, train_labels_int, train_labels_onehot),
        "valid": (valid_features, valid_labels_int, valid_labels_onehot),
        "test":  (test_features,  test_labels_int,  test_labels_onehot)
    }, save_path)


def get_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    labels_predict = logits.argmax(dim=-1)
    accuracy = (labels == labels_predict).to(torch.float32).mean(dim=-1)
    return accuracy


def get_cross_entropy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(
        logits.movedim((-2, -1), (0, 1)),
        labels.broadcast_to(logits.shape[:-1]).movedim(-1, 0),
        reduction="none"
    ).mean(dim=0)

def get_brier_score(logits: torch.Tensor, labels_onehot: torch.Tensor) -> torch.Tensor:
    probs = F.softmax(logits, dim=-1)
    return ((probs - labels_onehot[None, ...]) ** 2).mean(dim=(-2, -1))

def compute_logits(X, W, b):
    #(batch, feature dim) X (ensemble dim, feature dim, target dim) -> (ensemble dim, batch, target dim)
    return torch.einsum("nd,edc->enc", X, W) + b[:, None, :]

num_lrs = len(config["learning_rates"])
N, D = train_features.shape
num_classes = 10

lr = torch.tensor(config["learning_rates"], device=config["device"])
W = torch.zeros(num_lrs, D, num_classes, device = config["device"])
b = torch.zeros(num_lrs, num_classes, device = config["device"])

results = {}

for loss_type in ["nll", "brier"]:
    # W shape: (7, 3072, 10) (learning rates, feature dim, target dim)
    W = torch.zeros_like(W, requires_grad = True)
    b = torch.zeros_like(b, requires_grad = True)
        
    for step in range(config["steps_num"]):
        logits = compute_logits(train_features, W, b)
            
        if loss_type == "nll":
            loss = get_cross_entropy(logits, train_labels_int)
        elif loss_type == "brier":
            loss = get_brier_score(logits, train_labels_onehot)
            
        total_loss = loss.sum()
        
        total_loss.backward()

        with torch.no_grad():
            W -= lr[:, None, None] * W.grad
            b -= lr[:, None] * b.grad
            W.grad.zero_()
            b.grad.zero_()

    with torch.no_grad():
        valid_logits = compute_logits(valid_features, W, b)
        val_acc = get_accuracy(valid_logits, valid_labels_int)
        
    results[loss_type] = val_acc.detach().numpy()


plt.plot(np.log10(lr.cpu()), results["nll"], label="NLL Loss", marker='o')
plt.plot(np.log10(lr.cpu()), results["brier"], label="Brier Loss", marker='s')
plt.ylabel("Accuracy")
plt.title("Accuracy vs Learning Rate")
plt.legend()
plt.show()