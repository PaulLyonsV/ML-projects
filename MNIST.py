from collections import defaultdict
from collections.abc import Generator, Sequence
import matplotlib.pyplot as plt
import scipy
import torch
import torch.nn.functional as F
import tqdm

from util_0926 import load_preprocessed_dataset

config = {
    "dataset_preprocessed_path": "data/mnist.pt",
    "device": "cpu",
    "ensemble_shape": (10,),
    "learning_rate": 1,
    "minibatch_size": 256,
    "seed": 1,
    "steps_num": 1000,
    "valid_interval": 10
}

torch.manual_seed(config["seed"])

(
    (train_features, train_labels),
    (valid_features, valid_labels),
    (test_features, test_labels)
) = load_preprocessed_dataset(
    config
)
    
def get_shuffled_indices(
    dataset_size: int,
    device="msp",
    ensemble_shape=(),
) -> torch.Tensor:
    
    total_shape = ensemble_shape + (dataset_size,)
    uniform = torch.rand(
        total_shape,
        device="cpu"
    )
    indices = uniform.argsort(dim=-1)

    return indices

get_shuffled_indices(5, ensemble_shape=(2, 4))


def get_random_reshuffler(
    dataset_size: int,
    minibatch_size: int,
    device="cpu",
    ensemble_shape=()
) -> Generator[torch.Tensor]:
#Generator function was given to us by our professor. 
# From my understanding, it efficiently samples each data point 
# in random order without replacement before 
# re-shuffling indices. 
    
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

    random_reshuffler = get_random_reshuffler(
        10,
        4,
        ensemble_shape=(3,)
    )

def get_dataloader_random_reshuffle(
    config: dict,
    features: torch.Tensor,
    labels: torch.Tensor
) -> Generator[tuple[torch.Tensor, torch.Tensor]]:
        for indices in get_random_reshuffler(
            len(labels),
            config["minibatch_size"],
            ensemble_shape=config["ensemble_shape"]
        ):
            yield features[indices], labels[indices]

train_dataloader = get_dataloader_random_reshuffle(
config,
train_features,
train_labels
)


def get_accuracy(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
          
    labels_predict = logits.argmax(dim=-1)
    accuracy = (labels == labels_predict).to(torch.float32).mean(dim=-1)

    return accuracy

minibatch_features, minibatch_labels = next(train_dataloader)

weights = torch.zeros(
    config["ensemble_shape"] + (valid_features.shape[1], 10),
    device=config["device"]
)

bias = torch.zeros_like(weights[..., 0:1, :])


minibatch_logits = minibatch_features @ weights + bias[..., :]
valid_logits = valid_features @ weights + bias[..., :]

print(
    get_accuracy(
        minibatch_logits,
        minibatch_labels,
    ),
    get_accuracy(
        valid_logits,
        valid_labels,
    ),
    sep='\n'
)


def get_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    return F.cross_entropy(
        logits.movedim((-2, -1), (0, 1)),
        labels.broadcast_to(logits.shape[:-1]).movedim(-1, 0),
        reduction="none"
    ).mean(dim=0)

print(
    get_cross_entropy(
        minibatch_logits,
        minibatch_labels,
    ),
    get_cross_entropy(
        valid_logits,
        valid_labels,
    ),
    sep='\n'
)


def train_logistic_regression(
    config: dict,
    label_num: int,
    train_dataloader: Generator[tuple[torch.Tensor, torch.Tensor]],
    valid_features: torch.Tensor,
    valid_labels: torch.Tensor,
    use_bias=True
) -> dict:
    
    device = valid_features.device
    features_dtype = valid_features.dtype
    output = defaultdict(list)

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
        config["ensemble_shape"] + (valid_features.shape[1], label_num),
        device=device,
        dtype=features_dtype,
        requires_grad=True
    )
    if use_bias:
        bias = torch.zeros_like(weights[..., 0:1, :])

    optimizer = torch.optim.SGD([weights], lr=config["learning_rate"])

    for minibatch_features, minibatch_labels in train_dataloader:
        minibatch_size = minibatch_labels.shape[-1]
        optimizer.zero_grad()
        logits = minibatch_features @ weights
        if use_bias:
            logits = logits + bias

        train_accuracies_step += get_accuracy(
            logits.detach(),
            minibatch_labels,
        ) * minibatch_size
        loss = get_cross_entropy(
            logits,
            minibatch_labels,
        )
        loss.sum().backward()
        optimizer.step()

        train_losses_step += loss.detach() * minibatch_size
        train_entries += minibatch_size

        progress_bar.update()
        step_id += 1
        if step_id % config["valid_interval"] == 0:
            
            with torch.no_grad():
                logits = valid_features @ weights
                if use_bias:
                    logits = logits + bias
                probs = torch.softmax(logits, dim=-1)
                prob_avg = probs.mean(dim=0)
                

            valid_accuracy = get_accuracy(
                logits,
                valid_labels,
            )
            
            valid_prob_accuracy = get_accuracy(prob_avg, valid_labels)  
                    
            valid_loss = get_cross_entropy(
                logits,
                valid_labels,
            )

            output["training accuracy"].append(
                (train_accuracies_step / train_entries).cpu()
            )
            output["training cross-entropy"].append(
                (train_losses_step / train_entries).cpu()
            )
            output["training steps"].append(step_id)
            output["validation accuracy"].append(valid_accuracy.cpu())
            output["validation ensemble accuracy"].append(valid_prob_accuracy.cpu())
            output["validation cross-entropy"].append(valid_loss.cpu())

            train_accuracies_step.zero_()
            train_entries = 0
            train_losses_step.zero_()

        if step_id >= config["steps_num"]:
            for key in (
                "training accuracy",
                "training cross-entropy",
                "validation accuracy",
                "validation ensemble accuracy",
                "validation cross-entropy"
            ):
                output[key] = torch.stack(output[key])

            output["weights"] = weights
            progress_bar.close()

            return output
        
output = train_logistic_regression(
    config,
    10,
    train_dataloader,
    valid_features,
    valid_labels
)

for key in (
    "training accuracy",
    "training cross-entropy",
    "validation accuracy",
    "validation ensemble accuracy",
    "validation cross-entropy"
):
    print(key, output[key][-1])
    
    
def line_plot_confidence_band(
    x: Sequence,
    y: torch.Tensor,
    color=None,
    confidence_level=.99,
    label="",
    opacity=.2
):
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


line_plot_confidence_band(
    output["training steps"],
    output["validation accuracy"],
    color="blue",
    label="validation accuracy"
)

plt.plot(
    output["training steps"],
    output["validation ensemble accuracy"],
    color="red",
    label="ensemble validation accuracy"
)


plt.legend()
plt.xlabel("Training steps")
plt.ylim(.85, .95)
plt.show()
plt.close()

