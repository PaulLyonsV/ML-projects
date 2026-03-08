import datasets
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch
import tqdm

config = {
    "dataset_path": "ylecun/mnist",
    "device": "cpu",
    "learning_rate": 1,
    "seed": 0,
    "steps_num": 100
}

mnist = datasets.load_dataset(
    config["dataset_path"]
).with_format(
    "torch",
    device=config["device"]
)
print(mnist)

train, test = (mnist[key] for key in ["train", "test"])
train_valid = train.train_test_split(
    seed=config["seed"],
    test_size=10_000
)

train, valid = (train_valid[key] for key in ["train", "test"])

train_features, train_labels = (train[key][:] for key in ["image", "label"])
for t in [train_features, train_labels]:
    print(t.shape, t.device, t.dtype)
    
def flatten_images(
    images: torch.Tensor,
    dtype=torch.float32,
    scale=1/255
) -> torch.Tensor:
    
    batch_size, channel_num, height, width = images.shape
    feature_dim = channel_num * height * width

    images = (
        images
       .reshape(batch_size, feature_dim)
       .to(dtype)
      * scale
    )

    return images

train_features = flatten_images(train_features)
print(train_features.shape, train_features.dtype)

def preprocess_dataset(
    dataset: datasets.Dataset,
    column_1s=False,
    device="cpu",
    features_dtype=torch.float32,
    features_scale=1 / 255,
    images_name="image",
    labels_name="label"
) -> tuple[torch.Tensor, torch.Tensor]:
    
    
        dataset = dataset.with_format("torch", device="cpu")
        feature_matrix = flatten_images(
        dataset[images_name][:],
        dtype=features_dtype,
        scale=features_scale
        )
        if column_1s:
            feature_matrix = torch.cat([feature_matrix, torch.ones_like(feature_matrix[:, 0])],
            axis=-1
            )

        labels = dataset[labels_name][:]

        return feature_matrix, labels

[
    (train_features, train_labels),
    (valid_features, valid_labels),
    (test_features, test_labels)
] = (
    preprocess_dataset(dataset, device=config["device"])
    for dataset in [train, valid, test]
)

for t in [
    train_features, train_labels,
    valid_features, valid_labels,
    test_features, test_labels
]:


    torch.save(
        {
            "train_features": train_features,
            "train_labels": train_labels,
            "valid_features": valid_features,
            "valid_labels": valid_labels,
            "test_features": test_features,
            "test_labels": test_labels
         },
        "data/mnist.pt"
    )
    
    