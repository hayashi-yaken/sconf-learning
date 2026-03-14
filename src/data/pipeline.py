import numpy as np
import torch

from src.data.mnist import load_mnist_binary
from src.device import device
from src.losses import logistic
from src.models import mlp_model
from src.pairing.factory import get_pair_dataset, get_sconf_training_data


def generate_confidence_scores(train_loader, v_train_loader):
    """Train an auxiliary MLP with true labels and return per-sample confidence."""
    model = mlp_model(input_dim=28 * 28, hidden_dim=500, output_dim=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    for _ in range(10):
        for images, label in train_loader:
            images, label = images.to(device), label.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = logistic(f=outputs, label=label)
            loss.backward()
            optimizer.step()

    conf = torch.Tensor([]).to(device)
    for images, label in v_train_loader:
        images, label = images.to(device), label.to(device)
        scores = model(images).detach().to(device)
        conf = torch.cat((conf, torch.sigmoid(scores).to(device)), 0).to(device)
    return conf


def prepare_mnist_data(batch_size, pair_strategy='iid', pair_kwargs=None, pair_csv_path=None):
    """MNIST data pipeline.

    Args:
        batch_size (int): Batch size
        pair_strategy (str): Pair generation strategy 'iid' | 'anchor_type1' | 'anchor_type2'
        pair_kwargs (dict or None): Additional keyword arguments passed to the pair generation function
        pair_csv_path (str or None): If provided, save pair dataset metadata to this CSV path.

    Returns:
        v_train_loader, sconf_loader, test_loader, sd_loader, pair_loader, prior
    """
    if pair_kwargs is None:
        pair_kwargs = {}

    image, labels, test_image, test_labels = load_mnist_binary()

    train_dataset = torch.utils.data.TensorDataset(image, labels)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=False
    )
    v_train_loader = train_loader
    prior = torch.Tensor([0.7])

    conf = generate_confidence_scores(train_loader, v_train_loader)

    perm = torch.randperm(len(labels))
    sconf = torch.zeros(len(labels)).to(device)
    for i in range(int(np.floor(len(labels) / 2))):
        sconf[perm[2 * i]] = (
            conf[perm[2 * i]] * conf[perm[2 * i + 1]]
            + (1 - conf[perm[2 * i]]) * (1 - conf[perm[2 * i + 1]])
        )
        sconf[perm[2 * i + 1]] = sconf[perm[2 * i]]
    if len(labels) % 2 == 1:
        sconf[perm[len(labels) - 1]] = 0.5

    anchor_sconf = get_sconf_training_data(
        pair_strategy,
        image.to("cpu").numpy(),
        labels.numpy(),
        perm,
        conf=conf.to("cpu").numpy(),
        true_labels=labels.numpy(),
        **pair_kwargs
    )
    if anchor_sconf is not None:
        sconf_images, sconf_vals = anchor_sconf
        sconf_dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(sconf_images).to(device),
            torch.from_numpy(sconf_vals).to(device),
        )
    else:
        sconf_dataset = torch.utils.data.TensorDataset(image, sconf.to(device))
    sconf_loader = torch.utils.data.DataLoader(
        dataset=sconf_dataset, batch_size=batch_size, shuffle=True
    )

    sd_label = sconf.clone().detach().to(device)
    for i in range(int(np.floor(len(labels) / 2))):
        sd_label[perm[2 * i]] = torch.bernoulli(sd_label[perm[2 * i]])
        sd_label[perm[2 * i + 1]] = sd_label[perm[2 * i]]
    sd_dataset = torch.utils.data.TensorDataset(image, sd_label.to(device))
    sd_loader = torch.utils.data.DataLoader(
        dataset=sd_dataset, batch_size=batch_size, shuffle=True
    )

    pair_dataset = get_pair_dataset(
        pair_strategy,
        image.to("cpu").numpy(),
        sd_label.to("cpu").numpy(),
        perm,
        conf=conf.to("cpu").numpy(),
        true_labels=labels.numpy(),
        **pair_kwargs
    )
    if pair_csv_path is not None and hasattr(pair_dataset, "save_csv"):
        pair_dataset.save_csv(pair_csv_path)
    pair_loader = torch.utils.data.DataLoader(
        pair_dataset, batch_size=batch_size, shuffle=True
    )

    test_dataset = torch.utils.data.TensorDataset(test_image, test_labels)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=False
    )

    return v_train_loader, sconf_loader, test_loader, sd_loader, pair_loader, prior
