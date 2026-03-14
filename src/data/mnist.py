import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms


def load_mnist_binary(root='./data/mnist'):
    """Load MNIST and binarize labels (0-6 -> +1, 7-9 -> -1)."""
    ordinary_train_dataset = dsets.MNIST(
        root=root, train=True, transform=transforms.ToTensor(), download=True
    )
    train_loader = torch.utils.data.DataLoader(
        dataset=ordinary_train_dataset,
        batch_size=len(ordinary_train_dataset),
        shuffle=False,
    )
    train_image, train_labels = None, None
    for _, (train_image, train_labels) in enumerate(train_loader):
        continue
    for i in range(len(train_labels)):
        train_labels[i] = 1 if train_labels[i] < 7 else -1

    test_dataset = dsets.MNIST(
        root=root, train=False, transform=transforms.ToTensor()
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=len(test_dataset),
        shuffle=False,
    )
    test_image, test_labels = None, None
    for _, (test_image, test_labels) in enumerate(test_loader):
        continue
    for i in range(len(test_labels)):
        test_labels[i] = 1 if test_labels[i] < 7 else -1

    return train_image, train_labels, test_image, test_labels
