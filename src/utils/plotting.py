"""
Training curve visualization utilities

Saves per-epoch train loss and test accuracy as plot images.
"""
import os

import matplotlib.pyplot as plt


def save_training_curves(results, output_path, title=None):
    """Save training curves (train loss / test accuracy) as a plot image.

    Args:
        results (list[dict]): Per-epoch records. Each element is a dict with the following keys:
            - 'epoch' (int): Epoch number
            - 'train_loss' (float): Mean train loss for the epoch
            - 'test_accuracy' (float): Test accuracy (%)
        output_path (str): Destination file path (including extension, e.g. .png).
        title (str or None): Plot title. If None, no title is shown.
    """
    epochs = [r['epoch'] for r in results]
    train_losses = [r['train_loss'] for r in results]
    test_accuracies = [r['test_accuracy'] for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Train loss
    ax1.plot(epochs, train_losses, color='steelblue', linewidth=1.5)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Train Loss')
    ax1.set_title('Train Loss')
    ax1.grid(True, alpha=0.3)

    # Test accuracy
    ax2.plot(epochs, test_accuracies, color='darkorange', linewidth=1.5)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Test Accuracy (%)')
    ax2.set_title('Test Accuracy')
    ax2.grid(True, alpha=0.3)

    if title:
        fig.suptitle(title, fontsize=13)

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
