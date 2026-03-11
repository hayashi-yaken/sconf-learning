"""
Sconf 学習の1回分のトレーニングループ
Single training run for Sconf-based loss methods

複数実験で共通して使える学習関数を提供する。
Provides a reusable training function shared across experiment scripts.
"""
import torch

from src.models import mlp_model
from src.losses import Sconf_loss
from src.engine.evaluate import accuracy_check

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_sconf_one_run(sconf_loader, test_loader, prior,
                        method='u', lr=1e-3, weight_decay=1e-3, epochs=60):
    """Sconf 損失を使った1回分の学習を実行し、エポックごとの記録を返す。
    Run a single training session with Sconf loss and return per-epoch records.

    Args:
        sconf_loader: Sconf 値付きの訓練データローダー /
            Training data loader with per-sample sconf values.
        test_loader: テストデータローダー /
            Test data loader.
        prior (torch.Tensor): 事前確率 /
            Class prior probability.
        method (str): 損失バリアント 'u' | 'abs' | 'nn' (default: 'u') /
            Loss variant 'u' | 'abs' | 'nn' (default: 'u').
        lr (float): 学習率 (default: 1e-3) /
            Learning rate (default: 1e-3).
        weight_decay (float): 重み減衰 (default: 1e-3) /
            Weight decay (default: 1e-3).
        epochs (int): エポック数 (default: 60) /
            Number of training epochs (default: 60).

    Returns:
        list[dict]: エポックごとの記録。各 dict は以下のキーを持つ。 /
            Per-epoch records. Each dict has the following keys:
            - 'epoch' (int)
            - 'train_loss' (float): エポック平均 train loss / Mean train loss
            - 'test_accuracy' (float): テスト精度（%） / Test accuracy (%)
    """
    model = mlp_model(input_dim=28 * 28, hidden_dim=500, output_dim=1).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), weight_decay=weight_decay, lr=lr
    )

    results = []

    # Epoch 0: initial accuracy before training
    model.eval()
    with torch.no_grad():
        acc = float(accuracy_check(loader=test_loader, model=model).to("cpu").numpy()[0])
    results.append({'epoch': 0, 'train_loss': float('nan'), 'test_accuracy': acc})

    for epoch in range(1, epochs):
        # Learning rate schedule (step decay at epoch 20 and 40)
        if epoch == 20:
            for pg in optimizer.param_groups:
                pg['lr'] /= 10
        if epoch == 40:
            for pg in optimizer.param_groups:
                pg['lr'] /= 10

        model.train()
        total_loss, n_batches = 0.0, 0
        for images, sconf in sconf_loader:
            images, sconf = images.to(device), sconf.to(device)
            optimizer.zero_grad()
            outputs = model(images).to(device)
            loss = Sconf_loss(f=outputs, prior=prior, sconf=sconf, loss_name=method)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        train_loss = total_loss / n_batches

        model.eval()
        with torch.no_grad():
            acc = float(accuracy_check(loader=test_loader, model=model).to("cpu").numpy()[0])
        results.append({'epoch': epoch, 'train_loss': train_loss, 'test_accuracy': acc})

    return results
