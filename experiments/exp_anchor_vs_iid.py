"""
アンカー型ペア生成 vs i.i.d Baseline の比較実験
Comparison experiment: anchor-type pair generation vs i.i.d baseline

エポックごとのテスト精度を CSV に保存する。
Saves per-epoch test accuracy to a CSV file.

使用例 / Usage:
    # i.i.d Baseline
    python experiments/exp_anchor_vs_iid.py --pair_strategy iid --seed 0 --epochs 3

    # アンカー型 Type1 / Anchor-type 1
    python experiments/exp_anchor_vs_iid.py --pair_strategy anchor_type1 --K 100 --n_pairs 30000 --seed 0 --epochs 3
"""
import argparse
import csv
import os
import warnings

import numpy as np
import torch

from src.data import prepare_mnist_data
from src.models import mlp_model
from src.losses import Sconf_loss
from src.engine import accuracy_check

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Sconf learning: anchor_type1 vs iid pair strategy comparison"
)
parser.add_argument('--method', choices=['u', 'abs', 'nn'], default='u',
                    help='Sconf loss variant (default: u)')
parser.add_argument('--pair_strategy', choices=['iid', 'anchor_type1'], default='iid',
                    help='Pair generation strategy (default: iid)')
parser.add_argument('--n_pairs', type=int, default=30000,
                    help='Total number of pairs for anchor_type1 (default: 30000)')
parser.add_argument('--K', type=int, default=100,
                    help='Number of anchors for anchor_type1 (default: 100)')
parser.add_argument('--K_pos', type=int, default=None,
                    help='Number of positive-class anchors (None = random selection)')
parser.add_argument('--K_neg', type=int, default=None,
                    help='Number of negative-class anchors (None = random selection)')
parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3,
                    help='Learning rate (default: 1e-3)')
parser.add_argument('-bs', '--batch_size', type=int, default=3000,
                    help='Batch size (default: 3000)')
parser.add_argument('-e', '--epochs', type=int, default=60,
                    help='Number of epochs (default: 60)')
parser.add_argument('-wd', '--weight_decay', type=float, default=1e-3,
                    help='Weight decay (default: 1e-3)')
parser.add_argument('--seed', type=int, default=0,
                    help='Random seed (default: 0)')
parser.add_argument('--output_dir', type=str, default='./outputs',
                    help='Directory to save results (default: ./outputs)')

args = parser.parse_args()

# ---------------------------------------------------------------------------
# Random seed
# ---------------------------------------------------------------------------
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

# ---------------------------------------------------------------------------
# Build pair_kwargs for the selected strategy
# ---------------------------------------------------------------------------
pair_kwargs = {}
if args.pair_strategy == 'anchor_type1':
    pair_kwargs['n_pairs'] = args.n_pairs
    pair_kwargs['K'] = args.K
    if args.K_pos is not None:
        pair_kwargs['K_pos'] = args.K_pos
    if args.K_neg is not None:
        pair_kwargs['K_neg'] = args.K_neg

# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------
print(f"[INFO] pair_strategy={args.pair_strategy}, method={args.method}, seed={args.seed}")
print(f"[INFO] pair_kwargs={pair_kwargs}")

v_train_loader, sconf_loader, test_loader, sd_loader, pair_loader, prior = prepare_mnist_data(
    batch_size=args.batch_size,
    pair_strategy=args.pair_strategy,
    pair_kwargs=pair_kwargs,
)

# ---------------------------------------------------------------------------
# Model and optimizer
# ---------------------------------------------------------------------------
model = mlp_model(input_dim=28 * 28, hidden_dim=500, output_dim=1).to(device)
optimizer = torch.optim.Adam(
    model.parameters(), weight_decay=args.weight_decay, lr=args.learning_rate
)

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
results = []  # list of (epoch, test_accuracy)

test_accuracy = accuracy_check(loader=test_loader, model=model).to("cpu")
print(f"Epoch: 0. Test Accuracy: {test_accuracy.numpy()[0]:.2f}%")
results.append((0, float(test_accuracy.numpy()[0])))

for epoch in range(1, args.epochs):
    # Learning rate schedule (mirrors demo.py)
    if epoch == 20:
        for pg in optimizer.param_groups:
            pg['lr'] /= 10
    if epoch == 40:
        for pg in optimizer.param_groups:
            pg['lr'] /= 10

    model.train()
    for images, sconf in sconf_loader:
        images, sconf = images.to(device), sconf.to(device)
        optimizer.zero_grad()
        outputs = model(images).to(device)
        loss = Sconf_loss(f=outputs, prior=prior, sconf=sconf, loss_name=args.method)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        test_accuracy = accuracy_check(loader=test_loader, model=model).to("cpu")
        acc = float(test_accuracy.numpy()[0])
    print(f"Epoch: {epoch}. Test Accuracy: {acc:.2f}%")
    results.append((epoch, acc))

# ---------------------------------------------------------------------------
# Save results to CSV
# ---------------------------------------------------------------------------
os.makedirs(args.output_dir, exist_ok=True)

if args.pair_strategy == 'anchor_type1':
    filename = f"anchor_type1_K{args.K}_npairs{args.n_pairs}_seed{args.seed}.csv"
else:
    filename = f"iid_npairs{args.n_pairs}_seed{args.seed}.csv"

output_path = os.path.join(args.output_dir, filename)
with open(output_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['epoch', 'test_accuracy'])
    writer.writerows(results)

print(f"\n[INFO] Results saved to: {output_path}")
