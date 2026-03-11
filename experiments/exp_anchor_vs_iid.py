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
import sys
import warnings

# Add project root to sys.path so that `src` is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from src.device import device
from src.data import prepare_mnist_data
from src.engine import train_sconf_one_run
from src.utils import save_training_curves

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
# Training
# ---------------------------------------------------------------------------
results = train_sconf_one_run(
    sconf_loader=sconf_loader,
    test_loader=test_loader,
    prior=prior,
    method=args.method,
    lr=args.learning_rate,
    weight_decay=args.weight_decay,
    epochs=args.epochs,
)

for r in results:
    loss_str = f"{r['train_loss']:.4f}" if not (r['train_loss'] != r['train_loss']) else '-'
    print(f"Epoch: {r['epoch']:3d}. Train Loss: {loss_str:>8s}  Test Accuracy: {r['test_accuracy']:.2f}%")

# ---------------------------------------------------------------------------
# Save results to CSV
# ---------------------------------------------------------------------------
os.makedirs(args.output_dir, exist_ok=True)

if args.pair_strategy == 'anchor_type1':
    filename = f"anchor_type1_K{args.K}_npairs{args.n_pairs}_seed{args.seed}.csv"
else:
    filename = f"iid_npairs{args.n_pairs}_seed{args.seed}.csv"

stem = filename.replace('.csv', '')
csv_path = os.path.join(args.output_dir, filename)
plot_path = os.path.join(args.output_dir, f"{stem}.png")

with open(csv_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['epoch', 'train_loss', 'test_accuracy'])
    writer.writeheader()
    writer.writerows(results)

plot_title = f"{args.pair_strategy} | method={args.method} | K={args.K} | n_pairs={args.n_pairs} | seed={args.seed}"
save_training_curves(results, plot_path, title=plot_title)

print(f"\n[INFO] CSV   saved to: {csv_path}")
print(f"[INFO] Plot  saved to: {plot_path}")
