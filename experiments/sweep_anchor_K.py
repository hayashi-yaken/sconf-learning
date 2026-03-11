"""
アンカー数 K を 1〜100 でスイープする比較実験
Sweep experiment varying anchor count K from 1 to 100

設定 / Configuration:
    - method: u / abs / nn（全3種）
    - K: 1〜100（1刻み）
    - n_pairs: 30000（固定）
    - seed: 0〜9（10回）
    - epochs: 60（固定）
    - bias: なし（K_pos / K_neg 未指定）

出力 / Output:
    outputs/sweep_anchor_K/
        {method}_K{K:03d}_seed{seed}.csv
        {method}_K{K:03d}_seed{seed}.png
"""
import argparse
import csv
import itertools
import os
import sys
import warnings

# Add project root to sys.path so that `src` is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from src.data import prepare_mnist_data
from src.engine import train_sconf_one_run
from src.utils import save_training_curves

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Sweep experiment: vary anchor count K from 1 to 100"
)
parser.add_argument('--methods', nargs='+', choices=['u', 'abs', 'nn'],
                    default=['u', 'abs', 'nn'],
                    help='Sconf loss variants to sweep (default: all)')
parser.add_argument('--K_min', type=int, default=1,
                    help='Minimum anchor count (default: 1)')
parser.add_argument('--K_max', type=int, default=100,
                    help='Maximum anchor count (default: 100)')
parser.add_argument('--K_step', type=int, default=1,
                    help='Step size for K sweep (default: 1)')
parser.add_argument('--n_pairs', type=int, default=30000,
                    help='Total number of pairs (default: 30000)')
parser.add_argument('--n_seeds', type=int, default=10,
                    help='Number of seeds (default: 10)')
parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3,
                    help='Learning rate (default: 1e-3)')
parser.add_argument('-bs', '--batch_size', type=int, default=3000,
                    help='Batch size (default: 3000)')
parser.add_argument('-e', '--epochs', type=int, default=60,
                    help='Number of epochs (default: 60)')
parser.add_argument('-wd', '--weight_decay', type=float, default=1e-3,
                    help='Weight decay (default: 1e-3)')
parser.add_argument('--output_dir', type=str, default='./outputs/sweep_anchor_K',
                    help='Directory to save results (default: ./outputs/sweep_anchor_K)')
parser.add_argument('--skip_existing', action='store_true',
                    help='Skip runs whose CSV already exists')

args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

# ---------------------------------------------------------------------------
# Sweep configuration
# ---------------------------------------------------------------------------
K_values = list(range(args.K_min, args.K_max + 1, args.K_step))
seeds = list(range(args.n_seeds))
total_runs = len(args.methods) * len(K_values) * len(seeds)

print(f"[INFO] methods={args.methods}")
print(f"[INFO] K range: {args.K_min} ~ {args.K_max} (step={args.K_step}), {len(K_values)} values")
print(f"[INFO] seeds: {seeds}")
print(f"[INFO] total runs: {total_runs}")
print(f"[INFO] output_dir: {args.output_dir}")
print()

# ---------------------------------------------------------------------------
# Main sweep loop
# ---------------------------------------------------------------------------
run_idx = 0
for method, K, seed in itertools.product(args.methods, K_values, seeds):
    run_idx += 1
    stem = f"{method}_K{K:03d}_seed{seed}"
    csv_path = os.path.join(args.output_dir, f"{stem}.csv")
    plot_path = os.path.join(args.output_dir, f"{stem}.png")

    print(f"[{run_idx:4d}/{total_runs}] method={method}  K={K:3d}  seed={seed}", end="  ", flush=True)

    # Skip if already done
    if args.skip_existing and os.path.exists(csv_path):
        print("skipped (already exists)")
        continue

    # Fix random seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Data preparation
    pair_kwargs = {'n_pairs': args.n_pairs, 'K': K}
    _, sconf_loader, test_loader, _, _, prior = prepare_mnist_data(
        batch_size=args.batch_size,
        pair_strategy='anchor_type1',
        pair_kwargs=pair_kwargs,
    )

    # Training
    results = train_sconf_one_run(
        sconf_loader=sconf_loader,
        test_loader=test_loader,
        prior=prior,
        method=method,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
    )

    final_acc = results[-1]['test_accuracy']
    print(f"-> final test acc: {final_acc:.2f}%")

    # Save CSV
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['epoch', 'train_loss', 'test_accuracy'])
        writer.writeheader()
        writer.writerows(results)

    # Save plot
    title = f"anchor_type1 | method={method} | K={K} | n_pairs={args.n_pairs} | seed={seed}"
    save_training_curves(results, plot_path, title=title)

print(f"\n[INFO] All runs complete. Results saved to: {args.output_dir}")
