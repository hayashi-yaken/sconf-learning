"""
Sweep experiment varying anchor count K from 1 to 100

Configuration:
    - method: u / abs / nn (all 3 variants)
    - K: 1 to 100 (step 1)
    - n_pairs: 30000 (fixed)
    - train_seeds: swept explicitly
    - pair_seeds: swept explicitly
    - epochs: 60 (fixed)
    - bias: none (K_pos / K_neg not specified)

Output:
    outputs/sweep_anchor_K/
        {method}_K{K:03d}_tseed{train_seed}_pseed{pair_seed}.csv
        {method}_K{K:03d}_tseed{train_seed}_pseed{pair_seed}.png
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
from src.utils import init_wandb_run, save_training_curves

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Sweep experiment: vary anchor count K from 1 to 100")
parser.add_argument(
    "--methods",
    nargs="+",
    choices=["u", "abs", "nn"],
    default=["u", "abs", "nn"],
    help="Sconf loss variants to sweep (default: all)",
)
parser.add_argument("--K_min", type=int, default=1, help="Minimum anchor count (default: 1)")
parser.add_argument("--K_max", type=int, default=100, help="Maximum anchor count (default: 100)")
parser.add_argument("--K_step", type=int, default=1, help="Step size for K sweep (default: 1)")
parser.add_argument("--n_pairs", type=int, default=30000, help="Total number of pairs (default: 30000)")
parser.add_argument(
    "--train_seeds",
    nargs="+",
    type=int,
    default=list(range(10)),
    help="Training seeds to sweep (default: 0 1 2 3 4 5 6 7 8 9)",
)
parser.add_argument(
    "--pair_seeds", nargs="+", type=int, default=[0], help="Pair-generation seeds to sweep (default: 0)"
)
parser.add_argument("-lr", "--learning_rate", type=float, default=1e-3, help="Learning rate (default: 1e-3)")
parser.add_argument("-bs", "--batch_size", type=int, default=3000, help="Batch size (default: 3000)")
parser.add_argument("-e", "--epochs", type=int, default=60, help="Number of epochs (default: 60)")
parser.add_argument("-wd", "--weight_decay", type=float, default=1e-3, help="Weight decay (default: 1e-3)")
parser.add_argument(
    "--output_dir",
    type=str,
    default="./outputs/sweep_anchor_K",
    help="Directory to save results (default: ./outputs/sweep_anchor_K)",
)
parser.add_argument("--skip_existing", action="store_true", help="Skip runs whose CSV already exists")
parser.add_argument(
    "--save_pair_csv", action="store_true", help="Save generated pair dataset metadata as CSV for each run"
)
parser.add_argument("--wandb", action="store_true", help="Log each run to Weights & Biases")
parser.add_argument(
    "--wandb_group", type=str, default="sweep_anchor_K", help="wandb group name (default: sweep_anchor_K)"
)

args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

# ---------------------------------------------------------------------------
# Sweep configuration
# ---------------------------------------------------------------------------
K_values = list(range(args.K_min, args.K_max + 1, args.K_step))
train_seeds = args.train_seeds
pair_seeds = args.pair_seeds
total_runs = len(args.methods) * len(K_values) * len(train_seeds) * len(pair_seeds)

print(f"[INFO] methods={args.methods}")
print(f"[INFO] K range: {args.K_min} ~ {args.K_max} (step={args.K_step}), {len(K_values)} values")
print(f"[INFO] train_seeds: {train_seeds}")
print(f"[INFO] pair_seeds: {pair_seeds}")
print(f"[INFO] total runs: {total_runs}")
print(f"[INFO] output_dir: {args.output_dir}")
print()

# ---------------------------------------------------------------------------
# Main sweep loop
# ---------------------------------------------------------------------------
run_idx = 0
for method, K, pair_seed, train_seed in itertools.product(args.methods, K_values, pair_seeds, train_seeds):
    run_idx += 1
    stem = f"{method}_K{K:03d}_tseed{train_seed}_pseed{pair_seed}"
    csv_path = os.path.join(args.output_dir, f"{stem}.csv")
    plot_path = os.path.join(args.output_dir, f"{stem}.png")
    pair_csv_path = os.path.join(args.output_dir, f"{stem}_pairs.csv") if args.save_pair_csv else None

    print(
        f"[{run_idx:4d}/{total_runs}] method={method}  K={K:3d}  " f"train_seed={train_seed}  pair_seed={pair_seed}",
        end="  ",
        flush=True,
    )

    # Skip if already done
    if args.skip_existing and os.path.exists(csv_path):
        print("skipped (already exists)")
        continue

    # Fix random seeds
    np.random.seed(train_seed)
    torch.manual_seed(train_seed)
    torch.cuda.manual_seed_all(train_seed)

    # Data preparation
    pair_kwargs = {"n_pairs": args.n_pairs, "K": K, "seed": pair_seed}
    _, sconf_loader, test_loader, _, _, prior = prepare_mnist_data(
        batch_size=args.batch_size,
        pair_strategy="anchor_type1",
        pair_kwargs=pair_kwargs,
        pair_csv_path=pair_csv_path,
    )

    wandb_run = None
    if args.wandb:
        group = f"{args.wandb_group}_{method}_K{K:03d}_pseed{pair_seed}"
        wandb_run = init_wandb_run(
            {
                "script": "experiments/sweep_anchor_K.py",
                "method": method,
                "pair_strategy": "anchor_type1",
                "K": K,
                "n_pairs": args.n_pairs,
                "train_seed": train_seed,
                "pair_seed": pair_seed,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "epochs": args.epochs,
                "save_pair_csv": args.save_pair_csv,
            },
            group=group,
            name=stem,
            tags=["anchor_type1", "sweep_anchor_K", method],
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

    if wandb_run is not None:
        for record in results:
            wandb_run.log(record, step=record["epoch"])
        best_acc = max(r["test_accuracy"] for r in results)
        wandb_run.summary["final_test_accuracy"] = results[-1]["test_accuracy"]
        wandb_run.summary["best_test_accuracy"] = best_acc
        wandb_run.summary["output_csv"] = csv_path
        wandb_run.summary["output_plot"] = plot_path
        if pair_csv_path is not None:
            wandb_run.summary["pair_csv"] = pair_csv_path

    final_acc = results[-1]["test_accuracy"]
    print(f"-> final test acc: {final_acc:.2f}%")

    # Save CSV
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "test_accuracy"])
        writer.writeheader()
        writer.writerows(results)

    # Save plot
    title = (
        f"anchor_type1 | method={method} | K={K} | n_pairs={args.n_pairs} | "
        f"train_seed={train_seed} | pair_seed={pair_seed}"
    )
    save_training_curves(results, plot_path, title=title)

    if wandb_run is not None:
        wandb_run.finish()

print(f"\n[INFO] All runs complete. Results saved to: {args.output_dir}")
