"""
Minimal Weights & Biases helper.

Example:
    from src.utils import init_wandb_run

    run = init_wandb_run(
        {
            "learning_rate": 1e-3,
            "method": "u",
            "pair_strategy": "anchor_type1",
        }
    )
"""

import os

import wandb

DEFAULT_WANDB_ENTITY = "hayashinaofumi-waseda-university"
DEFAULT_WANDB_PROJECT = "sconf-learning"


def init_wandb_run(config, **kwargs):
    """Initialize a wandb run with repo-level defaults.

    Args:
        config (dict): Hyperparameters and metadata to log.
        **kwargs: Additional keyword arguments forwarded to `wandb.init()`.

    Returns:
        wandb.sdk.wandb_run.Run: Initialized wandb run.
    """
    entity = kwargs.pop("entity", os.getenv("WANDB_ENTITY", DEFAULT_WANDB_ENTITY))
    project = kwargs.pop("project", os.getenv("WANDB_PROJECT", DEFAULT_WANDB_PROJECT))

    return wandb.init(
        entity=entity,
        project=project,
        config=dict(config),
        **kwargs,
    )
