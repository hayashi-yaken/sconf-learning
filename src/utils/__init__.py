def save_training_curves(*args, **kwargs):
    from .plotting import save_training_curves as _save_training_curves

    return _save_training_curves(*args, **kwargs)


def init_wandb_run(*args, **kwargs):
    from .wandb import init_wandb_run as _init_wandb_run

    return _init_wandb_run(*args, **kwargs)
