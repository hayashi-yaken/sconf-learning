from .iid import create_iid_pairs
from .anchor_type1 import create_anchor_type1_pairs, create_anchor_sconf_data
from .anchor_type2 import create_anchor_type2_pairs

PAIR_STRATEGIES = {
    'iid': create_iid_pairs,
    'anchor_type1': create_anchor_type1_pairs,
    'anchor_type2': create_anchor_type2_pairs,
}

# Strategies that provide custom sconf training data for sconf_loader.
# iid is absent because src.data.pipeline handles it with the existing perm-based code.
SCONF_DATA_STRATEGIES = {
    'anchor_type1': create_anchor_sconf_data,
}


def get_pair_dataset(strategy, data, label, perm, **kwargs):
    """Select a pair generation strategy by name and return a PairDataset.

    Args:
        strategy (str): 'iid' | 'anchor_type1' | 'anchor_type2'
        data: Image data (numpy)
        label: Label data (numpy)
        perm: Shuffle index (torch.Tensor)
    """
    if strategy not in PAIR_STRATEGIES:
        raise ValueError(
            "Unknown strategy: {}. Choose from {}".format(strategy, list(PAIR_STRATEGIES.keys()))
        )
    return PAIR_STRATEGIES[strategy](data, label, perm, **kwargs)


def get_sconf_training_data(strategy, data, label, perm, **kwargs):
    """Returns flat (images, sconf_values) for building sconf_loader in anchor strategies.

    Returns None for 'iid' (src.data.pipeline falls back to its own perm-based computation).

    Args:
        strategy (str): 'iid' | 'anchor_type1' | 'anchor_type2'
        data, label, perm, **kwargs: Same as get_pair_dataset.

    Returns:
        (images, sconf_values) tuple, or None if strategy == 'iid'.
    """
    if strategy not in SCONF_DATA_STRATEGIES:
        return None
    return SCONF_DATA_STRATEGIES[strategy](data, label, perm, **kwargs)
