"""
Anchor-type 1 pair generation

Fix an anchor set {a_1, ..., a_K} and assign multiple partner instances
to each anchor a_i to form pairs.
This results in a non-i.i.d structure where the same instance appears in multiple pairs.
"""
import numpy as np

from .iid import PairDataset


def _generate_anchor_pairs(data, label, n_pairs, K, K_pos, K_neg, true_labels, conf):
    """Core pair generation logic for anchor-type 1.

    Returns:
        x0 (np.ndarray): anchor images (n_pairs, 1, 28, 28)
        x1 (np.ndarray): partner images (n_pairs, 1, 28, 28)
        sconf_vals (np.ndarray): raw sconf values per pair (n_pairs,)
        label_sd (np.ndarray): bernoulli-sampled binary labels (n_pairs,)
    """
    N = len(data)

    # Select anchor indices
    if K_pos is not None or K_neg is not None:
        if true_labels is None:
            raise ValueError("true_labels is required when K_pos or K_neg is specified")
        pos_idx = np.where(true_labels == 1)[0]
        neg_idx = np.where(true_labels == -1)[0]

        k_pos = K_pos if K_pos is not None else (K - (K_neg or 0))
        k_neg = K_neg if K_neg is not None else (K - k_pos)

        anchor_pos = np.random.choice(pos_idx, size=k_pos, replace=len(pos_idx) < k_pos)
        anchor_neg = np.random.choice(neg_idx, size=k_neg, replace=len(neg_idx) < k_neg)
        anchor_idx = np.concatenate([anchor_pos, anchor_neg])
    else:
        anchor_idx = np.random.choice(N, size=K, replace=False)

    # Distribute n_pairs evenly across anchors
    M_base = n_pairs // K
    remainder = n_pairs % K

    partner_pool = np.arange(N)

    x0_list, x1_list, sconf_list, label_list = [], [], [], []

    for i, a_i in enumerate(anchor_idx):
        n_i = M_base + (1 if i < remainder else 0)

        # Sample partners with replacement
        partner_idx = np.random.choice(partner_pool, size=n_i, replace=True)

        # Compute pair labels
        if conf is not None:
            # sconf(a_i, x_j) = p_a * p_j + (1 - p_a) * (1 - p_j)
            p_a = conf[a_i]
            p_j = conf[partner_idx]
            sv = (p_a * p_j + (1 - p_a) * (1 - p_j)).astype(np.float32)
            pl = np.random.binomial(1, sv).astype(np.int32)
        else:
            # Fallback: use anchor's sd_label, sconf set to 0.5 (uninformative)
            sv = np.full(n_i, 0.5, dtype=np.float32)
            pl = np.full(n_i, int(label[a_i]), dtype=np.int32)

        x0_list.append(np.tile(data[a_i], (n_i, 1, 1, 1)))
        x1_list.append(data[partner_idx])
        sconf_list.append(sv)
        label_list.append(pl)

    return (
        np.concatenate(x0_list).astype(np.float32),
        np.concatenate(x1_list).astype(np.float32),
        np.concatenate(sconf_list).astype(np.float32),
        np.concatenate(label_list).astype(np.int32),
    )


def create_anchor_type1_pairs(data, label, perm, n_pairs=30000, K=100,
                        K_pos=None, K_neg=None,
                        true_labels=None, conf=None, **kwargs):
    """Returns a PairDataset for anchor-type 1 (used for pair_loader).

    Args:
        data (np.ndarray): Image data (N, 1, 28, 28) float32
        label (np.ndarray): sd_label (N,) -- fallback when conf is not provided
        perm: Unused. Accepted only to keep a unified interface with iid.
        n_pairs (int): Total number of pairs (default: 30000)
        K (int): Number of anchors (default: 100)
        K_pos (int or None): Number of positive-class anchors. None -> random selection.
        K_neg (int or None): Number of negative-class anchors. None -> random selection.
        true_labels (np.ndarray or None): ±1 class labels (required when K_pos or K_neg is specified)
        conf (np.ndarray or None): Per-sample confidence scores (N,) used for pair label computation.

    Returns:
        PairDataset: PairDataset containing (x0, x1, label) pairs.
    """
    x0, x1, _, label_sd = _generate_anchor_pairs(
        data, label, n_pairs, K, K_pos, K_neg, true_labels, conf
    )
    return PairDataset(x0, x1, label_sd)


def create_anchor_sconf_data(data, label, perm, n_pairs=30000, K=100,
                             K_pos=None, K_neg=None,
                             true_labels=None, conf=None, **kwargs):
    """Returns flat (images, sconf_values) training data reflecting anchor pair structure.

    Unrolls each pair (anchor, partner) into two independent training examples,
    so that anchors appearing in many pairs contribute proportionally more to the loss.

    Args:
        data, label, perm, n_pairs, K, K_pos, K_neg, true_labels, conf:
            Same as create_anchor_pairs.

    Returns:
        images (np.ndarray): (2 * n_pairs, 1, 28, 28) -- anchor + partner images
        sconf_values (np.ndarray): (2 * n_pairs,) -- corresponding raw sconf values
    """
    x0, x1, sconf_vals, _ = _generate_anchor_pairs(
        data, label, n_pairs, K, K_pos, K_neg, true_labels, conf
    )
    images = np.concatenate([x0, x1], axis=0).astype(np.float32)
    sconf_flat = np.concatenate([sconf_vals, sconf_vals], axis=0).astype(np.float32)
    return images, sconf_flat
