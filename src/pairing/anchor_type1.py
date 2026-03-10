"""
アンカー型 Type1 のペア生成 / Anchor-type 1 pair generation

アンカー集合 {a_1, ..., a_K} を固定し、
各アンカー a_i に対して複数のインスタンス {x_1, ...} を割り当ててペアを作る。
同一インスタンスが複数ペアに重複する非 i.i.d 構造になる。

Fix an anchor set {a_1, ..., a_K} and assign multiple partner instances
to each anchor a_i to form pairs.
This results in a non-i.i.d structure where the same instance appears in multiple pairs.
"""
import numpy as np

from .iid import PairDataset


def create_anchor_type1_pairs(data, label, perm, n_pairs=30000, K=100,
                        K_pos=None, K_neg=None,
                        true_labels=None, conf=None, **kwargs):
    """アンカー型 Type1 のペア生成。
    Anchor-type 1 pair generation.

    アンカー集合 {a_1, ..., a_K} を固定し、各アンカーに複数のパートナーを
    割り当ててペアを作る（同一インスタンスが複数ペアに重複する非 i.i.d 構造）。
    Fix an anchor set {a_1, ..., a_K} and assign multiple partners to each anchor,
    creating a non-i.i.d structure where the same instance appears in multiple pairs.

    Args:
        data (np.ndarray): 画像データ (N, 1, 28, 28) float32 /
            Image data (N, 1, 28, 28) float32
        label (np.ndarray): sd_label (N,) -- conf がない場合の fallback /
            sd_label (N,) -- fallback when conf is not provided
        perm: 未使用。iid との interface 統一のため受け取るだけ /
            Unused. Accepted only to keep a unified interface with iid.
        n_pairs (int): 総ペア数（デフォルト 30000）/ Total number of pairs (default: 30000)
        K (int): アンカー数（デフォルト 100）/ Number of anchors (default: 100)
        K_pos (int or None): 正例アンカー数。None → ランダム選択 /
            Number of positive-class anchors. None → random selection.
        K_neg (int or None): 負例アンカー数。None → ランダム選択 /
            Number of negative-class anchors. None → random selection.
        true_labels (np.ndarray or None): ±1 ラベル (K_pos/K_neg 指定時に必須) /
            ±1 class labels (required when K_pos or K_neg is specified)
        conf (np.ndarray or None): per-sample 信頼度 (N,) -- ペアラベル計算に使用 /
            Per-sample confidence scores (N,) used for pair label computation.

    Returns:
        PairDataset: (x0, x1, label) のペアデータセット /
            PairDataset containing (x0, x1, label) pairs.
    """
    N = len(data)

    # 1. Select anchor indices
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

    # 2. Generate pairs (distribute n_pairs evenly across anchors)
    M_base = n_pairs // K
    remainder = n_pairs % K

    partner_pool = np.arange(N)

    x0_list = []
    x1_list = []
    label_list = []

    for i, a_i in enumerate(anchor_idx):
        n_i = M_base + (1 if i < remainder else 0)

        # Sample partners with replacement
        partner_idx = np.random.choice(partner_pool, size=n_i, replace=True)

        # 3. Compute pair labels
        if conf is not None:
            # sconf(a_i, x_j) = p_a * p_j + (1 - p_a) * (1 - p_j)
            p_a = conf[a_i]
            p_j = conf[partner_idx]
            sconf_vals = p_a * p_j + (1 - p_a) * (1 - p_j)
            pair_labels = np.random.binomial(1, sconf_vals).astype(np.int32)
        else:
            # Fallback: use the anchor's sd_label
            pair_labels = np.full(n_i, label[a_i], dtype=np.int32)

        x0_list.append(np.tile(data[a_i], (n_i, 1, 1, 1)))
        x1_list.append(data[partner_idx])
        label_list.append(pair_labels)

    x0_data = np.concatenate(x0_list, axis=0).astype(np.float32)
    x1_data = np.concatenate(x1_list, axis=0).astype(np.float32)
    label_sd = np.concatenate(label_list, axis=0).astype(np.int32)

    return PairDataset(x0_data, x1_data, label_sd)
