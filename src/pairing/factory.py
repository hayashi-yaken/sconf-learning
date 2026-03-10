from .iid import create_iid_pairs
from .anchor_type1 import create_anchor_type1_pairs
from .anchor_type2 import create_anchor_type2_pairs

PAIR_STRATEGIES = {
    'iid': create_iid_pairs,
    'anchor_type1': create_anchor_type1_pairs,
    'anchor_type2': create_anchor_type2_pairs,
}


def get_pair_dataset(strategy, data, label, perm, **kwargs):
    """ペア生成戦略を名前で選択して実行する。

    Args:
        strategy (str): 'iid' | 'anchor_type1' | 'anchor_type2'
        data: 画像データ (numpy)
        label: ラベルデータ (numpy)
        perm: シャッフルインデックス (torch.Tensor)
    """
    if strategy not in PAIR_STRATEGIES:
        raise ValueError(
            "Unknown strategy: {}. Choose from {}".format(strategy, list(PAIR_STRATEGIES.keys()))
        )
    return PAIR_STRATEGIES[strategy](data, label, perm, **kwargs)
