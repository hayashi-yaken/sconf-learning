"""
アンカー型 Type1 のペア生成（実装予定）

アンカー集合 {a_1, ..., a_K} を固定し、
各アンカー a_i に対して複数のインスタンス {x_1, ...} を割り当ててペアを作る。
同一インスタンスが複数ペアに重複する非 i.i.d 構造になる。
"""


def create_anchor_type1_pairs(data, label, anchors, n_pairs_per_anchor, **kwargs):
    raise NotImplementedError("アンカー型 Type1 は未実装です")
