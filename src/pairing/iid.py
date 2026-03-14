import numpy as np

from .dataset import PairDataset


def create_iid_pairs(data, label, perm):
    """Create a PairDataset of i.i.d random pairs (Baseline)."""
    x0_data = []
    x1_data = []
    label_sd = []
    x0_indices = []
    x1_indices = []
    for i in range(int(np.floor(len(label) / 2))):
        x0_data.append(data[2 * i])
        x1_data.append(data[2 * i + 1])
        label_sd.append(label[2 * i])
        x0_indices.append(2 * i)
        x1_indices.append(2 * i + 1)
    x0_data = np.array(x0_data, dtype=np.float32)
    x1_data = np.array(x1_data, dtype=np.float32)
    label_sd = np.array(label_sd, dtype=np.int32)
    metadata = {
        "x0_index": np.array(x0_indices, dtype=np.int64),
        "x1_index": np.array(x1_indices, dtype=np.int64),
        "sd_label": label_sd,
    }
    return PairDataset(x0_data, x1_data, label_sd, metadata=metadata)
