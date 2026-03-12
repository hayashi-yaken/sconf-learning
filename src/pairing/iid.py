import numpy as np
import torch


class PairDataset(object):
    """Dataset of image pairs and binary SD labels."""
    def __init__(self, x0, x1, label):
        self.size = label.shape[0]
        self.x0 = torch.from_numpy(x0)
        self.x1 = torch.from_numpy(x1)
        self.label = torch.from_numpy(label)

    def __getitem__(self, index):
        return (self.x0[index], self.x1[index], self.label[index])

    def __len__(self):
        return self.size


def create_iid_pairs(data, label, perm):
    """Create a PairDataset of i.i.d random pairs (Baseline)."""
    x0_data = []
    x1_data = []
    label_sd = []
    for i in range(int(np.floor(len(label) / 2))):
        x0_data.append(data[2 * i])
        x1_data.append(data[2 * i + 1])
        label_sd.append(label[2 * i])
    x0_data = np.array(x0_data, dtype=np.float32)
    x1_data = np.array(x1_data, dtype=np.float32)
    label_sd = np.array(label_sd, dtype=np.int32)
    return PairDataset(x0_data, x1_data, label_sd)
