import csv
import os

import torch


class PairDataset(object):
    """Dataset of image pairs and binary SD labels."""

    def __init__(self, x0, x1, label, metadata=None):
        self.size = label.shape[0]
        self.x0 = torch.from_numpy(x0)
        self.x1 = torch.from_numpy(x1)
        self.label = torch.from_numpy(label)
        self.metadata = metadata or {}

    def __getitem__(self, index):
        return (self.x0[index], self.x1[index], self.label[index])

    def __len__(self):
        return self.size

    def save_csv(self, path):
        """Save pair metadata to CSV for reproducibility/inspection."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        fieldnames = ["pair_index"] + list(self.metadata.keys())
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for i in range(self.size):
                row = {"pair_index": i}
                for key, values in self.metadata.items():
                    row[key] = values[i]
                writer.writerow(row)
