"""
Placeholder for Multi Cancer dataset loader.
"""

import torch
from torch.utils.data import Dataset

class CancerDataset(Dataset):
    def __init__(self, root):
        # placeholder
        self.data = torch.randn(200, 3, 224, 224)
        self.labels = torch.randint(0, 10, (200,))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
