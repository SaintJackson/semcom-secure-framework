"""
Base class for Semantic Communication models.
"""

import torch
import torch.nn as nn

class BaseSemCom(nn.Module):
    def encode(self, x):
        raise NotImplementedError
    
    def decode(self, s):
        raise NotImplementedError

    def forward(self, x):
        s = self.encode(x)
        out = self.decode(s)
        return s, out
