"""
    Script for the Decoder
    ---
        MLPDecoder: 2-layer MLP (64 → 128 → 49) with ReLU.
"""

import torch
import torch.nn as nn
import numpy as np
from .utils.modules import *


class MLPDecoder(nn.Module):
    def __init__(self, emb_dim, patch_size, hidden_dim=128):
        super(MLPDecoder, self).__init__()
        self.fc1 = nn.Linear(emb_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, patch_size)

    def forward(self, encoded_patch):
        x = self.fc1(encoded_patch)
        x = self.activation(x)
        x = self.fc2(x)
        return x
