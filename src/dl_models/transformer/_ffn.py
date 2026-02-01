import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class FeedForwardNet(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        self.linear1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        assert x.dim() == 3, "Input tensor must be 3-dimensional"
        batch_size, seq_len, d_model = x.shape

        assert d_model == self.d_model, "Input feature dimension must match d_model"

        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)

        out: Tensor = self.dropout(out)

        return out
