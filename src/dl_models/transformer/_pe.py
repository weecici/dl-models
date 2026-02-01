import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        assert (
            d_model % 2 == 0
        ), "d_model must be even for sinusoidal positional encoding"

        self.d_model = d_model
        self.max_len = max_len

        # calculate pos / 10000^{2i/d_model}
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        i = torch.arange(0, d_model, 2, dtype=torch.float32)
        angle_rates = 1.0 / (10000 ** (i / d_model))
        angles = pos * angle_rates

        # calculate positional encodings for max_len positions
        # PE(2i, pos) = sin(pos / 10000^{2i/d_model})
        # PE(2i+1, pos) = cos(pos / 10000^{2i/d_model})
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(angles)
        pe[:, 1::2] = torch.cos(angles)

        self.register_buffer("pe", pe.unsqueeze(0))  # shape: [1, max_len, d_model]
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        assert x.dim() == 3, "Input tensor must be 3-dimensional"
        batch_size, seq_len, d_model = x.shape

        assert d_model == self.d_model, "Input feature dimension must match d_model"
        assert seq_len <= self.max_len, "Input sequence length exceeds max_len"

        # get positional encodings based on the input sequence length
        pe = self.pe[:, :seq_len, :].type_as(x)
        out = self.dropout(x + pe)

        return out


if __name__ == "__main__":
    batch_size = 2
    seq_length = 8
    d_emb = 8
    x = torch.zeros((batch_size, seq_length, d_emb))
    pe = PositionalEncoding(d_emb)
    ...
