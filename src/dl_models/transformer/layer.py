import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ._attn import MultiHeadAttention
from ._ffn import FeedForwardNet


class EncoderLayer(nn.Module):
    def __init__(
        self, d_model: int, d_ff: int, num_heads: int = 1, dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model

        self.mha = MultiHeadAttention(d_model, num_heads, dropout)
        self.layer_norm1 = nn.LayerNorm(d_model)

        self.ffn = FeedForwardNet(d_model, d_ff, dropout)
        self.layer_norm2 = nn.LayerNorm(d_model)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        assert x.dim() == 3, "Input tensor must be 3-dimensional"
        batch_size, seq_len, d_model = x.shape

        assert d_model == self.d_model, "Input feature dimension must match d_model"

        # multi-head self attention
        attn_out = self.mha(x, x, x, mask)
        attn_out = self.layer_norm1(x + attn_out)

        # position-wise feed-forward network
        ffn_out = self.ffn(attn_out)
        out: Tensor = self.layer_norm2(attn_out + ffn_out)

        return out


class DecoderLayer(nn.Module):
    def __init__(
        self, d_model: int, d_ff: int, num_heads: int = 1, dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model

        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.layer_norm1 = nn.LayerNorm(d_model)

        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.layer_norm2 = nn.LayerNorm(d_model)

        self.ffn = FeedForwardNet(d_model, d_ff, dropout)
        self.layer_norm3 = nn.LayerNorm(d_model)

    def forward(
        self,
        x: Tensor,
        enc_out: Tensor,
        self_mask: Tensor | None = None,
        cross_mask: Tensor | None = None,
    ) -> Tensor:
        assert x.dim() == enc_out.dim() == 3, "Input tensors must be 3-dimensional"

        assert (
            x.shape[0] == enc_out.shape[0]
        ), "Input tensors must have the same batch size"

        assert (
            x.shape[2] == enc_out.shape[2]
        ), "Input tensors must have the same feature dimension"

        batch_size, seq_len, d_model = x.shape

        assert d_model == self.d_model, "Input feature dimension must match d_model"

        # masked multi-head self attention
        self_attn_out = self.self_attn(x, x, x, self_mask)
        self_attn_out = self.layer_norm1(x + self_attn_out)

        # multi-head cross attention with encoder output
        cross_attn_out = self.cross_attn(self_attn_out, enc_out, enc_out, cross_mask)
        cross_attn_out = self.layer_norm2(self_attn_out + cross_attn_out)

        # position-wise feed-forward network
        ffn_out = self.ffn(cross_attn_out)
        out: Tensor = self.layer_norm3(cross_attn_out + ffn_out)

        return out
