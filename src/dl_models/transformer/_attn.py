import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float):
        super().__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.num_heads = num_heads
        self.d_model = d_model
        self.d_head = d_model // num_heads
        self.scalar = 1.0 / math.sqrt(self.d_head)

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

    def forward(
        self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor | None = None
    ) -> Tensor:

        assert (
            query.dim() == key.dim() == value.dim() == 3
        ), "Input tensors must be 3-dimensional"

        assert (
            query.shape[0] == key.shape[0] == value.shape[0]
        ), "Input tensors must have the same batch size"

        assert (
            query.shape[2] == key.shape[2] == value.shape[2]
        ), "Input tensors must have the same feature dimension"

        assert (
            mask is None or mask.dim() == 3 or mask.dim() == 4
        ), "Mask tensor must be 3D or 4D"

        batch_size, seq_len1, d_model = query.shape
        seq_len2 = key.shape[1]

        assert d_model == self.d_model, "Input feature dimension must match d_model"

        Q: Tensor = self.W_q(query)
        K: Tensor = self.W_k(key)
        V: Tensor = self.W_v(value)

        # reshape all tensors to shape [B, H, L1/L2, D_head] to calculate multi-head attention
        multihead_Q = Q.reshape(
            batch_size, seq_len1, self.num_heads, self.d_head
        ).transpose(1, 2)
        multihead_K = K.reshape(
            batch_size, seq_len2, self.num_heads, self.d_head
        ).transpose(1, 2)
        multihead_V = V.reshape(
            batch_size, seq_len2, self.num_heads, self.d_head
        ).transpose(1, 2)

        # matrix multiplications to calculate attention scores for each head, shape: [B, H, L1, L2]
        attn_scores = (
            torch.matmul(multihead_Q, multihead_K.transpose(-2, -1)) * self.scalar
        )

        if mask is not None:
            # allow mask shapes: [B, L2], [B, L1, L2], [B, 1, L1, L2], or [B, H, L1, L2]
            if mask.dim() == 2:  # [B, L2]
                mask = mask.unsqueeze(1).unsqueeze(2)  # -> [B, 1, 1, L2]
            elif mask.dim() == 3:  # [B, L1, L2]
                mask = mask.unsqueeze(1)  # -> [B, 1, L1, L2]
            # True = mask, False = no mask
            if mask.dtype == torch.bool:
                attn_scores = attn_scores.masked_fill(mask, -1e9)
            else:
                attn_scores = attn_scores.masked_fill(mask == 1, -1e9)

        # calculate attention probabilities using softmax on the last dimension, shape: [B, H, L1, L2]
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.attn_dropout(attn_probs)

        # calculate weighted sum of values for each head, shape: [B, H, L1, D_head]
        multihead_attn_out = torch.matmul(attn_probs, multihead_V)

        # concatenate multi-head attention outputs, shape: [B, L1, H * D_head = D_model]
        concat_attn_out = (
            multihead_attn_out.transpose(1, 2)
            .contiguous()
            .reshape(batch_size, seq_len1, self.d_model)
        )

        # apply linear projection to concatenated attention output, shape: [B, L1, D_model]
        out = self.W_o(concat_attn_out)

        out: Tensor = self.out_dropout(out)

        return out


if __name__ == "__main__":
    batch_size = 2
    seq_length = 4
    d_model = 8
    num_heads = 2

    x = torch.rand((batch_size, seq_length, d_model))
    attn = MultiHeadAttention(d_model, num_heads)
    a = torch.ones((batch_size, seq_length, seq_length - 2), dtype=torch.bool)
    b = torch.zeros((batch_size, seq_length, 2), dtype=torch.bool)
    mask = torch.cat([a, b], dim=-1)
    print(mask)
    output = attn(x, x, x, mask)
    print(output)
