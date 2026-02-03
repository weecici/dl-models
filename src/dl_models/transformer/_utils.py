import torch
from torch import BoolTensor, LongTensor, Tensor


def get_pad_mask(seq: LongTensor, pad_idx: int) -> BoolTensor:
    assert seq.dim() == 2  # (batch_size, seq_length)

    # shape: (batch_size, 1, seq_length)
    return (seq == pad_idx).unsqueeze(1)


def get_subsequent_mask(seq: LongTensor) -> BoolTensor:
    assert seq.dim() == 2  # (batch_size, seq_length)
    seq_len = seq.shape[1]

    # shape: (1, seq_length, seq_length)
    return torch.triu(torch.ones((1, seq_len, seq_len), dtype=torch.bool), diagonal=1)
