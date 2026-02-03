import math
import torch
import torchinfo
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, LongTensor

from .layer import EncoderLayer, DecoderLayer
from ._pe import PositionalEncoding
from ._utils import get_pad_mask, get_subsequent_mask
from ._const import *


class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        d_ff: int,
        num_heads: int,
        dropout: float,
        max_len: int,
        num_layers: int,
        pad_idx: int,
    ):
        super().__init__()
        self.scalar = math.sqrt(d_model)

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout=dropout)
        self.layer_stack = nn.ModuleList(
            [EncoderLayer(d_model, d_ff, num_heads, dropout) for _ in range(num_layers)]
        )

    def forward(self, x: LongTensor, mask: Tensor | None = None) -> Tensor:

        emb = self.embedding(x)
        # - "In the embedding layers, we multiply those weights by sqrt(d_model)"
        emb *= self.scalar
        out = self.pos_encoding(emb)

        for enc_layer in self.layer_stack:
            out: Tensor = enc_layer(out, mask)

        return out


class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        d_ff: int,
        num_heads: int,
        dropout: float,
        max_len: int,
        num_layers: int,
        pad_idx: int,
    ):
        super().__init__()
        self.scalar = math.sqrt(d_model)

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout=dropout)
        self.layer_stack = nn.ModuleList(
            [DecoderLayer(d_model, d_ff, num_heads, dropout) for _ in range(num_layers)]
        )

    def forward(
        self,
        x: LongTensor,
        enc_out: Tensor,
        self_mask: Tensor | None = None,
        cross_mask: Tensor | None = None,
    ) -> Tensor:

        emb = self.embedding(x)
        # - "In the embedding layers, we multiply those weights by sqrt(d_model)"
        emb *= self.scalar
        out = self.pos_encoding(emb)

        for dec_layer in self.layer_stack:
            out: Tensor = dec_layer(out, enc_out, self_mask, cross_mask)

        return out


class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = DEFAULT_D_MODEL,
        d_ff: int = DEFAULT_D_FF,
        num_heads: int = DEFAULT_NUM_HEADS,
        dropout: float = DEFAULT_DROPOUT,
        max_len: int = DEFAULT_MAX_LEN,
        num_encoder_layers: int = DEFAULT_NUM_LAYERS,
        num_decoder_layers: int = DEFAULT_NUM_LAYERS,
        pad_idx: int = DEFAULT_PAD_IDX,
    ):
        super().__init__()

        self.encoder = Encoder(
            vocab_size,
            d_model,
            d_ff,
            num_heads,
            dropout,
            max_len,
            num_encoder_layers,
            pad_idx=pad_idx,
        )
        self.decoder = Decoder(
            vocab_size,
            d_model,
            d_ff,
            num_heads,
            dropout,
            max_len,
            num_decoder_layers,
            pad_idx=pad_idx,
        )
        self.linear = nn.Linear(d_model, vocab_size, bias=False)

        #  - "In our model, we share the same weight matrix between the two embedding layers and the pre-softmax linear transformation..."
        self.linear.weight = self.encoder.embedding.weight
        self.decoder.embedding.weight = self.encoder.embedding.weight

        self.pad_idx = pad_idx

    def forward(
        self,
        src: LongTensor,
        tgt: LongTensor,
    ) -> Tensor:
        src_mask = get_pad_mask(src, self.pad_idx)
        tgt_mask = get_pad_mask(tgt, self.pad_idx) & get_subsequent_mask(tgt)

        assert src.dim() == tgt.dim() == 2, "Input tensors must be 2-dimensional"
        assert (
            src.shape[0] == tgt.shape[0]
        ), "Input tensors must have the same batch size"

        enc_out = self.encoder(src, src_mask)
        dec_out = self.decoder(tgt, enc_out, tgt_mask, src_mask)
        logits = self.linear(dec_out)

        return logits
