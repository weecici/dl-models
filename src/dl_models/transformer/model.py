import math
import torch
import torchinfo
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, LongTensor

from .layer import EncoderLayer, DecoderLayer
from ._pe import PositionalEncoding


class Encoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        vocab_size: int,
        d_model: int,
        d_ff: int,
        num_heads: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.scalar = math.sqrt(d_model)

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout=dropout)
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
        num_layers: int,
        vocab_size: int,
        d_model: int,
        d_ff: int,
        num_heads: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.scalar = math.sqrt(d_model)

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout=dropout)
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
        num_encoder_layers: int,
        num_decoder_layers: int,
        vocab_size: int,
        d_model: int,
        d_ff: int,
        num_heads: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.encoder = Encoder(
            num_encoder_layers, vocab_size, d_model, d_ff, num_heads, dropout
        )
        self.decoder = Decoder(
            num_decoder_layers, vocab_size, d_model, d_ff, num_heads, dropout
        )
        self.linear = nn.Linear(d_model, vocab_size, bias=False)

        #  - "In our model, we share the same weight matrix between the two embedding layers and the pre-softmax linear transformation..."
        self.linear.weight = self.encoder.embedding.weight
        self.decoder.embedding.weight = self.encoder.embedding.weight

    def forward(
        self,
        src: LongTensor,
        tgt: LongTensor,
        src_mask: Tensor | None = None,
        tgt_mask: Tensor | None = None,
    ) -> Tensor:

        enc_out = self.encoder(src, src_mask)
        print("enc_out shape:", enc_out.shape)
        dec_out = self.decoder(tgt, enc_out, tgt_mask, src_mask)
        logits = self.linear(dec_out)

        return logits


if __name__ == "__main__":
    batch_size = 2
    src_seq_length = 10
    tgt_seq_length = 9
    vocab_size = 100
    d_model = 32
    d_ff = 64
    num_heads = 4
    num_encoder_layers = 2
    num_decoder_layers = 2

    src = torch.randint(0, vocab_size, (batch_size, src_seq_length), dtype=torch.long)
    tgt = torch.randint(0, vocab_size, (batch_size, tgt_seq_length), dtype=torch.long)

    model = Transformer(
        num_encoder_layers,
        num_decoder_layers,
        vocab_size,
        d_model,
        d_ff,
        num_heads,
    )

    output = model(src, tgt)
    print(
        "Output shape:", output.shape
    )  # Expected: [batch_size, tgt_seq_length, vocab_size]

    # Print model summary
    torchinfo.summary(
        model,
        input_data=(src, tgt),
        col_names=["input_size", "output_size", "num_params", "trainable"],
        depth=4,
    )
