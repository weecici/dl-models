import torch
import torchinfo
from dl_models.transformer.model import Transformer


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
        vocab_size=vocab_size,
        d_model=d_model,
        d_ff=d_ff,
        num_heads=num_heads,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
    )

    torchinfo.summary(
        model,
        input_data=(src, tgt),
        col_names=["input_size", "output_size", "num_params", "trainable"],
        depth=4,
    )

    output = model(src, tgt)
    print(
        "Output shape:", output.shape
    )  # Expected: [batch_size, tgt_seq_length, vocab_size]
