import torch
from torch import nn
from zeta import (
    AutoregressiveWrapper,
    Decoder,
    Encoder,
    Transformer,
    ViTransformerWrapper,
)


class RT2(nn.Module):
    def __init__(
        self,
        image_size=256,
        patch_size=32,
        encoder_dim=512,
        encoder_depth=6,
        encoder_heads=8,
        num_tokens=20000,
        max_seq_len=1024,
        decoder_dim=512,
        decoder_depth=6,
        decoder_heads=8,
        attn_kv_heads=2,
        use_abs_pos_emb=False,
        cross_attend=True,
        attn_flash=True,
        qk_norm=True,
    ):
        super(RT2, self).__init__()

        self.encoder = ViTransformerWrapper(
            image_size=image_size,
            patch_size=patch_size,
            attn_layers=Encoder(
                dim=encoder_dim, depth=encoder_depth, heads=encoder_heads
            ),
        )

        self.decoder = Transformer(
            num_tokens=num_tokens,
            max_seq_len=max_seq_len,
            use_abs_pos_emb=use_abs_pos_emb,
            attn_layers=Decoder(
                dim=decoder_dim,
                depth=decoder_depth,
                heads=decoder_heads,
                cross_attend=cross_attend,
                attn_kv_heads=attn_kv_heads,
                attn_flash=attn_flash,
                qk_norm=qk_norm,
            ),
        )

        self.decoder = AutoregressiveWrapper(self.decoder)

    def forward(self, img: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        try:
            encoded = self.encoder(img, return_embeddings=True)
            return self.decoder(text, context=encoded)
        except Exception as error:
            print(f"Failed in forward method: {error}")
            raise
