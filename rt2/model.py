import torch
from torch import nn
from zeta.structs import (
    AutoRegressiveWrapper,
    Decoder,
    Encoder,
    Transformer,
    ViTransformerWrapper,
)


class RT2(nn.Module):
    """
    RT2 model implementation.

    Args:
        image_size (int): Size of the input image.
        patch_size (int): Size of each image patch.
        encoder_dim (int): Dimension of the encoder.
        encoder_depth (int): Depth of the encoder.
        encoder_heads (int): Number of attention heads in the encoder.
        num_tokens (int): Number of tokens in the decoder.
        max_seq_len (int): Maximum sequence length in the decoder.
        decoder_dim (int): Dimension of the decoder.
        decoder_depth (int): Depth of the decoder.
        decoder_heads (int): Number of attention heads in the decoder.
        attn_kv_heads (int): Number of attention heads for key-value projection.
        use_abs_pos_emb (bool): Whether to use absolute positional embeddings.
        cross_attend (bool): Whether to enable cross-attention in the decoder.
        attn_flash (bool): Whether to enable flash attention in the decoder.
        qk_norm (bool): Whether to normalize queries and keys in attention.

    Attributes:
        encoder (ViTransformerWrapper): Encoder module.
        decoder (AutoRegressiveWrapper): Decoder module.

    """

    def __init__(
        self,
        image_size: int = 256,
        patch_size: int = 32,
        encoder_dim: int = 512,
        encoder_depth: int = 6,
        encoder_heads: int = 8,
        num_tokens: int = 20000,
        max_seq_len: int = 1024,
        decoder_dim: int = 512,
        decoder_depth: int = 6,
        decoder_heads: int = 8,
        attn_kv_heads: int = 2,
        use_abs_pos_emb: bool = False,
        cross_attend: bool = True,
        attn_flash: bool = True,
        qk_norm: bool = True,
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

        self.decoder = AutoRegressiveWrapper(self.decoder)

    def forward(self, img: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the RT2 model.

        Args:
            img (torch.Tensor): Input image tensor.
            text (torch.Tensor): Input text tensor.

        Returns:
            torch.Tensor: Output tensor.

        Raises:
            Exception: If an error occurs during the forward pass.

        """
        try:
            encoded = self.encoder(img, return_embeddings=True)
            return self.decoder(text, context=encoded)
        except Exception as error:
            print(f"Failed in forward method: {error}")
            raise
