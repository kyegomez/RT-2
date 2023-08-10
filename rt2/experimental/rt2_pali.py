from typing import List, Optional

import torch
import torch.nn.functional as F
from beartype import beartype
from classifier_free_guidance_pytorch import (
    AttentionTextConditioner,
    TextConditioner,
    classifier_free_guidance,
)
from einops import pack, rearrange, reduce, repeat, unpack
from einops.layers.torch import Rearrange
from pali import VitModel, Pali
from torch import nn


#helpers
def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def cast_tupel(val, length=1):
    return val if isinstance(val, tuple) else ((val,) * length) 

def pack_one(x, pattern):
    return pack([x], pattern)

def unpack_one(x, ps, pattern):
    return unpack(x, ps, pattern)[0]


#sinusoidla positions

def pos_emb_sincos_1d(seq, dim, temperature=10000, device=None, dtype=torch.float16):
    n = torch.arange(seq, device=device)
    omega = torch.arange(dim // 2, device=device) / (dim // 2 - 1)
    omega = 1. / (temperature ** omega)

    n = n[:, None] * omega[None, :]
    pos_emb = torch.cat((n.sin(), n.cos()), dim=1)
    return pos_emb.type(dtype)

    
class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))
    
    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)


#MBCONV


class TokenLearner(nn.Module):
    def __init__(
            self,
            *,
            dim,
            ff_mult=2,
            num_output_tokens = 8,
            num_layers=2
        ):
            super().__init__()
            inner_dim = dim * ff_mult * num_output_tokens

            self.num_output_tokens = num_output_tokens
            self.net = nn.Sequential(
                nn.Conv2d(dim * num_output_tokens, inner_dim, 1, groups=num_output_tokens),
                nn.GELU(),
                nn.Conv2d(inner_dim, num_output_tokens, 1, groups = num_output_tokens),
            )
    
    def forward(self, x):
        x, ps = pack_one(x, '* c h w')
        x = repeat(x, 'b c h w -> b (g x) h w', g = self.num_output_tokens)
        attn = self.net(x)

        attn = rearrange(attn, 'b g h w -> b 1 g h w')
        x = rearrange(x, 'b (g c) h w -> b c g h w', g =self.num_output_tokens)

        x = reduce(x * attn, 'b c g h w -> b c g', 'mean')
        x = unpack_one(x, ps, '8 c n')
        return x
    


@beartype
class RT2Pali(nn.Module):
    def __init__(
        self,
        *,
        pali: Pali,
        vit_model: VitModel,
        num_actions = 11,
        action_bins = 256,
        depth = 6,
        heads = 8,
        dim_head = 64,
        token_learner_ff_mult = 2,
        token_learner_num_layers = 2,
        token_learner_num_output_tokens = 8,
        cond_drop_prob = 0.2,
        use_attn_conditioner = False,
        conditioner_kwargs: dict = dict()
    ):
        super().__init__()
        self.vit_model = vit_model
        self.pali = pali

        self.num_pali_stages = len(pali.cond_hidden_dims)

        conditioner_klass = AttentionTextConditioner if use_attn_conditioner else TextConditioner

        self.conditioner = conditioner_klass(
            hidden_dims = (*tuple(pali.cond_hidden_dims), *((pali.embed_dim,) * depth * 2)),
            hiddens_channel_first = (*((True,) * self.num_pali_stages), *((False,) * depth * 2)),
            cond_drop_prob = cond_drop_prob,
            **conditioner_kwargs
        )

        self.token_learner = TokenLearner(
            dim = pali.embed_dim,
            ff_mult = token_learner_ff_mult,
            num_output_tokens = token_learner_num_output_tokens,
            num_layers = token_learner_num_layers
        )

        self.num_learned_tokens = token_learner_num_output_tokens

        self.transformer_depth = depth

        self.cond_drop_prob = cond_drop_prob

        self.to_logits = nn.Sequential(
            LayerNorm(pali.embed_dim),
            nn.Linear(pali.embed_dim, num_actions * action_bins),
            Rearrange('... (a b) -> ... a b', b = action_bins)
        )

    @classifier_free_guidance
    def forward(
        self,
        video,
        texts: Optional[List[str]] = None,
        cond_drop_prob = 0.
    ):
        depth = self.transformer_depth
        cond_drop_prob = default(cond_drop_prob, self.cond_drop_prob)

        frames, device = video.shape[2], video.device

        cond_fns = self.conditioner(
            texts,
            cond_drop_prob = cond_drop_prob,
            repeat_batch = (*((frames,) * self.num_pali_stages), *((1,) * self.transformer_depth * 2))
        )

        pali_cond_fns, transformer_cond_fns = cond_fns[:-(depth * 2)], cond_fns[-(depth * 2):]

        video = rearrange(video, 'b c f h w -> b f c h w')
        images, packed_shape = pack_one(video, '* c h w')
        img_embeds = self.vit_model.process(images)
        

        tokens = self.pali(
            texts,
            image_embeds = img_embeds,
            cond_fns = pali_cond_fns,
            cond_drop_prob = cond_drop_prob,
            return_embeddings = True
        )

        tokens = unpack_one(tokens, packed_shape, '* c h w')
        learned_tokens = self.token_learner(tokens)

        learned_tokens = rearrange(learned_tokens, 'b f c n -> b (f n) c')

        # causal attention mask

        attn_mask = torch.ones((frames, frames), dtype = torch.bool, device = device).triu(1)
        attn_mask = repeat(attn_mask, 'i j -> (i r1) (j r2)', r1 = self.num_learned_tokens, r2 = self.num_learned_tokens)

        # sinusoidal positional embedding

        pos_emb = pos_emb_sincos_1d(frames, learned_tokens.shape[-1], dtype = learned_tokens.dtype, device = learned_tokens.device)

        learned_tokens = learned_tokens + repeat(pos_emb, 'n d -> (n r) d', r = self.num_learned_tokens)

        # attention
        attended_tokens = self.pali(learned_tokens, cond_fns = transformer_cond_fns, attn_mask = ~attn_mask)

        pooled = reduce(attended_tokens, 'b (f n) d -> b f d', 'mean', f = frames)

        logits = self.to_logits(pooled)
        return logits
