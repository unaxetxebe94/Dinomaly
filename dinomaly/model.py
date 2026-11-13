"""Dinomaly: frozen DINOv2 encoder + noisy MLP bottleneck + linear-attention decoder.

Faithful re-implementation of the official ViTill model
(https://github.com/guojiajeremy/Dinomaly, CVPR 2025).  The only difference is
that the DINOv2 backbone is loaded through ``timm`` instead of a vendored copy
of the DINOv2 repository, which keeps this project small and easy to install.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from functools import partial

import timm
import torch
import torch.nn.functional as F
from torch import nn

# ---------------------------------------------------------------------------
# Decoder building blocks (copied/adapted from the official implementation)
# ---------------------------------------------------------------------------


def drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)


class DropKey(nn.Module):
    """Regularises softmax attention by randomly dropping keys during training."""

    def __init__(self, p: float = 0.0) -> None:
        super().__init__()
        self.p = p

    def forward(self, attn: torch.Tensor) -> torch.Tensor:
        if self.training and self.p > 0:
            attn = attn + torch.bernoulli(torch.ones_like(attn) * self.p) * -1e12
        return attn


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer: type[nn.Module] = nn.GELU,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return self.drop(x)


class bMlp(nn.Module):
    """Noisy bottleneck: Linear -> GELU -> Linear with dropout around each."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer: type[nn.Module] = nn.GELU,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return self.drop(x)


class LinearAttention2(nn.Module):
    """O(N) linear attention used by the Dinomaly decoder."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None):
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = F.elu(q) + 1.0
        k = F.elu(k) + 1.0
        kv = torch.einsum("...sd,...se->...de", k, v)
        denom = torch.einsum("...sd,...d->...s", q, k.sum(dim=-2))
        # In bf16 a collapsed head can make the denominator underflow to zero,
        # which turns z into inf and poisons the whole network with NaN.
        z = 1.0 / torch.clamp(denom, min=1e-6)
        x = torch.einsum("...de,...sd,...s->...se", kv, q, z)
        x = x.transpose(1, 2).reshape(b, n, c)
        x = self.proj(x)
        return self.proj_drop(x), kv


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path_rate: float = 0.0,
        act_layer: type[nn.Module] = nn.GELU,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        attn: type[nn.Module] = LinearAttention2,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = attn(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop
        )
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None, return_attention: bool = False):
        y, attn = self.attn(self.norm1(x), attn_mask=attn_mask)
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return (x, attn) if return_attention else x


# ---------------------------------------------------------------------------
# DINOv2 encoder (timm)
# ---------------------------------------------------------------------------

_ENCODER_ALIASES = {
    "dinov2_vit_small_14": "vit_small_patch14_dinov2.lvd142m",
    "dinov2reg_vit_small_14": "vit_small_patch14_reg4_dinov2.lvd142m",
    "dinov2_vit_base_14": "vit_base_patch14_dinov2.lvd142m",
    "dinov2reg_vit_base_14": "vit_base_patch14_reg4_dinov2.lvd142m",
    "dinov2_vit_large_14": "vit_large_patch14_dinov2.lvd142m",
    "dinov2reg_vit_large_14": "vit_large_patch14_reg4_dinov2.lvd142m",
}



class DinoV2Encoder(nn.Module):
    """Frozen DINOv2 returning the tokens after the requested transformer blocks.

    All weights are frozen; ``forward`` runs under ``torch.no_grad``.  The
    positional embedding is interpolated by timm for the actual input size, so
    the same model works for 392x392 and 518x518 inputs.
    """

    def __init__(self, name: str = "dinov2reg_vit_base_14", pretrained: bool = True, img_size: int = 518) -> None:
        super().__init__()
        if name not in _ENCODER_ALIASES and ".lvd" not in name:
            raise ValueError(f"Unknown encoder {name!r}; known aliases: {sorted(_ENCODER_ALIASES)}")
        timm_name = _ENCODER_ALIASES.get(name, name)
        self.arch = name
        self.vit = timm.create_model(
            timm_name,
            pretrained=pretrained,
            num_classes=0,
            img_size=img_size,
            dynamic_img_size=True,
        )
        self.vit.requires_grad_(False)
        self.vit.eval()
        self.patch_size = int(self.vit.patch_embed.patch_size[0])
        self.embed_dim = int(self.vit.embed_dim)
        self.num_prefix_tokens = int(self.vit.num_prefix_tokens)
        self.num_register_tokens = self.num_prefix_tokens - 1
        self.num_heads = int(self.vit.blocks[0].attn.num_heads)
        self.num_blocks = len(self.vit.blocks)

    @property
    def blocks(self) -> nn.ModuleList:
        return self.vit.blocks

    @torch.no_grad()
    def forward(self, x: torch.Tensor, target_layers: Sequence[int]) -> list[torch.Tensor]:
        x = self.vit.patch_embed(x)
        x = self.vit._pos_embed(x)
        features: list[torch.Tensor] = []
        wanted = {int(i) for i in target_layers}
        last = max(wanted)
        for i, block in enumerate(self.vit.blocks):
            x = block(x)
            if i in wanted:
                features.append(x)
            if i >= last:
                break
        return features


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------


class Dinomaly(nn.Module):
    def __init__(
        self,
        encoder: DinoV2Encoder,
        decoder_blocks: int = 8,
        bottleneck_drop: float = 0.2,
        bottleneck_ratio: float = 4.0,
        decoder_mlp_ratio: float = 4.0,
        target_layers: Sequence[int] = (2, 3, 4, 5, 6, 7, 8, 9),
        fuse_layer_encoder: Sequence[Sequence[int]] = ((0, 1, 2, 3), (4, 5, 6, 7)),
        fuse_layer_decoder: Sequence[Sequence[int]] = ((0, 1, 2, 3), (4, 5, 6, 7)),
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.target_layers = tuple(int(i) for i in target_layers)
        self.fuse_layer_encoder = tuple(tuple(int(i) for i in g) for g in fuse_layer_encoder)
        self.fuse_layer_decoder = tuple(tuple(int(i) for i in g) for g in fuse_layer_decoder)

        n_targets = len(self.target_layers)
        for group in self.fuse_layer_encoder + self.fuse_layer_decoder:
            if max(group) >= n_targets:
                raise ValueError(f"Fusion group {group} indexes {n_targets} target layers")
        if len(self.fuse_layer_encoder) != len(self.fuse_layer_decoder):
            raise ValueError("fuse_layer_encoder and fuse_layer_decoder need the same number of groups")

        embed_dim = encoder.embed_dim
        self.bottleneck = nn.Sequential(
            bMlp(embed_dim, int(embed_dim * bottleneck_ratio), embed_dim, drop=bottleneck_drop)
        )
        self.decoder = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=encoder.num_heads,
                    mlp_ratio=decoder_mlp_ratio,
                    qkv_bias=True,
                    norm_layer=partial(nn.LayerNorm, eps=1e-8),
                    attn_drop=0.0,
                    attn=LinearAttention2,
                )
                for _ in range(decoder_blocks)
            ]
        )
        self.apply(self._init_weights)
        self.trainable_modules = nn.ModuleList([self.bottleneck, self.decoder])

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.01, a=-0.03, b=0.03)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.weight, 1.0)
            nn.init.constant_(module.bias, 0)

    def train(self, mode: bool = True):
        super().train(mode)
        # The frozen encoder must never switch to train mode (no dropout/BN).
        self.encoder.eval()
        return self

    def forward(self, x: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        en_list = self.encoder(x, self.target_layers)
        prefix = self.encoder.num_prefix_tokens
        side = round(math.sqrt(en_list[0].shape[1] - prefix))

        x = torch.stack(en_list, dim=1).mean(dim=1)
        x = self.bottleneck(x)

        de_list = []
        for block in self.decoder:
            x = block(x)
            de_list.append(x)
        de_list = de_list[::-1]

        en = [torch.stack([en_list[i] for i in group], dim=1).mean(dim=1) for group in self.fuse_layer_encoder]
        de = [torch.stack([de_list[i] for i in group], dim=1).mean(dim=1) for group in self.fuse_layer_decoder]

        en = [e[:, prefix:, :] for e in en]
        de = [d[:, prefix:, :] for d in de]
        en = [e.permute(0, 2, 1).reshape(e.shape[0], -1, side, side).contiguous() for e in en]
        de = [d.permute(0, 2, 1).reshape(d.shape[0], -1, side, side).contiguous() for d in de]
        return en, de

    @property
    def trainable_parameters(self):
        return self.trainable_modules.parameters()


def anomaly_maps(
    en: Sequence[torch.Tensor],
    de: Sequence[torch.Tensor],
    out_size: int | tuple[int, int],
    sigma: float = 4.0,
    kernel_size: int = 5,
) -> torch.Tensor:
    """Cosine distance between encoder/decoder features, upsampled and blurred."""
    from torchvision.transforms.functional import gaussian_blur

    if not isinstance(out_size, (tuple, list)):
        out_size = (int(out_size), int(out_size))
    maps = []
    for target, prediction in zip(en, de):
        distance = 1 - F.cosine_similarity(target, prediction, dim=1)
        maps.append(F.interpolate(distance.unsqueeze(1), size=out_size, mode="bilinear", align_corners=False))
    anomaly_map = torch.stack(maps, dim=0).mean(dim=0)
    if sigma > 0:
        anomaly_map = gaussian_blur(anomaly_map, kernel_size=kernel_size, sigma=sigma)
    return anomaly_map


def build_model(cfg: dict) -> Dinomaly:
    encoder_cfg = cfg.get("encoder", {})
    name = cfg.get("name", "dinov2reg_vit_base_14")
    encoder = DinoV2Encoder(
        name=name,
        pretrained=bool(encoder_cfg.get("pretrained", True)),
        img_size=int(encoder_cfg.get("img_size", 518)),
    )
    return Dinomaly(
        encoder=encoder,
        decoder_blocks=int(cfg.get("decoder_blocks", 8)),
        bottleneck_drop=float(cfg.get("bottleneck_drop", 0.2)),
        bottleneck_ratio=float(cfg.get("bottleneck_ratio", 4.0)),
        decoder_mlp_ratio=float(cfg.get("decoder_mlp_ratio", 4.0)),
        target_layers=cfg.get("target_layers", [2, 3, 4, 5, 6, 7, 8, 9]),
        fuse_layer_encoder=cfg.get("fuse_layer_encoder", [[0, 1, 2, 3], [4, 5, 6, 7]]),
        fuse_layer_decoder=cfg.get("fuse_layer_decoder", [[0, 1, 2, 3], [4, 5, 6, 7]]),
    )
