"""Dinomaly losses."""

from __future__ import annotations

from functools import partial

import torch


def _modify_grad(x: torch.Tensor, inds: torch.Tensor, factor: float = 0.0) -> torch.Tensor:
    inds = inds.expand_as(x)
    x[inds] *= factor
    return x


def global_cosine_hm_percent(
    en: list[torch.Tensor],
    de: list[torch.Tensor],
    p: float = 0.9,
    factor: float = 0.1,
) -> torch.Tensor:
    """Cosine reconstruction loss with hard-mining on the decoder gradients.

    The easiest ``p`` fraction of patch positions contributes only ``factor``
    times the normal gradient.  This prevents the decoder from spending all of
    its capacity on the (many) trivially reconstructable normal patches, which
    is the key trick behind Dinomaly's localisation performance.
    """
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0.0
    for target, prediction in zip(en, de):
        target = target.detach()
        with torch.no_grad():
            point_dist = (1 - cos_loss(target, prediction)).unsqueeze(1)
        k = max(1, int(point_dist.numel() * (1 - p)))
        threshold = torch.topk(point_dist.reshape(-1), k=k)[0][-1]
        loss = loss + torch.mean(
            1 - cos_loss(target.reshape(target.shape[0], -1), prediction.reshape(prediction.shape[0], -1))
        )
        prediction.register_hook(partial(_modify_grad, inds=point_dist < threshold, factor=factor))
    return loss / len(en)


def global_cosine(en: list[torch.Tensor], de: list[torch.Tensor]) -> torch.Tensor:
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0.0
    for target, prediction in zip(en, de):
        loss = loss + torch.mean(
            1
            - cos_loss(
                target.detach().reshape(target.shape[0], -1),
                prediction.reshape(prediction.shape[0], -1),
            )
        )
    return loss / len(en)
