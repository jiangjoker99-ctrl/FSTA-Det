"""Residual composition of TAFE, CSTA and foreground prediction."""

import torch
import torch.nn as nn

from .csta import CSTA
from .foreground import ForegroundHead
from .tafe import TAFE


class FSTABlock(nn.Module):
    """Apply TAFE then CSTA and return explicit foreground logits."""

    def __init__(self,
                 channels=256,
                 enable_tafe=True,
                 tafe_use_foreground_gate=True,
                 tafe_use_texture_gate=True,
                 enable_csta=True,
                 csta_use_foreground=True,
                 csta_foreground_bias=False,
                 csta_foreground_bias_weight=1.0,
                 inner_dim=64,
                 window_size=8,
                 group_size=4,
                 alpha_init=1e-3,
                 beta_init=1e-3,
                 need_foreground_head=True):
        super().__init__()
        self.enable_tafe = enable_tafe
        self.enable_csta = enable_csta
        self.tafe_use_foreground_gate = tafe_use_foreground_gate
        self.csta_use_foreground = csta_use_foreground
        self.need_foreground_head = need_foreground_head

        self.foreground_head = (
            ForegroundHead(channels) if need_foreground_head else None)
        self.tafe = TAFE(
            channels,
            use_foreground_gate=tafe_use_foreground_gate,
            use_texture_gate=tafe_use_texture_gate,
            alpha_init=alpha_init) if enable_tafe else None
        self.csta = CSTA(
            channels,
            inner_dim=inner_dim,
            window_size=window_size,
            group_size=group_size,
            use_foreground=csta_use_foreground,
            foreground_bias=csta_foreground_bias,
            foreground_bias_weight=csta_foreground_bias_weight
        ) if enable_csta else None
        # Do not register an unused trainable parameter in TAFE-only/disabled
        # ablations.  This also keeps those variants safe with DDP's default
        # ``find_unused_parameters=False`` setting.
        self.beta = (nn.Parameter(
            torch.full((1, channels, 1, 1), float(beta_init)))
                     if enable_csta else None)

    def forward(self, x, valid_mask=None):
        if valid_mask is None:
            masked_x = x
        else:
            masked_x = x * valid_mask.to(dtype=x.dtype)
        foreground_logits = None
        foreground_prob = None
        if self.foreground_head is not None:
            foreground_logits = self.foreground_head(masked_x)
            foreground_prob = torch.sigmoid(foreground_logits)
            if valid_mask is not None:
                foreground_prob = foreground_prob * valid_mask.to(x.dtype)

        out = x
        if self.enable_tafe:
            out, _ = self.tafe(
                out,
                foreground_prob=foreground_prob,
                valid_mask=valid_mask)
        if self.enable_csta:
            csta_foreground = (
                foreground_prob if self.csta_use_foreground else None)
            delta = self.csta(
                out,
                foreground_prob=csta_foreground,
                valid_mask=valid_mask)
            out = out + self.beta * delta
        return out, foreground_logits
