"""Scale-residual detail enhancement. Adapted from CUST; see third_party/."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TAFE(nn.Module):
    """Tiny-object-Aware Frequency Enhancement.

    The module estimates frequency-related residuals using spatial scale
    differences.  It is not an FFT or an exact spectral decomposition.
    """

    def __init__(self,
                 channels,
                 use_foreground_gate=True,
                 use_texture_gate=True,
                 alpha_init=1e-3):
        super().__init__()
        self.use_foreground_gate = use_foreground_gate
        self.use_texture_gate = use_texture_gate
        self.refiner = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                3,
                padding=2,
                dilation=2,
                groups=channels,
                bias=False), nn.GELU(), nn.Conv2d(channels, channels, 1))
        if use_texture_gate:
            hidden = max(channels // 4, 1)
            self.texture_gate = nn.Sequential(
                nn.Conv2d(channels * 2, hidden, 1), nn.GELU(),
                nn.Conv2d(hidden, 1, 1), nn.Sigmoid())
        else:
            self.texture_gate = None
        self.alpha = nn.Parameter(
            torch.full((1, channels, 1, 1), float(alpha_init)))

    def forward(self, x, foreground_prob=None, valid_mask=None):
        height, width = x.shape[-2:]
        if valid_mask is None:
            valid = x.new_ones((x.size(0), 1, height, width))
        else:
            valid = valid_mask.to(dtype=x.dtype)
        # Invalid batch padding must not influence pooling or convolutions in
        # the added enhancement path.  The original FPN feature is retained in
        # the residual output; only the learned increment uses masked input.
        masked_x = x * valid
        half_size = (max(height // 2, 1), max(width // 2, 1))
        quarter_size = (max(height // 4, 1), max(width // 4, 1))

        down2 = F.adaptive_avg_pool2d(masked_x, half_size)
        low2 = F.interpolate(
            down2, size=(height, width), mode='bilinear', align_corners=False)
        # Match the hierarchical construction in CUST: the second pooling is
        # applied to the half-resolution feature.
        down4 = F.adaptive_avg_pool2d(down2, quarter_size)
        low4 = F.interpolate(
            down4, size=(height, width), mode='bilinear', align_corners=False)

        error2 = masked_x - low2
        error4 = low2 - low4
        refined = self.refiner(error2 + error4)

        if self.use_texture_gate:
            texture_gate = self.texture_gate(
                torch.cat((error2.abs(), error4.abs()), dim=1))
        else:
            texture_gate = x.new_ones((x.size(0), 1, height, width))

        gate = texture_gate
        if self.use_foreground_gate:
            if foreground_prob is None:
                raise ValueError('TAFE foreground gating requires a probability map')
            gate = gate * foreground_prob
        gate = gate * valid

        delta = refined * gate
        return x + self.alpha * delta, delta
