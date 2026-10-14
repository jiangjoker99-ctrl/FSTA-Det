# Copyright (c) OpenMMLab. All rights reserved.
"""FPN integration for the FSTA feature modules."""

import torch
import torch.nn as nn
from mmcv.cnn import bias_init_with_prob

from ..builder import NECKS
from ..plugins.fsta import CSTA, TAFE, FSTABlock, ForegroundHead
from .fpn import FPN


@NECKS.register_module()
class FSTAFPN(FPN):
    """Feature pyramid that enhances selected high-resolution levels."""

    def __init__(self,
                 *args,
                 enhance_levels=(0, 1),
                 feature_strides=(4, 8, 16, 32),
                 shared_weights=False,
                 enable_tafe=True,
                 tafe_use_foreground_gate=True,
                 tafe_use_texture_gate=True,
                 enable_csta=True,
                 csta_use_foreground=True,
                 csta_foreground_bias=False,
                 csta_foreground_bias_weight=1.0,
                 csta_inner_dim=64,
                 window_size=8,
                 group_size=4,
                 alpha_init=1e-3,
                 beta_init=1e-3,
                 loss_fg_weight=0.1,
                 loss_fg_gamma=2.0,
                 loss_fg_alpha=0.25,
                 foreground_prior_prob=0.01,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.enhance_levels = tuple(enhance_levels)
        self.feature_strides = tuple(feature_strides)
        self.shared_weights = shared_weights
        self.loss_fg_weight = float(loss_fg_weight)
        self.loss_fg_gamma = float(loss_fg_gamma)
        self.loss_fg_alpha = float(loss_fg_alpha)
        self.foreground_prior_prob = float(foreground_prior_prob)
        self._foreground_initialized = False

        if len(self.feature_strides) != self.num_outs:
            raise ValueError('feature_strides must match FPN num_outs')
        if any(level < 0 or level >= self.num_outs
               for level in self.enhance_levels):
            raise ValueError('enhance_levels contains an invalid FPN index')

        need_foreground_head = (
            (enable_tafe and tafe_use_foreground_gate)
            or (enable_csta and csta_use_foreground)
            or self.loss_fg_weight > 0)
        block_kwargs = dict(
            channels=self.out_channels,
            enable_tafe=enable_tafe,
            tafe_use_foreground_gate=tafe_use_foreground_gate,
            tafe_use_texture_gate=tafe_use_texture_gate,
            enable_csta=enable_csta,
            csta_use_foreground=csta_use_foreground,
            csta_foreground_bias=csta_foreground_bias,
            csta_foreground_bias_weight=csta_foreground_bias_weight,
            inner_dim=csta_inner_dim,
            window_size=window_size,
            group_size=group_size,
            alpha_init=alpha_init,
            beta_init=beta_init,
            need_foreground_head=need_foreground_head)
        if shared_weights:
            self.shared_block = FSTABlock(**block_kwargs)
            self.blocks = None
        else:
            self.shared_block = None
            self.blocks = nn.ModuleList(
                [FSTABlock(**block_kwargs) for _ in self.enhance_levels])

    def init_weights(self):
        super().init_weights()
        if self._foreground_initialized:
            return
        # FPN's Xavier init reaches all nested convolutions.  Reinitialize the
        # binary foreground predictors with a sparse-object prior, matching the
        # standard FCOS classification-head practice and avoiding a dominant
        # all-background auxiliary loss at startup.
        blocks = ([self.shared_block] if self.shared_weights else self.blocks)
        for block in blocks:
            if block.foreground_head is None:
                continue
            predictor = block.foreground_head.layers[-1]
            nn.init.normal_(predictor.weight, std=0.01)
            nn.init.constant_(
                predictor.bias,
                bias_init_with_prob(self.foreground_prior_prob))
        self._foreground_initialized = True

    def _valid_masks(self, outputs, img_metas):
        masks = []
        for level, (feature, stride) in enumerate(
                zip(outputs, self.feature_strides)):
            batch, _, height, width = feature.shape
            if img_metas is None:
                masks.append(
                    torch.ones((batch, 1, height, width),
                               dtype=torch.bool,
                               device=feature.device))
                continue
            if len(img_metas) != batch:
                raise ValueError('img_metas length does not match batch size')
            level_masks = []
            ys = (torch.arange(height, device=feature.device) + 0.5) * stride
            xs = (torch.arange(width, device=feature.device) + 0.5) * stride
            for meta in img_metas:
                image_height, image_width = meta['img_shape'][:2]
                valid_y = ys < image_height
                valid_x = xs < image_width
                level_masks.append(valid_y[:, None] & valid_x[None, :])
            masks.append(torch.stack(level_masks, dim=0).unsqueeze(1))
        return masks

    def forward(self, inputs, img_metas=None, return_aux=False):
        outputs = list(super().forward(inputs))
        valid_masks = self._valid_masks(outputs, img_metas)
        foreground_logits = {}
        for block_index, level in enumerate(self.enhance_levels):
            block = self.shared_block if self.shared_weights else self.blocks[
                block_index]
            outputs[level], logits = block(outputs[level], valid_masks[level])
            if logits is not None:
                foreground_logits[level] = logits

        outputs = tuple(outputs)
        if not return_aux:
            return outputs
        auxiliary = dict(
            foreground_logits=foreground_logits, valid_masks=valid_masks)
        return outputs, auxiliary
