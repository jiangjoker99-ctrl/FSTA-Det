#!/usr/bin/env python
"""Two-rank Gloo check for FSTA foreground-loss collective ordering."""

import os
import tempfile
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from mmdet.models import build_head
from mmdet.models.detectors.fsta_det import FSTADet


def worker(rank, world_size, init_file):
    dist.init_process_group(
        'gloo',
        init_method=f'file://{init_file}',
        rank=rank,
        world_size=world_size)
    try:
        head = build_head(dict(
            type='FCOSHead',
            num_classes=1,
            in_channels=4,
            feat_channels=4,
            stacked_convs=1,
            strides=[4, 8],
            regress_ranges=((-1, 32), (32, 1e8)),
            norm_cfg=None))
        owner = SimpleNamespace(
            bbox_head=head,
            neck=SimpleNamespace(
                loss_fg_weight=.1,
                loss_fg_alpha=.25,
                loss_fg_gamma=2.,
                feature_strides=(4, 8)),
            _ignore_mask=FSTADet._ignore_mask)
        features = [
            torch.zeros(1, 4, 2, 2, dtype=torch.float16),
            torch.zeros(1, 4, 1, 1, dtype=torch.float16),
        ]
        logits = {
            0: torch.randn(1, 1, 2, 2, requires_grad=True),
            1: torch.randn(1, 1, 1, 1, requires_grad=True),
        }
        valid = {
            0: torch.full((1, 1, 2, 2), rank == 1, dtype=torch.bool),
            1: torch.full((1, 1, 1, 1), rank == 1, dtype=torch.bool),
        }
        if rank == 1:
            boxes = [torch.tensor([[0., 0., 7., 7.]])]
            labels = [torch.tensor([0], dtype=torch.long)]
        else:
            boxes = [torch.zeros((0, 4))]
            labels = [torch.zeros((0, ), dtype=torch.long)]
        losses, statistics = FSTADet._foreground_loss(
            owner,
            features,
            dict(foreground_logits=logits, valid_masks=valid),
            boxes,
            labels,
            [torch.zeros((0, 4))])
        loss = losses['loss_fsta_fg']
        assert torch.isfinite(loss)
        assert all(torch.isfinite(value) for value in statistics.values())
        loss.backward()
        assert all(value.grad is not None for value in logits.values())
    finally:
        dist.destroy_process_group()


def main():
    with tempfile.TemporaryDirectory() as directory:
        init_file = os.path.join(directory, 'gloo-init')
        mp.spawn(worker, args=(2, init_file), nprocs=2, join=True)
    print('PASS two-rank foreground loss with asymmetric valid masks')


if __name__ == '__main__':
    main()
