#!/usr/bin/env python
"""Benchmark end-to-end FSTA-Det inference with preprocessing and NMS."""

import argparse
import statistics
import time

import torch

from mmdet.apis import inference_detector, init_detector


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/visdrone/a6_fsta_det.py')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--image', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--repeats', type=int, default=50)
    return parser.parse_args()


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def main():
    args = parse_args()
    if args.warmup < 0 or args.repeats < 1:
        raise ValueError('warmup must be non-negative and repeats positive')
    device = torch.device(args.device)
    model = init_detector(args.config, args.checkpoint, device=args.device)
    for _ in range(args.warmup):
        inference_detector(model, args.image)
    synchronize(device)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)

    timings = []
    for _ in range(args.repeats):
        started = time.perf_counter()
        inference_detector(model, args.image)
        synchronize(device)
        timings.append(time.perf_counter() - started)

    print('config:', args.config)
    print('checkpoint:', args.checkpoint)
    print('image:', args.image)
    print('device:', args.device)
    print('batch_size: 1')
    print('precision: model checkpoint/default runtime')
    print('timing_scope: image loading + preprocessing + model + NMS')
    print('warmup:', args.warmup, 'repeats:', args.repeats)
    print('mean_seconds:', round(statistics.mean(timings), 6))
    print('median_seconds:', round(statistics.median(timings), 6))
    if device.type == 'cuda':
        print('inference_peak_memory_mib:', round(
            torch.cuda.max_memory_allocated(device) / 1024**2, 2))


if __name__ == '__main__':
    main()
