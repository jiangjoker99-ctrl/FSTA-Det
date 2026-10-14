#!/usr/bin/env python
"""Low-resource model, loss, inference and checkpoint smoke test."""

import argparse
import json
import os
import tempfile
import time

import mmcv
import numpy as np
import torch
from mmcv import Config

from mmdet.models import build_detector


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', default='configs/visdrone/a6_fsta_det.py')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--real-image', default=None)
    parser.add_argument(
        '--annotation',
        default=None)
    return parser.parse_args()


def prepare_image(path, device):
    if path:
        image = mmcv.imread(path)
        original_shape = image.shape
        original_h, original_w = original_shape[:2]
        image = mmcv.imrescale(image, (160, 96))
        resized_h, resized_w = image.shape[:2]
        scale_factor = np.array([
            resized_w / original_w, resized_h / original_h,
            resized_w / original_w, resized_h / original_h
        ], dtype=np.float32)
        image = mmcv.imnormalize(
            image,
            np.array([123.675, 116.28, 103.53], dtype=np.float32),
            np.array([58.395, 57.12, 57.375], dtype=np.float32),
            to_rgb=True)
        tensor = torch.from_numpy(image.transpose(2, 0, 1)).float()[None]
    else:
        tensor = torch.randn(1, 3, 96, 160)
        original_shape = (96, 160, 3)
        scale_factor = np.ones(4, dtype=np.float32)
    return tensor.to(device), original_shape, scale_factor


def load_annotations(annotation_path, image_path, original_shape, resized_shape,
                     device):
    if image_path is None or not annotation_path or not os.path.isfile(annotation_path):
        return ([torch.tensor([[20., 16., 42., 38.]], device=device)],
                [torch.tensor([3], dtype=torch.long, device=device)],
                [torch.tensor([[120., 60., 150., 88.]], device=device)])

    with open(annotation_path, 'r') as file:
        coco = json.load(file)
    filename = os.path.basename(image_path)
    image_info = next(item for item in coco['images']
                      if item['file_name'] == filename)
    category_to_label = {
        item['id']: index for index, item in enumerate(coco['categories'])
    }
    original_h, original_w = original_shape[:2]
    resized_h, resized_w = resized_shape
    scale_x, scale_y = resized_w / original_w, resized_h / original_h
    boxes, labels, ignored = [], [], []
    for annotation in coco['annotations']:
        if annotation['image_id'] != image_info['id']:
            continue
        x, y, width, height = annotation['bbox']
        box = [x * scale_x, y * scale_y, (x + width) * scale_x,
               (y + height) * scale_y]
        if annotation.get('ignore', 0) or annotation.get('iscrowd', 0):
            ignored.append(box)
        else:
            boxes.append(box)
            labels.append(category_to_label[annotation['category_id']])
    boxes = torch.tensor(boxes, dtype=torch.float32, device=device).reshape(-1, 4)
    labels = torch.tensor(labels, dtype=torch.long, device=device)
    ignored = torch.tensor(
        ignored, dtype=torch.float32, device=device).reshape(-1, 4)
    return [boxes], [labels], [ignored]


def build_smoke_model(config_path, device):
    cfg = Config.fromfile(config_path)
    cfg.model.backbone.init_cfg = None
    # Preserve the external 256-channel FPN interface while reducing attention
    # work for a CPU smoke test.
    cfg.model.neck.csta_inner_dim = 16
    cfg.model.neck.window_size = 4
    cfg.model.neck.group_size = 2
    cfg.model.test_cfg.score_thr = 0.0
    cfg.model.test_cfg.nms_pre = 200
    cfg.model.test_cfg.max_per_img = 20
    model = build_detector(cfg.model)
    model.init_weights()
    return model.to(device), cfg


def main():
    args = parse_args()
    torch.set_num_threads(2)
    device = torch.device(args.device)
    model, cfg = build_smoke_model(args.config, device)
    image, original_shape, scale_factor = prepare_image(
        args.real_image, device)
    height, width = image.shape[-2:]
    metas = [
        dict(
            img_shape=(height, width, 3),
            ori_shape=original_shape,
            pad_shape=(height, width, 3),
            scale_factor=scale_factor,
            flip=False)
    ]
    boxes, labels, ignored = load_annotations(
        args.annotation, args.real_image, original_shape, (height, width),
        device)

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-5, momentum=0.9)
    model.train()
    losses = model.forward_train(
        image, metas, boxes, labels, gt_bboxes_ignore=ignored)
    total_loss = sum(value.mean() for key, value in losses.items()
                     if 'loss' in key)
    assert torch.isfinite(total_loss), losses
    total_loss.backward()
    gradient_names = [
        name for name, parameter in model.named_parameters()
        if ('tafe' in name or 'csta' in name or 'foreground_head' in name)
        and parameter.grad is not None
    ]
    if not gradient_names:
        raise AssertionError('No gradient reached the FSTA modules')
    optimizer.step()

    # Empty-GT images are common in detection pipelines and must remain
    # numerically stable for both FCOS and the auxiliary foreground loss.
    model.zero_grad(set_to_none=True)
    empty_losses = model.forward_train(
        image,
        metas,
        [boxes[0].new_zeros((0, 4))],
        [labels[0].new_zeros((0, ), dtype=torch.long)],
        gt_bboxes_ignore=[ignored[0].new_zeros((0, 4))])
    empty_total = sum(value.mean() for key, value in empty_losses.items()
                      if 'loss' in key)
    assert torch.isfinite(empty_total), empty_losses

    # Exercise a padded two-image batch with mixed non-empty/empty GT and an
    # ignore region on the smaller logical image.
    mixed_image = image.repeat(2, 1, 1, 1)
    mixed_metas = [metas[0].copy(), metas[0].copy()]
    mixed_metas[1]['img_shape'] = (max(height - 17, 1),
                                   max(width - 23, 1), 3)
    mixed_losses = model.forward_train(
        mixed_image,
        mixed_metas,
        [boxes[0], boxes[0].new_zeros((0, 4))],
        [labels[0], labels[0].new_zeros((0, ), dtype=torch.long)],
        gt_bboxes_ignore=[
            ignored[0],
            boxes[0].new_tensor([[0., 0., 8., 8.]])
        ])
    mixed_total = sum(value.mean() for key, value in mixed_losses.items()
                      if 'loss' in key)
    assert torch.isfinite(mixed_total), mixed_losses

    model.eval()
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.no_grad():
        result = model.simple_test(image, metas, rescale=False)
        original_result = model.simple_test(image, metas, rescale=True)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    detections = sum(len(per_class) for per_class in result[0])
    if detections == 0:
        raise AssertionError('Smoke inference produced no structured boxes')
    for per_class in result[0]:
        if len(per_class):
            assert per_class.shape[1] == 5
            assert np.isfinite(per_class).all()
    for resized, original in zip(result[0], original_result[0]):
        assert len(resized) == len(original)
        if len(resized):
            np.testing.assert_allclose(
                original[:, :4] * scale_factor,
                resized[:, :4],
                rtol=1e-5,
                atol=1e-4)
            np.testing.assert_allclose(
                original[:, 4], resized[:, 4], rtol=0, atol=0)

    with tempfile.TemporaryDirectory() as directory:
        checkpoint = os.path.join(directory, 'smoke.pth')
        torch.save(dict(state_dict=model.state_dict()), checkpoint)
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        missing, unexpected = model.load_state_dict(
            state['state_dict'], strict=False)
        assert not missing and not unexpected, (missing, unexpected)

    print('config:', args.config)
    print('feature_strides:', cfg.model.neck.feature_strides)
    print('real_gt_boxes:', int(boxes[0].shape[0]))
    print('real_ignored_boxes:', int(ignored[0].shape[0]))
    print('losses:', {key: float(value.detach().mean().cpu())
                      for key, value in losses.items()})
    print('empty_gt_losses:', {
        key: float(value.detach().mean().cpu())
        for key, value in empty_losses.items()
    })
    print('mixed_batch_losses:', {
        key: float(value.detach().mean().cpu())
        for key, value in mixed_losses.items()
    })
    print('fsta_gradient_parameters:', len(gradient_names))
    print('detections:', detections)
    print('rescale_to_original: passed')
    print('single_inference_seconds:', round(elapsed, 4))
    if device.type == 'cuda':
        print('peak_memory_mib:', round(
            torch.cuda.max_memory_allocated(device) / 1024**2, 2))


if __name__ == '__main__':
    main()
