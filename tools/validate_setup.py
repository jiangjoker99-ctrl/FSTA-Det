#!/usr/bin/env python
"""Validate configs, data/evaluator wiring and checkpoint behavior."""

import argparse
import copy
import logging
import os
import tempfile

import numpy as np
import torch
import torch.nn as nn
from mmcv import Config
from mmcv.runner import EpochBasedRunner
from torch.utils.data import DataLoader, Dataset

from mmdet.core.evaluation.eval_hooks import EvalHook, _is_valid_key_score
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config-dir', default='configs/visdrone',
        help='Directory containing the A0-A6 experiment configs.')
    return parser.parse_args()


def validate_configs(config_dir):
    rows = []
    for name in sorted(os.listdir(config_dir)):
        if not name.endswith('.py'):
            continue
        path = os.path.join(config_dir, name)
        cfg = Config.fromfile(path)
        model = build_detector(cfg.model)
        parameters = sum(parameter.numel() for parameter in model.parameters())
        trainable = sum(parameter.numel() for parameter in model.parameters()
                        if parameter.requires_grad)
        rows.append((name, cfg.model.type, cfg.model.neck.type,
                     list(cfg.model.bbox_head.strides), parameters, trainable))
        assert cfg.evaluation.interval == 1
        assert cfg.evaluation.save_best == 'bbox_mAP'
        assert cfg.checkpoint_config is None
    return rows


def validate_evaluator(config_path):
    cfg = Config.fromfile(config_path)
    dataset = build_dataset(cfg.data.val)
    num_classes = len(dataset.CLASSES)
    results = []
    for image_index in range(len(dataset)):
        per_class = [
            np.zeros((0, 5), dtype=np.float32)
            for _ in range(num_classes)
        ]
        if image_index == 0:
            per_class[0] = np.array(
                [[0., 0., 1., 1., 0.001]], dtype=np.float32)
        results.append(per_class)
    eval_kwargs = dict(
        metric='bbox',
        proposal_nums=tuple(cfg.evaluation.proposal_nums),
        metric_items=list(cfg.evaluation.metric_items))
    metrics = dataset.evaluate(results, **eval_kwargs)
    empty_results = [[
        np.zeros((0, 5), dtype=np.float32) for _ in range(num_classes)
    ] for _ in range(len(dataset))]
    empty_metrics = dataset.evaluate(empty_results, **eval_kwargs)
    required = {
        'bbox_mAP', 'bbox_mAP_50', 'bbox_mAP_75', 'bbox_mAP_s',
        'bbox_AR@100', 'bbox_AR@300', 'bbox_AR@1000'
    }
    assert required <= metrics.keys(), metrics
    assert required <= empty_metrics.keys(), empty_metrics
    assert all(empty_metrics[key] == 0 for key in required)
    return len(dataset), metrics


def validate_real_ignore_pipeline(config_path):
    cfg = Config.fromfile(config_path)
    train_cfg = copy.deepcopy(cfg.data.train)
    for transform in train_cfg.pipeline:
        if transform.type == 'RandomFlip':
            transform.flip_ratio = 0.0
    dataset = build_dataset(train_cfg)
    sample_index = None
    annotation = None
    for index in range(len(dataset)):
        candidate = dataset.get_ann_info(index)
        if len(candidate['bboxes_ignore']):
            sample_index = index
            annotation = candidate
            break
    assert sample_index is not None
    sample = dataset[sample_index]
    assert 'gt_bboxes_ignore' in sample
    meta = sample['img_metas'].data
    scale = np.asarray(meta['scale_factor'], dtype=np.float32)
    expected = annotation['bboxes_ignore'] * scale
    actual = sample['gt_bboxes_ignore'].data.cpu().numpy()
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-4)

    flipped_cfg = copy.deepcopy(cfg.data.train)
    for transform in flipped_cfg.pipeline:
        if transform.type == 'RandomFlip':
            transform.flip_ratio = 1.0
    flipped_dataset = build_dataset(flipped_cfg)
    flipped_sample = flipped_dataset[sample_index]
    flipped_meta = flipped_sample['img_metas'].data
    assert flipped_meta['flip']
    expected_flipped = expected.copy()
    image_width = flipped_meta['img_shape'][1]
    expected_flipped[:, 0] = image_width - expected[:, 2]
    expected_flipped[:, 2] = image_width - expected[:, 0]
    actual_flipped = flipped_sample['gt_bboxes_ignore'].data.cpu().numpy()
    np.testing.assert_allclose(
        actual_flipped, expected_flipped, rtol=1e-5, atol=1e-4)

    loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=0,
        dist=False,
        shuffle=False)
    batch = next(iter(loader))
    assert 'gt_bboxes_ignore' in batch
    return sample_index, len(actual)


def validate_disabled_equivalence(config_path):
    cfg = Config.fromfile(config_path)
    fsta_model = build_detector(cfg.model).eval()
    reference_cfg = copy.deepcopy(cfg.model)
    reference_cfg.type = 'FCOS'
    reference_cfg.neck.type = 'FPN'
    custom_keys = [
        'enhance_levels', 'feature_strides', 'shared_weights', 'enable_tafe',
        'tafe_use_foreground_gate', 'tafe_use_texture_gate', 'enable_csta',
        'csta_use_foreground', 'csta_foreground_bias',
        'csta_foreground_bias_weight', 'csta_inner_dim', 'window_size',
        'group_size', 'alpha_init', 'beta_init', 'loss_fg_weight',
        'loss_fg_gamma', 'loss_fg_alpha', 'foreground_prior_prob'
    ]
    for key in custom_keys:
        reference_cfg.neck.pop(key, None)
    reference_model = build_detector(reference_cfg).eval()
    reference_model.load_state_dict(fsta_model.state_dict(), strict=True)
    image = torch.randn(1, 3, 96, 160)
    with torch.no_grad():
        fsta_features = fsta_model.extract_feat(image)
        reference_features = reference_model.extract_feat(image)
        fsta_predictions = fsta_model.bbox_head(fsta_features)
        reference_predictions = reference_model.bbox_head(reference_features)
    for actual, expected in zip(fsta_features, reference_features):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for actual_group, expected_group in zip(fsta_predictions,
                                            reference_predictions):
        for actual, expected in zip(actual_group, expected_group):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def validate_best_checkpoint_logic():
    assert _is_valid_key_score(0.0)
    assert not _is_valid_key_score(None)
    assert not _is_valid_key_score(float('nan'))
    class TinyModel(nn.Linear):
        def train_step(self, data, optimizer):
            del data, optimizer
            return dict(loss=self.weight.sum() * 0, num_samples=1)

        def val_step(self, data, optimizer=None):
            return self.train_step(data, optimizer)

    class ScoreDataset(Dataset):
        def __init__(self):
            self.score = 0.0

        def __len__(self):
            return 1

        def __getitem__(self, index):
            return torch.tensor([float(index)])

        def evaluate(self, results, logger=None, **kwargs):
            del results, logger, kwargs
            return {'bbox_mAP': self.score}

    dataset = ScoreDataset()
    loader = DataLoader(dataset, batch_size=1)

    def test_fn(model, data_loader):
        del model
        return [None] * len(data_loader.dataset)

    with tempfile.TemporaryDirectory() as directory:
        model = TinyModel(2, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=.01, momentum=.9)
        runner = EpochBasedRunner(
            model=model,
            optimizer=optimizer,
            work_dir=directory,
            logger=logging.getLogger('fsta-best-check'),
            meta={})
        hook = EvalHook(
            loader,
            interval=1,
            save_best='bbox_mAP',
            rule='greater',
            test_fn=test_fn)
        hook.before_run(runner)
        runner._epoch = 0
        hook._do_evaluate(runner)
        first_path = runner.meta['hook_msgs']['best_ckpt']
        assert os.path.isfile(first_path)
        first_checkpoint = torch.load(
            first_path, map_location='cpu', weights_only=False)
        assert {'meta', 'state_dict', 'optimizer'} <= first_checkpoint.keys()

        runner._epoch = 1
        hook._do_evaluate(runner)
        assert runner.meta['hook_msgs']['best_ckpt'] == first_path
        assert os.path.isfile(first_path)

        dataset.score = .20
        runner._epoch = 2
        hook._do_evaluate(runner)
        final_path = runner.meta['hook_msgs']['best_ckpt']
        assert final_path != first_path
        assert not os.path.exists(first_path)
        assert os.path.isfile(final_path)
        checkpoints = [
            name for name in os.listdir(directory) if name.endswith('.pth')
        ]
        assert checkpoints == [os.path.basename(final_path)]

        restored_model = TinyModel(2, 1)
        restored_optimizer = torch.optim.SGD(
            restored_model.parameters(), lr=.01, momentum=.9)
        restored_runner = EpochBasedRunner(
            model=restored_model,
            optimizer=restored_optimizer,
            work_dir=directory,
            logger=logging.getLogger('fsta-resume-check'),
            meta={})
        restored_runner.resume(final_path, map_location='cpu')
        for expected, actual in zip(model.parameters(),
                                    restored_model.parameters()):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def main():
    args = parse_args()
    rows = validate_configs(args.config_dir)
    for row in rows:
        name, detector, neck, strides, parameters, trainable = row
        print(f'{name}: detector={detector}, neck={neck}, strides={strides}, '
              f'params={parameters:,}, trainable={trainable:,}')
    config_path = os.path.join(args.config_dir, 'a6_fsta_det.py')
    val_size, metrics = validate_evaluator(config_path)
    ignore_index, ignore_count = validate_real_ignore_pipeline(config_path)
    validate_disabled_equivalence(
        os.path.join(args.config_dir, 'a1_fcos_p2_p5.py'))
    validate_best_checkpoint_logic()
    print('validation_images:', val_size)
    print('evaluation_keys:', sorted(metrics))
    print('bbox_mAP:', metrics['bbox_mAP'])
    print('real_ignore_pipeline:', ignore_index, ignore_count)
    print('disabled_equivalence: exact features and raw FCOS predictions')
    print('best_checkpoint_logic: AP=0 saved, restored, one best retained')


if __name__ == '__main__':
    main()
