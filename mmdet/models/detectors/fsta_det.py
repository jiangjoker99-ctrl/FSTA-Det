# Copyright (c) OpenMMLab. All rights reserved.
"""FSTA-Det detector integration for MMDetection 2.x."""

import torch
import torch.nn.functional as F

from mmdet.core import bbox2result, reduce_mean
from ..builder import DETECTORS
from .single_stage import SingleStageDetector


@DETECTORS.register_module()
class FSTADet(SingleStageDetector):
    """FCOS-compatible detector with explicit FSTA auxiliary outputs."""

    def _extract_feat_with_aux(self, img, img_metas=None, return_aux=False):
        features = self.backbone(img)
        auxiliary = None
        if self.with_neck:
            if return_aux:
                features, auxiliary = self.neck(
                    features, img_metas=img_metas, return_aux=True)
            else:
                features = self.neck(features, img_metas=img_metas)
        return features, auxiliary

    def extract_feat(self, img):
        features, _ = self._extract_feat_with_aux(
            img, img_metas=None, return_aux=False)
        return features

    @staticmethod
    def _ignore_mask(points, ignore_boxes):
        if ignore_boxes is None or ignore_boxes.numel() == 0:
            return points.new_zeros((points.size(0), ), dtype=torch.bool)
        xs = points[:, 0:1]
        ys = points[:, 1:2]
        left = xs - ignore_boxes[None, :, 0]
        top = ys - ignore_boxes[None, :, 1]
        right = ignore_boxes[None, :, 2] - xs
        bottom = ignore_boxes[None, :, 3] - ys
        distances = torch.stack((left, top, right, bottom), dim=-1)
        return (distances.min(dim=-1)[0] > 0).any(dim=1)

    def _foreground_loss(self,
                         features,
                         auxiliary,
                         gt_bboxes,
                         gt_labels,
                         gt_bboxes_ignore=None):
        logits_by_level = auxiliary.get('foreground_logits', {})
        if not logits_by_level or self.neck.loss_fg_weight <= 0:
            return {}, {}

        feature_sizes = [feature.shape[-2:] for feature in features]
        points = self.bbox_head.prior_generator.grid_priors(
            feature_sizes,
            # Keep target assignment in FP32.  In particular, the final FCOS
            # regression range uses 1e8, which is not representable in FP16.
            dtype=torch.float32,
            device=features[0].device)
        target_bboxes = [boxes.float() for boxes in gt_bboxes]
        labels_by_level, _ = self.bbox_head.get_targets(
            points, target_bboxes, gt_labels)

        batch_size = features[0].size(0)
        if gt_bboxes_ignore is None:
            gt_bboxes_ignore = [None] * batch_size

        first_logits = next(iter(logits_by_level.values())).float()
        # A one-element zero keeps the graph connected without reducing a
        # large half-precision feature map, which could overflow before * 0.
        total_loss = first_logits.reshape(-1)[:1].sum() * 0
        total_positive = features[0].new_tensor(0.0)
        statistics = {}

        for level, logits in logits_by_level.items():
            height, width = logits.shape[-2:]
            labels = labels_by_level[level].view(batch_size, height, width)
            target = ((labels >= 0)
                      & (labels < self.bbox_head.num_classes)).to(torch.float32)
            valid = auxiliary['valid_masks'][level][:, 0] & (labels >= 0)

            for image_index, ignore_boxes in enumerate(gt_bboxes_ignore):
                ignored = self._ignore_mask(points[level], ignore_boxes)
                valid[image_index] &= ~ignored.view(height, width)

            valid_flat = valid.reshape(-1)
            target_flat = target.reshape(-1)
            logits_flat = logits[:, 0].reshape(-1).float()
            positive = (target_flat * valid_flat.to(target_flat.dtype)).sum()
            total_positive = total_positive + positive

            cross_entropy = F.binary_cross_entropy_with_logits(
                logits_flat, target_flat, reduction='none')
            probability = logits_flat.sigmoid()
            probability_t = probability * target_flat + (
                1 - probability) * (1 - target_flat)
            alpha = self.neck.loss_fg_alpha
            alpha_t = alpha * target_flat + (1 - alpha) * (1 - target_flat)
            focal_weight = alpha_t * (
                1 - probability_t).pow(self.neck.loss_fg_gamma)
            total_loss = total_loss + (
                focal_weight * cross_entropy * valid_flat.float()).sum()

            stride = self.neck.feature_strides[level]
            # All ranks visit the same enhanced levels, so logging collectives
            # remain ordered even when one rank contains only ignored pixels.
            statistics[f'fsta_fg_pos_s{stride}'] = reduce_mean(
                positive.detach())

        # This collective is unconditional.  Conditional all-reduce calls can
        # deadlock when different DDP ranks have different valid/positive data.
        normalizer = reduce_mean(total_positive.detach()).clamp_min(1.0)
        loss = total_loss / normalizer
        losses = dict(loss_fsta_fg=loss * self.neck.loss_fg_weight)
        return losses, statistics

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        # Run BaseDetector's metadata bookkeeping without invoking the regular
        # SingleStageDetector extraction path a second time.
        super(SingleStageDetector, self).forward_train(img, img_metas)
        features, auxiliary = self._extract_feat_with_aux(
            img, img_metas=img_metas, return_aux=True)
        losses = self.bbox_head.forward_train(features, img_metas, gt_bboxes,
                                              gt_labels, gt_bboxes_ignore)
        foreground_losses, statistics = self._foreground_loss(
            features, auxiliary, gt_bboxes, gt_labels, gt_bboxes_ignore)
        losses.update(foreground_losses)
        losses.update(statistics)
        return losses

    def simple_test(self, img, img_metas, rescale=False):
        features, _ = self._extract_feat_with_aux(
            img, img_metas=img_metas, return_aux=False)
        results_list = self.bbox_head.simple_test(
            features, img_metas, rescale=rescale)
        return [
            bbox2result(det_bboxes, det_labels,
                        self.bbox_head.num_classes)
            for det_bboxes, det_labels in results_list
        ]

    def aug_test(self, imgs, img_metas, rescale=False):
        features = [
            self._extract_feat_with_aux(
                image, img_metas=meta, return_aux=False)[0]
            for image, meta in zip(imgs, img_metas)
        ]
        results_list = self.bbox_head.aug_test(
            features, img_metas, rescale=rescale)
        return [
            bbox2result(det_bboxes, det_labels,
                        self.bbox_head.num_classes)
            for det_bboxes, det_labels in results_list
        ]
