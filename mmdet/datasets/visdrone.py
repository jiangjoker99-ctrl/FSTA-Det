# Copyright (c) OpenMMLab. All rights reserved.
"""VisDrone dataset adapter with an explicit, internally consistent protocol."""

import copy

import numpy as np

from .api_wrappers import COCO, COCOeval
from .builder import DATASETS
from .coco import CocoDataset


class VisDroneCOCOeval(COCOeval):
    """COCOeval whose six AP entries use the same final maxDet value."""

    def summarize(self):
        if not self.eval:
            raise RuntimeError('Please run accumulate() first')
        if self.params.iouType not in ('bbox', 'segm'):
            return super().summarize()

        def summarize_one(ap=True, iou_thr=None, area_rng='all', max_dets=None):
            params = self.params
            max_dets = params.maxDets[-1] if max_dets is None else max_dets
            title = 'Average Precision' if ap else 'Average Recall'
            metric = '(AP)' if ap else '(AR)'
            iou_text = (f'{params.iouThrs[0]:0.2f}:{params.iouThrs[-1]:0.2f}'
                        if iou_thr is None else f'{iou_thr:0.2f}')
            area_index = [
                index for index, label in enumerate(params.areaRngLbl)
                if label == area_rng
            ]
            det_index = [
                index for index, value in enumerate(params.maxDets)
                if value == max_dets
            ]
            if ap:
                values = self.eval['precision']
                if iou_thr is not None:
                    values = values[np.where(iou_thr == params.iouThrs)[0]]
                values = values[:, :, :, area_index, det_index]
            else:
                values = self.eval['recall']
                if iou_thr is not None:
                    values = values[np.where(iou_thr == params.iouThrs)[0]]
                values = values[:, :, area_index, det_index]
            score = np.mean(values[values > -1]) if np.any(values > -1) else -1
            print(' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>4d} ] '
                  '= {:0.3f}'.format(title, metric, iou_text, area_rng,
                                    max_dets, score))
            return score

        first, second, final = self.params.maxDets
        self.stats = np.array([
            summarize_one(True, max_dets=final),
            summarize_one(True, iou_thr=.5, max_dets=final),
            summarize_one(True, iou_thr=.75, max_dets=final),
            summarize_one(True, area_rng='small', max_dets=final),
            summarize_one(True, area_rng='medium', max_dets=final),
            summarize_one(True, area_rng='large', max_dets=final),
            summarize_one(False, max_dets=first),
            summarize_one(False, max_dets=second),
            summarize_one(False, max_dets=final),
            summarize_one(False, area_rng='small', max_dets=final),
            summarize_one(False, area_rng='medium', max_dets=final),
            summarize_one(False, area_rng='large', max_dets=final),
        ])


@DATASETS.register_module()
class VisDroneDataset(CocoDataset):
    """Ten-class VisDrone detection dataset.

    VisDrone ignored regions have no semantic category.  The converted COCO
    files store them as crowd annotations under one placeholder category.
    During evaluation this adapter duplicates only those crowd annotations in
    memory for every evaluated category, so detections of any class can be
    ignored without rewriting the source JSON.
    """

    CLASSES = (
        'pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle',
        'awning-tricycle', 'bus', 'motor')

    def _build_coco_eval(self, coco_gt, coco_det, iou_type):
        return VisDroneCOCOeval(coco_gt, coco_det, iou_type)

    def _expand_category_agnostic_ignores(self, coco_gt):
        expanded = COCO()
        expanded.dataset = copy.deepcopy(coco_gt.dataset)
        annotations = expanded.dataset.get('annotations', [])
        ignored = [
            annotation for annotation in annotations
            if annotation.get('visdrone_ignore', 0)
            or annotation.get('iscrowd', 0)
        ]
        next_id = max(
            (annotation.get('id', 0) for annotation in annotations),
            default=0) + 1
        duplicates = []
        for annotation in ignored:
            for category_id in self.cat_ids:
                if category_id == annotation['category_id']:
                    continue
                duplicate = copy.deepcopy(annotation)
                duplicate['id'] = next_id
                duplicate['category_id'] = category_id
                duplicate['iscrowd'] = 1
                duplicate['ignore'] = 0
                duplicate['visdrone_ignore'] = 1
                duplicates.append(duplicate)
                next_id += 1
        annotations.extend(duplicates)
        expanded.createIndex()
        return expanded

    def evaluate_det_segm(self,
                          results,
                          result_files,
                          coco_gt,
                          metrics,
                          logger=None,
                          classwise=False,
                          proposal_nums=(100, 300, 1000),
                          iou_thrs=None,
                          metric_items=None):
        expanded_gt = self._expand_category_agnostic_ignores(coco_gt)
        return super().evaluate_det_segm(
            results,
            result_files,
            expanded_gt,
            metrics,
            logger=logger,
            classwise=classwise,
            proposal_nums=proposal_nums,
            iou_thrs=iou_thrs,
            metric_items=metric_items)
