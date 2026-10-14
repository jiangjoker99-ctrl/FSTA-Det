from mmdet.datasets.api_wrappers import COCO
from mmdet.datasets.visdrone import VisDroneCOCOeval, VisDroneDataset


def build_coco(images, annotations, categories):
    coco = COCO()
    coco.dataset = dict(
        images=images,
        annotations=annotations,
        categories=categories,
        licenses=[],
        info={})
    coco.createIndex()
    return coco


def test_ap_uses_final_max_det_consistently_for_dense_images():
    annotations = []
    detections = []
    for index in range(150):
        x = float((index % 15) * 4)
        y = float((index // 15) * 4)
        annotations.append(dict(
            id=index + 1,
            image_id=1,
            category_id=1,
            bbox=[x, y, 2., 2.],
            area=4.,
            iscrowd=0))
        detections.append(dict(
            image_id=1,
            category_id=1,
            bbox=[x, y, 2., 2.],
            score=1. - index * 1e-4))
    ground_truth = build_coco(
        [dict(id=1, width=64, height=48)],
        annotations,
        [dict(id=1, name='object')])
    evaluator = VisDroneCOCOeval(
        ground_truth, ground_truth.loadRes(detections), 'bbox')
    evaluator.params.maxDets = [100, 300, 1000]
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    assert evaluator.stats[0] > 0.99


def test_ignore_region_is_category_agnostic_during_evaluation():
    ground_truth = build_coco(
        [dict(id=1, width=32, height=32)],
        [
            dict(
                id=1,
                image_id=1,
                category_id=1,
                bbox=[0., 0., 10., 10.],
                area=100.,
                iscrowd=1,
                visdrone_ignore=1),
            dict(
                id=2,
                image_id=1,
                category_id=2,
                bbox=[20., 20., 5., 5.],
                area=25.,
                iscrowd=0),
        ],
        [dict(id=1, name='a'), dict(id=2, name='b')])
    dataset = object.__new__(VisDroneDataset)
    dataset.cat_ids = [1, 2]
    expanded = dataset._expand_category_agnostic_ignores(ground_truth)
    category_two_crowds = [
        annotation for annotation in expanded.dataset['annotations']
        if annotation['category_id'] == 2 and annotation.get('iscrowd', 0)
    ]
    assert len(category_two_crowds) == 1

    detections = [
        dict(
            image_id=1,
            category_id=2,
            bbox=[0., 0., 10., 10.],
            score=.99),
        dict(
            image_id=1,
            category_id=2,
            bbox=[20., 20., 5., 5.],
            score=.90),
    ]
    evaluator = VisDroneCOCOeval(
        expanded, expanded.loadRes(detections), 'bbox')
    evaluator.params.catIds = [2]
    evaluator.params.maxDets = [100, 300, 1000]
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    assert evaluator.stats[0] > 0.99
