import os

dataset_type = 'VisDroneDataset'
data_root = os.path.join(os.environ.get('VISDRONE_ROOT', 'data/visdrone'), '')
classes = (
    'pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle',
    'awning-tricycle', 'bus', 'motor')

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1600, 1000), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['img', 'gt_bboxes', 'gt_labels', 'gt_bboxes_ignore']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1600, 1000),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/visdrone_train_coco.json',
        img_prefix=data_root + 'VisDrone2019-DET-train/images/',
        classes=classes,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/visdrone_val_coco.json',
        img_prefix=data_root + 'VisDrone2019-DET-val/images/',
        classes=classes,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/visdrone_val_coco.json',
        img_prefix=data_root + 'VisDrone2019-DET-val/images/',
        classes=classes,
        pipeline=test_pipeline))

# Validate after every epoch. The top-level experiment config disables regular
# per-epoch checkpoints, while EvalHook retains the best bbox_mAP model.
evaluation = dict(
    interval=1,
    metric='bbox',
    proposal_nums=(100, 300, 1000),
    metric_items=[
        'mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l',
        'AR@100', 'AR@300', 'AR@1000', 'AR_s@1000', 'AR_m@1000',
        'AR_l@1000'
    ],
    save_best='bbox_mAP',
    rule='greater')
