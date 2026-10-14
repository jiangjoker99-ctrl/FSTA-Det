import os

dataset_type = "CocoDataset"
data_root = os.path.join(os.environ.get("AITOD_ROOT", "data/aitod"), "")
classes = (
    "airplane", "bridge", "storage-tank", "ship",
    "swimming-pool", "vehicle", "person", "wind-mill",
)

# Official AI-TOD split: 11,214 train, 2,804 validation, and 14,018 test
# images. The mirror stores train and validation images together in the
# data/images/train directory and the held-out test images in data/images/val.
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True,
)

train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(type="Resize", img_scale=(800, 800), keep_ratio=True),
    dict(type="RandomFlip", flip_ratio=0.5),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="Pad", size_divisor=32),
    dict(type="DefaultFormatBundle"),
    dict(
        type="Collect",
        keys=["img", "gt_bboxes", "gt_labels", "gt_bboxes_ignore"]),
]

test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(800, 800),
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="RandomFlip"),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="Pad", size_divisor=32),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ]),
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file=data_root + "annotations/aitod_train.json",
        img_prefix=data_root + "data/images/train/",
        classes=classes,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + "annotations/aitod_val.json",
        img_prefix=data_root + "data/images/train/",
        classes=classes,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + "annotations/aitod_val_coco.json",
        img_prefix=data_root + "data/images/val/",
        classes=classes,
        pipeline=test_pipeline),
)

# Use maxDets=(100, 300, 1000), as in the released SET/AI-TOD protocol.
evaluation = dict(
    interval=12,
    metric="bbox",
    proposal_nums=(100, 300, 1000),
    metric_items=[
        "mAP", "mAP_50", "mAP_75", "mAP_s", "mAP_m", "mAP_l",
        "AR@100", "AR@300", "AR@1000",
        "AR_s@1000", "AR_m@1000", "AR_l@1000",
    ],
    save_best="bbox_mAP",
    rule="greater",
)
