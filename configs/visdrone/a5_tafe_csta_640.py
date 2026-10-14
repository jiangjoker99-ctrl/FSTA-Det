"""A5: TAFE+CSTA with fixed 640x640 input for a controlled run."""

_base_ = ['./a5_tafe_csta_appearance.py']

# Re-declare the normalization constants because MMCV config files do not
# expose variables defined in an indirect base file while executing this file.
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True)

# Keep aspect ratio during resize, then pad to an exact square tensor.
# This matches a 640x640 detector input while avoiding geometric distortion.
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(640, 640), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=(640, 640)),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['img', 'gt_bboxes', 'gt_labels', 'gt_bboxes_ignore']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(640, 640),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size=(640, 640)),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline))

runner = dict(type='EpochBasedRunner', max_epochs=12)
evaluation = dict(interval=1, save_best='bbox_mAP', rule='greater')
checkpoint_config = None
