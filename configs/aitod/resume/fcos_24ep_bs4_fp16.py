"""Continue a batch-16 FCOS checkpoint with batch 4 and FP16."""

_base_ = ['../fcos_p3_p7.py']

data = dict(samples_per_gpu=4)
optimizer = dict(lr=0.00125)
lr_config = dict(_delete_=True, policy='step', warmup=None, step=[20, 23])
fp16 = dict(loss_scale='dynamic')
runner = dict(type='EpochBasedRunner', max_epochs=24)
checkpoint_config = dict(interval=24, max_keep_ckpts=2)
evaluation = dict(interval=24)
