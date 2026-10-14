"""Continue a 12-epoch FSTA checkpoint through epoch 24."""

_base_ = ['../tafe_csta_p3_p4.py']

lr_config = dict(_delete_=True, policy='step', warmup=None, step=[20, 23])
runner = dict(type='EpochBasedRunner', max_epochs=24)
checkpoint_config = dict(interval=24, max_keep_ckpts=2)
evaluation = dict(interval=24)
