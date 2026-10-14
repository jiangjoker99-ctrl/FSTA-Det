"""A2: frequency residual and texture gate without foreground gating."""

_base_ = ['./a1_fcos_p2_p5.py']

model = dict(
    neck=dict(
        enable_tafe=True,
        tafe_use_foreground_gate=False,
        tafe_use_texture_gate=True,
        enable_csta=False,
        csta_use_foreground=False,
        loss_fg_weight=0.0))
