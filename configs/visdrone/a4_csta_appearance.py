"""A4: appearance-only CSTA without TAFE or foreground guidance."""

_base_ = ['./a1_fcos_p2_p5.py']

model = dict(
    neck=dict(
        enable_tafe=False,
        tafe_use_foreground_gate=False,
        enable_csta=True,
        csta_use_foreground=False,
        csta_foreground_bias=False,
        loss_fg_weight=0.0))
