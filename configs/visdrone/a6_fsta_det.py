"""A6: full FSTA-Det with foreground-guided TAFE and CSTA."""

_base_ = ['./a1_fcos_p2_p5.py']

model = dict(
    neck=dict(
        enable_tafe=True,
        tafe_use_foreground_gate=True,
        tafe_use_texture_gate=True,
        enable_csta=True,
        csta_use_foreground=True,
        csta_foreground_bias=False,
        csta_inner_dim=64,
        window_size=8,
        group_size=4,
        loss_fg_weight=0.1))
