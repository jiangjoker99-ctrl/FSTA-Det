"""AI-TOD appearance-only CSTA ablation on P3/P4."""

_base_ = ['./tafe_csta_p3_p4.py']

model = dict(
    neck=dict(
        enable_tafe=False,
        tafe_use_foreground_gate=False,
        loss_fg_weight=0.0))
