"""AI-TOD TAFE ablation on P3/P4."""

_base_ = ['./tafe_csta_p3_p4.py']

model = dict(neck=dict(enable_csta=False, csta_use_foreground=False))
