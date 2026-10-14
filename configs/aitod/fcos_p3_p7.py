"""AI-TOD FCOS P3-P7 baseline, batch 16 and 12 epochs."""

_base_ = ['./tafe_csta_p3_p4.py']

model = dict(
    type='FCOS',
    neck=dict(
        _delete_=True,
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=5,
        relu_before_extra_convs=True))
