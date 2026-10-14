"""Count parameters and registered-module MACs for one detector input."""

import argparse

import torch
from mmcv import Config
from mmcv.cnn import get_model_complexity_info

from mmdet.models import build_detector


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    parser.add_argument('--shape', type=int, nargs=2, default=[1000, 1600],
                        metavar=('HEIGHT', 'WIDTH'))
    args = parser.parse_args()
    torch.set_num_threads(2)
    cfg = Config.fromfile(args.config)
    cfg.model.backbone.init_cfg = None
    model = build_detector(cfg.model).eval()
    parameters = sum(value.numel() for value in model.parameters())
    model.forward = model.forward_dummy
    with torch.no_grad():
        macs, _ = get_model_complexity_info(
            model, (3, *args.shape), as_strings=False,
            print_per_layer_stat=False)
    print(f'Parameters: {parameters / 1e6:.4f} M')
    print(f'Registered-module MACs: {macs / 1e9:.4f} G')
    print('Functional CSTA attention, interpolation and detection postprocessing '
          'are outside this counter; this is not the full inference FLOP count.')


if __name__ == '__main__':
    main()
