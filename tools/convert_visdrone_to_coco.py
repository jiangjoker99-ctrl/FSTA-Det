#!/usr/bin/env python3
"""Convert VisDrone-DET annotations to the COCO format used by MMDetection."""

import argparse
import json
from pathlib import Path

from PIL import Image


CLASSES = (
    'pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle',
    'awning-tricycle', 'bus', 'motor')


def annotation_rows(path):
    """Yield valid VisDrone rows as (x, y, w, h, score, category)."""
    with path.open('r', encoding='utf-8') as handle:
        for line_number, line in enumerate(handle, 1):
            fields = [field.strip() for field in line.strip().split(',')]
            if not line.strip():
                continue
            if len(fields) < 6:
                raise ValueError(f'{path}:{line_number}: expected at least 6 fields')
            try:
                x, y, width, height, score, category = map(int, fields[:6])
            except ValueError as error:
                raise ValueError(f'{path}:{line_number}: malformed annotation') from error
            if width > 0 and height > 0:
                yield x, y, width, height, score, category


def convert_split(root, split, output_path):
    split_root = root / f'VisDrone2019-DET-{split}'
    image_dir = split_root / 'images'
    annotation_dir = split_root / 'annotations'
    image_paths = sorted(image_dir.glob('*.jpg'))
    if not image_paths:
        raise FileNotFoundError(f'No JPG images found in {image_dir}')

    coco = {
        'info': {
            'description': 'VisDrone2019-DET converted for MMDetection',
            'version': '1.0',
        },
        'licenses': [],
        'images': [],
        'annotations': [],
        'categories': [
            {'id': index, 'name': name, 'supercategory': 'object'}
            for index, name in enumerate(CLASSES, 1)
        ],
    }
    annotation_id = 1
    ignored_count = 0
    foreground_count = 0

    for image_id, image_path in enumerate(image_paths, 1):
        annotation_path = annotation_dir / f'{image_path.stem}.txt'
        if not annotation_path.is_file():
            raise FileNotFoundError(f'Missing annotation file: {annotation_path}')
        with Image.open(image_path) as image:
            width, height = image.size
        coco['images'].append({
            'id': image_id,
            'file_name': image_path.name,
            'width': width,
            'height': height,
        })
        for x, y, box_width, box_height, score, category in annotation_rows(annotation_path):
            # VisDrone category 0 and score 0 denote ignored regions. Category 11
            # (others) is excluded from the ten detection classes and is also ignored.
            is_ignored = score == 0 or category == 0 or category == 11
            if not is_ignored and category not in range(1, len(CLASSES) + 1):
                continue
            coco_category = 1 if is_ignored else category
            coco['annotations'].append({
                'id': annotation_id,
                'image_id': image_id,
                'category_id': coco_category,
                'bbox': [x, y, box_width, box_height],
                'area': box_width * box_height,
                'iscrowd': int(is_ignored),
                'ignore': 0,
                'visdrone_ignore': int(is_ignored),
            })
            annotation_id += 1
            if is_ignored:
                ignored_count += 1
            else:
                foreground_count += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as handle:
        json.dump(coco, handle, ensure_ascii=False)
    print(f'{split}: {len(image_paths)} images, {foreground_count} foreground boxes, '
          f'{ignored_count} ignored boxes -> {output_path}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True,
                        help='Directory containing VisDrone2019-DET-train and -val')
    parser.add_argument('--splits', nargs='+', default=['train', 'val'],
                        choices=['train', 'val'])
    parser.add_argument('--output-dir', type=Path, default=None,
                        help='Defaults to <root>/annotations')
    parser.add_argument(
        '--output-suffix',
        default='',
        help='Optional suffix inserted before .json.')
    args = parser.parse_args()
    output_dir = args.output_dir or args.root / 'annotations'
    for split in args.splits:
        convert_split(
            args.root,
            split,
            output_dir / f'visdrone_{split}_coco{args.output_suffix}.json')


if __name__ == '__main__':
    main()
