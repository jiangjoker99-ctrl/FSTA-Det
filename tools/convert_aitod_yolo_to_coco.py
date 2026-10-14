#!/usr/bin/env python
"""Convert the Kaggle AI-TOD YOLO mirror into COCO JSON for MMDetection 2.x.

The mirror contains already generated 800x800 AI-TOD tiles. This converter
does not resize or alter boxes; it only changes the annotation container.
"""
import argparse
import json
from pathlib import Path

from PIL import Image


CLASSES = (
    "airplane", "bridge", "storage-tank", "ship",
    "swimming-pool", "vehicle", "person", "wind-mill",
)


def convert_split(root, split):
    image_dir = root / "data" / "images" / split
    label_dir = root / "data" / "labels" / split
    out_dir = root / "annotations"
    out_dir.mkdir(parents=True, exist_ok=True)
    image_paths = sorted(image_dir.glob("*.png"))
    if not image_paths:
        raise FileNotFoundError(f"No PNG images found in {image_dir}")

    images = []
    annotations = []
    ann_id = 1
    skipped = 0
    for image_id, image_path in enumerate(image_paths, start=1):
        with Image.open(image_path) as image:
            width, height = image.size
        images.append(
            dict(
                id=image_id,
                file_name=image_path.name,
                width=width,
                height=height,
            ))
        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue
        for raw in label_path.read_text(
                encoding="utf-8", errors="ignore").splitlines():
            values = raw.strip().split()
            if len(values) < 5:
                continue
            try:
                category = int(float(values[0]))
                cx, cy, bw, bh = (float(v) for v in values[1:5])
            except ValueError:
                skipped += 1
                continue
            if category < 0 or category >= len(CLASSES):
                skipped += 1
                continue
            x = (cx - bw / 2.0) * width
            y = (cy - bh / 2.0) * height
            box_w = bw * width
            box_h = bh * height
            x = max(0.0, min(x, float(width)))
            y = max(0.0, min(y, float(height)))
            box_w = max(0.0, min(box_w, float(width) - x))
            box_h = max(0.0, min(box_h, float(height) - y))
            if box_w <= 0.0 or box_h <= 0.0:
                skipped += 1
                continue
            annotations.append(
                dict(
                    id=ann_id,
                    image_id=image_id,
                    category_id=category + 1,
                    bbox=[x, y, box_w, box_h],
                    area=box_w * box_h,
                    iscrowd=0,
                    ignore=0,
                    segmentation=[],
                ))
            ann_id += 1

    payload = dict(
        info=dict(
            description="AI-TOD tiles converted from YOLO labels",
            version="1.0",
            year=2026,
        ),
        licenses=[],
        images=images,
        annotations=annotations,
        categories=[
            dict(id=index + 1, name=name, supercategory="object")
            for index, name in enumerate(CLASSES)
        ],
    )
    output = out_dir / f"aitod_{split}_coco.json"
    output.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8")
    print(
        f"{split}: images={len(images)} annotations={len(annotations)} "
        f"skipped={skipped} -> {output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    args = parser.parse_args()
    for split in args.splits:
        convert_split(args.root, split)


if __name__ == "__main__":
    main()
