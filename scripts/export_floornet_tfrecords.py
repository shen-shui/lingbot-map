#!/usr/bin/env python3
"""Export FloorNet TFRecords into masks, corners, and point arrays."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


def _bytes_feature(example: Any, name: str) -> bytes:
    values = example.features.feature[name].bytes_list.value
    if not values:
        raise KeyError(f"Missing bytes feature: {name}")
    return bytes(values[0])


def _int64_feature(example: Any, name: str) -> np.ndarray:
    return np.asarray(example.features.feature[name].int64_list.value, dtype=np.int64)


def _float_feature(example: Any, name: str) -> np.ndarray:
    return np.asarray(example.features.feature[name].float_list.value, dtype=np.float32)


def _decode_label_mask(payload: bytes) -> np.ndarray:
    values = np.frombuffer(payload, dtype=np.uint8)
    side = int(round(values.size ** 0.5))
    if side * side != values.size:
        raise ValueError(f"Expected square uint8 mask, got {values.size} bytes")
    return values.reshape(side, side)


def _label_to_rgb(mask: np.ndarray) -> Image.Image:
    palette = np.array(
        [
            [0, 0, 0],
            [230, 25, 75],
            [60, 180, 75],
            [255, 225, 25],
            [0, 130, 200],
            [245, 130, 48],
            [145, 30, 180],
            [70, 240, 240],
            [240, 50, 230],
            [210, 245, 60],
            [250, 190, 190],
            [0, 128, 128],
            [230, 190, 255],
            [170, 110, 40],
            [255, 250, 200],
            [128, 0, 0],
            [170, 255, 195],
            [128, 128, 0],
            [255, 215, 180],
            [0, 0, 128],
        ],
        dtype=np.uint8,
    )
    return Image.fromarray(palette[mask % len(palette)])


def _save_corner_overlay(path: Path, room_mask: np.ndarray, corners: np.ndarray) -> None:
    image = _label_to_rgb(room_mask).convert("RGB")
    draw = ImageDraw.Draw(image)
    for x, y, label in corners:
        x_i, y_i = int(x), int(y)
        if x_i < 0 or y_i < 0:
            continue
        color = (255, 255, 255) if int(label) == 0 else (20, 20, 20)
        draw.ellipse((x_i - 2, y_i - 2, x_i + 2, y_i + 2), fill=color, outline=(255, 0, 0))
    image.save(path)


def _export_example(example: Any, output_dir: Path, index: int, save_points: bool) -> dict[str, Any]:
    scan_id = _bytes_feature(example, "image_path").decode("utf-8", errors="replace")
    safe_scan_id = scan_id.replace("/", "_")
    prefix = f"{index:06d}_{safe_scan_id}"

    room = _decode_label_mask(_bytes_feature(example, "room"))
    icon = _decode_label_mask(_bytes_feature(example, "icon"))
    corners_raw = _int64_feature(example, "corner").reshape(-1, 3)
    num_corners = int(_int64_feature(example, "num_corners")[0])
    corners = corners_raw[:num_corners]
    flags = _int64_feature(example, "flags")

    np.save(output_dir / f"{prefix}_room.npy", room)
    np.save(output_dir / f"{prefix}_icon.npy", icon)
    np.save(output_dir / f"{prefix}_corners.npy", corners)
    Image.fromarray(room).save(output_dir / f"{prefix}_room_label.png")
    Image.fromarray(icon).save(output_dir / f"{prefix}_icon_label.png")
    _label_to_rgb(room).save(output_dir / f"{prefix}_room_color.png")
    _label_to_rgb(icon).save(output_dir / f"{prefix}_icon_color.png")
    _save_corner_overlay(output_dir / f"{prefix}_corners_overlay.png", room, corners)

    points_shape = None
    if save_points:
        points = _float_feature(example, "points").reshape(-1, 3)
        point_indices = _int64_feature(example, "point_indices")
        np.save(output_dir / f"{prefix}_points.npy", points)
        np.save(output_dir / f"{prefix}_point_indices.npy", point_indices)
        points_shape = list(points.shape)

    metadata = {
        "index": int(index),
        "scan_id": scan_id,
        "prefix": prefix,
        "flags": flags.astype(int).tolist(),
        "num_corners": int(num_corners),
        "room_shape": list(room.shape),
        "room_labels": sorted(int(v) for v in np.unique(room)),
        "icon_shape": list(icon.shape),
        "icon_labels": sorted(int(v) for v in np.unique(icon)),
        "corners_shape": list(corners.shape),
        "points_shape": points_shape,
        "outputs": {
            "room_label": f"{prefix}_room_label.png",
            "room_color": f"{prefix}_room_color.png",
            "icon_label": f"{prefix}_icon_label.png",
            "icon_color": f"{prefix}_icon_color.png",
            "corners_overlay": f"{prefix}_corners_overlay.png",
        },
    }
    with (output_dir / f"{prefix}_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)
    return metadata


def export_floornet_tfrecords(
    tfrecords_path: str | Path,
    output_dir: str | Path,
    max_examples: int = 20,
    save_points: bool = False,
) -> dict[str, Any]:
    import tensorflow as tf

    tfrecords_path = Path(tfrecords_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    examples = []
    dataset = tf.data.TFRecordDataset(str(tfrecords_path))
    for index, raw in enumerate(dataset.take(max_examples)):
        example = tf.train.Example()
        example.ParseFromString(bytes(raw.numpy()))
        examples.append(_export_example(example, output_dir, index, save_points))

    summary = {
        "tfrecords": str(tfrecords_path),
        "max_examples": int(max_examples),
        "save_points": bool(save_points),
        "example_count": len(examples),
        "examples": examples,
    }
    with (output_dir / "floornet_export_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tfrecords", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-examples", type=int, default=20)
    parser.add_argument("--save-points", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = export_floornet_tfrecords(
        args.tfrecords,
        args.output_dir,
        max_examples=args.max_examples,
        save_points=args.save_points,
    )
    print(f"Exported {summary['example_count']} FloorNet examples to {args.output_dir}")
    for example in summary["examples"]:
        print(
            f"  {example['prefix']}: room_labels={example['room_labels']} "
            f"icon_labels={example['icon_labels']} corners={example['num_corners']}"
        )


if __name__ == "__main__":
    sys.exit(main())
