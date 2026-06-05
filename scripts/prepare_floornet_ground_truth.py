#!/usr/bin/env python3
"""Prepare FloorNet room masks as floorplan ground truth for evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw


def _bytes_feature(example: Any, name: str) -> bytes:
    values = example.features.feature[name].bytes_list.value
    if not values:
        raise KeyError(f"Missing bytes feature: {name}")
    return bytes(values[0])


def _int64_feature(example: Any, name: str) -> np.ndarray:
    return np.asarray(example.features.feature[name].int64_list.value, dtype=np.int64)


def _decode_label_mask(payload: bytes) -> np.ndarray:
    values = np.frombuffer(payload, dtype=np.uint8)
    side = int(round(values.size ** 0.5))
    if side * side != values.size:
        raise ValueError(f"Expected square uint8 mask, got {values.size} bytes")
    return values.reshape(side, side)


def _extract_outer_polygon(mask: np.ndarray, epsilon_ratio: float) -> np.ndarray:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No foreground region in room mask")
    contour = max(contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(contour, True)
    polygon = cv2.approxPolyDP(contour, epsilon_ratio * perimeter, True)[:, 0, :]
    return polygon.astype(np.float32)


def _boundary(mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    return cv2.morphologyEx(mask.astype(np.uint8) * 255, cv2.MORPH_GRADIENT, kernel)


def _save_overlay(
    output_path: Path,
    room_mask: np.ndarray,
    polygon: np.ndarray,
    corners: np.ndarray,
) -> None:
    image = Image.fromarray((room_mask.astype(np.uint8) * 235)).convert("RGB")
    draw = ImageDraw.Draw(image)
    closed = [tuple(point) for point in polygon] + [tuple(polygon[0])]
    draw.line(closed, fill=(230, 35, 35), width=3)
    for index, point in enumerate(polygon):
        x, y = (int(round(value)) for value in point)
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=(20, 85, 230))
        draw.text((x + 4, y + 2), str(index), fill=(20, 20, 20))
    for x, y, label in corners:
        if x < 0 or y < 0:
            continue
        draw.ellipse((int(x) - 1, int(y) - 1, int(x) + 1, int(y) + 1), fill=(30, 30, 30))
    image.save(output_path)


def _prepare_example(
    example: Any,
    output_dir: Path,
    index: int,
    epsilon_ratio: float,
) -> dict[str, Any]:
    scan_id = _bytes_feature(example, "image_path").decode("utf-8", errors="replace")
    prefix = f"{index:06d}_{scan_id.replace('/', '_')}"
    room_labels = _decode_label_mask(_bytes_feature(example, "room"))
    room_mask = room_labels > 0
    corners_raw = _int64_feature(example, "corner").reshape(-1, 3)
    num_corners = int(_int64_feature(example, "num_corners")[0])
    corners = corners_raw[:num_corners]
    polygon = _extract_outer_polygon(room_mask, epsilon_ratio)
    boundary = _boundary(room_mask)

    Image.fromarray(room_labels).save(output_dir / f"{prefix}_room_labels.png")
    Image.fromarray(room_mask.astype(np.uint8) * 255).save(output_dir / f"{prefix}_gt_room_mask.png")
    Image.fromarray(boundary).save(output_dir / f"{prefix}_gt_boundary.png")
    np.save(output_dir / f"{prefix}_gt_room_mask.npy", room_mask)
    np.save(output_dir / f"{prefix}_gt_corners.npy", corners)
    _save_overlay(output_dir / f"{prefix}_gt_overlay.png", room_mask, polygon, corners)

    floorplan = {
        "method": "floornet_room_mask_outer_polygon_ground_truth",
        "scan_id": scan_id,
        "image_shape": list(room_mask.shape),
        "pixel_polygon": polygon.astype(float).tolist(),
        "corner_points": corners.astype(int).tolist(),
        "room_labels": sorted(int(value) for value in np.unique(room_labels)),
        "outputs": {
            "room_labels": f"{prefix}_room_labels.png",
            "room_mask": f"{prefix}_gt_room_mask.png",
            "boundary": f"{prefix}_gt_boundary.png",
            "overlay": f"{prefix}_gt_overlay.png",
        },
    }
    with (output_dir / f"{prefix}_gt_floorplan.json").open("w", encoding="utf-8") as file:
        json.dump(floorplan, file, ensure_ascii=False, indent=2)
    return {
        "index": int(index),
        "scan_id": scan_id,
        "prefix": prefix,
        "room_labels": floorplan["room_labels"],
        "room_pixels": int(room_mask.sum()),
        "polygon_vertices": int(len(polygon)),
        "num_corners": int(num_corners),
        "gt_floorplan": f"{prefix}_gt_floorplan.json",
    }


def prepare_floornet_ground_truth(
    tfrecords_path: str | Path,
    output_dir: str | Path,
    max_examples: int = 50,
    epsilon_ratio: float = 0.005,
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
        examples.append(_prepare_example(example, output_dir, index, epsilon_ratio))

    summary = {
        "tfrecords": str(tfrecords_path),
        "max_examples": int(max_examples),
        "epsilon_ratio": float(epsilon_ratio),
        "example_count": len(examples),
        "examples": examples,
    }
    with (output_dir / "floornet_ground_truth_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tfrecords", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-examples", type=int, default=50)
    parser.add_argument("--epsilon-ratio", type=float, default=0.005)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = prepare_floornet_ground_truth(
        args.tfrecords,
        args.output_dir,
        max_examples=args.max_examples,
        epsilon_ratio=args.epsilon_ratio,
    )
    print(f"Prepared {summary['example_count']} FloorNet ground-truth examples")
    for example in summary["examples"][:20]:
        print(
            f"  {example['prefix']}: pixels={example['room_pixels']} "
            f"vertices={example['polygon_vertices']} corners={example['num_corners']}"
        )


if __name__ == "__main__":
    sys.exit(main())
