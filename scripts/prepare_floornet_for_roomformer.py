#!/usr/bin/env python3
"""Convert FloorNet TFRecords into a RoomFormer-style COCO dataset."""

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


def _float_feature(example: Any, name: str) -> np.ndarray:
    return np.asarray(example.features.feature[name].float_list.value, dtype=np.float32)


def _decode_label_mask(payload: bytes) -> np.ndarray:
    values = np.frombuffer(payload, dtype=np.uint8)
    side = int(round(values.size ** 0.5))
    if side * side != values.size:
        raise ValueError(f"Expected square uint8 mask, got {values.size} bytes")
    return values.reshape(side, side)


def _normalize(values: np.ndarray, percentile: float = 99.0) -> np.ndarray:
    values = values.astype(np.float32, copy=False)
    positive = values[values > 0]
    if positive.size == 0:
        return np.zeros_like(values, dtype=np.float32)
    scale = np.percentile(positive, percentile)
    return np.clip(values / max(float(scale), 1e-8), 0.0, 1.0)


def _parse_int_list(value: str) -> set[int]:
    if not value:
        return set()
    return {int(item.strip()) for item in value.split(",") if item.strip()}


def _parse_axis_pair(value: str) -> tuple[int, int]:
    parts = [int(item.strip()) for item in value.split(",") if item.strip()]
    if len(parts) != 2 or any(part not in {0, 1, 2} for part in parts) or parts[0] == parts[1]:
        raise ValueError("--point-axes must contain two distinct axes from {0,1,2}, e.g. 0,1")
    return parts[0], parts[1]


def _points_to_density(
    points: np.ndarray,
    image_size: int,
    bounds_margin: float,
    point_axes: tuple[int, int],
    flip_x: bool,
    flip_y: bool,
) -> np.ndarray:
    horizontal = points[:, list(point_axes)]
    valid = np.isfinite(horizontal).all(axis=1)
    horizontal = horizontal[valid]
    lower = np.percentile(horizontal, 1, axis=0)
    upper = np.percentile(horizontal, 99, axis=0)
    margin = (upper - lower) * float(bounds_margin)
    lower -= margin
    upper += margin
    scale = (image_size - 1) / np.maximum(upper - lower, 1e-8)
    xy = ((horizontal - lower) * scale).astype(np.int32)
    keep = np.logical_and(xy >= 0, xy < image_size).all(axis=1)
    xy = xy[keep]
    counts = np.zeros((image_size, image_size), dtype=np.uint32)
    x = image_size - 1 - xy[:, 0] if flip_x else xy[:, 0]
    y = xy[:, 1] if flip_y else image_size - 1 - xy[:, 1]
    np.add.at(counts, (y, x), 1)
    density = _normalize(np.log1p(counts))
    return density.astype(np.float32)


def _point_indices_to_density(point_indices: np.ndarray, image_size: int) -> np.ndarray:
    valid = point_indices[np.logical_and(point_indices >= 0, point_indices < image_size * image_size)]
    counts = np.zeros((image_size, image_size), dtype=np.uint32)
    if valid.size:
        y = valid // image_size
        x = valid % image_size
        np.add.at(counts, (y.astype(np.int32), x.astype(np.int32)), 1)
    density = _normalize(np.log1p(counts))
    return density.astype(np.float32)


def _density_to_rgb(density: np.ndarray) -> np.ndarray:
    gray = (255.0 * (1.0 - np.sqrt(np.clip(density, 0.0, 1.0)))).astype(np.uint8)
    return np.repeat(gray[:, :, None], 3, axis=2)


def _density_to_gray(density: np.ndarray) -> np.ndarray:
    return (255.0 * (1.0 - np.sqrt(np.clip(density, 0.0, 1.0)))).astype(np.uint8)


def _extract_room_polygons(
    room_labels: np.ndarray,
    min_area: int,
    epsilon_ratio: float,
    include_labels: set[int],
    exclude_labels: set[int],
) -> list[dict[str, Any]]:
    polygons: list[dict[str, Any]] = []
    for room_label in sorted(int(value) for value in np.unique(room_labels) if value > 0):
        if include_labels and room_label not in include_labels:
            continue
        if room_label in exclude_labels:
            continue
        mask = (room_labels == room_label).astype(np.uint8)
        count, components, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for component_id in range(1, count):
            area = int(stats[component_id, cv2.CC_STAT_AREA])
            if area < min_area:
                continue
            component = (components == component_id).astype(np.uint8)
            contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            contour = max(contours, key=cv2.contourArea)
            perimeter = cv2.arcLength(contour, True)
            polygon = cv2.approxPolyDP(contour, epsilon_ratio * perimeter, True)[:, 0, :]
            if len(polygon) < 3:
                continue
            x, y, width, height = cv2.boundingRect(polygon.astype(np.int32))
            polygons.append(
                {
                    "room_label": int(room_label),
                    "polygon": polygon.astype(float),
                    "area": float(cv2.contourArea(polygon.astype(np.float32))),
                    "bbox": [float(x), float(y), float(width), float(height)],
                }
            )
    return polygons


def _save_overlay(path: Path, density: np.ndarray, polygons: list[dict[str, Any]]) -> None:
    image = Image.fromarray(_density_to_rgb(density)).convert("RGB")
    draw = ImageDraw.Draw(image)
    colors = [
        (230, 25, 75),
        (60, 180, 75),
        (255, 225, 25),
        (0, 130, 200),
        (245, 130, 48),
        (145, 30, 180),
        (70, 240, 240),
        (240, 50, 230),
    ]
    for idx, item in enumerate(polygons):
        polygon = item["polygon"]
        color = colors[idx % len(colors)]
        closed = [tuple(point) for point in polygon] + [tuple(polygon[0])]
        draw.line(closed, fill=color, width=2)
    image.save(path)


def _convert_example(
    example: Any,
    image_id: int,
    annotation_start_id: int,
    image_dir: Path,
    overlay_dir: Path,
    image_size: int,
    min_room_area: int,
    epsilon_ratio: float,
    bounds_margin: float,
    point_axes: tuple[int, int],
    flip_x: bool,
    flip_y: bool,
    density_source: str,
    include_labels: set[int],
    exclude_labels: set[int],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    scan_id = _bytes_feature(example, "image_path").decode("utf-8", errors="replace")
    file_stem = f"{image_id:06d}_{scan_id.replace('/', '_')}"
    room_labels = _decode_label_mask(_bytes_feature(example, "room"))
    points = _float_feature(example, "points").reshape(-1, 3)
    if density_source == "point_indices":
        point_indices = _int64_feature(example, "point_indices")
        density = _point_indices_to_density(point_indices, image_size)
    else:
        density = _points_to_density(points, image_size, bounds_margin, point_axes, flip_x, flip_y)
    polygons = _extract_room_polygons(
        room_labels,
        min_room_area,
        epsilon_ratio,
        include_labels,
        exclude_labels,
    )

    image_path = image_dir / f"{file_stem}.png"
    Image.fromarray(_density_to_gray(density), mode="L").save(image_path)
    _save_overlay(overlay_dir / f"{file_stem}_overlay.png", density, polygons)

    image_record = {
        "id": int(image_id),
        "file_name": image_path.name,
        "height": int(image_size),
        "width": int(image_size),
        "scan_id": scan_id,
    }
    annotations = []
    for local_idx, item in enumerate(polygons):
        polygon = item["polygon"]
        annotations.append(
            {
                "id": int(annotation_start_id + local_idx),
                "image_id": int(image_id),
                "category_id": 1,
                "room_label": int(item["room_label"]),
                "segmentation": [polygon.reshape(-1).astype(float).tolist()],
                "bbox": item["bbox"],
                "area": float(item["area"]),
                "iscrowd": 0,
            }
        )
    summary = {
        "image_id": int(image_id),
        "scan_id": scan_id,
        "file_name": image_path.name,
        "room_polygon_count": len(polygons),
        "room_labels": sorted(int(value) for value in np.unique(room_labels) if value > 0),
        "used_room_labels": sorted(
            int(value)
            for value in np.unique(room_labels)
            if value > 0
            and (not include_labels or int(value) in include_labels)
            and int(value) not in exclude_labels
        ),
    }
    return image_record, annotations, summary


def prepare_floornet_for_roomformer(
    tfrecords_path: str | Path,
    output_dir: str | Path,
    split: str,
    max_examples: int,
    image_size: int = 256,
    min_room_area: int = 80,
    epsilon_ratio: float = 0.005,
    bounds_margin: float = 0.05,
    point_axes: tuple[int, int] = (0, 1),
    flip_x: bool = False,
    flip_y: bool = False,
    density_source: str = "point_indices",
    include_labels: set[int] | None = None,
    exclude_labels: set[int] | None = None,
) -> dict[str, Any]:
    import tensorflow as tf

    output_dir = Path(output_dir)
    image_dir = output_dir / split
    annotation_dir = output_dir / "annotations"
    overlay_dir = output_dir / "overlays" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    annotation_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)
    include_labels = include_labels or set()
    exclude_labels = exclude_labels if exclude_labels is not None else {15, 16}

    dataset = tf.data.TFRecordDataset(str(tfrecords_path))
    images = []
    annotations = []
    examples = []
    next_annotation_id = 1
    for image_id, raw in enumerate(dataset.take(max_examples), start=1):
        example = tf.train.Example()
        example.ParseFromString(bytes(raw.numpy()))
        image_record, image_annotations, summary = _convert_example(
            example,
            image_id,
            next_annotation_id,
            image_dir,
            overlay_dir,
            image_size,
            min_room_area,
            epsilon_ratio,
            bounds_margin,
            point_axes,
            flip_x,
            flip_y,
            density_source,
            include_labels,
            exclude_labels,
        )
        images.append(image_record)
        annotations.extend(image_annotations)
        examples.append(summary)
        next_annotation_id += len(image_annotations)

    coco = {
        "info": {
            "description": "FloorNet converted for RoomFormer-style room polygon training",
            "source_tfrecords": str(tfrecords_path),
            "split": split,
        },
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "room"}],
    }
    annotation_path = annotation_dir / f"{split}.json"
    with annotation_path.open("w", encoding="utf-8") as file:
        json.dump(coco, file, ensure_ascii=False)

    summary = {
        "tfrecords": str(tfrecords_path),
        "output_dir": str(output_dir),
        "split": split,
        "image_count": len(images),
        "annotation_count": len(annotations),
        "parameters": {
            "image_size": int(image_size),
            "min_room_area": int(min_room_area),
            "epsilon_ratio": float(epsilon_ratio),
            "bounds_margin": float(bounds_margin),
            "point_axes": list(point_axes),
            "flip_x": bool(flip_x),
            "flip_y": bool(flip_y),
            "density_source": density_source,
            "include_labels": sorted(include_labels),
            "exclude_labels": sorted(exclude_labels),
        },
        "annotation_path": str(annotation_path),
        "examples": examples,
    }
    with (output_dir / f"{split}_conversion_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tfrecords", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--max-examples", type=int, default=100)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--min-room-area", type=int, default=80)
    parser.add_argument("--epsilon-ratio", type=float, default=0.005)
    parser.add_argument("--bounds-margin", type=float, default=0.05)
    parser.add_argument("--point-axes", type=_parse_axis_pair, default=(0, 1))
    parser.add_argument("--flip-x", action="store_true")
    parser.add_argument("--flip-y", action="store_true")
    parser.add_argument(
        "--density-source",
        choices=("point_indices", "points"),
        default="point_indices",
        help="Use FloorNet raster point_indices by default; points mode projects raw xyz coordinates.",
    )
    parser.add_argument("--include-labels", type=_parse_int_list, default=set())
    parser.add_argument("--exclude-labels", type=_parse_int_list, default={15, 16})
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = prepare_floornet_for_roomformer(
        args.tfrecords,
        args.output_dir,
        args.split,
        args.max_examples,
        image_size=args.image_size,
        min_room_area=args.min_room_area,
        epsilon_ratio=args.epsilon_ratio,
        bounds_margin=args.bounds_margin,
        point_axes=args.point_axes,
        flip_x=args.flip_x,
        flip_y=args.flip_y,
        density_source=args.density_source,
        include_labels=args.include_labels,
        exclude_labels=args.exclude_labels,
    )
    print(
        f"Converted {summary['image_count']} images and "
        f"{summary['annotation_count']} room polygons to {summary['output_dir']}"
    )
    print(f"Annotation: {summary['annotation_path']}")


if __name__ == "__main__":
    sys.exit(main())
