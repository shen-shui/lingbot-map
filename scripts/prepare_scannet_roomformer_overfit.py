#!/usr/bin/env python3
"""Prepare a one-scene ScanNet weak-polygon dataset for RoomFormer overfit tests."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--density", type=Path, required=True, help="RoomFormer input density PNG.")
    parser.add_argument("--structure-mask", type=Path, required=True, help="ScanNet structure_mask.png.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scene-id", type=str, default="scene0000_00")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--mask-threshold", type=int, default=128)
    parser.add_argument(
        "--label-mode",
        choices=["interior", "structure"],
        default="interior",
        help=(
            "Use interior free-space polygons inferred from structure lines, or the raw "
            "structure contours. RoomFormer expects room/interior polygons."
        ),
    )
    parser.add_argument(
        "--interior-fallback",
        choices=["hull", "structure", "none"],
        default="hull",
        help="Fallback when no enclosed interior can be flood-filled from structure lines.",
    )
    parser.add_argument("--close-kernel", type=int, default=9)
    parser.add_argument("--dilate-kernel", type=int, default=5)
    parser.add_argument("--dilate-iterations", type=int, default=1)
    parser.add_argument("--minimum-area", type=float, default=120.0)
    parser.add_argument("--epsilon-ratio", type=float, default=0.01)
    parser.add_argument("--max-polygons", type=int, default=20)
    return parser.parse_args()


def load_roomformer_density(path: Path, image_size: int) -> Image.Image:
    return Image.open(path).convert("L").resize((image_size, image_size), Image.BILINEAR)


def load_structure_mask(path: Path, image_size: int, threshold: int) -> np.ndarray:
    gray = np.asarray(Image.open(path).convert("L").resize((image_size, image_size), Image.NEAREST))
    # Our ScanNet mask visualization uses black structure on white background.
    return gray < threshold


def mask_to_polygons(mask: np.ndarray, args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.label_mode == "interior":
        image = infer_interior_mask(mask, args)
        polygons = binary_mask_to_polygons(image, args)
        if polygons:
            return polygons
        if args.interior_fallback == "none":
            return []
        if args.interior_fallback == "hull":
            hull = structure_hull_mask(mask, args)
            return binary_mask_to_polygons(hull, args)
        image = mask.astype(np.uint8)
    else:
        image = mask.astype(np.uint8)

    return binary_mask_to_polygons(image, args)


def infer_interior_mask(structure_mask: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    """Infer weak interior polygons from projected structure lines.

    ScanNet gives a structural line/mask cue, while RoomFormer is trained to output
    filled room polygons. For an overfit sanity check, close and thicken structure
    lines, flood-fill exterior background from the image border, then keep enclosed
    free-space components as weak room/interior labels.
    """
    walls = structure_mask.astype(np.uint8)
    if args.close_kernel > 1:
        kernel = np.ones((args.close_kernel, args.close_kernel), dtype=np.uint8)
        walls = cv2.morphologyEx(walls, cv2.MORPH_CLOSE, kernel)
    if args.dilate_kernel > 1 and args.dilate_iterations > 0:
        kernel = np.ones((args.dilate_kernel, args.dilate_kernel), dtype=np.uint8)
        walls = cv2.dilate(walls, kernel, iterations=args.dilate_iterations)

    free = (walls == 0).astype(np.uint8)
    flood = free.copy()
    height, width = flood.shape
    flood_mask = np.zeros((height + 2, width + 2), dtype=np.uint8)
    for x in range(width):
        if flood[0, x]:
            cv2.floodFill(flood, flood_mask, (x, 0), 2)
        if flood[height - 1, x]:
            cv2.floodFill(flood, flood_mask, (x, height - 1), 2)
    for y in range(height):
        if flood[y, 0]:
            cv2.floodFill(flood, flood_mask, (0, y), 2)
        if flood[y, width - 1]:
            cv2.floodFill(flood, flood_mask, (width - 1, y), 2)
    return (flood == 1).astype(np.uint8)


def structure_hull_mask(structure_mask: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    points = np.column_stack(np.nonzero(structure_mask.astype(bool)))
    if len(points) < 3:
        return np.zeros_like(structure_mask, dtype=np.uint8)
    xy = points[:, ::-1].astype(np.int32)
    hull = cv2.convexHull(xy)
    mask = np.zeros_like(structure_mask, dtype=np.uint8)
    cv2.fillPoly(mask, [hull], 1)
    if args.dilate_kernel > 1 and args.dilate_iterations > 0:
        kernel = np.ones((args.dilate_kernel, args.dilate_kernel), dtype=np.uint8)
        mask = cv2.erode(mask, kernel, iterations=max(1, args.dilate_iterations))
    return mask


def binary_mask_to_polygons(image: np.ndarray, args: argparse.Namespace) -> list[dict[str, Any]]:
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        area = float(cv2.contourArea(contour))
        if area < args.minimum_area:
            continue
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, args.epsilon_ratio * perimeter, True)[:, 0, :]
        if len(approx) < 3:
            continue
        x, y, width, height = cv2.boundingRect(approx.astype(np.int32))
        polygons.append(
            {
                "polygon": approx.astype(np.float32),
                "area": float(cv2.contourArea(approx.astype(np.float32))),
                "bbox": [float(x), float(y), float(width), float(height)],
            }
        )
        if len(polygons) >= args.max_polygons:
            break
    return polygons


def save_overlay(path: Path, density: Image.Image, polygons: list[dict[str, Any]]) -> None:
    image = density.convert("RGB")
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
        poly = item["polygon"]
        pts = [tuple(float(v) for v in point) for point in poly]
        draw.line(pts + [pts[0]], fill=colors[idx % len(colors)], width=2)
    image.save(path)


def build_coco(split: str, file_name: str, polygons: list[dict[str, Any]], image_size: int) -> dict[str, Any]:
    annotations = []
    for idx, item in enumerate(polygons, 1):
        polygon = item["polygon"]
        annotations.append(
            {
                "id": idx,
                "image_id": 1,
                "category_id": 1,
                "segmentation": [polygon.reshape(-1).astype(float).tolist()],
                "bbox": item["bbox"],
                "area": float(item["area"]),
                "iscrowd": 0,
                "weak_label_source": "scannet_structure_mask",
            }
        )
    return {
        "images": [
            {
                "id": 1,
                "file_name": file_name,
                "height": int(image_size),
                "width": int(image_size),
                "split": split,
            }
        ],
        "annotations": annotations,
        "categories": [{"id": 1, "name": "room"}],
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "annotations").mkdir(exist_ok=True)
    (args.output_dir / "overlays").mkdir(exist_ok=True)

    density = load_roomformer_density(args.density, args.image_size)
    structure_mask = load_structure_mask(args.structure_mask, args.image_size, args.mask_threshold)
    polygons = mask_to_polygons(structure_mask, args)
    if not polygons:
        raise ValueError("No weak polygons extracted from structure mask")

    file_name = f"000001_{args.scene_id}_gravity_structural.png"
    for split in ["train", "val"]:
        split_dir = args.output_dir / split
        split_dir.mkdir(exist_ok=True)
        density.save(split_dir / file_name)
        coco = build_coco(split, file_name, polygons, args.image_size)
        with (args.output_dir / "annotations" / f"{split}.json").open("w", encoding="utf-8") as f:
            json.dump(coco, f, ensure_ascii=False, indent=2)

    save_overlay(args.output_dir / "overlays" / f"{args.scene_id}_weak_polygons.png", density, polygons)
    shutil.copy2(args.structure_mask, args.output_dir / "overlays" / f"{args.scene_id}_source_structure_mask.png")

    summary = {
        "scene_id": args.scene_id,
        "density": str(args.density),
        "structure_mask": str(args.structure_mask),
        "label_mode": args.label_mode,
        "image_size": int(args.image_size),
        "polygon_count": int(len(polygons)),
        "areas": [float(item["area"]) for item in polygons],
        "note": (
            "Weak polygon labels extracted from ScanNet projected structure mask. "
            "Use only for RoomFormer single-sample overfit sanity checks."
        ),
    }
    with (args.output_dir / "overfit_dataset_info.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Saved RoomFormer overfit dataset to {args.output_dir}; polygons={len(polygons)}")


if __name__ == "__main__":
    main()
