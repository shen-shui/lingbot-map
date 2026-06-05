#!/usr/bin/env python3
"""Create noisy RoomFormer density datasets from clean RoomFormer-style data."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw


def _load_gray(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8)


def _gray_to_density(gray: np.ndarray) -> np.ndarray:
    return 1.0 - gray.astype(np.float32) / 255.0


def _density_to_gray(density: np.ndarray) -> np.ndarray:
    density = np.clip(density, 0.0, 1.0)
    return (255.0 * (1.0 - density)).astype(np.uint8)


def _random_rect_mask(shape: tuple[int, int], rng: np.random.Generator, count: int, size_range: tuple[int, int]) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=np.float32)
    for _ in range(count):
        rh = int(rng.integers(size_range[0], size_range[1] + 1))
        rw = int(rng.integers(size_range[0], size_range[1] + 1))
        y0 = int(rng.integers(0, max(h - rh, 1)))
        x0 = int(rng.integers(0, max(w - rw, 1)))
        mask[y0 : y0 + rh, x0 : x0 + rw] = 1.0
    return mask


def _add_blob_noise(density: np.ndarray, rng: np.random.Generator, count: int) -> np.ndarray:
    noisy = density.copy()
    h, w = noisy.shape
    for _ in range(count):
        cx = int(rng.integers(0, w))
        cy = int(rng.integers(0, h))
        radius = int(rng.integers(3, 13))
        strength = float(rng.uniform(0.25, 0.8))
        cv2.circle(noisy, (cx, cy), radius, strength, thickness=-1)
    return np.maximum(density, noisy)


def _add_line_noise(density: np.ndarray, rng: np.random.Generator, count: int) -> np.ndarray:
    noisy = density.copy()
    h, w = noisy.shape
    for _ in range(count):
        x0 = int(rng.integers(0, w))
        y0 = int(rng.integers(0, h))
        x1 = int(np.clip(x0 + rng.normal(0, w * 0.25), 0, w - 1))
        y1 = int(np.clip(y0 + rng.normal(0, h * 0.25), 0, h - 1))
        thickness = int(rng.integers(1, 4))
        strength = float(rng.uniform(0.25, 0.75))
        cv2.line(noisy, (x0, y0), (x1, y1), strength, thickness=thickness)
    return np.maximum(density, noisy)


def degrade_density(density: np.ndarray, rng: np.random.Generator, severity: float) -> np.ndarray:
    """Apply LingBot-Map-like density degradations while keeping output in [0, 1]."""
    severity = float(np.clip(severity, 0.0, 1.0))
    noisy = density.astype(np.float32, copy=True)

    # Wall/support dropout: local rectangular holes and random point thinning.
    hole_count = int(rng.integers(2, 7) * (0.5 + severity))
    holes = _random_rect_mask(noisy.shape, rng, hole_count, (10, 42))
    noisy *= 1.0 - holes * float(rng.uniform(0.45, 0.95) * severity)
    keep_prob = 1.0 - float(rng.uniform(0.05, 0.25) * severity)
    noisy *= (rng.random(noisy.shape) < keep_prob).astype(np.float32)

    # Pose drift / reconstruction blur: thicken then blur density support.
    if rng.random() < 0.8:
        kernel_size = int(rng.integers(2, 5))
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        noisy = cv2.dilate(noisy, kernel, iterations=int(rng.integers(1, 3)))
    if rng.random() < 0.9:
        sigma = float(rng.uniform(0.4, 1.6) * (0.5 + severity))
        noisy = cv2.GaussianBlur(noisy, (0, 0), sigmaX=sigma, sigmaY=sigma)

    # Furniture / clutter: blobs and short line fragments unrelated to walls.
    noisy = _add_blob_noise(noisy, rng, int(rng.integers(10, 35) * (0.4 + severity)))
    noisy = _add_line_noise(noisy, rng, int(rng.integers(4, 16) * (0.4 + severity)))

    # Sensor speckle and mild contrast shift.
    speckle = rng.random(noisy.shape) < float(rng.uniform(0.001, 0.012) * (0.5 + severity))
    noisy[speckle] = np.maximum(noisy[speckle], rng.uniform(0.2, 0.8, size=int(speckle.sum())))
    gamma = float(rng.uniform(0.75, 1.35))
    noisy = np.power(np.clip(noisy, 0.0, 1.0), gamma)

    return np.clip(noisy, 0.0, 1.0)


def _draw_annotations(gray: np.ndarray, annotations: list[dict[str, Any]]) -> Image.Image:
    image = Image.fromarray(gray).convert("RGB")
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
    for idx, ann in enumerate(annotations):
        for segmentation in ann.get("segmentation", []):
            polygon = np.asarray(segmentation, dtype=np.float32).reshape(-1, 2)
            if len(polygon) < 3:
                continue
            points = [tuple(point) for point in polygon] + [tuple(polygon[0])]
            draw.line(points, fill=colors[idx % len(colors)], width=2)
    return image


def _copy_and_augment_split(
    input_dir: Path,
    output_dir: Path,
    split: str,
    variants: int,
    include_clean: bool,
    seed: int,
    severity: float,
    max_previews: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    ann_path = input_dir / "annotations" / f"{split}.json"
    data = json.loads(ann_path.read_text())
    image_by_id = {int(image["id"]): image for image in data["images"]}
    anns_by_image: dict[int, list[dict[str, Any]]] = {image_id: [] for image_id in image_by_id}
    for ann in data["annotations"]:
        anns_by_image.setdefault(int(ann["image_id"]), []).append(ann)

    split_dir = output_dir / split
    overlay_dir = output_dir / "overlays" / split
    split_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "annotations").mkdir(parents=True, exist_ok=True)

    new_images: list[dict[str, Any]] = []
    new_annotations: list[dict[str, Any]] = []
    next_image_id = 1
    next_annotation_id = 1
    preview_count = 0

    def add_example(source_image: dict[str, Any], gray: np.ndarray, suffix: str) -> None:
        nonlocal next_image_id, next_annotation_id, preview_count
        source_id = int(source_image["id"])
        stem = Path(source_image["file_name"]).stem
        file_name = f"{stem}_{suffix}.png"
        Image.fromarray(gray, mode="L").save(split_dir / file_name)

        image_record = dict(source_image)
        image_record["id"] = next_image_id
        image_record["file_name"] = file_name
        image_record["source_image_id"] = source_id
        image_record["degradation"] = suffix
        new_images.append(image_record)

        copied_anns = []
        for ann in anns_by_image.get(source_id, []):
            ann_record = dict(ann)
            ann_record["id"] = next_annotation_id
            ann_record["image_id"] = next_image_id
            ann_record["source_annotation_id"] = int(ann["id"])
            new_annotations.append(ann_record)
            copied_anns.append(ann_record)
            next_annotation_id += 1

        if preview_count < max_previews:
            _draw_annotations(gray, copied_anns).save(overlay_dir / f"{Path(file_name).stem}_overlay.png")
            preview_count += 1
        next_image_id += 1

    for image in data["images"]:
        source_path = input_dir / split / image["file_name"]
        gray = _load_gray(source_path)
        density = _gray_to_density(gray)
        if include_clean:
            add_example(image, gray, "clean")
        for variant_idx in range(variants):
            noisy_density = degrade_density(density, rng, severity=severity)
            noisy_gray = _density_to_gray(noisy_density)
            add_example(image, noisy_gray, f"noisy{variant_idx + 1:02d}")

    output_data = {
        "images": new_images,
        "annotations": new_annotations,
        "categories": data.get("categories", [{"id": 1, "name": "room"}]),
    }
    out_ann_path = output_dir / "annotations" / f"{split}.json"
    out_ann_path.write_text(json.dumps(output_data))
    return {
        "split": split,
        "source_images": len(data["images"]),
        "source_annotations": len(data["annotations"]),
        "output_images": len(new_images),
        "output_annotations": len(new_annotations),
        "annotation_path": str(out_ann_path),
    }


def augment_roomformer_density(
    input_dir: str | Path,
    output_dir: str | Path,
    splits: list[str],
    variants: int,
    include_clean: bool,
    seed: int,
    severity: float,
    max_previews: int,
) -> dict[str, Any]:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if (input_dir / "annotations").exists():
        shutil.copytree(input_dir / "annotations", output_dir / "source_annotations", dirs_exist_ok=True)

    summaries = []
    for split_idx, split in enumerate(splits):
        summaries.append(
            _copy_and_augment_split(
                input_dir=input_dir,
                output_dir=output_dir,
                split=split,
                variants=variants,
                include_clean=include_clean,
                seed=seed + split_idx * 10007,
                severity=severity,
                max_previews=max_previews,
            )
        )
    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "splits": summaries,
        "variants": int(variants),
        "include_clean": bool(include_clean),
        "severity": float(severity),
        "seed": int(seed),
    }
    (output_dir / "augmentation_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, help="Clean RoomFormer-style dataset root")
    parser.add_argument("--output-dir", required=True, help="Output noisy RoomFormer-style dataset root")
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--variants", type=int, default=3, help="Noisy variants per source image")
    parser.add_argument("--include-clean", action="store_true", help="Also include the original clean image")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--severity", type=float, default=0.7)
    parser.add_argument("--max-previews", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = augment_roomformer_density(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        splits=args.splits,
        variants=args.variants,
        include_clean=args.include_clean,
        seed=args.seed,
        severity=args.severity,
        max_previews=args.max_previews,
    )
    for split in summary["splits"]:
        print(
            f"{split['split']}: {split['source_images']} source images -> "
            f"{split['output_images']} augmented images; annotations={split['output_annotations']}"
        )
    print(f"Saved noisy RoomFormer dataset to {summary['output_dir']}")


if __name__ == "__main__":
    main()
