#!/usr/bin/env python3
"""Generate a cleaner structural density map from LingBot-Map point clouds."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from lingbot_map.utils.wall_density import (  # noqa: E402
    choose_vertical_axis,
    load_prediction_points,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--confidence-threshold", type=float, default=1.2)
    parser.add_argument("--point-stride", type=int, default=8)
    parser.add_argument("--vertical-axis", choices=["auto", "x", "y", "z"], default="auto")
    parser.add_argument("--grid-size", type=int, default=256)
    parser.add_argument("--bounds-low-percentile", type=float, default=1.0)
    parser.add_argument("--bounds-high-percentile", type=float, default=99.0)
    parser.add_argument("--bounds-margin", type=float, default=0.05)
    parser.add_argument("--height-low", type=float, default=0.12)
    parser.add_argument("--height-high", type=float, default=0.92)
    parser.add_argument("--height-bins", type=int, default=8)
    parser.add_argument("--min-height-bins", type=int, default=3)
    parser.add_argument("--min-column-points", type=int, default=4)
    parser.add_argument("--close-kernel", type=int, default=3)
    parser.add_argument("--open-kernel", type=int, default=3)
    parser.add_argument("--minimum-component-area", type=int, default=40)
    parser.add_argument("--dilate-iterations", type=int, default=0)
    return parser.parse_args()


def _normalize(counts: np.ndarray) -> np.ndarray:
    density = np.log1p(counts.astype(np.float32))
    positive = density[density > 0]
    if positive.size == 0:
        return density
    scale = np.percentile(positive, 99)
    return np.clip(density / max(float(scale), 1e-8), 0.0, 1.0)


def _save_density_png(path: Path, density: np.ndarray) -> None:
    image = (255.0 * (1.0 - np.sqrt(np.clip(density, 0.0, 1.0)))).astype(np.uint8)
    Image.fromarray(image).save(path)


def _raster_coords(
    horizontal_points: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    grid_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    scale = (grid_size - 1) / np.maximum(upper - lower, 1e-8)
    xy = ((horizontal_points - lower) * scale).astype(np.int32)
    valid = np.logical_and(xy >= 0, xy < grid_size).all(axis=1)
    xy = xy[valid]
    rows = grid_size - 1 - xy[:, 1]
    cols = xy[:, 0]
    return rows, cols


def _counts_from_coords(rows: np.ndarray, cols: np.ndarray, grid_size: int) -> np.ndarray:
    counts = np.zeros((grid_size, grid_size), dtype=np.uint32)
    np.add.at(counts, (rows, cols), 1)
    return counts


def _remove_small_components(mask: np.ndarray, minimum_area: int) -> np.ndarray:
    if minimum_area <= 0:
        return mask
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    filtered = np.zeros(mask.shape, dtype=np.uint8)
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] >= minimum_area:
            filtered[labels == label] = 1
    return filtered.astype(bool)


def _morphology(mask: np.ndarray, close_kernel: int, open_kernel: int, dilate_iterations: int) -> np.ndarray:
    image = mask.astype(np.uint8)
    if close_kernel > 1:
        kernel = np.ones((close_kernel, close_kernel), dtype=np.uint8)
        image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    if open_kernel > 1:
        kernel = np.ones((open_kernel, open_kernel), dtype=np.uint8)
        image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    if dilate_iterations > 0:
        kernel = np.ones((3, 3), dtype=np.uint8)
        image = cv2.dilate(image, kernel, iterations=dilate_iterations)
    return image.astype(bool)


def _save_diagnostics(
    path: Path,
    panels: list[tuple[str, np.ndarray]],
) -> None:
    cell = 256
    label_height = 26
    width = cell * len(panels)
    height = cell + label_height
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    for idx, (title, density) in enumerate(panels):
        image = (255.0 * (1.0 - np.sqrt(np.clip(density, 0.0, 1.0)))).astype(np.uint8)
        pil = Image.fromarray(image).convert("RGB").resize((cell, cell))
        x = idx * cell
        canvas.paste(pil, (x, label_height))
        draw.text((x + 6, 6), title, fill=(0, 0, 0))
    canvas.save(path)


def generate_structural_density(
    points: np.ndarray,
    output_dir: Path,
    args: argparse.Namespace,
    input_metadata: dict[str, Any],
) -> dict[str, Any]:
    if len(points) == 0:
        raise ValueError("No valid points available for structural density generation")
    output_dir.mkdir(parents=True, exist_ok=True)

    vertical_idx = choose_vertical_axis(points, args.vertical_axis)
    horizontal_indices = [idx for idx in range(3) if idx != vertical_idx]
    vertical = points[:, vertical_idx]
    horizontal = points[:, horizontal_indices]

    height_min, height_max = np.percentile(
        vertical,
        [args.bounds_low_percentile, args.bounds_high_percentile],
    )
    height_span = max(float(height_max - height_min), 1e-8)
    selected_low = height_min + args.height_low * height_span
    selected_high = height_min + args.height_high * height_span
    in_height_range = (vertical >= selected_low) & (vertical <= selected_high)

    lower = np.percentile(horizontal, args.bounds_low_percentile, axis=0)
    upper = np.percentile(horizontal, args.bounds_high_percentile, axis=0)
    margin = (upper - lower) * args.bounds_margin
    lower -= margin
    upper += margin

    full_rows, full_cols = _raster_coords(horizontal, lower, upper, args.grid_size)
    raw_counts = _counts_from_coords(full_rows, full_cols, args.grid_size)
    raw_density = _normalize(raw_counts)

    candidate_points = points[in_height_range]
    candidate_vertical = candidate_points[:, vertical_idx]
    candidate_horizontal = candidate_points[:, horizontal_indices]
    rows, cols = _raster_coords(candidate_horizontal, lower, upper, args.grid_size)
    candidate_counts = _counts_from_coords(rows, cols, args.grid_size)
    height_density = _normalize(candidate_counts)

    height_edges = np.linspace(selected_low, selected_high, args.height_bins + 1)
    bin_support = np.zeros((args.height_bins, args.grid_size, args.grid_size), dtype=bool)
    bin_counts = []
    for bin_idx in range(args.height_bins):
        low = height_edges[bin_idx]
        high = height_edges[bin_idx + 1]
        in_bin = (candidate_vertical >= low) & (candidate_vertical <= high)
        bin_rows, bin_cols = _raster_coords(
            candidate_horizontal[in_bin],
            lower,
            upper,
            args.grid_size,
        )
        counts = _counts_from_coords(bin_rows, bin_cols, args.grid_size)
        bin_support[bin_idx] = counts > 0
        bin_counts.append(counts)

    support_count = bin_support.sum(axis=0).astype(np.float32)
    vertical_support = np.clip(support_count / max(args.height_bins, 1), 0.0, 1.0)
    column_point_mask = candidate_counts >= args.min_column_points
    consistency_mask = support_count >= args.min_height_bins
    structural_mask = consistency_mask & column_point_mask
    structural_mask = _morphology(
        structural_mask,
        close_kernel=args.close_kernel,
        open_kernel=args.open_kernel,
        dilate_iterations=args.dilate_iterations,
    )
    structural_mask = _remove_small_components(structural_mask, args.minimum_component_area)

    structural_density = height_density * (0.45 + 0.55 * vertical_support)
    structural_density *= structural_mask.astype(np.float32)
    structural_density = np.clip(structural_density, 0.0, 1.0).astype(np.float32)

    np.save(output_dir / "raw_density.npy", raw_density)
    np.save(output_dir / "height_filtered_density.npy", height_density)
    np.save(output_dir / "vertical_support.npy", vertical_support.astype(np.float32))
    np.save(output_dir / "structural_mask.npy", structural_mask.astype(np.uint8))
    np.save(output_dir / "structural_density.npy", structural_density)
    _save_density_png(output_dir / "raw_density.png", raw_density)
    _save_density_png(output_dir / "height_filtered_density.png", height_density)
    _save_density_png(output_dir / "vertical_support.png", vertical_support)
    _save_density_png(output_dir / "structural_mask.png", structural_mask.astype(np.float32))
    _save_density_png(output_dir / "structural_density.png", structural_density)
    _save_diagnostics(
        output_dir / "structural_density_diagnostics.png",
        [
            ("raw", raw_density),
            ("height filtered", height_density),
            ("vertical support", vertical_support),
            ("structural mask", structural_mask.astype(np.float32)),
            ("structural", structural_density),
        ],
    )

    metadata = {
        "method": "height_consistent_structural_density",
        "input": input_metadata,
        "vertical_axis": "xyz"[vertical_idx],
        "horizontal_axes": ["xyz"[idx] for idx in horizontal_indices],
        "height_bounds": [float(height_min), float(height_max)],
        "selected_height_bounds": [float(selected_low), float(selected_high)],
        "horizontal_bounds": [lower.tolist(), upper.tolist()],
        "grid_size": int(args.grid_size),
        "height_bins": int(args.height_bins),
        "min_height_bins": int(args.min_height_bins),
        "min_column_points": int(args.min_column_points),
        "raw_nonzero_pixels": int((raw_counts > 0).sum()),
        "height_filtered_nonzero_pixels": int((candidate_counts > 0).sum()),
        "structural_nonzero_pixels": int((structural_density > 0).sum()),
        "height_filtered_points": int(in_height_range.sum()),
        "outputs": {
            "raw_density": "raw_density.png",
            "height_filtered_density": "height_filtered_density.png",
            "vertical_support": "vertical_support.png",
            "structural_mask": "structural_mask.png",
            "structural_density": "structural_density.png",
            "diagnostics": "structural_density_diagnostics.png",
        },
    }
    with (output_dir / "structural_density_info.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    return metadata


def main() -> None:
    args = parse_args()
    points, point_metadata = load_prediction_points(
        args.predictions,
        confidence_threshold=args.confidence_threshold,
        point_stride=args.point_stride,
    )
    print(
        f"Loaded {len(points):,} points from {point_metadata['point_source']} "
        f"(confidence >= {args.confidence_threshold}, stride={args.point_stride})"
    )
    metadata = generate_structural_density(
        points,
        args.output_dir,
        args,
        input_metadata={
            "predictions": str(args.predictions),
            **point_metadata,
        },
    )
    print(
        f"Saved structural density to {args.output_dir}; "
        f"vertical axis={metadata['vertical_axis']}, "
        f"structural pixels={metadata['structural_nonzero_pixels']}"
    )


if __name__ == "__main__":
    main()
