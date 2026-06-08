#!/usr/bin/env python3
"""Generate top-down density from vertical surfaces in LingBot-Map points."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from lingbot_map.utils.wall_density import choose_vertical_axis, load_prediction_points  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--confidence-threshold", type=float, default=1.2)
    parser.add_argument("--point-stride", type=int, default=8)
    parser.add_argument("--vertical-axis", choices=["auto", "x", "y", "z"], default="auto")
    parser.add_argument("--grid-size", type=int, default=512)
    parser.add_argument("--bounds-low-percentile", type=float, default=1.0)
    parser.add_argument("--bounds-high-percentile", type=float, default=99.0)
    parser.add_argument("--bounds-margin", type=float, default=0.05)
    parser.add_argument("--height-low", type=float, default=0.08)
    parser.add_argument("--height-high", type=float, default=0.95)
    parser.add_argument("--normal-neighbors", type=int, default=24)
    parser.add_argument("--max-normal-points", type=int, default=260000)
    parser.add_argument(
        "--vertical-normal-threshold",
        type=float,
        default=0.35,
        help="Keep surfaces whose normal has abs(dot(normal, up)) below this value.",
    )
    parser.add_argument("--min-height-bins", type=int, default=3)
    parser.add_argument("--height-bins", type=int, default=8)
    parser.add_argument("--min-column-points", type=int, default=3)
    parser.add_argument("--close-kernel", type=int, default=3)
    parser.add_argument("--open-kernel", type=int, default=3)
    parser.add_argument("--minimum-component-area", type=int, default=40)
    return parser.parse_args()


def normalize_counts(counts: np.ndarray) -> np.ndarray:
    density = np.log1p(counts.astype(np.float32))
    positive = density[density > 0]
    if positive.size == 0:
        return density
    scale = np.percentile(positive, 99)
    return np.clip(density / max(float(scale), 1e-8), 0.0, 1.0)


def save_density(path: Path, density: np.ndarray) -> None:
    image = (255.0 * (1.0 - np.sqrt(np.clip(density, 0.0, 1.0)))).astype(np.uint8)
    Image.fromarray(image).save(path)


def rasterize(
    horizontal: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    grid_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scale = (grid_size - 1) / np.maximum(upper - lower, 1e-8)
    coords = ((horizontal - lower) * scale).astype(np.int32)
    valid = np.logical_and(coords >= 0, coords < grid_size).all(axis=1)
    coords = coords[valid]
    rows = grid_size - 1 - coords[:, 1]
    cols = coords[:, 0]
    counts = np.zeros((grid_size, grid_size), dtype=np.uint32)
    np.add.at(counts, (rows, cols), 1)
    return counts, rows, cols


def remove_small_components(mask: np.ndarray, minimum_area: int) -> np.ndarray:
    if minimum_area <= 0:
        return mask
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    filtered = np.zeros(mask.shape, dtype=np.uint8)
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] >= minimum_area:
            filtered[labels == label] = 1
    return filtered.astype(bool)


def apply_morphology(mask: np.ndarray, close_kernel: int, open_kernel: int) -> np.ndarray:
    image = mask.astype(np.uint8)
    if close_kernel > 1:
        kernel = np.ones((close_kernel, close_kernel), dtype=np.uint8)
        image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    if open_kernel > 1:
        kernel = np.ones((open_kernel, open_kernel), dtype=np.uint8)
        image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    return image.astype(bool)


def estimate_vertical_surface_mask(
    points: np.ndarray,
    vertical_idx: int,
    neighbors: int,
    threshold: float,
    max_points: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    if len(points) > max_points:
        rng = np.random.default_rng(42)
        selected_indices = np.sort(rng.choice(len(points), size=max_points, replace=False))
        sampled = points[selected_indices]
    else:
        selected_indices = np.arange(len(points))
        sampled = points

    tree = cKDTree(sampled)
    k = min(max(neighbors, 4), len(sampled))
    _, neighbor_indices = tree.query(sampled, k=k, workers=-1)
    if neighbor_indices.ndim == 1:
        neighbor_indices = neighbor_indices[:, None]

    centered = sampled[neighbor_indices] - sampled[:, None, :]
    cov = np.einsum("nki,nkj->nij", centered, centered) / max(k - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    normals = eigvecs[:, :, 0]
    normal_up_abs = np.abs(normals[:, vertical_idx])
    sampled_vertical = normal_up_abs <= threshold

    if len(sampled) == len(points):
        mask = sampled_vertical
    else:
        full_tree = cKDTree(sampled)
        _, nearest = full_tree.query(points, k=1, workers=-1)
        mask = sampled_vertical[nearest]

    return mask.astype(bool), {
        "normal_sample_count": int(len(sampled)),
        "normal_neighbors": int(k),
        "vertical_normal_threshold": float(threshold),
        "sampled_vertical_surface_ratio": float(sampled_vertical.mean()),
    }


def height_consistency_mask(
    vertical: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    grid_size: int,
    height_bins: int,
    min_height_bins: int,
    min_column_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    if len(vertical) == 0:
        return np.zeros((grid_size, grid_size), dtype=bool), np.zeros((grid_size, grid_size), dtype=np.float32)
    edges = np.linspace(float(vertical.min()), float(vertical.max()), height_bins + 1)
    bin_support = np.zeros((height_bins, grid_size, grid_size), dtype=bool)
    column_counts = np.zeros((grid_size, grid_size), dtype=np.uint32)
    np.add.at(column_counts, (rows, cols), 1)
    for idx in range(height_bins):
        selected = (vertical >= edges[idx]) & (vertical <= edges[idx + 1])
        if not selected.any():
            continue
        support = np.zeros((grid_size, grid_size), dtype=bool)
        support[rows[selected], cols[selected]] = True
        bin_support[idx] = support
    support_count = bin_support.sum(axis=0)
    support_ratio = np.clip(support_count / max(height_bins, 1), 0.0, 1.0).astype(np.float32)
    mask = (support_count >= min_height_bins) & (column_counts >= min_column_points)
    return mask, support_ratio


def save_diagnostics(path: Path, panels: list[tuple[str, np.ndarray]]) -> None:
    cell = 256
    label_height = 26
    canvas = Image.new("RGB", (cell * len(panels), cell + label_height), "white")
    draw = ImageDraw.Draw(canvas)
    for idx, (title, density) in enumerate(panels):
        image = (255.0 * (1.0 - np.sqrt(np.clip(density, 0.0, 1.0)))).astype(np.uint8)
        pil = Image.fromarray(image).convert("RGB").resize((cell, cell))
        x = idx * cell
        canvas.paste(pil, (x, label_height))
        draw.text((x + 6, 6), title, fill=(0, 0, 0))
    canvas.save(path)


def main() -> None:
    args = parse_args()
    points, point_metadata = load_prediction_points(
        args.predictions,
        confidence_threshold=args.confidence_threshold,
        point_stride=args.point_stride,
    )
    if len(points) == 0:
        raise ValueError("No valid points after confidence filtering")

    output_dir = args.output_dir
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
    low = height_min + args.height_low * height_span
    high = height_min + args.height_high * height_span
    height_mask = (vertical >= low) & (vertical <= high)

    lower = np.percentile(horizontal, args.bounds_low_percentile, axis=0)
    upper = np.percentile(horizontal, args.bounds_high_percentile, axis=0)
    margin = (upper - lower) * args.bounds_margin
    lower -= margin
    upper += margin

    raw_counts, _, _ = rasterize(horizontal, lower, upper, args.grid_size)
    raw_density = normalize_counts(raw_counts)

    candidate_points = points[height_mask]
    normal_mask, normal_metadata = estimate_vertical_surface_mask(
        candidate_points,
        vertical_idx=vertical_idx,
        neighbors=args.normal_neighbors,
        threshold=args.vertical_normal_threshold,
        max_points=args.max_normal_points,
    )
    vertical_surface_points = candidate_points[normal_mask]
    vertical_surface_vertical = vertical_surface_points[:, vertical_idx]
    vertical_surface_horizontal = vertical_surface_points[:, horizontal_indices]

    vertical_counts, rows, cols = rasterize(
        vertical_surface_horizontal,
        lower,
        upper,
        args.grid_size,
    )
    vertical_density = normalize_counts(vertical_counts)
    consistency, support_ratio = height_consistency_mask(
        vertical_surface_vertical,
        rows,
        cols,
        args.grid_size,
        height_bins=args.height_bins,
        min_height_bins=args.min_height_bins,
        min_column_points=args.min_column_points,
    )
    consistency = apply_morphology(consistency, args.close_kernel, args.open_kernel)
    consistency = remove_small_components(consistency, args.minimum_component_area)
    wall_density = np.clip(vertical_density * (0.45 + 0.55 * support_ratio), 0.0, 1.0)
    wall_density *= consistency.astype(np.float32)

    np.save(output_dir / "raw_density.npy", raw_density.astype(np.float32))
    np.save(output_dir / "vertical_surface_density.npy", vertical_density.astype(np.float32))
    np.save(output_dir / "height_support.npy", support_ratio.astype(np.float32))
    np.save(output_dir / "wall_candidate_mask.npy", consistency.astype(np.uint8))
    np.save(output_dir / "wall_surface_density.npy", wall_density.astype(np.float32))
    save_density(output_dir / "raw_density.png", raw_density)
    save_density(output_dir / "vertical_surface_density.png", vertical_density)
    save_density(output_dir / "height_support.png", support_ratio)
    save_density(output_dir / "wall_candidate_mask.png", consistency.astype(np.float32))
    save_density(output_dir / "wall_surface_density.png", wall_density)
    save_diagnostics(
        output_dir / "vertical_surface_density_diagnostics.png",
        [
            ("raw", raw_density),
            ("vertical surfaces", vertical_density),
            ("height support", support_ratio),
            ("wall mask", consistency.astype(np.float32)),
            ("wall density", wall_density),
        ],
    )

    metadata = {
        "method": "local_pca_vertical_surface_density",
        "input": {
            "predictions": str(args.predictions),
            **point_metadata,
        },
        "vertical_axis": "xyz"[vertical_idx],
        "horizontal_axes": ["xyz"[idx] for idx in horizontal_indices],
        "height_bounds": [float(height_min), float(height_max)],
        "selected_height_bounds": [float(low), float(high)],
        "horizontal_bounds": [lower.tolist(), upper.tolist()],
        "grid_size": int(args.grid_size),
        "height_filtered_points": int(height_mask.sum()),
        "vertical_surface_points": int(len(vertical_surface_points)),
        "raw_nonzero_pixels": int((raw_counts > 0).sum()),
        "vertical_surface_nonzero_pixels": int((vertical_counts > 0).sum()),
        "wall_surface_nonzero_pixels": int((wall_density > 0).sum()),
        **normal_metadata,
    }
    with (output_dir / "vertical_surface_density_info.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(
        f"Saved vertical surface density to {output_dir}; "
        f"vertical axis={metadata['vertical_axis']}, "
        f"vertical points={metadata['vertical_surface_points']:,}, "
        f"wall pixels={metadata['wall_surface_nonzero_pixels']:,}"
    )


if __name__ == "__main__":
    main()
