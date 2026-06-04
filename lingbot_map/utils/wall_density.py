"""Fixed-height wall-density baseline for LingBot-Map predictions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from lingbot_map.utils.experiment_export import _unproject_depth_strided


_AXES = {"x": 0, "y": 1, "z": 2}


def load_prediction_points(
    predictions_path: str | Path,
    confidence_threshold: float = 1.5,
    point_stride: int = 4,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load or reconstruct confidence-filtered world points."""
    predictions_path = Path(predictions_path)
    with np.load(predictions_path) as data:
        confidence = data["world_points_conf"] if "world_points_conf" in data else data["depth_conf"]
        if "world_points" in data:
            points = data["world_points"][:, ::point_stride, ::point_stride]
            point_source = "world_points"
        else:
            required = {"depth", "extrinsic", "intrinsic"}
            missing = required.difference(data.files)
            if missing:
                raise ValueError(f"Missing required prediction arrays: {sorted(missing)}")
            points = _unproject_depth_strided(
                data["depth"], data["extrinsic"], data["intrinsic"], point_stride
            )
            point_source = "depth_unprojection"
        confidence = confidence[:, ::point_stride, ::point_stride]

    points = points.reshape(-1, 3).astype(np.float32, copy=False)
    confidence = confidence.reshape(-1).astype(np.float32, copy=False)
    valid = np.isfinite(points).all(axis=1)
    valid &= np.isfinite(confidence)
    valid &= confidence >= float(confidence_threshold)
    return points[valid], {
        "point_source": point_source,
        "point_stride": int(point_stride),
        "confidence_threshold": float(confidence_threshold),
        "point_count": int(valid.sum()),
    }


def choose_vertical_axis(points: np.ndarray, axis: str = "auto") -> int:
    if axis != "auto":
        return _AXES[axis]
    robust_span = np.percentile(points, 99, axis=0) - np.percentile(points, 1, axis=0)
    return int(np.argmin(robust_span))


def _normalize_density(counts: np.ndarray) -> np.ndarray:
    density = np.log1p(counts.astype(np.float32))
    positive = density[density > 0]
    if positive.size == 0:
        return density
    scale = np.percentile(positive, 99)
    return np.clip(density / max(float(scale), 1e-8), 0.0, 1.0)


def _save_density_png(path: Path, density: np.ndarray) -> None:
    image = (255.0 * (1.0 - np.sqrt(density))).clip(0, 255).astype(np.uint8)
    Image.fromarray(image).save(path)


def _local_support(mask: np.ndarray, radius: int) -> np.ndarray:
    """Dilate a binary mask without adding a scipy dependency."""
    if radius <= 0:
        return mask.astype(np.float32)
    padded = np.pad(mask, radius, mode="constant", constant_values=False)
    support = np.zeros(mask.shape, dtype=bool)
    size = 2 * radius + 1
    for row_offset in range(size):
        for column_offset in range(size):
            support |= padded[
                row_offset : row_offset + mask.shape[0],
                column_offset : column_offset + mask.shape[1],
            ]
    return support.astype(np.float32)


def _rasterize(
    horizontal_points: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    grid_size: int,
) -> np.ndarray:
    scale = (grid_size - 1) / np.maximum(upper - lower, 1e-8)
    coords = ((horizontal_points - lower) * scale).astype(np.int32)
    valid = np.logical_and(coords >= 0, coords < grid_size).all(axis=1)
    coords = coords[valid]
    counts = np.zeros((grid_size, grid_size), dtype=np.uint32)
    np.add.at(counts, (grid_size - 1 - coords[:, 1], coords[:, 0]), 1)
    return counts


def generate_fixed_height_densities(
    points: np.ndarray,
    output_dir: str | Path,
    vertical_axis: str = "auto",
    grid_size: int = 512,
    bounds_percentiles: tuple[float, float] = (1.0, 99.0),
    bounds_margin: float = 0.05,
    slices: dict[str, tuple[float, float]] | None = None,
    support_radius: int = 2,
    input_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate fixed-height densities and a tolerant cross-height wall baseline."""
    if len(points) == 0:
        raise ValueError("No valid points available for density generation")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    slices = slices or {
        "low": (0.05, 0.30),
        "middle": (0.35, 0.65),
        "high": (0.70, 0.95),
    }
    if support_radius < 0:
        raise ValueError("support_radius must be non-negative")

    vertical_idx = choose_vertical_axis(points, vertical_axis)
    horizontal_indices = [idx for idx in range(3) if idx != vertical_idx]
    vertical = points[:, vertical_idx]
    horizontal = points[:, horizontal_indices]

    height_min, height_max = np.percentile(vertical, bounds_percentiles)
    height_span = max(float(height_max - height_min), 1e-8)
    lower = np.percentile(horizontal, bounds_percentiles[0], axis=0)
    upper = np.percentile(horizontal, bounds_percentiles[1], axis=0)
    margin = (upper - lower) * float(bounds_margin)
    lower -= margin
    upper += margin

    full_counts = _rasterize(horizontal, lower, upper, grid_size)
    full_density = _normalize_density(full_counts)
    np.save(output_dir / "density_full_counts.npy", full_counts)
    np.save(output_dir / "density_full.npy", full_density)
    _save_density_png(output_dir / "density_full.png", full_density)

    normalized_slices = []
    slice_metadata = {}
    for name, (relative_low, relative_high) in slices.items():
        absolute_low = height_min + relative_low * height_span
        absolute_high = height_min + relative_high * height_span
        selected = (vertical >= absolute_low) & (vertical <= absolute_high)
        counts = _rasterize(horizontal[selected], lower, upper, grid_size)
        density = _normalize_density(counts)
        normalized_slices.append(density)
        np.save(output_dir / f"density_{name}_counts.npy", counts)
        np.save(output_dir / f"density_{name}.npy", density)
        _save_density_png(output_dir / f"density_{name}.png", density)
        slice_metadata[name] = {
            "relative_height": [float(relative_low), float(relative_high)],
            "absolute_height": [float(absolute_low), float(absolute_high)],
            "point_count": int(selected.sum()),
            "nonzero_pixels": int((counts > 0).sum()),
        }

    stacked = np.stack(normalized_slices, axis=0)
    strict_consistency = np.prod(stacked + 1e-6, axis=0) ** (1.0 / len(normalized_slices))
    strict_consistency[stacked.min(axis=0) <= 0] = 0
    np.save(
        output_dir / "density_wall_strict_consistency.npy",
        strict_consistency.astype(np.float32),
    )
    _save_density_png(
        output_dir / "density_wall_strict_consistency.png",
        strict_consistency,
    )

    preferred_weights = {"low": 0.35, "middle": 0.45, "high": 0.20}
    slice_weight_values = np.array(
        [preferred_weights.get(name, 1.0) for name in slices],
        dtype=np.float32,
    )
    slice_weight_values /= slice_weight_values.sum()
    slice_weights = slice_weight_values[:, None, None]
    weighted_density = (stacked * slice_weights).sum(axis=0)
    local_slice_support = np.stack(
        [_local_support(density > 0, support_radius) for density in normalized_slices],
        axis=0,
    ).mean(axis=0)
    fused_density = weighted_density * (0.5 + 0.5 * local_slice_support)
    fused_density = np.clip(fused_density, 0.0, 1.0).astype(np.float32)
    np.save(output_dir / "density_cross_height_support.npy", local_slice_support)
    np.save(output_dir / "density_wall_fused.npy", fused_density)
    _save_density_png(output_dir / "density_cross_height_support.png", local_slice_support)
    _save_density_png(output_dir / "density_wall_fused.png", fused_density)

    metadata = {
        "method": "fixed_height_tolerant_cross_slice_fusion_baseline",
        "input": input_metadata or {},
        "vertical_axis": "xyz"[vertical_idx],
        "horizontal_axes": ["xyz"[idx] for idx in horizontal_indices],
        "height_bounds": [float(height_min), float(height_max)],
        "horizontal_bounds": [lower.tolist(), upper.tolist()],
        "bounds_percentiles": list(bounds_percentiles),
        "bounds_margin": float(bounds_margin),
        "grid_size": int(grid_size),
        "support_radius_pixels": int(support_radius),
        "slice_weights": {
            name: float(weight)
            for name, weight in zip(slices, slice_weight_values)
        },
        "full_nonzero_pixels": int((full_counts > 0).sum()),
        "slices": slice_metadata,
        "outputs": {
            "full": "density_full.png",
            "wall_fused": "density_wall_fused.png",
            "cross_height_support": "density_cross_height_support.png",
            "strict_consistency_diagnostic": "density_wall_strict_consistency.png",
        },
    }
    with (output_dir / "density_info.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    return metadata
