#!/usr/bin/env python3
"""Estimate gravity, align LingBot-Map points, then generate 2D structural density."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from lingbot_map.utils.experiment_export import _world_to_camera_to_camera_to_world  # noqa: E402
from lingbot_map.utils.wall_density import load_prediction_points  # noqa: E402
from scripts.generate_structural_density import generate_structural_density  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--confidence-threshold", type=float, default=1.2)
    parser.add_argument("--point-stride", type=int, default=8)
    parser.add_argument(
        "--gravity-method",
        choices=["auto", "trajectory-pca", "plane-ransac"],
        default="auto",
    )
    parser.add_argument("--ransac-samples", type=int, default=80000)
    parser.add_argument("--ransac-iterations", type=int, default=400)
    parser.add_argument("--ransac-threshold", type=float, default=0.04)
    parser.add_argument("--grid-size", type=int, default=512)
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


def load_camera_centers(predictions_path: Path) -> np.ndarray | None:
    with np.load(predictions_path) as data:
        if "extrinsic" not in data:
            return None
        extrinsic = data["extrinsic"]
    camera_to_world = _world_to_camera_to_camera_to_world(extrinsic)
    centers = camera_to_world[:, :3, 3].astype(np.float32)
    valid = np.isfinite(centers).all(axis=1)
    centers = centers[valid]
    if len(centers) < 3:
        return None
    return centers


def orient_up_toward_cameras(up: np.ndarray, points: np.ndarray, camera_centers: np.ndarray | None) -> np.ndarray:
    up = up.astype(np.float64)
    up /= max(float(np.linalg.norm(up)), 1e-12)
    if camera_centers is not None and len(camera_centers) > 0:
        camera_center = np.median(camera_centers, axis=0)
        point_center = np.median(points, axis=0)
        if float(np.dot(camera_center - point_center, up)) < 0:
            up = -up
    return up.astype(np.float32)


def estimate_gravity_from_trajectory(
    camera_centers: np.ndarray,
    points: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    centered = camera_centers - np.mean(camera_centers, axis=0, keepdims=True)
    cov = centered.T @ centered / max(len(centered) - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)
    normal = eigvecs[:, order[0]]
    normal = orient_up_toward_cameras(normal, points, camera_centers)
    planar_ratio = float(eigvals[order[0]] / max(eigvals[order[-1]], 1e-12))
    return normal, {
        "gravity_method": "trajectory-pca",
        "trajectory_frame_count": int(len(camera_centers)),
        "trajectory_eigenvalues": eigvals.tolist(),
        "trajectory_planar_ratio": planar_ratio,
    }


def estimate_gravity_from_plane_ransac(
    points: np.ndarray,
    camera_centers: np.ndarray | None,
    sample_count: int,
    iterations: int,
    threshold: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    rng = np.random.default_rng(42)
    if len(points) > sample_count:
        sampled = points[rng.choice(len(points), size=sample_count, replace=False)]
    else:
        sampled = points

    best_normal = None
    best_offset = 0.0
    best_inliers = -1
    for _ in range(iterations):
        ids = rng.choice(len(sampled), size=3, replace=False)
        p0, p1, p2 = sampled[ids]
        normal = np.cross(p1 - p0, p2 - p0)
        norm = float(np.linalg.norm(normal))
        if norm < 1e-8:
            continue
        normal = normal / norm
        offset = -float(np.dot(normal, p0))
        distances = np.abs(sampled @ normal + offset)
        inliers = int((distances < threshold).sum())
        if inliers > best_inliers:
            best_inliers = inliers
            best_normal = normal
            best_offset = offset

    if best_normal is None:
        raise ValueError("RANSAC failed to estimate a plane")
    best_normal = orient_up_toward_cameras(best_normal, points, camera_centers)
    return best_normal, {
        "gravity_method": "plane-ransac",
        "ransac_sample_count": int(len(sampled)),
        "ransac_iterations": int(iterations),
        "ransac_threshold": float(threshold),
        "ransac_best_inliers": int(best_inliers),
        "ransac_best_inlier_ratio": float(best_inliers / max(len(sampled), 1)),
        "ransac_plane_offset": float(best_offset),
    }


def estimate_gravity(args: argparse.Namespace, points: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    camera_centers = load_camera_centers(args.predictions)
    if args.gravity_method in {"auto", "trajectory-pca"} and camera_centers is not None:
        up, metadata = estimate_gravity_from_trajectory(camera_centers, points)
        if args.gravity_method == "trajectory-pca" or metadata["trajectory_planar_ratio"] < 0.25:
            metadata["camera_centers_available"] = True
            return up, metadata

    if args.gravity_method == "trajectory-pca":
        raise ValueError("trajectory-pca requested, but usable camera centers were not found")

    up, metadata = estimate_gravity_from_plane_ransac(
        points,
        camera_centers,
        sample_count=args.ransac_samples,
        iterations=args.ransac_iterations,
        threshold=args.ransac_threshold,
    )
    metadata["camera_centers_available"] = camera_centers is not None
    if args.gravity_method == "auto":
        metadata["gravity_method"] = "auto:" + metadata["gravity_method"]
    return up, metadata


def rotation_to_align_vector_to_z(vector: np.ndarray) -> np.ndarray:
    source = vector.astype(np.float64)
    source /= max(float(np.linalg.norm(source)), 1e-12)
    target = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    cross = np.cross(source, target)
    sin_angle = float(np.linalg.norm(cross))
    cos_angle = float(np.dot(source, target))
    if sin_angle < 1e-8:
        if cos_angle > 0:
            return np.eye(3, dtype=np.float32)
        return np.diag([1.0, -1.0, -1.0]).astype(np.float32)
    vx = np.array(
        [
            [0.0, -cross[2], cross[1]],
            [cross[2], 0.0, -cross[0]],
            [-cross[1], cross[0], 0.0],
        ],
        dtype=np.float64,
    )
    rotation = np.eye(3, dtype=np.float64) + vx + vx @ vx * ((1.0 - cos_angle) / (sin_angle ** 2))
    return rotation.astype(np.float32)


def save_aligned_trajectory(
    output_path: Path,
    predictions_path: Path,
    rotation: np.ndarray,
) -> None:
    camera_centers = load_camera_centers(predictions_path)
    if camera_centers is None:
        return
    aligned = camera_centers @ rotation.T
    with output_path.open("w", encoding="utf-8") as f:
        f.write("# frame_idx aligned_camera_center_x aligned_camera_center_y aligned_camera_center_z\n")
        for idx, center in enumerate(aligned):
            f.write(f"{idx} {center[0]:.10f} {center[1]:.10f} {center[2]:.10f}\n")


def make_structural_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        vertical_axis="z",
        grid_size=args.grid_size,
        bounds_low_percentile=args.bounds_low_percentile,
        bounds_high_percentile=args.bounds_high_percentile,
        bounds_margin=args.bounds_margin,
        height_low=args.height_low,
        height_high=args.height_high,
        height_bins=args.height_bins,
        min_height_bins=args.min_height_bins,
        min_column_points=args.min_column_points,
        close_kernel=args.close_kernel,
        open_kernel=args.open_kernel,
        minimum_component_area=args.minimum_component_area,
        dilate_iterations=args.dilate_iterations,
    )


def main() -> None:
    args = parse_args()
    points, point_metadata = load_prediction_points(
        args.predictions,
        confidence_threshold=args.confidence_threshold,
        point_stride=args.point_stride,
    )
    if len(points) == 0:
        raise ValueError("No valid points after confidence filtering")

    up, gravity_metadata = estimate_gravity(args, points)
    rotation = rotation_to_align_vector_to_z(up)
    aligned_points = points @ rotation.T

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_aligned_trajectory(args.output_dir / "aligned_trajectory.txt", args.predictions, rotation)
    np.savetxt(args.output_dir / "gravity_rotation.txt", rotation)

    metadata = generate_structural_density(
        aligned_points,
        args.output_dir,
        make_structural_args(args),
        input_metadata={
            "predictions": str(args.predictions),
            **point_metadata,
            "gravity": {
                "estimated_up_vector_original": up.tolist(),
                "rotation_maps_estimated_up_to_z": rotation.tolist(),
                **gravity_metadata,
            },
        },
    )
    gravity_info = {
        "estimated_up_vector_original": up.tolist(),
        "rotation_maps_estimated_up_to_z": rotation.tolist(),
        **gravity_metadata,
        "structural_density": metadata,
    }
    with (args.output_dir / "gravity_alignment_info.json").open("w", encoding="utf-8") as f:
        json.dump(gravity_info, f, ensure_ascii=False, indent=2)

    print(
        f"Saved gravity-aligned density to {args.output_dir}; "
        f"method={gravity_metadata['gravity_method']}, "
        f"up={up.tolist()}, structural pixels={metadata['structural_nonzero_pixels']}"
    )


if __name__ == "__main__":
    main()
