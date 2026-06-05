"""Furniture-resistant wall confidence maps from top-down point densities."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image


def _save_confidence_png(path: Path, confidence: np.ndarray) -> None:
    image = (255.0 * (1.0 - np.sqrt(np.clip(confidence, 0.0, 1.0)))).clip(0, 255)
    Image.fromarray(image.astype(np.uint8)).save(path)


def _normalize(values: np.ndarray, percentile: float = 99.0) -> np.ndarray:
    values = values.astype(np.float32, copy=False)
    positive = values[values > 0]
    if positive.size == 0:
        return np.zeros_like(values, dtype=np.float32)
    scale = np.percentile(positive, percentile)
    return np.clip(values / max(float(scale), 1e-8), 0.0, 1.0).astype(np.float32)


def _odd(size: int) -> int:
    return size if size % 2 == 1 else size + 1


def _line_kernel(length: int, thickness: int, angle_degrees: float) -> np.ndarray:
    length = max(3, _odd(length))
    thickness = max(1, _odd(thickness))
    canvas_size = int(math.ceil(length * 1.5))
    canvas_size = max(canvas_size, length + thickness + 2)
    canvas_size = _odd(canvas_size)
    center = canvas_size // 2
    kernel = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
    half = length // 2
    cv2.line(
        kernel,
        (center - half, center),
        (center + half, center),
        1,
        thickness=thickness,
    )
    if abs(angle_degrees) > 1e-6:
        matrix = cv2.getRotationMatrix2D((center, center), angle_degrees, 1.0)
        kernel = cv2.warpAffine(kernel, matrix, (canvas_size, canvas_size), flags=cv2.INTER_NEAREST)
    return (kernel > 0).astype(np.uint8)


def _multi_angle_line_response(
    density: np.ndarray,
    line_lengths: tuple[int, ...],
    line_thickness: int,
    angle_step: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    response = np.zeros_like(density, dtype=np.float32)
    angles = np.arange(0.0, 180.0, float(angle_step), dtype=np.float32)
    for length in line_lengths:
        for angle in angles:
            kernel = _line_kernel(length, line_thickness, float(angle))
            opened = cv2.morphologyEx(density, cv2.MORPH_OPEN, kernel)
            response = np.maximum(response, opened)
    return _normalize(response), {
        "angles_degrees": [float(angle) for angle in angles],
        "line_lengths_pixels": [int(length) for length in line_lengths],
        "line_thickness_pixels": int(line_thickness),
    }


def _thin_structure_score(mask: np.ndarray, preferred_thickness: float) -> np.ndarray:
    distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
    excess = np.maximum(distance - float(preferred_thickness), 0.0)
    score = np.exp(-((excess / max(float(preferred_thickness), 1e-6)) ** 2))
    score[mask == 0] = 0.0
    return score.astype(np.float32)


def _world_to_pixel(
    points: np.ndarray,
    image_shape: tuple[int, int],
    horizontal_bounds: list[list[float]],
) -> np.ndarray:
    height, width = image_shape
    lower, upper = np.asarray(horizontal_bounds, dtype=np.float64)
    pixels = np.empty_like(points, dtype=np.float64)
    pixels[:, 0] = (points[:, 0] - lower[0]) / max(upper[0] - lower[0], 1e-8) * (width - 1)
    pixels[:, 1] = height - 1 - (
        (points[:, 1] - lower[1]) / max(upper[1] - lower[1], 1e-8) * (height - 1)
    )
    return pixels


def _load_trajectory_pixels(
    trajectory_path: str | Path,
    image_shape: tuple[int, int],
    density_info: dict[str, Any],
) -> np.ndarray:
    trajectory = np.loadtxt(trajectory_path)
    if trajectory.ndim == 1:
        trajectory = trajectory[None, :]
    if trajectory.shape[1] != 13:
        raise ValueError("Trajectory must contain frame index plus a row-major 3x4 camera-to-world matrix")
    centers = trajectory[:, [4, 8, 12]]
    axis_indices = ["xyz".index(axis) for axis in density_info["horizontal_axes"]]
    return _world_to_pixel(centers[:, axis_indices], image_shape, density_info["horizontal_bounds"])


def _trajectory_exterior_score(
    trajectory_pixels: np.ndarray,
    image_shape: tuple[int, int],
    near_radius: int,
    far_radius: int,
) -> np.ndarray:
    height, width = image_shape
    path = np.zeros((height, width), dtype=np.uint8)
    rounded = np.rint(trajectory_pixels).astype(np.int32)
    valid = np.logical_and(rounded >= 0, rounded < np.array([width, height])).all(axis=1)
    rounded = rounded[valid]
    if len(rounded) == 0:
        return np.ones((height, width), dtype=np.float32)
    path[rounded[:, 1], rounded[:, 0]] = 255
    if near_radius > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (_odd(near_radius * 2 + 1), _odd(near_radius * 2 + 1)),
        )
        path = cv2.dilate(path, kernel)
    distance = cv2.distanceTransform(255 - path, cv2.DIST_L2, 3)
    near = float(max(near_radius, 0))
    preferred = float(max(far_radius, near + 1.0))
    far = preferred * 2.0
    rising = np.clip((distance - near) / max(preferred - near, 1.0), 0.0, 1.0)
    falling = np.clip((far - distance) / max(far - preferred, 1.0), 0.0, 1.0)
    score = rising * falling
    return score.astype(np.float32)


def _draw_diagnostic_panel(output_path: Path, images: list[tuple[str, np.ndarray]]) -> None:
    panels = []
    label_height = 28
    for label, image in images:
        normalized = np.clip(image, 0.0, 1.0)
        gray = (255.0 * (1.0 - np.sqrt(normalized))).astype(np.uint8)
        panel = Image.new("RGB", (gray.shape[1], gray.shape[0] + label_height), "white")
        panel.paste(Image.fromarray(gray).convert("RGB"), (0, label_height))
        from PIL import ImageDraw

        draw = ImageDraw.Draw(panel)
        draw.text((8, 8), label, fill=(0, 0, 0))
        panels.append(panel)
    total_width = sum(panel.width for panel in panels)
    output = Image.new("RGB", (total_width, max(panel.height for panel in panels)), "white")
    x = 0
    for panel in panels:
        output.paste(panel, (x, 0))
        x += panel.width
    output.save(output_path)


def enhance_wall_confidence(
    density_path: str | Path,
    output_dir: str | Path,
    support_path: str | Path | None = None,
    density_info_path: str | Path | None = None,
    trajectory_path: str | Path | None = None,
    density_threshold: float = 0.12,
    support_weight: float = 0.30,
    line_weight: float = 0.55,
    thin_weight: float = 0.05,
    trajectory_weight: float = 0.10,
    line_lengths: tuple[int, ...] = (31, 63, 95),
    line_thickness: int = 3,
    angle_step: float = 15.0,
    preferred_thickness: float = 5.0,
    trajectory_near_radius: int = 8,
    trajectory_far_radius: int = 80,
    close_kernel: int = 5,
) -> dict[str, Any]:
    """Suppress furniture-like density while preserving long wall-like structures."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    density = np.load(density_path).astype(np.float32)
    if density.ndim != 2:
        raise ValueError("Density must be a two-dimensional array")
    support = (
        np.load(support_path).astype(np.float32)
        if support_path is not None
        else np.ones_like(density, dtype=np.float32)
    )
    if support.shape != density.shape:
        raise ValueError("Support and density must have the same shape")

    density_info: dict[str, Any] = {}
    if density_info_path is not None:
        with Path(density_info_path).open(encoding="utf-8") as file:
            density_info = json.load(file)

    base = _normalize(density)
    support_score = np.clip(support, 0.0, 1.0).astype(np.float32)
    line_response, line_metadata = _multi_angle_line_response(
        base,
        tuple(int(length) for length in line_lengths),
        int(line_thickness),
        float(angle_step),
    )
    candidate = (base >= float(density_threshold)).astype(np.uint8)
    thin_score = _thin_structure_score(candidate, float(preferred_thickness))

    trajectory_score = np.ones_like(base, dtype=np.float32)
    trajectory_points = 0
    if trajectory_path is not None and density_info:
        trajectory_pixels = _load_trajectory_pixels(trajectory_path, base.shape, density_info)
        trajectory_points = int(len(trajectory_pixels))
        trajectory_score = _trajectory_exterior_score(
            trajectory_pixels,
            base.shape,
            int(trajectory_near_radius),
            int(trajectory_far_radius),
        )

    weights = np.asarray(
        [support_weight, line_weight, thin_weight, trajectory_weight],
        dtype=np.float32,
    )
    weights = np.maximum(weights, 0.0)
    weights /= max(float(weights.sum()), 1e-8)
    structural_prior = (
        weights[0] * support_score
        + weights[1] * line_response
        + weights[2] * thin_score
        + weights[3] * trajectory_score
    )
    confidence = base * (0.25 + 0.75 * structural_prior)
    confidence *= 0.65 + 0.35 * support_score
    confidence = _normalize(confidence)

    if close_kernel > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (_odd(int(close_kernel)), _odd(int(close_kernel))),
        )
        confidence = cv2.morphologyEx(confidence, cv2.MORPH_CLOSE, kernel)
        confidence = _normalize(confidence)

    np.save(output_dir / "wall_confidence.npy", confidence.astype(np.float32))
    np.save(output_dir / "line_response.npy", line_response.astype(np.float32))
    np.save(output_dir / "thin_structure_score.npy", thin_score.astype(np.float32))
    np.save(output_dir / "trajectory_exterior_score.npy", trajectory_score.astype(np.float32))
    _save_confidence_png(output_dir / "wall_confidence.png", confidence)
    _save_confidence_png(output_dir / "line_response.png", line_response)
    _save_confidence_png(output_dir / "thin_structure_score.png", thin_score)
    _save_confidence_png(output_dir / "trajectory_exterior_score.png", trajectory_score)
    _draw_diagnostic_panel(
        output_dir / "wall_confidence_diagnostics.png",
        [
            ("input density", base),
            ("line response", line_response),
            ("thin score", thin_score),
            ("trajectory score", trajectory_score),
            ("wall confidence", confidence),
        ],
    )

    metadata = {
        "method": "line_thickness_trajectory_wall_confidence",
        "inputs": {
            "density": str(density_path),
            "support": str(support_path) if support_path else None,
            "density_info": str(density_info_path) if density_info_path else None,
            "trajectory": str(trajectory_path) if trajectory_path else None,
        },
        "parameters": {
            "density_threshold": float(density_threshold),
            "weights": {
                "support": float(weights[0]),
                "line": float(weights[1]),
                "thin": float(weights[2]),
                "trajectory": float(weights[3]),
            },
            "preferred_thickness_pixels": float(preferred_thickness),
            "trajectory_near_radius_pixels": int(trajectory_near_radius),
            "trajectory_far_radius_pixels": int(trajectory_far_radius),
            "close_kernel": int(close_kernel),
            **line_metadata,
        },
        "image_shape": list(base.shape),
        "trajectory_points": trajectory_points,
        "input_nonzero_pixels": int((base > 0).sum()),
        "confidence_nonzero_pixels": int((confidence > 0).sum()),
        "confidence_pixels_above_018": int((confidence >= 0.18).sum()),
        "outputs": {
            "wall_confidence": "wall_confidence.png",
            "line_response": "line_response.png",
            "thin_structure_score": "thin_structure_score.png",
            "trajectory_exterior_score": "trajectory_exterior_score.png",
            "diagnostics": "wall_confidence_diagnostics.png",
        },
    }
    with (output_dir / "wall_confidence_info.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)
    return metadata
