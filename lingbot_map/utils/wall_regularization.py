"""Regularize wall confidence maps into Manhattan-aligned line masks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw

from lingbot_map.utils.wall_lines import (
    build_wall_candidate_mask,
    detect_wall_segments,
    merge_manhattan_segments,
    skeletonize_mask,
)


def _estimate_manhattan_angle(mask: np.ndarray) -> float:
    edges = cv2.Canny(mask, 50, 150)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 360,
        threshold=30,
        minLineLength=max(20, min(mask.shape) // 16),
        maxLineGap=15,
    )
    if lines is None:
        return 0.0
    vectors = []
    for x1, y1, x2, y2 in lines[:, 0]:
        angle = np.arctan2(y2 - y1, x2 - x1)
        length = np.hypot(x2 - x1, y2 - y1)
        vectors.append(length * np.exp(4j * angle))
    mean_vector = np.sum(vectors)
    if abs(mean_vector) < 1e-8:
        return 0.0
    return float(np.degrees(np.angle(mean_vector) / 4.0))


def _rotation_matrix(image_shape: tuple[int, int], angle_degrees: float) -> np.ndarray:
    height, width = image_shape
    return cv2.getRotationMatrix2D(((width - 1) / 2.0, (height - 1) / 2.0), angle_degrees, 1.0)


def _save_mask_png(path: Path, mask: np.ndarray) -> None:
    Image.fromarray(mask.astype(np.uint8) * 255).save(path)


def _save_density_png(path: Path, density: np.ndarray) -> None:
    image = (255.0 * (1.0 - np.sqrt(np.clip(density, 0.0, 1.0)))).astype(np.uint8)
    Image.fromarray(image).save(path)


def _draw_segments_on_density(
    output_path: Path,
    density: np.ndarray,
    raw_segments: list[dict[str, Any]],
    merged_segments: list[dict[str, Any]],
) -> None:
    image = (255.0 * (1.0 - np.sqrt(np.clip(density, 0.0, 1.0)))).astype(np.uint8)
    canvas = Image.fromarray(image).convert("RGB")
    draw = ImageDraw.Draw(canvas)
    for segment in raw_segments:
        draw.line(segment["pixel"], fill=(80, 150, 255), width=1)
    for segment in merged_segments:
        draw.line(segment["pixel"], fill=(230, 40, 40), width=3)
    canvas.save(output_path)


def _rasterize_regularized_segments(
    segments: list[dict[str, Any]],
    shape: tuple[int, int],
    extend_pixels: int,
    thickness: int,
) -> np.ndarray:
    height, width = shape
    mask = np.zeros((height, width), dtype=np.uint8)
    for segment in segments:
        x1, y1, x2, y2 = segment["pixel"]
        if segment["orientation"] == "horizontal":
            x1 = max(0, x1 - extend_pixels)
            x2 = min(width - 1, x2 + extend_pixels)
        else:
            y1 = max(0, y1 - extend_pixels)
            y2 = min(height - 1, y2 + extend_pixels)
        cv2.line(mask, (x1, y1), (x2, y2), 255, thickness=thickness)
    return mask


def _complete_outer_frame(candidate_mask: np.ndarray, thickness: int) -> tuple[np.ndarray, dict[str, int]]:
    ys, xs = np.where(candidate_mask > 0)
    if len(xs) == 0:
        return np.zeros_like(candidate_mask, dtype=np.uint8), {
            "x_min": 0,
            "x_max": 0,
            "y_min": 0,
            "y_max": 0,
        }
    x_min, x_max = np.percentile(xs, [2, 98]).astype(int)
    y_min, y_max = np.percentile(ys, [2, 98]).astype(int)
    frame = np.zeros_like(candidate_mask, dtype=np.uint8)
    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), 255, thickness=int(thickness))
    return frame, {
        "x_min": int(x_min),
        "x_max": int(x_max),
        "y_min": int(y_min),
        "y_max": int(y_max),
    }


def regularize_wall_lines(
    density_path: str | Path,
    output_dir: str | Path,
    support_path: str | Path | None = None,
    density_threshold: float = 0.20,
    minimum_support: float = 1.0 / 3.0,
    close_kernel: int = 9,
    open_kernel: int = 3,
    minimum_component_area: int = 80,
    hough_threshold: int = 28,
    minimum_line_length: int = 28,
    maximum_line_gap: int = 20,
    angle_tolerance: float = 12.0,
    coordinate_tolerance: float = 12.0,
    merge_gap: float = 28.0,
    minimum_merged_length: float = 38.0,
    extend_pixels: int = 24,
    line_thickness: int = 5,
    final_close_kernel: int = 31,
    complete_outer_frame: bool = False,
) -> dict[str, Any]:
    """Rotate a density to its dominant Manhattan frame and regularize wall lines."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    density = np.load(density_path).astype(np.float32)
    support = np.load(support_path).astype(np.float32) if support_path else None
    if density.ndim != 2 or (support is not None and support.shape != density.shape):
        raise ValueError("Density and support must be matching two-dimensional arrays")

    initial_mask = build_wall_candidate_mask(
        density,
        support,
        density_threshold,
        minimum_support,
        close_kernel,
        open_kernel,
        minimum_component_area,
    )
    manhattan_angle = _estimate_manhattan_angle(initial_mask)
    rotation = _rotation_matrix(density.shape, manhattan_angle)
    aligned_density = cv2.warpAffine(
        density,
        rotation,
        density.shape[::-1],
        flags=cv2.INTER_LINEAR,
        borderValue=0,
    )
    aligned_support = (
        cv2.warpAffine(
            support,
            rotation,
            density.shape[::-1],
            flags=cv2.INTER_LINEAR,
            borderValue=0,
        )
        if support is not None
        else None
    )
    aligned_candidate = build_wall_candidate_mask(
        aligned_density,
        aligned_support,
        density_threshold,
        minimum_support,
        close_kernel,
        open_kernel,
        minimum_component_area,
    )
    skeleton = skeletonize_mask(aligned_candidate)
    raw_segments = detect_wall_segments(
        skeleton,
        hough_threshold,
        minimum_line_length,
        maximum_line_gap,
    )
    merged_segments = merge_manhattan_segments(
        raw_segments,
        angle_tolerance,
        coordinate_tolerance,
        merge_gap,
        minimum_merged_length,
    )
    regularized = _rasterize_regularized_segments(
        merged_segments,
        density.shape,
        int(extend_pixels),
        int(line_thickness),
    )
    if final_close_kernel > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (int(final_close_kernel), int(final_close_kernel)),
        )
        regularized = cv2.morphologyEx(regularized, cv2.MORPH_CLOSE, kernel)
    frame_metadata = None
    completed = regularized.copy()
    if complete_outer_frame:
        outer_frame, frame_metadata = _complete_outer_frame(aligned_candidate, line_thickness)
        completed = cv2.bitwise_or(completed, outer_frame)
        if final_close_kernel > 0:
            completed = cv2.morphologyEx(completed, cv2.MORPH_CLOSE, kernel)

    np.save(output_dir / "aligned_density.npy", aligned_density.astype(np.float32))
    np.save(output_dir / "aligned_wall_candidate_mask.npy", aligned_candidate > 0)
    np.save(output_dir / "aligned_regularized_wall_mask.npy", regularized > 0)
    np.save(output_dir / "aligned_completed_wall_mask.npy", completed > 0)
    _save_density_png(output_dir / "aligned_density.png", aligned_density)
    _save_mask_png(output_dir / "aligned_wall_candidate_mask.png", aligned_candidate > 0)
    _save_mask_png(output_dir / "aligned_regularized_wall_mask.png", regularized > 0)
    _save_mask_png(output_dir / "aligned_completed_wall_mask.png", completed > 0)
    _draw_segments_on_density(
        output_dir / "aligned_wall_lines_overlay.png",
        aligned_density,
        raw_segments,
        merged_segments,
    )
    metadata = {
        "method": "manhattan_aligned_wall_line_regularization",
        "inputs": {
            "density": str(density_path),
            "support": str(support_path) if support_path else None,
        },
        "parameters": {
            "density_threshold": float(density_threshold),
            "minimum_support": float(minimum_support),
            "close_kernel": int(close_kernel),
            "open_kernel": int(open_kernel),
            "minimum_component_area": int(minimum_component_area),
            "hough_threshold": int(hough_threshold),
            "minimum_line_length": int(minimum_line_length),
            "maximum_line_gap": int(maximum_line_gap),
            "angle_tolerance": float(angle_tolerance),
            "coordinate_tolerance": float(coordinate_tolerance),
            "merge_gap": float(merge_gap),
            "minimum_merged_length": float(minimum_merged_length),
            "extend_pixels": int(extend_pixels),
            "line_thickness": int(line_thickness),
            "final_close_kernel": int(final_close_kernel),
            "complete_outer_frame": bool(complete_outer_frame),
        },
        "outer_frame": frame_metadata,
        "manhattan_angle_degrees": float(manhattan_angle),
        "image_shape": list(density.shape),
        "raw_segment_count": len(raw_segments),
        "merged_segment_count": len(merged_segments),
        "regularized_pixels": int((regularized > 0).sum()),
        "raw_segments": raw_segments,
        "merged_segments": merged_segments,
        "outputs": {
            "aligned_density": "aligned_density.png",
            "aligned_candidate_mask": "aligned_wall_candidate_mask.png",
            "aligned_regularized_wall_mask": "aligned_regularized_wall_mask.png",
            "aligned_completed_wall_mask": "aligned_completed_wall_mask.png",
            "aligned_line_overlay": "aligned_wall_lines_overlay.png",
        },
    }
    with (output_dir / "wall_regularization_info.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)
    return metadata
