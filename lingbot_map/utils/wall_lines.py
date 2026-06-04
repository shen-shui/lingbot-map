"""Wall-line extraction baseline for top-down wall-density maps."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw


def _odd_kernel_size(size: int) -> int:
    if size <= 0:
        return 0
    return size if size % 2 == 1 else size + 1


def _filter_small_components(mask: np.ndarray, minimum_area: int) -> np.ndarray:
    if minimum_area <= 0:
        return mask
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    filtered = np.zeros_like(mask)
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] >= minimum_area:
            filtered[labels == label] = 255
    return filtered


def build_wall_candidate_mask(
    density: np.ndarray,
    support: np.ndarray | None = None,
    density_threshold: float = 0.18,
    minimum_support: float = 1.0 / 3.0,
    close_kernel: int = 7,
    open_kernel: int = 3,
    minimum_component_area: int = 80,
) -> np.ndarray:
    """Threshold and clean a wall-density map."""
    candidate = density >= density_threshold
    if support is not None:
        candidate &= support >= minimum_support
    mask = candidate.astype(np.uint8) * 255

    close_kernel = _odd_kernel_size(close_kernel)
    open_kernel = _odd_kernel_size(open_kernel)
    if close_kernel:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (close_kernel, close_kernel))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    if open_kernel:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (open_kernel, open_kernel))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return _filter_small_components(mask, minimum_component_area)


def skeletonize_mask(mask: np.ndarray) -> np.ndarray:
    """Reduce thick wall candidates to one-pixel centerline candidates."""
    skeleton = np.zeros_like(mask)
    remaining = mask.copy()
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while cv2.countNonZero(remaining):
        opened = cv2.morphologyEx(remaining, cv2.MORPH_OPEN, element)
        skeleton = cv2.bitwise_or(skeleton, cv2.subtract(remaining, opened))
        remaining = cv2.erode(remaining, element)
    return skeleton


def detect_wall_segments(
    mask: np.ndarray,
    hough_threshold: int = 30,
    minimum_line_length: int = 25,
    maximum_line_gap: int = 15,
) -> list[dict[str, Any]]:
    """Detect raw line segments from a cleaned wall candidate mask."""
    edges = cv2.Canny(mask, 50, 150, apertureSize=3)
    detected = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=hough_threshold,
        minLineLength=minimum_line_length,
        maxLineGap=maximum_line_gap,
    )
    if detected is None:
        return []

    segments = []
    for line in detected[:, 0]:
        x1, y1, x2, y2 = (int(value) for value in line)
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy)
        angle = math.degrees(math.atan2(dy, dx)) % 180.0
        segments.append(
            {
                "pixel": [x1, y1, x2, y2],
                "length_pixels": float(length),
                "angle_degrees": float(angle),
            }
        )
    return segments


def _snap_manhattan_segment(
    segment: dict[str, Any],
    angle_tolerance: float,
) -> dict[str, Any] | None:
    x1, y1, x2, y2 = segment["pixel"]
    angle = segment["angle_degrees"]
    horizontal_error = min(angle, 180.0 - angle)
    vertical_error = abs(angle - 90.0)
    if min(horizontal_error, vertical_error) > angle_tolerance:
        return None
    if horizontal_error <= vertical_error:
        coordinate = 0.5 * (y1 + y2)
        start, end = sorted((x1, x2))
        orientation = "horizontal"
    else:
        coordinate = 0.5 * (x1 + x2)
        start, end = sorted((y1, y2))
        orientation = "vertical"
    return {
        "orientation": orientation,
        "coordinate": float(coordinate),
        "start": float(start),
        "end": float(end),
        "source_count": 1,
    }


def _merge_group_intervals(
    segments: list[dict[str, Any]],
    maximum_gap: float,
) -> list[dict[str, Any]]:
    segments = sorted(segments, key=lambda item: item["start"])
    merged: list[dict[str, Any]] = []
    current = segments[0].copy()
    coordinates = [current["coordinate"]]
    for segment in segments[1:]:
        if segment["start"] <= current["end"] + maximum_gap:
            current["end"] = max(current["end"], segment["end"])
            current["source_count"] += segment["source_count"]
            coordinates.append(segment["coordinate"])
            current["coordinate"] = float(np.median(coordinates))
        else:
            merged.append(current)
            current = segment.copy()
            coordinates = [current["coordinate"]]
    merged.append(current)
    return merged


def merge_manhattan_segments(
    raw_segments: list[dict[str, Any]],
    angle_tolerance: float = 12.0,
    coordinate_tolerance: float = 10.0,
    maximum_gap: float = 20.0,
    minimum_merged_length: float = 35.0,
) -> list[dict[str, Any]]:
    """Snap near-Manhattan segments and merge nearby collinear intervals."""
    snapped = [
        snapped_segment
        for segment in raw_segments
        if (snapped_segment := _snap_manhattan_segment(segment, angle_tolerance)) is not None
    ]
    merged: list[dict[str, Any]] = []
    for orientation in ("horizontal", "vertical"):
        remaining = sorted(
            (segment for segment in snapped if segment["orientation"] == orientation),
            key=lambda item: item["coordinate"],
        )
        while remaining:
            seed = remaining.pop(0)
            group = [seed]
            outside = []
            for segment in remaining:
                if abs(segment["coordinate"] - seed["coordinate"]) <= coordinate_tolerance:
                    group.append(segment)
                else:
                    outside.append(segment)
            remaining = outside
            coordinate = float(np.median([segment["coordinate"] for segment in group]))
            for segment in group:
                segment["coordinate"] = coordinate
            merged.extend(_merge_group_intervals(group, maximum_gap))

    output = []
    for segment in merged:
        if segment["end"] - segment["start"] < minimum_merged_length:
            continue
        coordinate = int(round(segment["coordinate"]))
        start = int(round(segment["start"]))
        end = int(round(segment["end"]))
        if segment["orientation"] == "horizontal":
            pixel = [start, coordinate, end, coordinate]
        else:
            pixel = [coordinate, start, coordinate, end]
        output.append(
            {
                "orientation": segment["orientation"],
                "pixel": pixel,
                "length_pixels": float(segment["end"] - segment["start"]),
                "source_count": int(segment["source_count"]),
            }
        )
    return sorted(output, key=lambda item: item["length_pixels"], reverse=True)


def _pixel_to_world(
    x: float,
    y: float,
    image_shape: tuple[int, int],
    horizontal_bounds: list[list[float]],
) -> list[float]:
    height, width = image_shape
    lower, upper = np.asarray(horizontal_bounds, dtype=np.float64)
    first = lower[0] + x / max(width - 1, 1) * (upper[0] - lower[0])
    second = lower[1] + (height - 1 - y) / max(height - 1, 1) * (upper[1] - lower[1])
    return [float(first), float(second)]


def _draw_segments(
    density: np.ndarray,
    raw_segments: list[dict[str, Any]],
    merged_segments: list[dict[str, Any]],
    output_path: Path,
) -> None:
    gray = (255.0 * (1.0 - np.sqrt(np.clip(density, 0.0, 1.0)))).astype(np.uint8)
    image = Image.fromarray(gray).convert("RGB")
    draw = ImageDraw.Draw(image)
    for segment in raw_segments:
        draw.line(segment["pixel"], fill=(80, 150, 255), width=1)
    for segment in merged_segments:
        draw.line(segment["pixel"], fill=(230, 40, 40), width=3)
    image.save(output_path)


def extract_wall_lines(
    density_path: str | Path,
    output_dir: str | Path,
    support_path: str | Path | None = None,
    density_info_path: str | Path | None = None,
    density_threshold: float = 0.18,
    minimum_support: float = 1.0 / 3.0,
    close_kernel: int = 7,
    open_kernel: int = 3,
    minimum_component_area: int = 80,
    hough_threshold: int = 30,
    minimum_line_length: int = 25,
    maximum_line_gap: int = 15,
    angle_tolerance: float = 12.0,
    coordinate_tolerance: float = 10.0,
    merge_gap: float = 20.0,
    minimum_merged_length: float = 35.0,
) -> dict[str, Any]:
    """Run the complete wall-line extraction baseline and save its artifacts."""
    density_path = Path(density_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    density = np.load(density_path).astype(np.float32)
    support = np.load(support_path).astype(np.float32) if support_path else None
    if density.ndim != 2 or (support is not None and support.shape != density.shape):
        raise ValueError("Density and support must be matching two-dimensional arrays")

    mask = build_wall_candidate_mask(
        density,
        support,
        density_threshold,
        minimum_support,
        close_kernel,
        open_kernel,
        minimum_component_area,
    )
    skeleton = skeletonize_mask(mask)
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

    density_info = {}
    if density_info_path:
        with Path(density_info_path).open(encoding="utf-8") as file:
            density_info = json.load(file)
        bounds = density_info.get("horizontal_bounds")
        if bounds:
            for segment in merged_segments:
                x1, y1, x2, y2 = segment["pixel"]
                segment["world"] = [
                    _pixel_to_world(x1, y1, density.shape, bounds),
                    _pixel_to_world(x2, y2, density.shape, bounds),
                ]

    Image.fromarray(mask).save(output_dir / "wall_candidate_mask.png")
    Image.fromarray(skeleton).save(output_dir / "wall_candidate_skeleton.png")
    np.save(output_dir / "wall_candidate_mask.npy", mask > 0)
    np.save(output_dir / "wall_candidate_skeleton.npy", skeleton > 0)
    _draw_segments(density, raw_segments, merged_segments, output_dir / "wall_lines_overlay.png")
    metadata = {
        "method": "threshold_morphology_hough_manhattan_merge_baseline",
        "inputs": {
            "density": str(density_path),
            "support": str(support_path) if support_path else None,
            "density_info": str(density_info_path) if density_info_path else None,
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
        },
        "image_shape": list(density.shape),
        "horizontal_axes": density_info.get("horizontal_axes"),
        "candidate_pixels": int((mask > 0).sum()),
        "raw_segment_count": len(raw_segments),
        "merged_segment_count": len(merged_segments),
        "raw_segments": raw_segments,
        "merged_segments": merged_segments,
        "outputs": {
            "candidate_mask": "wall_candidate_mask.png",
            "candidate_skeleton": "wall_candidate_skeleton.png",
            "line_overlay": "wall_lines_overlay.png",
        },
    }
    with (output_dir / "wall_lines.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)
    return metadata
