"""Closed orthogonal floorplan baseline from wall candidate masks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw


def _find_largest_enclosed_region(wall_mask: np.ndarray) -> np.ndarray:
    free_space = (wall_mask == 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(free_space, connectivity=8)
    candidates = []
    height, width = wall_mask.shape
    for label in range(1, count):
        x, y, component_width, component_height, area = stats[label]
        touches_border = (
            x == 0
            or y == 0
            or x + component_width == width
            or y + component_height == height
        )
        if not touches_border:
            candidates.append((int(area), label))
    if not candidates:
        raise ValueError("No enclosed interior region found; increase --connection-kernel")
    _, largest_label = max(candidates)
    return (labels == largest_label).astype(np.uint8) * 255


def _orthogonalize_contour(points: np.ndarray) -> np.ndarray:
    edges = []
    for start, end in zip(points, np.roll(points, -1, axis=0)):
        delta = end - start
        length = float(np.linalg.norm(delta))
        if length <= 0:
            continue
        orientation = "horizontal" if abs(delta[0]) >= abs(delta[1]) else "vertical"
        coordinate = (
            0.5 * (start[1] + end[1])
            if orientation == "horizontal"
            else 0.5 * (start[0] + end[0])
        )
        edges.append([orientation, float(coordinate), length])
    if len(edges) < 4:
        raise ValueError("Contour has too few edges for orthogonalization")

    boundary_index = next(
        (index for index in range(len(edges)) if edges[index][0] != edges[index - 1][0]),
        0,
    )
    edges = edges[boundary_index:] + edges[:boundary_index]
    runs: list[list[Any]] = []
    for orientation, coordinate, length in edges:
        if runs and runs[-1][0] == orientation:
            total_length = runs[-1][2] + length
            runs[-1][1] = (
                runs[-1][1] * runs[-1][2] + coordinate * length
            ) / total_length
            runs[-1][2] = total_length
        else:
            runs.append([orientation, coordinate, length])
    if len(runs) < 4 or len(runs) % 2:
        raise ValueError("Unable to form an alternating orthogonal polygon")

    vertices = []
    for current, following in zip(runs, np.roll(np.asarray(runs, dtype=object), -1, axis=0)):
        if current[0] == "horizontal":
            vertices.append([following[1], current[1]])
        else:
            vertices.append([current[1], following[1]])
    return np.rint(np.asarray(vertices)).astype(np.int32)


def _pixel_to_world(
    point: np.ndarray,
    image_shape: tuple[int, int],
    horizontal_bounds: list[list[float]],
) -> list[float]:
    height, width = image_shape
    lower, upper = np.asarray(horizontal_bounds, dtype=np.float64)
    x, y = point
    first = lower[0] + x / max(width - 1, 1) * (upper[0] - lower[0])
    second = lower[1] + (height - 1 - y) / max(height - 1, 1) * (upper[1] - lower[1])
    return [float(first), float(second)]


def _polygon_area(points: np.ndarray) -> float:
    x = points[:, 0]
    y = points[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) * 0.5)


def _save_overlay(
    output_path: Path,
    polygon: np.ndarray,
    interior: np.ndarray,
    density_path: str | Path | None,
) -> None:
    if density_path:
        density = np.load(density_path).astype(np.float32)
        gray = (255.0 * (1.0 - np.sqrt(np.clip(density, 0.0, 1.0)))).astype(np.uint8)
        image = Image.fromarray(gray).convert("RGB")
    else:
        image = Image.new("RGB", (interior.shape[1], interior.shape[0]), "white")
    fill = Image.new("RGBA", image.size, (0, 0, 0, 0))
    fill_draw = ImageDraw.Draw(fill)
    fill_draw.polygon([tuple(point) for point in polygon], fill=(45, 180, 90, 70))
    image = Image.alpha_composite(image.convert("RGBA"), fill)
    draw = ImageDraw.Draw(image)
    closed_polygon = [tuple(point) for point in polygon] + [tuple(polygon[0])]
    draw.line(closed_polygon, fill=(220, 35, 35, 255), width=4)
    for index, point in enumerate(polygon):
        x, y = (int(value) for value in point)
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=(20, 80, 220, 255))
        draw.text((x + 4, y + 2), str(index), fill=(20, 20, 20, 255))
    image.convert("RGB").save(output_path)


def generate_floorplan_polygon(
    wall_mask_path: str | Path,
    output_dir: str | Path,
    density_info_path: str | Path | None = None,
    density_path: str | Path | None = None,
    connection_kernel: int = 15,
    approximation_epsilon: float = 0.005,
) -> dict[str, Any]:
    """Connect wall candidates and generate a closed orthogonal floorplan polygon."""
    wall_mask_path = Path(wall_mask_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    wall_mask = np.load(wall_mask_path)
    if wall_mask.ndim != 2:
        raise ValueError("Wall candidate mask must be two-dimensional")
    wall_mask = wall_mask.astype(np.uint8) * 255
    if connection_kernel <= 0:
        raise ValueError("connection_kernel must be positive")
    if approximation_epsilon <= 0:
        raise ValueError("approximation_epsilon must be positive")
    if connection_kernel % 2 == 0:
        connection_kernel += 1

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (connection_kernel, connection_kernel),
    )
    connected_wall_mask = cv2.morphologyEx(wall_mask, cv2.MORPH_CLOSE, kernel)
    interior = _find_largest_enclosed_region(connected_wall_mask)
    contours, _ = cv2.findContours(interior, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(contour, True)
    simplified = cv2.approxPolyDP(contour, approximation_epsilon * perimeter, True)
    polygon = _orthogonalize_contour(simplified[:, 0, :].astype(np.float64))

    polygon_mask = np.zeros_like(interior)
    cv2.fillPoly(polygon_mask, [polygon], 255)
    intersection = np.logical_and(polygon_mask > 0, interior > 0).sum()
    union = np.logical_or(polygon_mask > 0, interior > 0).sum()
    interior_iou = float(intersection / max(union, 1))

    density_info = {}
    world_polygon = None
    world_area = None
    if density_info_path:
        with Path(density_info_path).open(encoding="utf-8") as file:
            density_info = json.load(file)
        horizontal_bounds = density_info.get("horizontal_bounds")
        if horizontal_bounds:
            world_polygon = [
                _pixel_to_world(point, wall_mask.shape, horizontal_bounds)
                for point in polygon
            ]
            world_area = _polygon_area(np.asarray(world_polygon, dtype=np.float64))

    Image.fromarray(connected_wall_mask).save(output_dir / "connected_wall_mask.png")
    Image.fromarray(interior).save(output_dir / "interior_region.png")
    Image.fromarray(polygon_mask).save(output_dir / "floorplan_polygon_mask.png")
    _save_overlay(
        output_dir / "floorplan_polygon_overlay.png",
        polygon,
        interior,
        density_path,
    )
    metadata = {
        "method": "wall_connection_enclosed_region_orthogonal_polygon_baseline",
        "inputs": {
            "wall_mask": str(wall_mask_path),
            "density_info": str(density_info_path) if density_info_path else None,
            "density": str(density_path) if density_path else None,
        },
        "parameters": {
            "connection_kernel": int(connection_kernel),
            "approximation_epsilon": float(approximation_epsilon),
        },
        "image_shape": list(wall_mask.shape),
        "horizontal_axes": density_info.get("horizontal_axes"),
        "vertex_count": int(len(polygon)),
        "pixel_polygon": polygon.tolist(),
        "world_polygon": world_polygon,
        "pixel_area": _polygon_area(polygon.astype(np.float64)),
        "world_area": world_area,
        "world_area_units": "reconstruction_coordinate_units_squared",
        "metric_scale_available": False,
        "interior_iou": interior_iou,
        "outputs": {
            "connected_wall_mask": "connected_wall_mask.png",
            "interior_region": "interior_region.png",
            "polygon_mask": "floorplan_polygon_mask.png",
            "polygon_overlay": "floorplan_polygon_overlay.png",
        },
    }
    with (output_dir / "floorplan_polygon.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)
    return metadata
