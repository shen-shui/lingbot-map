"""Closed orthogonal floorplan baseline from wall candidate masks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw


def _find_enclosed_regions(wall_mask: np.ndarray) -> list[tuple[int, np.ndarray]]:
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
            candidates.append((int(area), (labels == label).astype(np.uint8) * 255))
    if not candidates:
        raise ValueError("No enclosed interior region found; increase --connection-kernel")
    return candidates


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


def _transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack([points, np.ones(len(points), dtype=np.float64)])
    return homogeneous @ matrix.T


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
    alignment_matrix: np.ndarray,
) -> np.ndarray:
    trajectory = np.loadtxt(trajectory_path)
    if trajectory.ndim == 1:
        trajectory = trajectory[None, :]
    if trajectory.shape[1] != 13:
        raise ValueError("Trajectory must contain frame index plus a row-major 3x4 camera-to-world matrix")
    centers = trajectory[:, [4, 8, 12]]
    axis_indices = ["xyz".index(axis) for axis in density_info["horizontal_axes"]]
    horizontal_centers = centers[:, axis_indices]
    pixels = _world_to_pixel(horizontal_centers, image_shape, density_info["horizontal_bounds"])
    return _transform_points(pixels, alignment_matrix)


def _select_interior_region(
    candidates: list[tuple[int, np.ndarray]],
    trajectory_pixels: np.ndarray | None,
    minimum_area_ratio: float,
    minimum_trajectory_coverage: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    image_area = candidates[0][1].size
    eligible = [
        (area, region)
        for area, region in candidates
        if area / image_area >= minimum_area_ratio
    ]
    if not eligible:
        largest_area = max(area for area, _ in candidates)
        raise ValueError(
            f"No enclosed region meets minimum area ratio {minimum_area_ratio:.3f}; "
            f"largest ratio={largest_area / image_area:.3f}"
        )
    scored = []
    for area, region in eligible:
        hits = 0
        valid_count = 0
        if trajectory_pixels is not None:
            rounded = np.rint(trajectory_pixels).astype(np.int32)
            valid = np.logical_and(rounded >= 0, rounded < np.asarray(region.shape[::-1])).all(axis=1)
            rounded = rounded[valid]
            valid_count = len(rounded)
            if valid_count:
                hits = int((region[rounded[:, 1], rounded[:, 0]] > 0).sum())
        hit_ratio = hits / max(valid_count, 1)
        scored.append((hit_ratio, area, region, hits, valid_count))
    hit_ratio, area, region, hits, valid_count = max(scored, key=lambda item: (item[0], item[1]))
    if trajectory_pixels is not None and hit_ratio < minimum_trajectory_coverage:
        raise ValueError(
            f"Best enclosed region covers only {hit_ratio:.3f} of valid trajectory points; "
            f"required {minimum_trajectory_coverage:.3f}"
        )
    return region, {
        "candidate_count": len(candidates),
        "eligible_candidate_count": len(eligible),
        "selected_area_pixels": int(area),
        "selected_area_ratio": float(area / image_area),
        "trajectory_points_inside": int(hits),
        "trajectory_points_valid": int(valid_count),
        "trajectory_coverage_ratio": float(hit_ratio),
    }


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
    trajectory_path: str | Path | None = None,
    connection_kernel: int = 15,
    approximation_epsilon: float = 0.005,
    auto_manhattan: bool = False,
    minimum_area_ratio: float = 0.02,
    minimum_trajectory_coverage: float = 0.5,
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
    if not 0 <= minimum_area_ratio < 1:
        raise ValueError("minimum_area_ratio must be in [0, 1)")
    if not 0 <= minimum_trajectory_coverage <= 1:
        raise ValueError("minimum_trajectory_coverage must be in [0, 1]")
    if connection_kernel % 2 == 0:
        connection_kernel += 1

    density_info = {}
    if density_info_path:
        with Path(density_info_path).open(encoding="utf-8") as file:
            density_info = json.load(file)

    manhattan_angle = _estimate_manhattan_angle(wall_mask) if auto_manhattan else 0.0
    alignment_matrix = _rotation_matrix(wall_mask.shape, manhattan_angle)
    aligned_wall_mask = cv2.warpAffine(
        wall_mask,
        alignment_matrix,
        wall_mask.shape[::-1],
        flags=cv2.INTER_NEAREST,
        borderValue=0,
    )
    trajectory_pixels = None
    if trajectory_path:
        if not density_info.get("horizontal_bounds") or not density_info.get("horizontal_axes"):
            raise ValueError("Trajectory-guided selection requires density_info with bounds and axes")
        trajectory_pixels = _load_trajectory_pixels(
            trajectory_path,
            wall_mask.shape,
            density_info,
            alignment_matrix,
        )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (connection_kernel, connection_kernel),
    )
    connected_wall_mask = cv2.morphologyEx(aligned_wall_mask, cv2.MORPH_CLOSE, kernel)
    interior, selection_info = _select_interior_region(
        _find_enclosed_regions(connected_wall_mask),
        trajectory_pixels,
        minimum_area_ratio,
        minimum_trajectory_coverage,
    )
    contours, _ = cv2.findContours(interior, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(contour, True)
    simplified = cv2.approxPolyDP(contour, approximation_epsilon * perimeter, True)
    aligned_polygon = _orthogonalize_contour(simplified[:, 0, :].astype(np.float64))

    polygon_mask = np.zeros_like(interior)
    cv2.fillPoly(polygon_mask, [aligned_polygon], 255)
    intersection = np.logical_and(polygon_mask > 0, interior > 0).sum()
    union = np.logical_or(polygon_mask > 0, interior > 0).sum()
    interior_iou = float(intersection / max(union, 1))

    inverse_alignment = cv2.invertAffineTransform(alignment_matrix)
    polygon = _transform_points(aligned_polygon.astype(np.float64), inverse_alignment)
    world_polygon = None
    world_area = None
    horizontal_bounds = density_info.get("horizontal_bounds")
    if horizontal_bounds:
        world_polygon = [
            _pixel_to_world(point, wall_mask.shape, horizontal_bounds)
            for point in polygon
        ]
        world_area = _polygon_area(np.asarray(world_polygon, dtype=np.float64))

    Image.fromarray(aligned_wall_mask).save(output_dir / "aligned_wall_mask.png")
    Image.fromarray(connected_wall_mask).save(output_dir / "connected_wall_mask.png")
    Image.fromarray(interior).save(output_dir / "interior_region.png")
    Image.fromarray(polygon_mask).save(output_dir / "floorplan_polygon_mask.png")
    _save_overlay(
        output_dir / "floorplan_polygon_aligned_overlay.png",
        aligned_polygon,
        interior,
        None,
    )
    if density_path:
        density = np.load(density_path).astype(np.float32)
        aligned_density = cv2.warpAffine(
            density,
            alignment_matrix,
            density.shape[::-1],
            flags=cv2.INTER_LINEAR,
            borderValue=0,
        )
        np.save(output_dir / "aligned_density.npy", aligned_density)
        _save_overlay(
            output_dir / "floorplan_polygon_aligned_overlay.png",
            aligned_polygon,
            interior,
            output_dir / "aligned_density.npy",
        )
        _save_overlay(
            output_dir / "floorplan_polygon_overlay.png",
            polygon,
            wall_mask,
            density_path,
        )
    else:
        _save_overlay(
            output_dir / "floorplan_polygon_overlay.png",
            polygon,
            wall_mask,
            None,
        )
    metadata = {
        "method": "wall_connection_enclosed_region_orthogonal_polygon_baseline",
        "inputs": {
            "wall_mask": str(wall_mask_path),
            "density_info": str(density_info_path) if density_info_path else None,
            "density": str(density_path) if density_path else None,
            "trajectory": str(trajectory_path) if trajectory_path else None,
        },
        "parameters": {
            "connection_kernel": int(connection_kernel),
            "approximation_epsilon": float(approximation_epsilon),
            "auto_manhattan": bool(auto_manhattan),
            "minimum_area_ratio": float(minimum_area_ratio),
            "minimum_trajectory_coverage": float(minimum_trajectory_coverage),
        },
        "manhattan_angle_degrees": float(manhattan_angle),
        "alignment_matrix": alignment_matrix.tolist(),
        "region_selection": selection_info,
        "image_shape": list(wall_mask.shape),
        "horizontal_axes": density_info.get("horizontal_axes"),
        "vertex_count": int(len(polygon)),
        "pixel_polygon": polygon.tolist(),
        "aligned_pixel_polygon": aligned_polygon.tolist(),
        "world_polygon": world_polygon,
        "pixel_area": _polygon_area(aligned_polygon.astype(np.float64)),
        "world_area": world_area,
        "world_area_units": "reconstruction_coordinate_units_squared",
        "metric_scale_available": False,
        "interior_iou": interior_iou,
        "outputs": {
            "connected_wall_mask": "connected_wall_mask.png",
            "aligned_wall_mask": "aligned_wall_mask.png",
            "interior_region": "interior_region.png",
            "polygon_mask": "floorplan_polygon_mask.png",
            "polygon_overlay": "floorplan_polygon_overlay.png",
            "aligned_polygon_overlay": "floorplan_polygon_aligned_overlay.png",
        },
    }
    with (output_dir / "floorplan_polygon.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)
    return metadata
