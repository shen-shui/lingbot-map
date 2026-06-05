"""Free-space carving for mobile-video floorplan reconstruction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw


def _camera_to_world_from_extrinsic(extrinsic: np.ndarray) -> np.ndarray:
    rotation = extrinsic[:, :3, :3]
    translation = extrinsic[:, :3, 3]
    camera_to_world = np.tile(np.eye(4, dtype=np.float32), (len(extrinsic), 1, 1))
    camera_to_world[:, :3, :3] = np.transpose(rotation, (0, 2, 1))
    camera_to_world[:, :3, 3] = -np.einsum("sji,sj->si", rotation, translation)
    return camera_to_world


def _normalize(values: np.ndarray, percentile: float = 99.0) -> np.ndarray:
    values = values.astype(np.float32, copy=False)
    positive = values[values > 0]
    if positive.size == 0:
        return np.zeros_like(values, dtype=np.float32)
    scale = np.percentile(positive, percentile)
    return np.clip(values / max(float(scale), 1e-8), 0.0, 1.0).astype(np.float32)


def _save_gray(path: Path, image: np.ndarray, invert: bool = True) -> None:
    image = np.clip(image.astype(np.float32), 0.0, 1.0)
    if invert:
        pixels = 255.0 * (1.0 - np.sqrt(image))
    else:
        pixels = 255.0 * np.sqrt(image)
    Image.fromarray(pixels.clip(0, 255).astype(np.uint8)).save(path)


def _unproject_frame_points(
    depth: np.ndarray,
    intrinsic: np.ndarray,
    world_to_camera: np.ndarray,
    pixel_stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    height, width = depth.shape
    rows, cols = np.meshgrid(
        np.arange(0, height, pixel_stride),
        np.arange(0, width, pixel_stride),
        indexing="ij",
    )
    sampled_depth = depth[::pixel_stride, ::pixel_stride]
    x = (cols - intrinsic[0, 2]) * sampled_depth / max(float(intrinsic[0, 0]), 1e-8)
    y = (rows - intrinsic[1, 2]) * sampled_depth / max(float(intrinsic[1, 1]), 1e-8)
    camera_points = np.stack((x, y, sampled_depth), axis=-1)
    rotation = world_to_camera[:3, :3]
    translation = world_to_camera[:3, 3]
    world_points = (camera_points - translation) @ rotation
    return world_points.reshape(-1, 3).astype(np.float32), sampled_depth.reshape(-1)


def _choose_bounds(
    horizontal_points: np.ndarray,
    density_info: dict[str, Any] | None,
    percentiles: tuple[float, float],
    margin: float,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    if density_info and density_info.get("horizontal_bounds") and density_info.get("horizontal_axes"):
        bounds = np.asarray(density_info["horizontal_bounds"], dtype=np.float32)
        return bounds[0], bounds[1], list(density_info["horizontal_axes"])
    lower = np.percentile(horizontal_points, percentiles[0], axis=0)
    upper = np.percentile(horizontal_points, percentiles[1], axis=0)
    pad = (upper - lower) * float(margin)
    return lower - pad, upper + pad, ["x", "z"]


def _world_to_pixel(
    horizontal: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    grid_size: int,
) -> np.ndarray:
    pixels = np.empty_like(horizontal, dtype=np.float32)
    pixels[:, 0] = (horizontal[:, 0] - lower[0]) / max(float(upper[0] - lower[0]), 1e-8)
    pixels[:, 1] = (horizontal[:, 1] - lower[1]) / max(float(upper[1] - lower[1]), 1e-8)
    pixels[:, 0] *= grid_size - 1
    pixels[:, 1] = (grid_size - 1) - pixels[:, 1] * (grid_size - 1)
    return pixels


def _filter_small_components(mask: np.ndarray, minimum_area: int) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    filtered = np.zeros_like(mask, dtype=np.uint8)
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] >= minimum_area:
            filtered[labels == label] = 255
    return filtered


def _largest_component(mask: np.ndarray) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if count <= 1:
        return mask.astype(np.uint8)
    label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == label).astype(np.uint8) * 255


def _extract_free_space_polygon(
    free_mask: np.ndarray,
    density: np.ndarray | None,
    output_dir: Path,
    approximation_epsilon: float,
    snap_boundary: bool = False,
    snap_search_radius: int = 18,
    snap_samples_per_edge: int = 96,
    orthogonalize_boundary: bool = False,
) -> dict[str, Any]:
    contours, _ = cv2.findContours(free_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No free-space component available for polygon extraction")
    contour = max(contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(contour, True)
    polygon = cv2.approxPolyDP(contour, approximation_epsilon * perimeter, True)[:, 0, :]
    snapped_polygon = None
    if snap_boundary and density is not None:
        snapped_polygon = _snap_polygon_to_evidence(
            polygon.astype(np.float32),
            density,
            search_radius=int(snap_search_radius),
            samples_per_edge=int(snap_samples_per_edge),
        )
    polygon_mask = np.zeros_like(free_mask, dtype=np.uint8)
    cv2.fillPoly(polygon_mask, [polygon.astype(np.int32)], 255)
    snapped_mask = None
    if snapped_polygon is not None:
        snapped_mask = np.zeros_like(free_mask, dtype=np.uint8)
        cv2.fillPoly(snapped_mask, [np.rint(snapped_polygon).astype(np.int32)], 255)

    if density is not None:
        gray = (255.0 * (1.0 - np.sqrt(np.clip(density, 0.0, 1.0)))).astype(np.uint8)
        image = Image.fromarray(gray).convert("RGB")
    else:
        image = Image.new("RGB", (free_mask.shape[1], free_mask.shape[0]), "white")
    fill = Image.new("RGBA", image.size, (0, 0, 0, 0))
    fill_draw = ImageDraw.Draw(fill)
    fill_draw.polygon([tuple(point) for point in polygon], fill=(45, 180, 90, 70))
    image = Image.alpha_composite(image.convert("RGBA"), fill)
    draw = ImageDraw.Draw(image)
    closed = [tuple(point) for point in polygon] + [tuple(polygon[0])]
    draw.line(closed, fill=(220, 35, 35, 255), width=4)
    for index, point in enumerate(polygon):
        x, y = (int(value) for value in point)
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=(20, 80, 220, 255))
        draw.text((x + 4, y + 2), str(index), fill=(20, 20, 20, 255))
    image.convert("RGB").save(output_dir / "free_space_polygon_overlay.png")
    Image.fromarray(polygon_mask).save(output_dir / "free_space_polygon_mask.png")
    outputs = {
        "polygon_overlay": "free_space_polygon_overlay.png",
        "polygon_mask": "free_space_polygon_mask.png",
    }
    snapped_info = None
    snapped_orthogonal_info = None
    if snapped_polygon is not None and snapped_mask is not None:
        _save_polygon_overlay(
            output_dir / "snapped_free_space_polygon_overlay.png",
            snapped_polygon,
            density,
            fill_color=(70, 150, 240, 70),
        )
        Image.fromarray(snapped_mask).save(output_dir / "snapped_free_space_polygon_mask.png")
        outputs["snapped_polygon_overlay"] = "snapped_free_space_polygon_overlay.png"
        outputs["snapped_polygon_mask"] = "snapped_free_space_polygon_mask.png"
        snapped_info = {
            "vertex_count": int(len(snapped_polygon)),
            "area_pixels": float(cv2.contourArea(snapped_polygon.astype(np.float32))),
            "polygon_pixel": snapped_polygon.astype(float).tolist(),
        }
        if orthogonalize_boundary:
            orthogonal_polygon = _orthogonalize_polygon_edges(snapped_polygon)
            orthogonal_mask = np.zeros_like(free_mask, dtype=np.uint8)
            cv2.fillPoly(orthogonal_mask, [np.rint(orthogonal_polygon).astype(np.int32)], 255)
            _save_polygon_overlay(
                output_dir / "orthogonal_free_space_polygon_overlay.png",
                orthogonal_polygon,
                density,
                fill_color=(245, 170, 40, 70),
            )
            Image.fromarray(orthogonal_mask).save(output_dir / "orthogonal_free_space_polygon_mask.png")
            outputs["orthogonal_polygon_overlay"] = "orthogonal_free_space_polygon_overlay.png"
            outputs["orthogonal_polygon_mask"] = "orthogonal_free_space_polygon_mask.png"
            snapped_orthogonal_info = {
                "vertex_count": int(len(orthogonal_polygon)),
                "area_pixels": float(cv2.contourArea(orthogonal_polygon.astype(np.float32))),
                "polygon_pixel": orthogonal_polygon.astype(float).tolist(),
            }
    result = {
        "vertex_count": int(len(polygon)),
        "area_pixels": float(cv2.contourArea(polygon.astype(np.float32))),
        "polygon_pixel": polygon.astype(float).tolist(),
        "snapped": snapped_info,
        "orthogonal": snapped_orthogonal_info,
        "outputs": outputs,
    }
    return result


def _sample_evidence(evidence: np.ndarray, points: np.ndarray) -> np.ndarray:
    height, width = evidence.shape
    x = np.clip(np.rint(points[:, 0]).astype(np.int32), 0, width - 1)
    y = np.clip(np.rint(points[:, 1]).astype(np.int32), 0, height - 1)
    return evidence[y, x]


def _line_from_points(start: np.ndarray, end: np.ndarray) -> tuple[float, float, float]:
    direction = end - start
    norm = float(np.linalg.norm(direction))
    if norm < 1e-6:
        return 0.0, 0.0, 0.0
    a = direction[1] / norm
    b = -direction[0] / norm
    c = -(a * start[0] + b * start[1])
    return float(a), float(b), float(c)


def _intersect_lines(
    first: tuple[float, float, float],
    second: tuple[float, float, float],
    fallback: np.ndarray,
) -> np.ndarray:
    a1, b1, c1 = first
    a2, b2, c2 = second
    det = a1 * b2 - a2 * b1
    if abs(det) < 1e-6:
        return fallback.astype(np.float32)
    x = (b1 * c2 - b2 * c1) / det
    y = (c1 * a2 - c2 * a1) / det
    return np.asarray([x, y], dtype=np.float32)


def _snap_polygon_to_evidence(
    polygon: np.ndarray,
    density: np.ndarray,
    search_radius: int,
    samples_per_edge: int,
) -> np.ndarray:
    if len(polygon) < 3 or search_radius <= 0:
        return polygon
    evidence = cv2.GaussianBlur(_normalize(density), (0, 0), sigmaX=2.0)
    shifted_lines = []
    shifted_endpoints = []
    offsets = np.arange(-search_radius, search_radius + 1, dtype=np.float32)
    for start, end in zip(polygon, np.roll(polygon, -1, axis=0)):
        direction = end - start
        length = float(np.linalg.norm(direction))
        if length < 1e-6:
            shifted_lines.append(_line_from_points(start, end))
            shifted_endpoints.append((start, end))
            continue
        normal = np.asarray([-direction[1], direction[0]], dtype=np.float32) / length
        t = np.linspace(0.0, 1.0, max(8, int(samples_per_edge)), dtype=np.float32)
        base_samples = start[None, :] * (1.0 - t[:, None]) + end[None, :] * t[:, None]
        best_offset = 0.0
        best_score = -1.0
        for offset in offsets:
            samples = base_samples + normal[None, :] * offset
            score = float(np.mean(_sample_evidence(evidence, samples)))
            # Mildly prefer small movements when evidence is similar.
            score -= 0.002 * abs(float(offset))
            if score > best_score:
                best_score = score
                best_offset = float(offset)
        snapped_start = start + normal * best_offset
        snapped_end = end + normal * best_offset
        shifted_endpoints.append((snapped_start, snapped_end))
        shifted_lines.append(_line_from_points(snapped_start, snapped_end))

    vertices = []
    for index in range(len(shifted_lines)):
        fallback = 0.5 * (
            shifted_endpoints[index - 1][1] + shifted_endpoints[index][0]
        )
        vertices.append(_intersect_lines(shifted_lines[index - 1], shifted_lines[index], fallback))
    snapped = np.asarray(vertices, dtype=np.float32)
    height, width = density.shape
    snapped[:, 0] = np.clip(snapped[:, 0], 0, width - 1)
    snapped[:, 1] = np.clip(snapped[:, 1], 0, height - 1)
    return snapped


def _save_polygon_overlay(
    output_path: Path,
    polygon: np.ndarray,
    density: np.ndarray | None,
    fill_color: tuple[int, int, int, int],
) -> None:
    if density is not None:
        gray = (255.0 * (1.0 - np.sqrt(np.clip(density, 0.0, 1.0)))).astype(np.uint8)
        image = Image.fromarray(gray).convert("RGB")
    else:
        x_max = int(np.ceil(polygon[:, 0].max())) + 16
        y_max = int(np.ceil(polygon[:, 1].max())) + 16
        image = Image.new("RGB", (max(64, x_max), max(64, y_max)), "white")
    fill = Image.new("RGBA", image.size, (0, 0, 0, 0))
    fill_draw = ImageDraw.Draw(fill)
    fill_draw.polygon([tuple(point) for point in polygon], fill=fill_color)
    image = Image.alpha_composite(image.convert("RGBA"), fill)
    draw = ImageDraw.Draw(image)
    closed = [tuple(point) for point in polygon] + [tuple(polygon[0])]
    draw.line(closed, fill=(25, 85, 230, 255), width=4)
    for index, point in enumerate(polygon):
        x, y = (int(round(value)) for value in point)
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=(220, 35, 35, 255))
        draw.text((x + 4, y + 2), str(index), fill=(20, 20, 20, 255))
    image.convert("RGB").save(output_path)


def _estimate_polygon_manhattan_angle(polygon: np.ndarray) -> float:
    vectors = []
    for start, end in zip(polygon, np.roll(polygon, -1, axis=0)):
        delta = end - start
        length = float(np.linalg.norm(delta))
        if length < 1e-6:
            continue
        angle = np.arctan2(delta[1], delta[0])
        vectors.append(length * np.exp(4j * angle))
    if not vectors:
        return 0.0
    mean_vector = np.sum(vectors)
    if abs(mean_vector) < 1e-8:
        return 0.0
    return float(np.angle(mean_vector) / 4.0)


def _rotate_points(points: np.ndarray, center: np.ndarray, angle_radians: float) -> np.ndarray:
    cos_a = float(np.cos(angle_radians))
    sin_a = float(np.sin(angle_radians))
    rotation = np.asarray([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)
    return (points - center) @ rotation.T + center


def _orthogonalize_polygon_edges(polygon: np.ndarray) -> np.ndarray:
    if len(polygon) < 4:
        return polygon.astype(np.float32)
    center = polygon.mean(axis=0)
    angle = _estimate_polygon_manhattan_angle(polygon)
    aligned = _rotate_points(polygon.astype(np.float32), center, -angle)
    lines = []
    for start, end in zip(aligned, np.roll(aligned, -1, axis=0)):
        delta = end - start
        if abs(float(delta[0])) >= abs(float(delta[1])):
            y = 0.5 * (start[1] + end[1])
            lines.append(("h", float(y), start, end))
        else:
            x = 0.5 * (start[0] + end[0])
            lines.append(("v", float(x), start, end))
    vertices = []
    for previous, current in zip(np.roll(np.asarray(lines, dtype=object), 1, axis=0), lines):
        if previous[0] == current[0]:
            vertices.append(current[2])
            continue
        if previous[0] == "h":
            x = current[1]
            y = previous[1]
        else:
            x = previous[1]
            y = current[1]
        vertices.append([x, y])
    orthogonal = np.asarray(vertices, dtype=np.float32)
    return _rotate_points(orthogonal, center, angle)


def generate_free_space_floorplan(
    predictions_path: str | Path,
    output_dir: str | Path,
    density_path: str | Path | None = None,
    density_info_path: str | Path | None = None,
    confidence_threshold: float = 1.5,
    frame_stride: int = 2,
    pixel_stride: int = 24,
    grid_size: int = 512,
    bounds_percentiles: tuple[float, float] = (1.0, 99.0),
    bounds_margin: float = 0.05,
    ray_shrink: float = 0.92,
    free_close_kernel: int = 31,
    free_open_kernel: int = 9,
    minimum_free_component_area: int = 2000,
    boundary_kernel: int = 7,
    approximation_epsilon: float = 0.01,
    snap_boundary: bool = False,
    snap_search_radius: int = 18,
    snap_samples_per_edge: int = 96,
    orthogonalize_boundary: bool = False,
) -> dict[str, Any]:
    """Carve visible free-space from camera-to-depth rays and extract its boundary."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    density = np.load(density_path).astype(np.float32) if density_path else None
    density_info = None
    if density_info_path:
        with Path(density_info_path).open(encoding="utf-8") as file:
            density_info = json.load(file)

    with np.load(predictions_path) as data:
        depth = data["depth"]
        confidence = data["depth_conf"] if "depth_conf" in data else None
        extrinsic = data["extrinsic"]
        intrinsic = data["intrinsic"]

    if depth.ndim == 4 and depth.shape[-1] == 1:
        depth_2d = depth[..., 0]
    else:
        depth_2d = depth
    camera_to_world = _camera_to_world_from_extrinsic(extrinsic)
    centers = camera_to_world[:, :3, 3].astype(np.float32)
    axis_indices = (
        ["xyz".index(axis) for axis in density_info["horizontal_axes"]]
        if density_info and density_info.get("horizontal_axes")
        else [0, 2]
    )

    sampled_points = []
    for frame_idx in range(0, len(depth_2d), max(1, int(frame_stride))):
        points, sampled_depth = _unproject_frame_points(
            depth_2d[frame_idx],
            intrinsic[frame_idx],
            extrinsic[frame_idx],
            max(1, int(pixel_stride)),
        )
        sampled_conf = (
            confidence[frame_idx, ::pixel_stride, ::pixel_stride].reshape(-1)
            if confidence is not None
            else np.ones(len(points), dtype=np.float32)
        )
        valid = np.isfinite(points).all(axis=1)
        valid &= np.isfinite(sampled_depth)
        valid &= sampled_depth > 0
        valid &= np.isfinite(sampled_conf)
        valid &= sampled_conf >= float(confidence_threshold)
        sampled_points.append(points[valid][:, axis_indices])
    all_horizontal = np.concatenate(sampled_points, axis=0)
    lower, upper, horizontal_axes = _choose_bounds(
        all_horizontal,
        density_info,
        bounds_percentiles,
        bounds_margin,
    )

    free_mask = np.zeros((grid_size, grid_size), dtype=np.uint8)
    hit_counts = np.zeros((grid_size, grid_size), dtype=np.uint32)
    ray_count = 0
    hit_count = 0
    for frame_idx in range(0, len(depth_2d), max(1, int(frame_stride))):
        points, sampled_depth = _unproject_frame_points(
            depth_2d[frame_idx],
            intrinsic[frame_idx],
            extrinsic[frame_idx],
            max(1, int(pixel_stride)),
        )
        sampled_conf = (
            confidence[frame_idx, ::pixel_stride, ::pixel_stride].reshape(-1)
            if confidence is not None
            else np.ones(len(points), dtype=np.float32)
        )
        valid = np.isfinite(points).all(axis=1)
        valid &= np.isfinite(sampled_depth)
        valid &= sampled_depth > 0
        valid &= np.isfinite(sampled_conf)
        valid &= sampled_conf >= float(confidence_threshold)
        if not valid.any():
            continue
        endpoints = points[valid]
        center = centers[frame_idx]
        free_endpoints = center + float(ray_shrink) * (endpoints - center)
        start_pixels = _world_to_pixel(
            np.repeat(center[None, axis_indices], len(endpoints), axis=0),
            lower,
            upper,
            grid_size,
        )
        free_pixels = _world_to_pixel(free_endpoints[:, axis_indices], lower, upper, grid_size)
        hit_pixels = np.rint(_world_to_pixel(endpoints[:, axis_indices], lower, upper, grid_size)).astype(np.int32)
        valid_hits = np.logical_and(hit_pixels >= 0, hit_pixels < grid_size).all(axis=1)
        if valid_hits.any():
            hp = hit_pixels[valid_hits]
            np.add.at(hit_counts, (hp[:, 1], hp[:, 0]), 1)
            hit_count += int(len(hp))
        for start, end in zip(start_pixels, free_pixels):
            x1, y1 = np.rint(start).astype(int)
            x2, y2 = np.rint(end).astype(int)
            if not (0 <= x1 < grid_size and 0 <= y1 < grid_size):
                continue
            if not (0 <= x2 < grid_size and 0 <= y2 < grid_size):
                continue
            cv2.line(free_mask, (x1, y1), (x2, y2), 255, thickness=1)
            ray_count += 1

    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (free_close_kernel, free_close_kernel))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (free_open_kernel, free_open_kernel))
    free_closed = cv2.morphologyEx(free_mask, cv2.MORPH_CLOSE, close_kernel)
    free_closed = cv2.morphologyEx(free_closed, cv2.MORPH_OPEN, open_kernel)
    free_closed = _filter_small_components(free_closed, int(minimum_free_component_area))
    free_largest = _largest_component(free_closed)
    boundary_kernel_mat = cv2.getStructuringElement(cv2.MORPH_RECT, (boundary_kernel, boundary_kernel))
    free_boundary = cv2.morphologyEx(free_largest, cv2.MORPH_GRADIENT, boundary_kernel_mat)
    hit_density = _normalize(hit_counts)
    boundary_confidence = (free_boundary > 0).astype(np.float32)
    if density is not None and density.shape == boundary_confidence.shape:
        dilated_density = cv2.dilate(density.astype(np.float32), np.ones((9, 9), np.uint8))
        boundary_confidence *= 0.5 + 0.5 * _normalize(dilated_density)

    np.save(output_dir / "free_space_raw.npy", free_mask > 0)
    np.save(output_dir / "free_space.npy", free_largest > 0)
    np.save(output_dir / "occupied_hit_counts.npy", hit_counts)
    np.save(output_dir / "occupied_hit_density.npy", hit_density)
    np.save(output_dir / "free_space_boundary.npy", free_boundary > 0)
    np.save(output_dir / "boundary_confidence.npy", boundary_confidence.astype(np.float32))
    _save_gray(output_dir / "free_space_raw.png", (free_mask > 0).astype(np.float32), invert=False)
    _save_gray(output_dir / "free_space.png", (free_largest > 0).astype(np.float32), invert=False)
    _save_gray(output_dir / "occupied_hit_density.png", hit_density)
    _save_gray(output_dir / "free_space_boundary.png", (free_boundary > 0).astype(np.float32), invert=False)
    _save_gray(output_dir / "boundary_confidence.png", boundary_confidence)
    polygon_info = _extract_free_space_polygon(
        free_largest,
        density,
        output_dir,
        approximation_epsilon=float(approximation_epsilon),
        snap_boundary=bool(snap_boundary),
        snap_search_radius=int(snap_search_radius),
        snap_samples_per_edge=int(snap_samples_per_edge),
        orthogonalize_boundary=bool(orthogonalize_boundary),
    )

    metadata = {
        "method": "camera_depth_ray_free_space_carving",
        "inputs": {
            "predictions": str(predictions_path),
            "density": str(density_path) if density_path else None,
            "density_info": str(density_info_path) if density_info_path else None,
        },
        "parameters": {
            "confidence_threshold": float(confidence_threshold),
            "frame_stride": int(frame_stride),
            "pixel_stride": int(pixel_stride),
            "grid_size": int(grid_size),
            "ray_shrink": float(ray_shrink),
            "free_close_kernel": int(free_close_kernel),
            "free_open_kernel": int(free_open_kernel),
            "minimum_free_component_area": int(minimum_free_component_area),
            "boundary_kernel": int(boundary_kernel),
            "approximation_epsilon": float(approximation_epsilon),
            "snap_boundary": bool(snap_boundary),
            "snap_search_radius": int(snap_search_radius),
            "snap_samples_per_edge": int(snap_samples_per_edge),
            "orthogonalize_boundary": bool(orthogonalize_boundary),
        },
        "horizontal_axes": horizontal_axes,
        "horizontal_bounds": [lower.astype(float).tolist(), upper.astype(float).tolist()],
        "frame_count": int(len(depth_2d)),
        "ray_count": int(ray_count),
        "hit_count": int(hit_count),
        "free_pixels": int((free_largest > 0).sum()),
        "boundary_pixels": int((free_boundary > 0).sum()),
        "polygon": polygon_info,
        "outputs": {
            "free_space": "free_space.png",
            "occupied_hit_density": "occupied_hit_density.png",
            "free_space_boundary": "free_space_boundary.png",
            "boundary_confidence": "boundary_confidence.png",
            **polygon_info["outputs"],
        },
    }
    with (output_dir / "free_space_info.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)
    return metadata
