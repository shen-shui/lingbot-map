"""Evaluation and metric-scale calibration for floorplan polygons."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw


def polygon_area(points: np.ndarray) -> float:
    x = points[:, 0]
    y = points[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) * 0.5)


def _load_floorplan(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as file:
        floorplan = json.load(file)
    if not floorplan.get("pixel_polygon"):
        raise ValueError(f"{path} does not contain pixel_polygon")
    if not floorplan.get("image_shape"):
        raise ValueError(f"{path} does not contain image_shape")
    return floorplan


def _polygon_mask(points: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(image_shape, dtype=np.uint8)
    cv2.fillPoly(mask, [np.rint(points).astype(np.int32)], 255)
    return mask


def _boundary(mask: np.ndarray) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    return cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel) > 0


def _boundary_metrics(
    prediction_mask: np.ndarray,
    ground_truth_mask: np.ndarray,
    tolerance: float,
) -> dict[str, float]:
    prediction_boundary = _boundary(prediction_mask)
    ground_truth_boundary = _boundary(ground_truth_mask)
    distance_to_ground_truth = cv2.distanceTransform(
        (~ground_truth_boundary).astype(np.uint8),
        cv2.DIST_L2,
        3,
    )
    distance_to_prediction = cv2.distanceTransform(
        (~prediction_boundary).astype(np.uint8),
        cv2.DIST_L2,
        3,
    )
    precision = float(
        (distance_to_ground_truth[prediction_boundary] <= tolerance).mean()
    )
    recall = float(
        (distance_to_prediction[ground_truth_boundary] <= tolerance).mean()
    )
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-8)
    return {
        "boundary_precision": precision,
        "boundary_recall": recall,
        "boundary_f1": float(f1),
    }


def _corner_metrics(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    tolerance: float,
) -> dict[str, float]:
    distances = np.linalg.norm(
        prediction[:, None, :] - ground_truth[None, :, :],
        axis=2,
    )
    prediction_to_ground_truth = distances.min(axis=1)
    ground_truth_to_prediction = distances.min(axis=0)
    return {
        "corner_error_pred_to_gt": float(prediction_to_ground_truth.mean()),
        "corner_error_gt_to_pred": float(ground_truth_to_prediction.mean()),
        "corner_error_symmetric": float(
            0.5 * (prediction_to_ground_truth.mean() + ground_truth_to_prediction.mean())
        ),
        "corner_precision": float((prediction_to_ground_truth <= tolerance).mean()),
        "corner_recall": float((ground_truth_to_prediction <= tolerance).mean()),
    }


def _save_evaluation_overlay(
    output_path: Path,
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    image_shape: tuple[int, int],
) -> None:
    image = Image.new("RGB", (image_shape[1], image_shape[0]), "white")
    draw = ImageDraw.Draw(image)
    draw.line(
        [tuple(point) for point in ground_truth] + [tuple(ground_truth[0])],
        fill=(35, 170, 75),
        width=4,
    )
    draw.line(
        [tuple(point) for point in prediction] + [tuple(prediction[0])],
        fill=(220, 45, 45),
        width=3,
    )
    image.save(output_path)


def evaluate_floorplan(
    prediction_path: str | Path,
    ground_truth_path: str | Path,
    output_dir: str | Path,
    boundary_tolerance: float = 5.0,
    corner_tolerance: float = 10.0,
) -> dict[str, Any]:
    """Evaluate a predicted pixel-space polygon against ground truth."""
    prediction_data = _load_floorplan(prediction_path)
    ground_truth_data = _load_floorplan(ground_truth_path)
    prediction_shape = tuple(prediction_data["image_shape"])
    ground_truth_shape = tuple(ground_truth_data["image_shape"])
    if prediction_shape != ground_truth_shape:
        raise ValueError(
            f"Image shapes do not match: {prediction_shape} versus {ground_truth_shape}"
        )
    prediction = np.asarray(prediction_data["pixel_polygon"], dtype=np.float64)
    ground_truth = np.asarray(ground_truth_data["pixel_polygon"], dtype=np.float64)
    prediction_mask = _polygon_mask(prediction, prediction_shape)
    ground_truth_mask = _polygon_mask(ground_truth, prediction_shape)
    intersection = np.logical_and(prediction_mask > 0, ground_truth_mask > 0).sum()
    union = np.logical_or(prediction_mask > 0, ground_truth_mask > 0).sum()
    prediction_area = polygon_area(prediction)
    ground_truth_area = polygon_area(ground_truth)

    metrics = {
        "region_iou": float(intersection / max(union, 1)),
        **_boundary_metrics(prediction_mask, ground_truth_mask, boundary_tolerance),
        **_corner_metrics(prediction, ground_truth, corner_tolerance),
        "prediction_area_pixels": prediction_area,
        "ground_truth_area_pixels": ground_truth_area,
        "area_relative_error": float(
            abs(prediction_area - ground_truth_area) / max(ground_truth_area, 1e-8)
        ),
        "prediction_vertex_count": int(len(prediction)),
        "ground_truth_vertex_count": int(len(ground_truth)),
        "vertex_count_absolute_error": int(abs(len(prediction) - len(ground_truth))),
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _save_evaluation_overlay(
        output_dir / "floorplan_evaluation_overlay.png",
        prediction,
        ground_truth,
        prediction_shape,
    )
    result = {
        "method": "pixel_polygon_floorplan_evaluation",
        "inputs": {
            "prediction": str(prediction_path),
            "ground_truth": str(ground_truth_path),
        },
        "parameters": {
            "boundary_tolerance_pixels": float(boundary_tolerance),
            "corner_tolerance_pixels": float(corner_tolerance),
        },
        "metrics": metrics,
        "outputs": {"overlay": "floorplan_evaluation_overlay.png"},
    }
    with (output_dir / "floorplan_evaluation.json").open("w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
    return result


def calibrate_floorplan_scale(
    floorplan_path: str | Path,
    output_path: str | Path,
    references: list[tuple[int, int, float]],
) -> dict[str, Any]:
    """Calibrate reconstruction coordinates using known vertex-pair lengths."""
    with Path(floorplan_path).open(encoding="utf-8") as file:
        floorplan = json.load(file)
    polygon = floorplan.get("world_polygon")
    if not polygon:
        raise ValueError("Floorplan does not contain world_polygon")
    if not references:
        raise ValueError("At least one scale reference is required")
    polygon_array = np.asarray(polygon, dtype=np.float64)
    scales = []
    reference_results = []
    for first, second, metric_length in references:
        if first == second or min(first, second) < 0 or max(first, second) >= len(polygon):
            raise ValueError(f"Invalid reference vertex indices: {first}, {second}")
        if metric_length <= 0:
            raise ValueError("Reference metric lengths must be positive")
        reconstruction_length = float(np.linalg.norm(polygon_array[first] - polygon_array[second]))
        if reconstruction_length <= 0:
            raise ValueError(f"Reference vertices {first} and {second} have zero distance")
        scale = float(metric_length / reconstruction_length)
        scales.append(scale)
        reference_results.append(
            {
                "vertices": [int(first), int(second)],
                "known_length_meters": float(metric_length),
                "reconstruction_length": reconstruction_length,
                "meters_per_reconstruction_unit": scale,
            }
        )
    metric_scale = float(np.median(scales))
    metric_polygon = (polygon_array * metric_scale).tolist()
    calibrated = dict(floorplan)
    calibrated.update(
        {
            "metric_scale_available": True,
            "meters_per_reconstruction_unit": metric_scale,
            "metric_polygon_meters": metric_polygon,
            "metric_area_square_meters": polygon_area(np.asarray(metric_polygon)),
            "scale_calibration": {
                "method": "median_known_vertex_pair_length",
                "references": reference_results,
                "scale_standard_deviation": float(np.std(scales)),
            },
        }
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(calibrated, file, ensure_ascii=False, indent=2)
    return calibrated
