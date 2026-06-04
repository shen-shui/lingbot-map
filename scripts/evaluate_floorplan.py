#!/usr/bin/env python3
"""Evaluate a predicted floorplan polygon against pixel-space ground truth."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from lingbot_map.utils.floorplan_evaluation import evaluate_floorplan  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--boundary-tolerance", type=float, default=5.0)
    parser.add_argument("--corner-tolerance", type=float, default=10.0)
    args = parser.parse_args()
    result = evaluate_floorplan(
        args.prediction,
        args.ground_truth,
        args.output_dir,
        boundary_tolerance=args.boundary_tolerance,
        corner_tolerance=args.corner_tolerance,
    )
    metrics = result["metrics"]
    print(
        f"Saved evaluation to {args.output_dir}; IoU={metrics['region_iou']:.3f}, "
        f"boundary F1={metrics['boundary_f1']:.3f}, "
        f"corner error={metrics['corner_error_symmetric']:.2f}px"
    )


if __name__ == "__main__":
    main()
