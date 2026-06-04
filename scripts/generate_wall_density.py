#!/usr/bin/env python3
"""Generate a fixed-height wall-density baseline from LingBot-Map predictions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from lingbot_map.utils.wall_density import (  # noqa: E402
    generate_fixed_height_densities,
    load_prediction_points,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--confidence-threshold", type=float, default=1.5)
    parser.add_argument("--point-stride", type=int, default=4)
    parser.add_argument("--vertical-axis", choices=["auto", "x", "y", "z"], default="auto")
    parser.add_argument("--grid-size", type=int, default=512)
    parser.add_argument("--bounds-low-percentile", type=float, default=1.0)
    parser.add_argument("--bounds-high-percentile", type=float, default=99.0)
    parser.add_argument("--bounds-margin", type=float, default=0.05)
    parser.add_argument("--support-radius", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    points, point_metadata = load_prediction_points(
        args.predictions,
        confidence_threshold=args.confidence_threshold,
        point_stride=args.point_stride,
    )
    print(
        f"Loaded {len(points):,} points from {point_metadata['point_source']} "
        f"(confidence >= {args.confidence_threshold}, stride={args.point_stride})"
    )
    metadata = generate_fixed_height_densities(
        points,
        args.output_dir,
        vertical_axis=args.vertical_axis,
        grid_size=args.grid_size,
        bounds_percentiles=(
            args.bounds_low_percentile,
            args.bounds_high_percentile,
        ),
        bounds_margin=args.bounds_margin,
        support_radius=args.support_radius,
        input_metadata={
            "predictions": str(args.predictions),
            **point_metadata,
        },
    )
    print(
        f"Saved density baseline to {args.output_dir}; "
        f"vertical axis={metadata['vertical_axis']}, "
        f"wall density={metadata['outputs']['wall_fused']}"
    )


if __name__ == "__main__":
    main()
