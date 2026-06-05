#!/usr/bin/env python3
"""Regularize wall confidence into a Manhattan-aligned wall-line mask."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from lingbot_map.utils.wall_regularization import regularize_wall_lines  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--density", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--support", type=Path)
    parser.add_argument("--density-threshold", type=float, default=0.20)
    parser.add_argument("--minimum-support", type=float, default=1.0 / 3.0)
    parser.add_argument("--close-kernel", type=int, default=9)
    parser.add_argument("--open-kernel", type=int, default=3)
    parser.add_argument("--minimum-component-area", type=int, default=80)
    parser.add_argument("--hough-threshold", type=int, default=28)
    parser.add_argument("--minimum-line-length", type=int, default=28)
    parser.add_argument("--maximum-line-gap", type=int, default=20)
    parser.add_argument("--angle-tolerance", type=float, default=12.0)
    parser.add_argument("--coordinate-tolerance", type=float, default=12.0)
    parser.add_argument("--merge-gap", type=float, default=28.0)
    parser.add_argument("--minimum-merged-length", type=float, default=38.0)
    parser.add_argument("--extend-pixels", type=int, default=24)
    parser.add_argument("--line-thickness", type=int, default=5)
    parser.add_argument("--final-close-kernel", type=int, default=31)
    parser.add_argument("--complete-outer-frame", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = regularize_wall_lines(
        args.density,
        args.output_dir,
        support_path=args.support,
        density_threshold=args.density_threshold,
        minimum_support=args.minimum_support,
        close_kernel=args.close_kernel,
        open_kernel=args.open_kernel,
        minimum_component_area=args.minimum_component_area,
        hough_threshold=args.hough_threshold,
        minimum_line_length=args.minimum_line_length,
        maximum_line_gap=args.maximum_line_gap,
        angle_tolerance=args.angle_tolerance,
        coordinate_tolerance=args.coordinate_tolerance,
        merge_gap=args.merge_gap,
        minimum_merged_length=args.minimum_merged_length,
        extend_pixels=args.extend_pixels,
        line_thickness=args.line_thickness,
        final_close_kernel=args.final_close_kernel,
        complete_outer_frame=args.complete_outer_frame,
    )
    print(
        f"Saved regularized wall mask to {args.output_dir}; "
        f"angle={metadata['manhattan_angle_degrees']:.2f}, "
        f"raw={metadata['raw_segment_count']}, merged={metadata['merged_segment_count']}"
    )


if __name__ == "__main__":
    main()
