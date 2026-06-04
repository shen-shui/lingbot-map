#!/usr/bin/env python3
"""Generate a closed orthogonal floorplan polygon from wall candidates."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from lingbot_map.utils.floorplan_polygon import generate_floorplan_polygon  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wall-mask", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--density-info", type=Path)
    parser.add_argument("--density", type=Path)
    parser.add_argument("--trajectory", type=Path)
    parser.add_argument("--connection-kernel", type=int, default=15)
    parser.add_argument("--approximation-epsilon", type=float, default=0.005)
    parser.add_argument("--auto-manhattan", action="store_true")
    parser.add_argument("--minimum-area-ratio", type=float, default=0.02)
    parser.add_argument("--minimum-trajectory-coverage", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = generate_floorplan_polygon(
        args.wall_mask,
        args.output_dir,
        density_info_path=args.density_info,
        density_path=args.density,
        trajectory_path=args.trajectory,
        connection_kernel=args.connection_kernel,
        approximation_epsilon=args.approximation_epsilon,
        auto_manhattan=args.auto_manhattan,
        minimum_area_ratio=args.minimum_area_ratio,
        minimum_trajectory_coverage=args.minimum_trajectory_coverage,
    )
    area = metadata["world_area"]
    area_text = f", reconstruction area={area:.3f}" if area is not None else ""
    print(
        f"Saved floorplan polygon to {args.output_dir}; "
        f"vertices={metadata['vertex_count']}, IoU={metadata['interior_iou']:.3f}"
        f"{area_text}"
    )


if __name__ == "__main__":
    main()
