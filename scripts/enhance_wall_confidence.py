#!/usr/bin/env python3
"""Enhance a wall-density map into a furniture-resistant wall confidence map."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from lingbot_map.utils.wall_confidence import enhance_wall_confidence  # noqa: E402


def _parse_line_lengths(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(",") if item.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--density", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--support", type=Path)
    parser.add_argument("--density-info", type=Path)
    parser.add_argument("--trajectory", type=Path)
    parser.add_argument("--density-threshold", type=float, default=0.12)
    parser.add_argument("--support-weight", type=float, default=0.30)
    parser.add_argument("--line-weight", type=float, default=0.55)
    parser.add_argument("--thin-weight", type=float, default=0.05)
    parser.add_argument("--trajectory-weight", type=float, default=0.10)
    parser.add_argument("--line-lengths", type=_parse_line_lengths, default=(31, 63, 95))
    parser.add_argument("--line-thickness", type=int, default=3)
    parser.add_argument("--angle-step", type=float, default=15.0)
    parser.add_argument("--preferred-thickness", type=float, default=5.0)
    parser.add_argument("--trajectory-near-radius", type=int, default=8)
    parser.add_argument("--trajectory-far-radius", type=int, default=80)
    parser.add_argument("--close-kernel", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = enhance_wall_confidence(
        args.density,
        args.output_dir,
        support_path=args.support,
        density_info_path=args.density_info,
        trajectory_path=args.trajectory,
        density_threshold=args.density_threshold,
        support_weight=args.support_weight,
        line_weight=args.line_weight,
        thin_weight=args.thin_weight,
        trajectory_weight=args.trajectory_weight,
        line_lengths=args.line_lengths,
        line_thickness=args.line_thickness,
        angle_step=args.angle_step,
        preferred_thickness=args.preferred_thickness,
        trajectory_near_radius=args.trajectory_near_radius,
        trajectory_far_radius=args.trajectory_far_radius,
        close_kernel=args.close_kernel,
    )
    print(
        f"Saved wall confidence to {args.output_dir}; "
        f"pixels>=0.18={metadata['confidence_pixels_above_018']}"
    )


if __name__ == "__main__":
    main()
