#!/usr/bin/env python3
"""Calibrate a floorplan to metric scale using known vertex-pair lengths."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from lingbot_map.utils.floorplan_evaluation import calibrate_floorplan_scale  # noqa: E402


def _parse_reference(value: str) -> tuple[int, int, float]:
    try:
        first, second, length = value.split(":")
        return int(first), int(second), float(length)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "reference must use FIRST_VERTEX:SECOND_VERTEX:LENGTH_METERS"
        ) from error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--floorplan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--reference",
        type=_parse_reference,
        action="append",
        required=True,
        help="Known length as FIRST_VERTEX:SECOND_VERTEX:LENGTH_METERS; repeatable.",
    )
    args = parser.parse_args()
    result = calibrate_floorplan_scale(args.floorplan, args.output, args.reference)
    print(
        f"Saved metric floorplan to {args.output}; "
        f"scale={result['meters_per_reconstruction_unit']:.6f} m/unit, "
        f"area={result['metric_area_square_meters']:.3f} m^2"
    )


if __name__ == "__main__":
    main()
