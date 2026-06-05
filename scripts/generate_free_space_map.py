#!/usr/bin/env python3
"""Generate a free-space floorplan candidate from LingBot-Map predictions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from lingbot_map.utils.free_space_floorplan import generate_free_space_floorplan  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--density", type=Path)
    parser.add_argument("--density-info", type=Path)
    parser.add_argument("--confidence-threshold", type=float, default=1.5)
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--pixel-stride", type=int, default=24)
    parser.add_argument("--grid-size", type=int, default=512)
    parser.add_argument("--bounds-low-percentile", type=float, default=1.0)
    parser.add_argument("--bounds-high-percentile", type=float, default=99.0)
    parser.add_argument("--bounds-margin", type=float, default=0.05)
    parser.add_argument("--ray-shrink", type=float, default=0.92)
    parser.add_argument("--free-close-kernel", type=int, default=31)
    parser.add_argument("--free-open-kernel", type=int, default=9)
    parser.add_argument("--minimum-free-component-area", type=int, default=2000)
    parser.add_argument("--boundary-kernel", type=int, default=7)
    parser.add_argument("--approximation-epsilon", type=float, default=0.01)
    parser.add_argument("--snap-boundary", action="store_true")
    parser.add_argument("--snap-search-radius", type=int, default=18)
    parser.add_argument("--snap-samples-per-edge", type=int, default=96)
    parser.add_argument("--orthogonalize-boundary", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = generate_free_space_floorplan(
        args.predictions,
        args.output_dir,
        density_path=args.density,
        density_info_path=args.density_info,
        confidence_threshold=args.confidence_threshold,
        frame_stride=args.frame_stride,
        pixel_stride=args.pixel_stride,
        grid_size=args.grid_size,
        bounds_percentiles=(args.bounds_low_percentile, args.bounds_high_percentile),
        bounds_margin=args.bounds_margin,
        ray_shrink=args.ray_shrink,
        free_close_kernel=args.free_close_kernel,
        free_open_kernel=args.free_open_kernel,
        minimum_free_component_area=args.minimum_free_component_area,
        boundary_kernel=args.boundary_kernel,
        approximation_epsilon=args.approximation_epsilon,
        snap_boundary=args.snap_boundary,
        snap_search_radius=args.snap_search_radius,
        snap_samples_per_edge=args.snap_samples_per_edge,
        orthogonalize_boundary=args.orthogonalize_boundary,
    )
    snapped = metadata["polygon"].get("snapped")
    snapped_text = f", snapped_vertices={snapped['vertex_count']}" if snapped else ""
    orthogonal = metadata["polygon"].get("orthogonal")
    orthogonal_text = f", orthogonal_vertices={orthogonal['vertex_count']}" if orthogonal else ""
    print(
        f"Saved free-space map to {args.output_dir}; "
        f"rays={metadata['ray_count']}, free_pixels={metadata['free_pixels']}, "
        f"vertices={metadata['polygon']['vertex_count']}{snapped_text}{orthogonal_text}"
    )


if __name__ == "__main__":
    main()
