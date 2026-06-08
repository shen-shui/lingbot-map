#!/usr/bin/env python3
"""Extract RGB frames from a ScanNet .sens file for LingBot-Map inference."""

from __future__ import annotations

import argparse
import io
import struct
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sens", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=120)
    parser.add_argument("--frame-stride", type=int, default=None)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--save-poses", action="store_true")
    return parser.parse_args()


def read_exact(handle, size: int) -> bytes:
    data = handle.read(size)
    if len(data) != size:
        raise EOFError(f"Expected {size} bytes, got {len(data)}")
    return data


def read_uint32(handle) -> int:
    return struct.unpack("I", read_exact(handle, 4))[0]


def read_uint64(handle) -> int:
    return struct.unpack("Q", read_exact(handle, 8))[0]


def read_int32(handle) -> int:
    return struct.unpack("i", read_exact(handle, 4))[0]


def read_float32(handle) -> float:
    return struct.unpack("f", read_exact(handle, 4))[0]


def read_matrix4(handle) -> np.ndarray:
    values = struct.unpack("f" * 16, read_exact(handle, 16 * 4))
    return np.asarray(values, dtype=np.float32).reshape(4, 4)


def skip_matrix4(handle) -> None:
    read_exact(handle, 16 * 4)


def read_header(handle) -> dict:
    version = read_uint32(handle)
    name_len = read_uint64(handle)
    sensor_name = read_exact(handle, name_len).decode("utf-8", errors="replace")

    # Intrinsics/extrinsics for color and depth streams.
    for _ in range(4):
        skip_matrix4(handle)

    color_compression_type = read_int32(handle)
    depth_compression_type = read_int32(handle)
    color_width = read_uint32(handle)
    color_height = read_uint32(handle)
    depth_width = read_uint32(handle)
    depth_height = read_uint32(handle)
    depth_shift = read_float32(handle)
    num_frames = read_uint64(handle)

    return {
        "version": version,
        "sensor_name": sensor_name,
        "color_compression_type": color_compression_type,
        "depth_compression_type": depth_compression_type,
        "color_size": [int(color_width), int(color_height)],
        "depth_size": [int(depth_width), int(depth_height)],
        "depth_shift": float(depth_shift),
        "num_frames": int(num_frames),
    }


def read_frame(handle) -> tuple[np.ndarray, int, int, bytes, bytes]:
    camera_to_world = read_matrix4(handle)
    timestamp_color = read_uint64(handle)
    timestamp_depth = read_uint64(handle)
    color_size_bytes = read_uint64(handle)
    depth_size_bytes = read_uint64(handle)
    color_data = read_exact(handle, color_size_bytes)
    depth_data = read_exact(handle, depth_size_bytes)
    return camera_to_world, timestamp_color, timestamp_depth, color_data, depth_data


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pose_dir = args.output_dir / "poses"
    if args.save_poses:
        pose_dir.mkdir(parents=True, exist_ok=True)

    with args.sens.open("rb") as handle:
        header = read_header(handle)
        num_frames = header["num_frames"]
        stride = args.frame_stride
        if stride is None:
            stride = max(num_frames // max(args.max_frames, 1), 1)

        saved = 0
        for frame_idx in range(num_frames):
            camera_to_world, timestamp_color, _, color_data, _ = read_frame(handle)
            if frame_idx % stride != 0:
                continue
            if saved >= args.max_frames:
                continue

            image = Image.open(io.BytesIO(color_data)).convert("RGB")
            image.save(args.output_dir / f"{saved:06d}.jpg", quality=args.jpeg_quality)
            if args.save_poses:
                np.savetxt(pose_dir / f"{saved:06d}.txt", camera_to_world)
            saved += 1

        metadata = {
            **header,
            "source_sens": str(args.sens),
            "frame_stride": int(stride),
            "saved_frames": int(saved),
        }
        import json

        with (args.output_dir / "sens_extract_info.json").open("w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"Saved {saved} RGB frames to {args.output_dir}")


if __name__ == "__main__":
    main()
