#!/usr/bin/env python3
"""Prepare short RGB pseudo-video sequences from Structured3D perspective renders.

Structured3D already ships rendered perspective RGB frames and camera poses. This
script turns those sparse renders into ordered short sequences that can be used
as RGB-video-like input for LingBot-Map. It does not perform new mesh rendering.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class FrameRecord:
    image: Path
    pose: Path
    scene_id: str
    camera_id: str
    view_id: str
    position: np.ndarray
    camera_pose_values: list[float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structured3d-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-sequences", type=int, default=10)
    parser.add_argument("--frames-per-sequence", type=int, default=12)
    parser.add_argument("--image-size", type=int, default=518)
    parser.add_argument("--stride-scenes", type=int, default=1)
    parser.add_argument("--min-frames-per-scene", type=int, default=8)
    parser.add_argument("--make-video", action="store_true")
    parser.add_argument("--fps", type=float, default=6.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_pose(path: Path) -> tuple[np.ndarray, list[float]]:
    values = [float(x) for x in path.read_text().strip().split()]
    if len(values) < 3:
        raise ValueError(f"Invalid camera pose file: {path}")
    # Structured3D camera positions are in millimeters.
    position = np.asarray(values[:3], dtype=np.float64) / 1000.0
    return position, values


def scene_id_from_path(scene_dir: Path) -> str:
    return scene_dir.name


def collect_scene_frames(scene_dir: Path) -> list[FrameRecord]:
    records: list[FrameRecord] = []
    scene_id = scene_id_from_path(scene_dir)
    for image in sorted(scene_dir.glob("2D_rendering/*/perspective/full/*/rgb_rawlight.png")):
        pose = image.with_name("camera_pose.txt")
        if not pose.exists():
            continue
        try:
            position, pose_values = read_pose(pose)
        except Exception:
            continue
        parts = image.parts
        try:
            idx = parts.index("2D_rendering")
            camera_id = parts[idx + 1]
            view_id = parts[idx + 4]
        except Exception:
            camera_id = image.parent.parent.parent.parent.name
            view_id = image.parent.name
        records.append(
            FrameRecord(
                image=image,
                pose=pose,
                scene_id=scene_id,
                camera_id=camera_id,
                view_id=view_id,
                position=position,
                camera_pose_values=pose_values,
            )
        )
    return records


def nearest_neighbor_order(records: list[FrameRecord], limit: int) -> list[FrameRecord]:
    if len(records) <= 1:
        return records[:limit]
    positions = np.stack([r.position for r in records], axis=0)
    centroid = positions.mean(axis=0)
    start = int(np.argmin(np.linalg.norm(positions - centroid[None, :], axis=1)))
    remaining = set(range(len(records)))
    order = [start]
    remaining.remove(start)
    while remaining and len(order) < limit:
        last = order[-1]
        next_idx = min(remaining, key=lambda i: float(np.linalg.norm(positions[i] - positions[last])))
        order.append(next_idx)
        remaining.remove(next_idx)
    return [records[i] for i in order]


def resize_frame(src: Path, dst: Path, image_size: int) -> tuple[int, int]:
    image = Image.open(src).convert("RGB")
    width, height = image.size
    if image_size > 0:
        scale = image_size / max(width, height)
        new_size = (int(round(width * scale)), int(round(height * scale)))
        image = image.resize(new_size, Image.BILINEAR)
    image.save(dst)
    return image.size


def write_video(frame_paths: list[Path], output_path: Path, fps: float) -> None:
    first = cv2.imread(str(frame_paths[0]), cv2.IMREAD_COLOR)
    if first is None:
        raise ValueError(f"Could not read first frame: {frame_paths[0]}")
    height, width = first.shape[:2]
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    try:
        for frame_path in frame_paths:
            frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            if frame is None:
                continue
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            writer.write(frame)
    finally:
        writer.release()


def write_sequence(sequence_dir: Path, records: list[FrameRecord], args: argparse.Namespace) -> dict[str, Any]:
    if sequence_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{sequence_dir} exists; pass --overwrite to replace it")
        shutil.rmtree(sequence_dir)
    frames_dir = sequence_dir / "frames"
    frames_dir.mkdir(parents=True)

    rows = []
    frame_paths = []
    output_size = None
    for idx, record in enumerate(records):
        frame_name = f"{idx:06d}.png"
        dst = frames_dir / frame_name
        output_size = resize_frame(record.image, dst, args.image_size)
        frame_paths.append(dst)
        rows.append(
            {
                "frame_index": idx,
                "frame": f"frames/{frame_name}",
                "source_image": str(record.image),
                "source_pose": str(record.pose),
                "scene_id": record.scene_id,
                "camera_id": record.camera_id,
                "view_id": record.view_id,
                "x_m": float(record.position[0]),
                "y_m": float(record.position[1]),
                "z_m": float(record.position[2]),
                "camera_pose_values": " ".join(f"{v:.9g}" for v in record.camera_pose_values),
            }
        )

    with (sequence_dir / "poses.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    path_length = 0.0
    for a, b in zip(records, records[1:]):
        path_length += float(np.linalg.norm(a.position - b.position))

    metadata = {
        "scene_id": records[0].scene_id,
        "num_frames": len(records),
        "frames_dir": "frames",
        "poses_csv": "poses.csv",
        "image_size": list(output_size) if output_size else None,
        "path_length_m": path_length,
        "note": "Pseudo-video sequence ordered from existing Structured3D perspective renders.",
    }
    if args.make_video:
        write_video(frame_paths, sequence_dir / "video.mp4", args.fps)
        metadata["video"] = "video.mp4"
        metadata["fps"] = args.fps

    with (sequence_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    return metadata


def iter_scene_dirs(root: Path) -> list[Path]:
    return sorted(p for p in root.glob("scene_*") if p.is_dir())


def main() -> None:
    args = parse_args()
    if not args.structured3d_root.exists():
        raise FileNotFoundError(args.structured3d_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    selected = []
    scenes = iter_scene_dirs(args.structured3d_root)[:: max(1, args.stride_scenes)]
    for scene_dir in scenes:
        records = collect_scene_frames(scene_dir)
        if len(records) < args.min_frames_per_scene:
            continue
        ordered = nearest_neighbor_order(records, args.frames_per_sequence)
        if len(ordered) < args.min_frames_per_scene:
            continue
        selected.append(ordered)
        if len(selected) >= args.num_sequences:
            break

    if not selected:
        raise RuntimeError("No Structured3D scenes with enough frames were found")

    manifest = []
    for idx, records in enumerate(selected):
        seq_name = f"seq_{idx:03d}_{records[0].scene_id}"
        seq_dir = args.output_dir / seq_name
        metadata = write_sequence(seq_dir, records, args)
        metadata["sequence_id"] = seq_name
        metadata["sequence_dir"] = str(seq_dir)
        manifest.append(metadata)
        print(f"{seq_name}: frames={metadata['num_frames']} path_length_m={metadata['path_length_m']:.2f}")

    with (args.output_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(manifest)} sequences to {args.output_dir}")


if __name__ == "__main__":
    main()
