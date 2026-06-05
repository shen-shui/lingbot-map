#!/usr/bin/env python3
"""Inspect FloorNet TFRecords schema and export a few decoded examples."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _feature_kind(feature: Any) -> tuple[str, int]:
    if feature.bytes_list.value:
        return "bytes", len(feature.bytes_list.value)
    if feature.float_list.value:
        return "float", len(feature.float_list.value)
    if feature.int64_list.value:
        return "int64", len(feature.int64_list.value)
    return "empty", 0


def _short_value(feature: Any) -> Any:
    if feature.bytes_list.value:
        value = feature.bytes_list.value[0]
        return {
            "byte_length": len(value),
            "prefix_hex": value[:16].hex(),
            "ascii_prefix": value[:32].decode("utf-8", errors="replace"),
        }
    if feature.float_list.value:
        values = list(feature.float_list.value[:8])
        return [float(value) for value in values]
    if feature.int64_list.value:
        values = list(feature.int64_list.value[:16])
        return [int(value) for value in values]
    return None


def inspect_tfrecords(
    tfrecords_path: str | Path,
    output_dir: str | Path,
    max_examples: int = 3,
) -> dict[str, Any]:
    import numpy as np
    import tensorflow as tf
    from PIL import Image

    tfrecords_path = Path(tfrecords_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = tf.data.TFRecordDataset(str(tfrecords_path))
    schema: dict[str, Any] = {}
    examples = []
    for index, raw in enumerate(dataset.take(max_examples)):
        example = tf.train.Example()
        example.ParseFromString(bytes(raw.numpy()))
        record_summary = {"index": index, "features": {}}
        for name, feature in sorted(example.features.feature.items()):
            kind, length = _feature_kind(feature)
            entry = {
                "kind": kind,
                "length": int(length),
                "preview": _short_value(feature),
            }
            record_summary["features"][name] = entry
            if name not in schema:
                schema[name] = {"kind": kind, "lengths": []}
            schema[name]["lengths"].append(int(length))

            if kind == "bytes" and feature.bytes_list.value:
                value = feature.bytes_list.value[0]
                if value.startswith(b"\x89PNG") or value.startswith(b"\xff\xd8"):
                    suffix = ".png" if value.startswith(b"\x89PNG") else ".jpg"
                    (output_dir / f"example_{index:03d}_{name}{suffix}").write_bytes(value)
                elif len(value) in {256 * 256, 128 * 128, 512 * 512}:
                    side = int(round(len(value) ** 0.5))
                    image = np.frombuffer(value, dtype=np.uint8).reshape(side, side)
                    Image.fromarray(image).save(output_dir / f"example_{index:03d}_{name}.png")
        examples.append(record_summary)

    compact_schema = {
        name: {
            "kind": item["kind"],
            "lengths": sorted(set(item["lengths"])),
        }
        for name, item in sorted(schema.items())
    }
    summary = {
        "tfrecords": str(tfrecords_path),
        "max_examples": int(max_examples),
        "schema": compact_schema,
        "examples": examples,
    }
    with (output_dir / "floornet_tfrecords_schema.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tfrecords", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-examples", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = inspect_tfrecords(
        args.tfrecords,
        args.output_dir,
        max_examples=args.max_examples,
    )
    print(f"Inspected {args.tfrecords}")
    print("Features:")
    for name, item in summary["schema"].items():
        print(f"  {name}: {item['kind']} lengths={item['lengths']}")
    print(f"Saved schema to {args.output_dir / 'floornet_tfrecords_schema.json'}")


if __name__ == "__main__":
    sys.exit(main())
