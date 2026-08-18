#!/usr/bin/env python3
"""Split a PyTorch state dict into balanced, independently loadable shards."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def tensor_bytes(value) -> int:
    return value.numel() * value.element_size() if torch.is_tensor(value) else 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--shards", required=True, type=int)
    args = parser.parse_args()

    state = torch.load(args.source, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise TypeError("checkpoint must contain a state dict")

    buckets = [dict() for _ in range(args.shards)]
    sizes = [0 for _ in range(args.shards)]
    for key, value in sorted(state.items(), key=lambda item: tensor_bytes(item[1]), reverse=True):
        index = min(range(args.shards), key=sizes.__getitem__)
        buckets[index][key] = value
        sizes[index] += tensor_bytes(value)

    args.destination.mkdir(parents=True, exist_ok=True)
    for index, bucket in enumerate(buckets):
        if not bucket:
            raise RuntimeError("requested more shards than checkpoint entries")
        target = args.destination / f"{args.prefix}-{index:03d}.pth"
        torch.save(bucket, target)
        print(f"{target.name}: {target.stat().st_size} bytes")


if __name__ == "__main__":
    main()
