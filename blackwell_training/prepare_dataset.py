#!/usr/bin/env python3
"""Stage the NERO dataset with normalization fitted on training episodes only."""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


def vector_stats(values):
    return {
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
        "mean": values.mean(axis=0).tolist(),
        "std": values.std(axis=0).tolist(),
        "count": [len(values)],
        **{f"q{int(q * 100):02d}": np.quantile(values, q, axis=0).tolist()
           for q in (0.01, 0.10, 0.50, 0.90, 0.99)},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--staged", required=True, type=Path)
    parser.add_argument("--split", required=True, type=Path)
    args = parser.parse_args()
    if args.staged.exists():
        raise SystemExit(f"Staged dataset already exists: {args.staged}")
    manifest = json.loads(args.split.read_text())
    train_ids = manifest["train_episodes"]
    val_ids = manifest["validation_episodes"]
    test_ids = manifest["test_episodes"]
    parquet_files = sorted((args.source / "data").rglob("*.parquet"))
    if not parquet_files:
        raise SystemExit(f"No source data found: {args.source}")
    frames = pd.concat(
        [pd.read_parquet(p, columns=["episode_index", "observation.state", "action"])
         for p in parquet_files],
        ignore_index=True,
    )
    counts = frames.groupby("episode_index").size().to_dict()
    expected = list(range(len(counts)))
    if sorted(counts) != expected or sorted(train_ids + val_ids + test_ids) != expected:
        raise SystemExit("Split must cover each exported episode exactly once.")
    if not train_ids or not val_ids or not test_ids:
        raise SystemExit("Train, validation, and test splits must all be nonempty.")
    if len(manifest["source_episodes"]) != len(counts):
        raise SystemExit("Split source episode count does not match the dataset.")
    train = frames[frames["episode_index"].isin(train_ids)]
    stats = json.loads((args.source / "meta/stats.json").read_text())
    for key in ("observation.state", "action"):
        values = np.stack(train[key].to_numpy()).astype(np.float64)
        stats[key] = vector_stats(values)
    manifest["frames"] = {
        "train": sum(counts[i] for i in train_ids),
        "validation": sum(counts[i] for i in val_ids),
        "test": sum(counts[i] for i in test_ids),
    }
    manifest["fps"] = json.loads((args.source / "meta/info.json").read_text())["fps"]
    shutil.copytree(args.source, args.staged)
    (args.staged / "meta/stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    (args.staged / "meta/train_split.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Staged dataset: {args.staged}")
    print(f"Split frames: {manifest['frames']}")


if __name__ == "__main__":
    main()
