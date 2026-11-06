#!/usr/bin/env python3
"""Merge 76 base demos with three copies of the 50 correction episodes."""

import argparse
from pathlib import Path

from lerobot.datasets.aggregate import aggregate_datasets
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--corrections", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"Output already exists: {args.output}")

    base = LeRobotDatasetMetadata("local/nero_pi05", root=args.base)
    corrections = LeRobotDatasetMetadata("local/nero_pi05_dagger", root=args.corrections)
    if base.total_episodes != 76 or corrections.total_episodes != 50:
        raise SystemExit("Expected 76 base and 50 correction episodes")

    aggregate_datasets(
        repo_ids=["local/nero_pi05"] + ["local/nero_pi05_dagger"] * 3,
        aggr_repo_id="local/nero_pi05",
        roots=[args.base] + [args.corrections] * 3,
        aggr_root=args.output,
        concatenate_videos=False,
        concatenate_data=False,
    )
    merged = LeRobotDatasetMetadata("local/nero_pi05", root=args.output)
    if merged.total_episodes != 226 or merged.total_frames != base.total_frames + 3 * corrections.total_frames:
        raise SystemExit("Merged dataset has the wrong episode or frame count")
    print(f"Merged {merged.total_episodes} episodes and {merged.total_frames} frames")


if __name__ == "__main__":
    main()
