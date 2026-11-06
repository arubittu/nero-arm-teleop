#!/usr/bin/env python3
"""Keep the 76-demo holdouts while repeating correction episodes in training."""

import argparse
import json
from pathlib import Path


SUFFIXES = ("_joints.h5", "_camera.h5", "_global_camera.h5")


def episode_names(folder):
    return sorted(path.name.removesuffix("_joints.h5")
                  for path in folder.glob("ep_*_joints.h5"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--corrections", type=Path, required=True)
    parser.add_argument("--base-split", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--repeat", type=int, default=3)
    args = parser.parse_args()

    if args.repeat < 1 or args.split.exists():
        raise SystemExit("Repeat must be positive; split must be a new path")
    original = json.loads(args.base_split.read_text())
    base = episode_names(args.base)
    corrections = episode_names(args.corrections)
    if base != original["source_episodes"] or not corrections:
        raise SystemExit("Base split does not match the source episodes, or corrections are empty")

    sources = [(f"ep_b{i:03d}_{name}", args.base / name)
               for i, name in enumerate(base)]
    sources += [(f"ep_c{repeat:02d}_{i:03d}_{name}", args.corrections / name)
                for repeat in range(args.repeat)
                for i, name in enumerate(corrections)]
    for _, source in sources:
        for suffix in SUFFIXES:
            if not source.with_name(source.name + suffix).is_file():
                raise SystemExit(f"Missing {source.name + suffix}")

    split = {
        "repo_id": original["repo_id"],
        "source_episodes": [alias for alias, _ in sources],
        "train_episodes": original["train_episodes"] + list(range(len(base), len(sources))),
        "validation_episodes": original["validation_episodes"],
        "test_episodes": original["test_episodes"],
    }
    args.split.write_text(json.dumps(split, indent=2) + "\n")
    print(f"Split {len(base)} base episodes and {len(corrections)} corrections "
          f"repeated {args.repeat} times; {len(sources)} episodes")
    print(f"Train/validation/test: {len(split['train_episodes'])}/"
          f"{len(split['validation_episodes'])}/{len(split['test_episodes'])}")


if __name__ == "__main__":
    main()
