#!/usr/bin/env python3
"""Print final π0.5 validation and untouched test losses offline."""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.utils.collate import lerobot_collate_fn


def evaluate(policy, preprocessor, meta, dataset_root, repo_id, episodes, label):
    dataset = LeRobotDataset(
        repo_id,
        root=dataset_root,
        episodes=episodes,
        delta_timestamps=resolve_delta_timestamps(policy.config, meta),
        return_uint8=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=4,
        collate_fn=lerobot_collate_fn if dataset.meta.has_language_columns else None,
    )
    weighted_loss = 0.0
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for batch in loader:
            for key in dataset.meta.camera_keys:
                if batch[key].dtype == torch.uint8:
                    batch[key] = batch[key].float() / 255.0
            loss, _ = policy.forward(preprocessor(batch))
            weighted_loss += loss.item() * len(batch["action"])
    print(f"{label}_loss={weighted_loss / len(dataset):.6f}  frames={len(dataset)}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args()
    model_dir = args.checkpoint / "pretrained_model"
    if not model_dir.is_dir():
        raise SystemExit(f"Missing checkpoint model: {model_dir}")
    torch.manual_seed(1000)
    policy = PI05Policy.from_pretrained(model_dir).to("cuda").eval()
    preprocessor, _ = make_pre_post_processors(policy.config, pretrained_path=str(model_dir))
    split = json.loads((args.dataset / "meta/train_split.json").read_text())
    repo_id = split["repo_id"]
    meta = LeRobotDatasetMetadata(repo_id, root=args.dataset)
    evaluate(policy, preprocessor, meta, args.dataset, repo_id,
             split["validation_episodes"], "validation")
    evaluate(policy, preprocessor, meta, args.dataset, repo_id,
             split["test_episodes"], "test")


if __name__ == "__main__":
    main()
