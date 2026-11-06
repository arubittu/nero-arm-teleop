#!/usr/bin/env python3
"""Export timestamped NERO teleop episodes to a local LeRobot v3 dataset."""

import argparse
import io
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

from nero_user_config import DATASET_FPS, RECORD_DIR, SAVED_IMAGE_SIZE


def nearest(times, target, max_gap):
    right = int(np.searchsorted(times, target))
    choices = [i for i in (right - 1, right) if 0 <= i < len(times)]
    index = min(choices, key=lambda i: abs(times[i] - target))
    if abs(times[index] - target) > max_gap:
        raise ValueError(f"stream gap near {target:.3f} s exceeds {max_gap:.3f} s")
    return index


def max_grid_gap(times, grid):
    right = np.clip(np.searchsorted(times, grid), 0, len(times) - 1)
    left = np.clip(right - 1, 0, len(times) - 1)
    return float(np.minimum(abs(times[left] - grid), abs(times[right] - grid)).max())


def image_at(dataset, index):
    rgb = np.asarray(Image.open(io.BytesIO(dataset[index].tobytes())).convert("RGB"))
    if rgb.shape != (SAVED_IMAGE_SIZE[1], SAVED_IMAGE_SIZE[0], 3):
        raise ValueError(f"RGB image has shape {rgb.shape}, expected 224 x 224 x 3")
    return rgb


def checked_times(h5, label):
    times = np.asarray(h5["timestamps"], dtype=np.float64)
    if len(times) < 2 or not np.all(np.isfinite(times)) or not np.all(np.diff(times) > 0):
        raise ValueError(f"{label}: missing, invalid, or non-increasing timestamps")
    if np.max(np.diff(times)) > 0.15:
        raise ValueError(f"{label}: recording gap exceeds 150 ms")
    return times


def align_wrist_times(wrist_times, global_times):
    """Align older OAK recordings whose device clock had a different origin."""
    if abs(wrist_times[0] - global_times[0]) < 1.0:
        return wrist_times
    offset = ((wrist_times[0] - global_times[0])
              + (wrist_times[-1] - global_times[-1])) / 2
    aligned = wrist_times - offset
    if (abs(aligned[0] - global_times[0]) > 0.1
            or abs(aligned[-1] - global_times[-1]) > 0.1):
        raise ValueError("wrist and global camera spans cannot be aligned")
    return aligned


def inspect_episode(joints_path):
    base = joints_path.name.removesuffix("_joints.h5")
    wrist_path = joints_path.with_name(base + "_camera.h5")
    global_path = joints_path.with_name(base + "_global_camera.h5")
    if not (wrist_path.exists() and global_path.exists()):
        raise ValueError(f"{base}: both RGB camera files are required")
    with (h5py.File(joints_path) as joints, h5py.File(wrist_path) as wrist,
          h5py.File(global_path) as global_camera):
        if len({f.attrs.get("episode") for f in (joints, wrist, global_camera)}) != 1:
            raise ValueError(f"{base}: episode IDs differ between streams")
        task = str(joints.attrs.get("task", "")).strip()
        if not task:
            raise ValueError(f"{base}: task text is missing")
        jt = checked_times(joints, "joints")
        wt = checked_times(wrist, "wrist RGB")
        gt = checked_times(global_camera, "global RGB")
        wt = align_wrist_times(wt, gt)
        state = np.column_stack((joints["follower/joint_angles"][:],
                                 joints["follower/gripper"][:]))
        action = np.column_stack((joints["action_target/joint_angles"][:],
                                  joints["action_target/gripper"][:]))
        if state.shape != (len(jt), 8) or action.shape != (len(jt), 8):
            raise ValueError(f"{base}: expected eight state and action values per sample")
        if not np.all(np.isfinite(state)) or not np.all(np.isfinite(action)):
            raise ValueError(f"{base}: state or action contains NaN/Inf")
        for label, camera, times in (("wrist", wrist, wt), ("global", global_camera, gt)):
            if len(camera["color_jpeg"]) != len(times):
                raise ValueError(f"{base}: {label} image count differs from timestamps")
            rate = (len(times) - 1) / (times[-1] - times[0])
            if rate < DATASET_FPS * 0.8:
                raise ValueError(f"{base}: {label} camera averaged only {rate:.1f} fps")
            if times[0] > jt[0] + 0.12 or times[-1] < jt[-1] - 0.12:
                raise ValueError(f"{base}: {label} camera missed part of the episode")
            for index in {0, len(times) // 2, len(times) - 1}:
                rgb = image_at(camera["color_jpeg"], index)
                if float(rgb.mean()) < 5 and float(rgb.std()) < 3:
                    raise ValueError(f"{base}: {label} camera has a black frame")
        start = max(jt[0], wt[0], gt[0])
        end = min(jt[-1], wt[-1], gt[-1])
        count = int((end - start) * DATASET_FPS) + 1
        if count < DATASET_FPS:
            raise ValueError(f"{base}: less than one second of synchronized data")
        grid = start + np.arange(count) / DATASET_FPS
        for label, times in (("joints", jt), ("wrist RGB", wt), ("global RGB", gt)):
            if max_grid_gap(times, grid) > 0.06:
                raise ValueError(f"{base}: {label} has a gap over 60 ms on the export grid")
        return base, task, count


def export_episode(dataset, joints_path):
    base = joints_path.name.removesuffix("_joints.h5")
    with (h5py.File(joints_path) as joints,
          h5py.File(joints_path.with_name(base + "_camera.h5")) as wrist,
          h5py.File(joints_path.with_name(base + "_global_camera.h5")) as global_camera):
        jt = joints["timestamps"][:]
        wt = wrist["timestamps"][:]
        gt = global_camera["timestamps"][:]
        wt = align_wrist_times(wt, gt)
        start = max(jt[0], wt[0], gt[0])
        end = min(jt[-1], wt[-1], gt[-1])
        count = int((end - start) * DATASET_FPS) + 1
        state = np.column_stack((joints["follower/joint_angles"][:],
                                 joints["follower/gripper"][:])).astype(np.float32)
        action = np.column_stack((joints["action_target/joint_angles"][:],
                                  joints["action_target/gripper"][:])).astype(np.float32)
        for step in range(count):
            t = start + step / DATASET_FPS
            j = nearest(jt, t, 0.06)
            w = nearest(wt, t, 0.06)
            g = nearest(gt, t, 0.06)
            dataset.add_frame({
                "observation.state": state[j],
                "action": action[j],
                "observation.images.base_0_rgb": image_at(global_camera["color_jpeg"], g),
                "observation.images.left_wrist_0_rgb": image_at(wrist["color_jpeg"], w),
                "task": str(joints.attrs["task"]),
            })
        dataset.save_episode()
        print(f"{base}: {count} synchronized frames at {DATASET_FPS} fps")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=RECORD_DIR)
    parser.add_argument("--output", type=Path, default=Path("lerobot_nero_pi05"))
    parser.add_argument("--repo-id", default="local/nero_pi05")
    args = parser.parse_args()
    paths = sorted(args.input.glob("ep_*_joints.h5"))
    if not paths:
        raise SystemExit(f"No recorded episodes in {args.input}")
    if args.output.exists():
        raise SystemExit(f"Output already exists: {args.output}. Choose a new --output path.")
    for path in paths:
        base, task, count = inspect_episode(path)
        print(f"Checked {base}: {count} frames, task: {task}")

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    image = {"dtype": "video", "shape": (224, 224, 3),
             "names": ["height", "width", "channel"]}
    vector = {"dtype": "float32", "shape": (8,),
              "names": ["j1", "j2", "j3", "j4", "j5", "j6", "j7", "gripper"]}
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id, root=args.output, fps=DATASET_FPS,
        robot_type="agilex_nero_single_arm", use_videos=True,
        features={"observation.state": vector, "action": vector,
                  "observation.images.base_0_rgb": image,
                  "observation.images.left_wrist_0_rgb": image},
    )
    try:
        for path in paths:
            export_episode(dataset, path)
    finally:
        dataset.finalize()
    for key in ("observation.state", "action"):
        stats = dataset.meta.stats.get(key, {})
        if "q01" not in stats or "q99" not in stats:
            raise SystemExit(f"{key} is missing q01/q99 normalization statistics")
    print(f"Exported {len(paths)} episodes to {args.output}")


if __name__ == "__main__":
    main()
