# AgileX NERO on macOS

The arm scripts use the USB-CAN adapter IDs and saved HOME pose in `nero_user_config.py`. Check those values before using another setup.

## Setup

```sh
conda env create -f environment.yml
conda activate nero-arm
```

Keep both arms supported when motors are off. Clear the full path before any HOME move and keep the physical E-stop within reach.

## `nero_leader_follower.py`

This script handles status, HOME, alignment, modes, following, and release.

```sh
python nero_leader_follower.py status
python nero_leader_follower.py home --preview
python nero_leader_follower.py home
python nero_leader_follower.py run
python nero_leader_follower.py --help
```

Run `home --preview` before moving from a supported rest pose. If it refuses the path, do not bypass its checks. Follow the prompts during the HOME move.

## `nero_apanda_mac.py`

This is a macOS adaptation of [`apanda-champ/AgileX_Nero_setup`](https://github.com/apanda-champ/AgileX_Nero_setup), commit `26370f5a6fd58d92ea59ab150dbafc29d445fb52`, with a 50 Hz leader-follower loop and two-camera RGB recording.

```sh
python nero_apanda_mac.py --check --no-home
python nero_apanda_mac.py --no-home
python nero_apanda_mac.py --no-home --preview-cameras
python nero_apanda_mac.py --no-home --preview-cameras --record-dir episodes_round2
python nero_apanda_mac.py --help
```

Use `--no-home` after both arms have reached the saved HOME pose. The Panda script's `--dry-home` uses a straight joint ramp and is not a general route from a released table-rest pose. While running, press `r` to start a demo, `e` to save it, `d` to discard, `h` to return HOME, and `q` to quit.

Use `--record-dir` to keep a new collection in its own folder. Existing episode numbers in that folder are continued, and existing HDF5 files are never overwritten.

Before collecting demos, set `TASK_TEXT` in `nero_user_config.py` to the exact task instruction. This file also holds the arm IDs, HOME pose, gripper mapping, camera settings, and output folder. The script will refuse to start a demo if either RGB feed or task text is missing. For one task, repeat `h -> r -> demonstrate -> e` ten times; `h` is only allowed between demos.

The OAK wrist and RealSense global cameras capture at 30 fps. Their RGB frames are saved as 224 × 224 JPEGs, fitted with black padding to keep the whole view. Depth is off by default. The 50 Hz arm loop and both camera streams keep their own timestamps. The RealSense requires `ffmpeg` (`brew install ffmpeg`). `--preview-cameras` opens the live side-by-side view but is optional for recording. If macOS lists only the RealSense depth feed, reconnect its USB cable and rerun the command.

Each episode saves `*_joints.h5`, `*_camera.h5` (wrist), and `*_global_camera.h5`. The joint file includes the follower's measured state and the exact targets sent to it. These HDF5 files are the raw recordings, not yet the LeRobot dataset.

## Export for π0.5

After recording, copy the repo or `episodes/` folder to a Python environment with `lerobot`, `h5py`, `numpy`, and `Pillow` installed, then run:

```sh
python export_nero_lerobot.py
```

The exporter checks all episodes first, matches robot and camera timestamps, and writes `lerobot_nero_pi05/` at 30 fps in LeRobot v3 format. It stores `observation.state` and `action` as `[J1, J2, J3, J4, J5, J6, J7, gripper_width_m]`, with global and wrist RGB views. Both arm joint arrays use radians. It refuses an existing output folder; use `--output new_folder` for another export. Use `--input path/to/episodes` when the recordings are elsewhere. This step stays separate from teleop so video encoding cannot slow the 50 Hz control loop.

The first nine episodes recorded on 31 October 2026 used the OAK device clock, which has a different origin from the Mac's monotonic clock. The exporter aligns their wrist timestamps to the global camera using the start and end of each episode. New episodes use host timestamps for both cameras.

## One-arm π0.5 inference

Run the server on Blackwell from `/mnt/storage/aryan/nero_pi05`. Choose the GPU and checkpoint explicitly:

```sh
python scripts/pi05_inference.py --gpu 1 \
  --checkpoint outputs/pi05_76_color_gpu1_bs4/checkpoints/last/pretrained_model
```

Both `--gpu` and `--checkpoint` are required; the `last` link currently points to step 20,585 of the 76-demo run. In an earlier no-motion check with the older checkpoint, guided calls took 1.11 s on GPU 1. `run --rtc` checks a guided request without moving and sends no RTC motion unless it is under 1.10 s. GPU load can change this result. Plain `run` remains available at slower inference speed.

In another terminal on the Mac, with `nero-arm` active and from this folder:

```sh
python nero_pi05_mac.py check     # read-only arm, two-camera and server check
python nero_pi05_mac.py dry-run       # one model request; no motor commands
python nero_pi05_mac.py dry-run --rtc # also time guided RTC; no motor commands
python nero_pi05_mac.py run --actions-per-chunk 35
python nero_pi05_mac.py run --actions-per-chunk 35 --preview-cameras
python nero_pi05_mac.py run --rtc     # RTC for the whole run
python nero_pi05_mac.py run --rtc --preview-cameras
python nero_pi05_mac.py run --actions-per-chunk 35 --task "Pick up the blocks and place them in the plate."
```

Use the existing HOME procedure first. `run` requires the follower already enabled in position control and within 0.10 rad of the recorded HOME pose; it never enables motors itself. Keep the physical E-stop close. It commands only the follower, with no leader arm connection. `run --rtc` checks guided inference speed without moving first, then starts a new Blackwell request after 15 of 50 actions, with 35 remaining, while the Mac keeps sending actions at the dataset's 30 Hz. The server guides the new chunk against the unplayed actions and the Mac skips actions that elapsed during inference. If a reply arrives after the queue empties, the Mac holds the follower until it arrives, then resumes with the remaining actions. Joint targets are clamped to the arm's firmware limits; a failed request or rejected motor command still stops the run. Plain `run` executes 35 actions per chunk by default and waits 30 ms before the next observation. `--preview-cameras` shows the live global and wrist RGB views in a separate window; its `s`, `p`, `h`, and `q` keys work like the terminal keys. The arm stays powered in position hold on exit. Server address and plain-run defaults are in `nero_user_config.py`.

During either run, press `s` to start, `p` to pause and hold, and `s` again to resume the same episode from the held pose with a fresh observation. To start a new episode, press `p`, then `h` to move the follower directly to HOME at 50% speed and command its gripper closed, then `s`. Press `q` or Ctrl-C to quit. If `h` is pressed during motion, it pauses first; press `h` again to move HOME. Keep the full HOME path clear and the gripper empty before pressing `h`. The script does not detect task success or save inference episodes.

`--task` overrides `TASK_TEXT` in `nero_user_config.py` for that run. With `--dagger`, press `i` to align the Leader, `r` to record a correction, and `e` to save it. The saved correction uses the same task text sent to the model. Use `--record-dir` to choose its output folder.

## Blackwell scripts

`pi05_inference.py` runs the model server on Blackwell. `blackwell_training/` contains the dataset preparation, training, and evaluation scripts used for the 76-demo and correction runs. On Blackwell these files live under `/mnt/storage/aryan/nero_pi05/scripts/`; the datasets, model weights, Conda environment, and output checkpoints live separately and are not included in this repository. The training launchers expect that existing Blackwell layout and the paths specified inside each script.
