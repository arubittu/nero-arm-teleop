# AgileX NERO on macOS

Two Python scripts for this two-arm setup. They use the USB-CAN adapter IDs and saved HOME pose in the files. Check those values before using another setup.

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

This is a macOS adaptation of [`apanda-champ/AgileX_Nero_setup`](https://github.com/apanda-champ/AgileX_Nero_setup), commit `26370f5a6fd58d92ea59ab150dbafc29d445fb52`, with a 50 Hz leader-follower loop and OAK camera recording.

```sh
python nero_apanda_mac.py --check --no-home
python nero_apanda_mac.py --no-home
python nero_apanda_mac.py --help
```

Use `--no-home` after both arms have reached the saved HOME pose. The Panda script's `--dry-home` uses a straight joint ramp and is not a general route from a released table-rest pose. While running, press `r` to start a demo, `e` to save it, `h` to return HOME, and `q` to quit.

The Panda script records the OAK camera. The separate RealSense D435 global camera is not part of this code.
