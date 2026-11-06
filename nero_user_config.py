"""Settings for this Mac's NERO teleop and one-arm π0.5 dataset."""

from pathlib import Path


# Hardware and motion. Keep these values matched to the physical setup.
LEADER_CHANNEL = "004900334148570D20343133:0"
FOLLOWER_CHANNEL = "003800294148570A20343133:0"
LOOP_HZ = 50.0
HOME_ON_START = True
HOME_POSE = [-0.056, -0.329, 0.164, 1.730, -0.001, 0.091, 0.281]
HOME_FOLLOWER_TOO = True
HOME_DURATION_S = 4.0
HOME_HOLD_S = 0.5
HOME_MAX_DELTA_RAD = 1.8
HOME_WARN_S = 3.0
HOME_WARN_MID_S = 0.0
START_WITH_GC = True
GC_WARN_S = 3.0

# Gripper target sent to the follower is the leader width times this gain.
LEADER_GRIP_FORCE = 0.05
LEADER_GRIP_TARGET = 0.1
GRIP_REASSERT_S = 1.0
GRIP_GAIN = 1.5
CLAMP_TO_FOLLOWER_RANGE = True
FOLLOWER_MIN_M = 0.0
FOLLOWER_MAX_M = 0.1
FOLLOWER_GRIP_FORCE = 2.0
MIRROR_GRIPPER = True

# Set this to the exact instruction used for every demonstration in this dataset.
TASK_TEXT = "Pick up the blocks and place them in the plate."
RECORD = True
RECORD_DIR = Path(__file__).resolve().parent / "episodes"
RECORD_FOLLOWER = True
H5_COMPRESSION = "gzip"
DATASET_FPS = 30

# Camera capture stays at native size; saved RGB frames are 224 x 224.
WRIST_CAPTURE_SIZE = (640, 400)
GLOBAL_CAPTURE_SIZE = (640, 480)
CAMERA_FPS = 30
PREVIEW_FPS = 15
SAVED_IMAGE_SIZE = (224, 224)
JPEG_QUALITY = 90
CAMERA_DEPTH = False  # π0.5 uses RGB; depth greatly increases RAM and disk use.

# Sequential one-arm π0.5 inference. Blackwell runs pi05_inference.py on GPU 1.
PI05_SERVER_URL = "http://10.72.18.159:8765"
PI05_REQUEST_TIMEOUT_S = 30.0
PI05_ACTIONS_PER_CHUNK = 35  # of the model's 50 predicted actions
PI05_POST_ACTION_WAIT_S = 0.03
