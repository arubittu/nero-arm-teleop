#!/usr/bin/env python3
"""π0.5 rollout on one NERO follower arm, with optional RTC."""

import argparse
import base64
import json
import math
import mmap
import os
import select
import subprocess
import sys
import tempfile
import termios
import threading
import time
import tty
import urllib.request

import cv2
import depthai as dai
import numpy as np
from pyAgxArm import AgxArmFactory, ArmModel, NeroFW, create_agx_arm_config

from nero_camera_preview import real_sense_rgb_index
from nero_user_config import (
    CAMERA_FPS, DATASET_FPS, FOLLOWER_CHANNEL, FOLLOWER_GRIP_FORCE,
    FOLLOWER_MAX_M, FOLLOWER_MIN_M, GLOBAL_CAPTURE_SIZE, HOME_MAX_DELTA_RAD,
    HOME_POSE, JPEG_QUALITY,
    PI05_ACTIONS_PER_CHUNK, PI05_POST_ACTION_WAIT_S, PI05_REQUEST_TIMEOUT_S,
    PI05_SERVER_URL, SAVED_IMAGE_SIZE, TASK_TEXT, WRIST_CAPTURE_SIZE,
)


NERO_JOINT_LIMITS = create_agx_arm_config(robot=ArmModel.NERO, firmeware_version=NeroFW.V121)["joint_limits"]
HOME_TOLERANCE_RAD = 0.10
CAMERA_TIMEOUT_S = 0.5
RTC_REPLAN_REMAINING = 35
RTC_EXECUTION_HORIZON = 35
HOME_SPEED_PERCENT = 50
HOME_TIMEOUT_S = 20
PREVIEW_FRAME_BYTES = WRIST_CAPTURE_SIZE[0] * WRIST_CAPTURE_SIZE[1] * 3


class Keyboard:
    def __init__(self, preview_process=None):
        self.preview_process = preview_process

    def __enter__(self):
        self.fd = sys.stdin.fileno()
        self.original = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        return self

    def __exit__(self, *_):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.original)

    def get(self):
        if select.select([self.fd], [], [], 0)[0]:
            return os.read(self.fd, 1).decode("ascii", errors="ignore").lower()
        if self.preview_process is not None:
            if self.preview_process.poll() is not None:
                return "q"
            output = self.preview_process.stdout
            if select.select([output], [], [], 0)[0]:
                return os.read(output.fileno(), 1).decode("ascii", errors="ignore").lower()
        return None


def saved_rgb(frame):
    """Match the letterboxed 224x224 BGR-to-JPEG export used in training."""
    width, height = SAVED_IMAGE_SIZE
    h, w = frame.shape[:2]
    scale = min(width / w, height / h)
    resized = cv2.resize(frame, (round(w * scale), round(h * scale)),
                         interpolation=cv2.INTER_AREA)
    out = np.zeros((height, width, 3), dtype=np.uint8)
    y, x = (height - resized.shape[0]) // 2, (width - resized.shape[1]) // 2
    out[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    ok, jpeg = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise RuntimeError("Could not encode camera frame")
    return jpeg.tobytes()


class Cameras:
    def __init__(self):
        self.running = False
        self.frames = {}
        self.lock = threading.Lock()
        self.threads = []
        self.process = None
        self.error = None
        self.preview_map = None
        self.frame_sink = None

    def start(self):
        index = real_sense_rgb_index()
        self.running = True
        for name, target in (("wrist", self._wrist), ("base", lambda: self._base(index))):
            thread = threading.Thread(target=self._guard, args=(name, target), daemon=True)
            thread.start()
            self.threads.append(thread)
        self.latest_after(0, timeout=8)

    def _guard(self, name, target):
        try:
            target()
            if self.running:
                raise RuntimeError(f"{name} camera stopped")
        except Exception as exc:
            self.error = f"{name} camera: {exc}"

    def _put(self, name, frame):
        if name == "base" and np.array_equal(frame[:, :, 0], frame[:, :, 1]) and np.array_equal(frame[:, :, 1], frame[:, :, 2]):
            raise RuntimeError("RealSense returned grayscale Depth/IR instead of RGB")
        jpeg = saved_rgb(frame)
        if self.preview_map is not None:
            if frame.shape[:2] != (WRIST_CAPTURE_SIZE[1], WRIST_CAPTURE_SIZE[0]):
                frame = cv2.resize(frame, WRIST_CAPTURE_SIZE, interpolation=cv2.INTER_AREA)
            index = 0 if name == "wrist" else 1
            slot = 1 - self.preview_map[index]
            offset = 2 + (2 * index + slot) * PREVIEW_FRAME_BYTES
            np.ndarray(frame.shape, dtype=np.uint8,
                       buffer=self.preview_map, offset=offset)[:] = frame
            self.preview_map[index] = slot
        stamp = time.monotonic()
        if self.frame_sink is not None:
            self.frame_sink(name, stamp, jpeg)
        with self.lock:
            self.frames[name] = (stamp, jpeg, float(frame.mean()))

    def _wrist(self):
        while self.running:
            try:
                with dai.Pipeline() as pipeline:
                    cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
                    queue = cam.requestOutput(
                        WRIST_CAPTURE_SIZE, dai.ImgFrame.Type.BGR888i,
                        fps=CAMERA_FPS).createOutputQueue(maxSize=1, blocking=False)
                    pipeline.start()
                    while self.running and pipeline.isRunning():
                        frame = queue.tryGet()
                        if frame is None:
                            time.sleep(0.002)
                        else:
                            self._put("wrist", frame.getCvFrame())
            except Exception as exc:
                if self.running:
                    print(f"Wrist camera disconnected: {exc}. Retrying in 1 s.")
            if self.running:
                with self.lock:
                    self.frames.pop("wrist", None)
                time.sleep(1)

    def _base(self, index):
        width, height = GLOBAL_CAPTURE_SIZE
        self.process = subprocess.Popen(
            ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
             "-f", "avfoundation", "-framerate", str(CAMERA_FPS),
             "-video_size", f"{width}x{height}", "-i", f"{index}:none",
             "-an", "-pix_fmt", "bgr24", "-f", "rawvideo", "pipe:1"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0,
        )
        frame_bytes = width * height * 3
        while self.running:
            chunks, left = [], frame_bytes
            while left:
                part = os.read(self.process.stdout.fileno(), left)
                if not part:
                    raise RuntimeError("RealSense RGB stopped returning frames")
                chunks.append(part)
                left -= len(part)
            frame = np.frombuffer(b"".join(chunks), dtype=np.uint8).reshape(height, width, 3)
            self._put("base", frame)

    def latest_after(self, after, timeout=CAMERA_TIMEOUT_S):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.error:
                raise RuntimeError(self.error)
            with self.lock:
                pair = {name: self.frames.get(name) for name in ("base", "wrist")}
            if all(item is not None and item[0] > after and item[2] >= 5
                   for item in pair.values()):
                return {name: item[1] for name, item in pair.items()}
            time.sleep(0.005)
        raise TimeoutError("Fresh, non-black RGB frames did not arrive from both cameras")

    def fresh(self):
        if self.error:
            return False
        with self.lock:
            return all(name in self.frames and time.monotonic() - self.frames[name][0] < 0.5
                       and self.frames[name][2] >= 5
                       for name in ("base", "wrist"))

    def stop(self):
        self.running = False
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
        for thread in self.threads:
            thread.join(timeout=0.5)


def connect_follower():
    config = create_agx_arm_config(robot=ArmModel.NERO, firmeware_version=NeroFW.V121,
                                   interface="candle", channel=FOLLOWER_CHANNEL)
    arm = AgxArmFactory.create_arm(config)
    arm.connect()
    gripper = arm.init_effector(arm.OPTIONS.EFFECTOR.AGX_GRIPPER)
    return arm, gripper


def read_firmware_joint_limits(arm):
    limits = {}
    for joint in range(1, 8):
        reply = arm.get_joint_angle_vel_limits(joint, timeout=1.5, min_interval=0)
        if reply is None:
            raise RuntimeError(f"Could not read firmware limit for J{joint}")
        low, high = reply.msg.min_angle_limit, reply.msg.max_angle_limit
        if not np.isfinite([low, high]).all() or low >= high:
            raise RuntimeError(f"Invalid firmware limit for J{joint}")
        limits[f"joint{joint}"] = [low, high]
    NERO_JOINT_LIMITS.update(limits)


def read_state(arm, gripper):
    joints = arm.get_joint_angles()
    grip = gripper.get_gripper_status()
    if joints is None or grip is None:
        raise RuntimeError("Missing follower joint or gripper feedback")
    state = np.asarray([*joints.msg, grip.msg.value], dtype=np.float32)
    if state.shape != (8,) or not np.isfinite(state).all():
        raise RuntimeError("Invalid follower state")
    return state


def require_position_mode(arm):
    status = arm.get_arm_status()
    if status is None or status.msg.ctrl_mode != arm.ARM_STATUS.CtrlMode.CAN_CTRL:
        raise RuntimeError("Follower must be in position mode; joint feedback is stale in gravity mode")


def require_position_control(arm):
    require_position_mode(arm)
    if not all(arm.get_joints_enable_status_list()):
        raise RuntimeError("Follower motors must already be enabled")


def server_request(path, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(PI05_SERVER_URL + path, data=data,
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=PI05_REQUEST_TIMEOUT_S) as response:
        return json.load(response)


def predict(state, frames, cycle, *, rtc=False, prefix=None, delay=0,
            actions_per_chunk=PI05_ACTIONS_PER_CHUNK, task=TASK_TEXT):
    start = time.perf_counter()
    payload = {
        "cycle": cycle, "task": task, "state": state.tolist(),
        "base_rgb": base64.b64encode(frames["base"]).decode("ascii"),
        "wrist_rgb": base64.b64encode(frames["wrist"]).decode("ascii"),
        "rtc": rtc, "inference_delay": delay,
    }
    if prefix is not None:
        payload["prev_chunk_left_over"] = prefix.tolist()
    response = server_request("/predict", payload)
    roundtrip_ms = (time.perf_counter() - start) * 1000
    actions = np.asarray(response["actions"], dtype=np.float32)
    if response["cycle"] != cycle or actions.shape != (50, 8) or not np.isfinite(actions).all():
        raise RuntimeError("Invalid or stale action chunk from Blackwell")
    raw = None
    if rtc:
        raw = np.asarray(response["raw_actions"], dtype=np.float32)
        if raw.shape != (50, 8) or not np.isfinite(raw).all():
            raise RuntimeError("Invalid RTC prefix from Blackwell")
    used = actions if rtc else actions[:actions_per_chunk]
    low, high = float(used[:, 7].min()), float(used[:, 7].max())
    print(f"Gripper: current {state[7]:.3f} m, next {len(used)} raw targets {low:.3f}..{high:.3f} m")
    return used, raw, response["inference_ms"], roundtrip_ms


def clamp_joint_actions(actions):
    lows, highs = np.asarray([NERO_JOINT_LIMITS[f"joint{j}"] for j in range(1, 8)]).T
    joints = actions[:, :7]
    outside = (joints < lows) | (joints > highs)
    for joint in np.flatnonzero(outside.any(axis=0)):
        print(f"J{joint + 1}: clamped {outside[:, joint].sum()} targets to "
              f"{lows[joint]:.3f}..{highs[joint]:.3f} rad")
    np.clip(joints, lows, highs, out=joints)


def rtc_guided_probe(state, frames, cycle, raw, roundtrip_ms, task=TASK_TEXT):
    delay = min(RTC_EXECUTION_HORIZON,
                max(1, math.ceil(roundtrip_ms * DATASET_FPS / 1000) + 2))
    _, _, _, guided_ms = predict(state, frames, cycle, rtc=True,
                                 prefix=raw[:RTC_EXECUTION_HORIZON], delay=delay, task=task)
    return guided_ms


def execute(arm, gripper, actions, current, keyboard, cameras, dagger=False):
    last_sent = None
    last_grip = float(current[7])
    for i, action in enumerate(actions):
        if last_sent is not None:
            time.sleep(max(0, last_sent + 1 / DATASET_FPS - time.monotonic()))
        key = keyboard.get()
        if key in ("p", "h") or (dagger and key == "i"):
            return key
        if key == "q":
            raise KeyboardInterrupt
        if not cameras.fresh():
            return "camera"
        last_sent = time.monotonic()
        result = arm.move_js(action[:7].tolist())
        if result is False:
            raise RuntimeError("Follower rejected a joint target")
        target_grip = min(FOLLOWER_MAX_M, max(FOLLOWER_MIN_M, float(action[7])))
        if abs(target_grip - last_grip) >= 0.002:
            result = gripper.move_gripper_m(value=target_grip, force=FOLLOWER_GRIP_FORCE)
            if result is False:
                raise RuntimeError("Follower rejected a gripper target")
            last_grip = target_grip
    last_command = time.monotonic()
    time.sleep(PI05_POST_ACTION_WAIT_S)
    return last_command


class RTCPlayback:
    """Play queued actions at 30 Hz while one Blackwell request runs in a thread."""

    def __init__(self, task=TASK_TEXT):
        self.task = task
        self.actions = self.raw = self.last_action = None
        self.index = self.cycle = 0
        self.pending = None
        self.roundtrip_ms = 0
        self.next_tick = 0
        self.last_grip = 0
        self.stalled = False

    def wait_pending(self):
        if self.pending is not None:
            self.pending["thread"].join()
            self.pending = None

    def step(self, arm, gripper, cameras):
        if self.actions is None:
            require_position_control(arm)
            frames = cameras.latest_after(time.monotonic() - 0.1)
            state = read_state(arm, gripper)
            _, raw, _, roundtrip_ms = predict(state, frames, self.cycle, rtc=True, task=self.task)
            guided_ms = rtc_guided_probe(state, frames, self.cycle, raw, roundtrip_ms, task=self.task)
            if guided_ms > (RTC_REPLAN_REMAINING - 2) * 1000 / DATASET_FPS:
                raise RuntimeError(
                    f"RTC guided round trip {guided_ms:.0f} ms is too slow for the "
                    f"{RTC_REPLAN_REMAINING}-action overlap; no RTC motion sent. "
                    "Use plain run or a faster/free GPU."
                )
            frames = cameras.latest_after(time.monotonic() - 0.1)
            state = read_state(arm, gripper)
            actions, raw, ms, self.roundtrip_ms = predict(state, frames, self.cycle, rtc=True,
                                                         task=self.task)
            clamp_joint_actions(actions)
            self.roundtrip_ms = max(self.roundtrip_ms, guided_ms)
            self.actions, self.raw, self.last_action = actions, raw, state
            self.last_grip = float(state[7])
            self.next_tick = time.monotonic()
            print(f"RTC cycle {self.cycle}: model {ms} ms, round trip {self.roundtrip_ms:.1f} ms")
            self.cycle += 1
            return

        if self.pending is not None and not self.pending["thread"].is_alive():
            pending = self.pending
            self.pending = None
            if "error" in pending:
                raise pending["error"]
            actions, raw, ms, self.roundtrip_ms = pending["result"]
            elapsed = self.index - pending["index"]
            if elapsed > RTC_EXECUTION_HORIZON:
                raise RuntimeError(f"RTC reply took {elapsed} actions; holding instead of joining stale chunks")
            actions, raw = actions[elapsed:], raw[elapsed:]
            clamp_joint_actions(actions)
            self.actions, self.raw, self.index = actions, raw, 0
            if self.stalled:
                self.next_tick = time.monotonic()
                self.stalled = False
            print(f"RTC cycle {self.cycle}: model {ms} ms, round trip {self.roundtrip_ms:.1f} ms, overlap {elapsed}")
            self.cycle += 1

        remaining = len(self.actions) - self.index
        if remaining <= RTC_REPLAN_REMAINING and self.pending is None:
            require_position_control(arm)
            frames = cameras.latest_after(time.monotonic() - 0.1, timeout=0.05)
            state = read_state(arm, gripper)
            delay = min(remaining, RTC_EXECUTION_HORIZON,
                        max(1, math.ceil(self.roundtrip_ms * DATASET_FPS / 1000) + 2))
            pending = {"index": self.index}
            prefix = self.raw[self.index:].copy()

            def request():
                try:
                    pending["result"] = predict(state, frames, self.cycle, rtc=True,
                                                prefix=prefix, delay=delay, task=self.task)
                except Exception as exc:
                    pending["error"] = exc

            pending["thread"] = threading.Thread(target=request, daemon=True)
            self.pending = pending
            pending["thread"].start()

        if self.index >= len(self.actions):
            if not self.stalled:
                self.last_action = hold(arm, gripper)
                self.last_grip = float(self.last_action[7])
                self.stalled = True
                print("RTC waiting for a late reply; follower holding its current pose.")
            time.sleep(0.02)
            return
        time.sleep(max(0, self.next_tick - time.monotonic()))
        sent = time.monotonic()
        action = self.actions[self.index]
        result = arm.move_js(action[:7].tolist())
        if result is False:
            raise RuntimeError("Follower rejected a joint target")
        target_grip = min(FOLLOWER_MAX_M, max(FOLLOWER_MIN_M, float(action[7])))
        if abs(target_grip - self.last_grip) >= 0.002:
            result = gripper.move_gripper_m(value=target_grip, force=FOLLOWER_GRIP_FORCE)
            if result is False:
                raise RuntimeError("Follower rejected a gripper target")
            self.last_grip = target_grip
        self.last_action = action
        self.index += 1
        self.next_tick = max(self.next_tick + 1 / DATASET_FPS,
                             sent + 1 / DATASET_FPS)


def hold(arm, gripper):
    state = read_state(arm, gripper)
    arm.move_j(state[:7].tolist())
    return state


def home_follower(arm, gripper):
    require_position_control(arm)
    current = read_state(arm, gripper)
    gap = max(abs(float(a - b)) for a, b in zip(current[:7], HOME_POSE))
    if gap > HOME_MAX_DELTA_RAD:
        print(f"HOME refused: {gap:.3f} rad exceeds the {HOME_MAX_DELTA_RAD:.2f} rad limit.")
        return False
    if any(
        not NERO_JOINT_LIMITS[f"joint{joint}"][0] <= target <= NERO_JOINT_LIMITS[f"joint{joint}"][1]
        for joint, target in enumerate(HOME_POSE, 1)
    ):
        print("HOME refused: HOME is outside the joint limits.")
        return False
    print(f"Moving follower to HOME at {HOME_SPEED_PERCENT}% speed; max joint change {gap:.3f} rad.")
    arm.set_speed_percent(HOME_SPEED_PERCENT)
    arm.move_j(list(HOME_POSE))
    deadline = time.monotonic() + HOME_TIMEOUT_S
    while time.monotonic() < deadline:
        current = read_state(arm, gripper)
        gap = max(abs(float(a - b)) for a, b in zip(current[:7], HOME_POSE))
        if gap <= 0.03:
            hold(arm, gripper)
            result = gripper.move_gripper_m(value=0.0, force=FOLLOWER_GRIP_FORCE)
            if result is False:
                raise RuntimeError("Follower rejected the gripper close command")
            print("At HOME. Gripper commanded closed. Press s to start the next episode.")
            return True
        time.sleep(0.05)
    raise TimeoutError("Follower did not reach HOME; holding its current pose")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "dry-run", "run"))
    parser.add_argument("--rtc", action="store_true", help="use real-time chunking for the whole run")
    parser.add_argument("--actions-per-chunk", type=int,
                        help=f"plain run only; default {PI05_ACTIONS_PER_CHUNK}, maximum 50")
    parser.add_argument("--preview-cameras", action="store_true",
                        help="show global and wrist RGB during a run")
    parser.add_argument("--dagger", action="store_true",
                        help="allow Leader takeover and record human corrections")
    parser.add_argument("--record-dir", default="episodes_dagger",
                        help="folder for DAgger correction episodes (default: episodes_dagger)")
    parser.add_argument("--task", default=TASK_TEXT,
                        help="language prompt for inference and DAgger recordings")
    args = parser.parse_args()
    actions_per_chunk = PI05_ACTIONS_PER_CHUNK if args.actions_per_chunk is None else args.actions_per_chunk
    if not 1 <= actions_per_chunk <= 50 or (args.rtc and args.actions_per_chunk is not None):
        parser.error("--actions-per-chunk must be 1..50 and applies only without --rtc")
    if args.preview_cameras and args.command != "run":
        parser.error("--preview-cameras applies only to run")
    if args.dagger and args.command != "run":
        parser.error("--dagger applies only to run")
    args.task = args.task.strip()
    if not args.task:
        parser.error("--task must contain text")
    arm = None
    intervention = None
    motion_started = False
    cameras = Cameras()
    preview_path = preview_map = preview_process = None
    try:
        if args.preview_cameras:
            preview_script = os.path.join(os.path.dirname(__file__), "nero_camera_preview.py")
            check = subprocess.run([sys.executable, preview_script, "--check"],
                                   capture_output=True, text=True, timeout=8)
            if check.returncode != 0:
                raise RuntimeError(check.stderr.strip() or "Camera preview is unavailable")
            with tempfile.NamedTemporaryFile(prefix="nero_cameras_", delete=False) as frame_file:
                preview_path = frame_file.name
                frame_file.truncate(2 + 4 * PREVIEW_FRAME_BYTES)
                preview_map = mmap.mmap(frame_file.fileno(), 0)
            cameras.preview_map = preview_map
        arm, gripper = connect_follower()
        if args.dagger:
            from nero_dagger import Intervention
            intervention = Intervention(args.record_dir, args.task)
            cameras.frame_sink = intervention.recorder.frame
        read_firmware_joint_limits(arm)
        cameras.start()
        health = server_request("/health")
        if health.get("chunk_size") != 50 or health.get("action_dim") != 8 or not health.get("rtc"):
            raise RuntimeError("Blackwell server has the wrong model contract")
        if args.rtc and health.get("rtc_execution_horizon") != RTC_EXECUTION_HORIZON:
            raise RuntimeError("Update and restart pi05_inference.py on Blackwell for 35-action RTC")
        if args.preview_cameras:
            preview_process = subprocess.Popen(
                [sys.executable, preview_script, preview_path], stdout=subprocess.PIPE,
                start_new_session=True)
        require_position_mode(arm)
        state = read_state(arm, gripper)
        print(f"Follower state: {np.round(state, 3).tolist()}")
        print("Both 224x224 RGB cameras and Blackwell server are ready.")
        if args.command == "check":
            return
        if args.command == "dry-run":
            frames = cameras.latest_after(0)
            actions, raw, ms, roundtrip_ms = predict(state, frames, cycle=0, rtc=args.rtc,
                                                    actions_per_chunk=actions_per_chunk,
                                                    task=args.task)
            clamp_joint_actions(actions)
            print(f"Dry run: {len(actions)} safe actions; model {ms} ms, round trip {roundtrip_ms:.1f} ms. No motor command sent.")
            if args.rtc:
                guided_ms = rtc_guided_probe(state, frames, cycle=1, raw=raw,
                                             roundtrip_ms=roundtrip_ms, task=args.task)
                print(f"RTC guided round trip: {guided_ms:.1f} ms; needs under "
                      f"{(RTC_REPLAN_REMAINING - 2) * 1000 / DATASET_FPS:.0f} ms to start.")
            return
        require_position_control(arm)
        gap = max(abs(float(a - b)) for a, b in zip(state[:7], HOME_POSE))
        if gap > HOME_TOLERANCE_RAD:
            raise RuntimeError(f"Follower is {gap:.3f} rad from HOME; run HOME first")
        prompt = "Clear the workspace and keep the E-stop within reach."
        if intervention:
            prompt += " Keep both grippers empty and clear. Support the Leader before pressing i, r, or h."
        input(prompt + " Press Enter for controls: ")
        motion_started = True
        hold(arm, gripper)
        cycle = 0
        episode = 0
        paused = True
        new_episode = True
        frame_after = 0
        rtc = args.rtc
        playback = RTCPlayback(args.task)
        keys = "s=start/resume, p=pause, h=HOME while paused, q=quit"
        if intervention:
            keys += ", i=intervene, r=record, e=save, d=discard"
        print(f"Paused at HOME. RTC {'ON' if rtc else 'OFF'}. Keys: {keys}.")
        aligned = False
        with Keyboard(preview_process) as keyboard:
            while True:
                key = keyboard.get()
                if key == "q":
                    break
                if intervention and intervention.recorder.active:
                    if not cameras.fresh():
                        hold(arm, gripper)
                        intervention.lock()
                        intervention.recorder.stop(save=False)
                        paused = True
                        aligned = False
                        print("Camera feed lost; correction discarded. Press s after both views recover.")
                        continue
                    if key in ("e", "d"):
                        hold(arm, gripper)
                        intervention.lock()
                        intervention.recorder.stop(save=key == "e")
                        paused = True
                        aligned = False
                        print("Correction ended. Leader locked; follower held. Press s to resume or h for HOME.")
                    else:
                        intervention.step(arm, gripper)
                    continue
                if paused:
                    if key == "s":
                        require_position_control(arm)
                        try:
                            cameras.latest_after(time.monotonic() - 0.1, timeout=1)
                        except TimeoutError:
                            print("Waiting for fresh RGB from both cameras; follower remains held.")
                            continue
                        if intervention and intervention.gc:
                            intervention.lock()
                        if new_episode:
                            state = read_state(arm, gripper)
                            gap = max(abs(float(a - b)) for a, b in zip(state[:7], HOME_POSE))
                            if gap > HOME_TOLERANCE_RAD:
                                print(f"Follower is {gap:.3f} rad from HOME. Press h before starting a new episode.")
                                continue
                            episode += 1
                            print(f"Episode {episode} started. Press p to pause.")
                        else:
                            print(f"Episode {episode} resumed from the current pose.")
                        new_episode = False
                        paused = False
                        frame_after = 0
                        playback = RTCPlayback(args.task)
                        aligned = False
                    elif key == "r" and intervention and aligned:
                        print("Support the Leader as gravity control engages.")
                        intervention.start()
                    elif key == "h":
                        homed = (intervention.home(lambda: home_follower(arm, gripper))
                                 if intervention else home_follower(arm, gripper))
                        if homed:
                            new_episode = True
                            aligned = False
                    elif key == "i" and intervention:
                        require_position_control(arm)
                        state = hold(arm, gripper)
                        playback.wait_pending()
                        intervention.align(state)
                        aligned = True
                    else:
                        time.sleep(0.02)
                    continue

                if key in ("p", "h") or (intervention and key == "i"):
                    hold(arm, gripper)
                    playback.wait_pending()
                    paused = True
                    if key == "i" and intervention:
                        intervention.align(read_state(arm, gripper))
                        aligned = True
                    else:
                        print(f"Episode {episode} paused. Press s to resume here, or h to HOME.")
                    continue
                if rtc:
                    if not cameras.fresh():
                        hold(arm, gripper)
                        playback.wait_pending()
                        paused = True
                        print("Camera feed lost; follower held. Press s after both views recover.")
                        continue
                    try:
                        playback.step(arm, gripper, cameras)
                    except TimeoutError as exc:
                        hold(arm, gripper)
                        playback.wait_pending()
                        paused = True
                        print(f"Camera pause: {exc}. Press s after both views recover.")
                    continue
                require_position_control(arm)
                try:
                    frames = cameras.latest_after(frame_after)
                except TimeoutError as exc:
                    hold(arm, gripper)
                    paused = True
                    print(f"Camera pause: {exc}. Press s after both views recover.")
                    continue
                state = read_state(arm, gripper)
                actions, _, ms, roundtrip_ms = predict(state, frames, cycle,
                                                      actions_per_chunk=actions_per_chunk,
                                                      task=args.task)
                key = keyboard.get()  # A pause pressed during inference must block the next chunk.
                if key == "q":
                    break
                if key in ("p", "h") or (intervention and key == "i"):
                    hold(arm, gripper)
                    paused = True
                    if key == "i" and intervention:
                        intervention.align(read_state(arm, gripper))
                        aligned = True
                    else:
                        print(f"Episode {episode} paused. Press s to resume here, or h to HOME.")
                    continue
                clamp_joint_actions(actions)
                last_command = execute(arm, gripper, actions, state, keyboard, cameras,
                                       dagger=intervention is not None)
                hold(arm, gripper)
                if last_command in ("p", "h", "i", "camera"):
                    paused = True
                    if last_command == "i" and intervention:
                        intervention.align(read_state(arm, gripper))
                        aligned = True
                    elif last_command == "camera":
                        print("Camera feed lost; follower held. Press s after both views recover.")
                    else:
                        print(f"Episode {episode} paused. Press s to resume here, or h to HOME.")
                    continue
                frame_after = last_command + PI05_POST_ACTION_WAIT_S
                print(f"cycle {cycle}: {len(actions)} actions, model {ms} ms, round trip {roundtrip_ms:.1f} ms")
                cycle += 1
    except KeyboardInterrupt:
        print("Stopped by operator.")
    finally:
        if intervention is not None:
            try:
                intervention.close()
            except Exception as exc:
                print(f"Could not confirm Leader lock: {exc}")
            if intervention.recorder.active:
                intervention.recorder.stop(save=False)
        if arm is not None and motion_started:
            try:
                hold(arm, gripper)
                print("Follower remains powered in position hold.")
            except Exception as exc:
                print(f"Could not confirm position hold: {exc}")
        try:
            cameras.stop()
        except KeyboardInterrupt:
            print("Finishing camera cleanup.")
        if preview_process is not None:
            if preview_process.poll() is None:
                preview_process.terminate()
            try:
                preview_process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                preview_process.kill()
                preview_process.wait()
            preview_process.stdout.close()
        if preview_map is not None:
            preview_map.close()
        if preview_path is not None:
            os.unlink(preview_path)
        if arm is not None:
            arm.disconnect()


if __name__ == "__main__":
    main()
