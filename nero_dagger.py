"""Human correction for the single-arm π0.5 runner."""

import datetime
import threading
import time
from pathlib import Path

import h5py
import numpy as np
from pyAgxArm import AgxArmFactory, ArmModel, NeroFW, create_agx_arm_config

from nero_user_config import (
    CAMERA_FPS, FOLLOWER_CHANNEL, FOLLOWER_GRIP_FORCE, FOLLOWER_MAX_M,
    FOLLOWER_MIN_M, GRIP_GAIN, GRIP_REASSERT_S, H5_COMPRESSION, HOME_POSE,
    JPEG_QUALITY, LEADER_CHANNEL, LEADER_GRIP_FORCE, LEADER_GRIP_TARGET,
    LOOP_HZ, SAVED_IMAGE_SIZE,
)


class CorrectionRecorder:
    """Write only human-controlled samples in the existing teleop HDF5 layout."""

    def __init__(self, directory, task):
        self.directory = Path(directory).expanduser().resolve()
        self.task = task
        self.directory.mkdir(parents=True, exist_ok=True)
        numbers = [int(p.name.split("_")[1]) for p in self.directory.glob("ep_*_joints.h5")
                   if p.name.split("_")[1].isdigit()]
        self.episode = max(numbers, default=0)
        self.lock = threading.Lock()
        self.active = False
        self.clear()
        print(f"Corrections: {self.directory} (next episode {self.episode + 1:04d})")

    def clear(self):
        self.t0 = None
        self.joints = []
        self.wrist = []
        self.base = []

    def start(self):
        with self.lock:
            self.clear()
            self.t0 = time.monotonic()
            self.started = datetime.datetime.now().isoformat(timespec="seconds")
            self.active = True
        print("Correction recording started. Press e to save or d to discard.")

    def frame(self, name, stamp, jpeg):
        with self.lock:
            if self.active:
                (self.wrist if name == "wrist" else self.base).append((stamp - self.t0, jpeg))

    def sample(self, leader, leader_width, follower, target):
        with self.lock:
            if self.active:
                self.joints.append((time.monotonic() - self.t0, leader.copy(),
                                    float(leader_width), follower.copy(), target.copy()))

    def stop(self, save):
        with self.lock:
            if not self.active:
                return
            self.active = False
        if not save:
            self.clear()
            print("Correction discarded.")
            return
        if len(self.joints) < LOOP_HZ or len(self.wrist) < CAMERA_FPS or len(self.base) < CAMERA_FPS:
            self.clear()
            print("Correction too short or missing camera frames; nothing saved.")
            return
        count = len(self.joints)
        self.episode += 1
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base = self.directory / f"ep_{self.episode:04d}_{stamp}"
        paths = [Path(str(base) + suffix) for suffix in
                 ("_joints.h5", "_camera.h5", "_global_camera.h5")]
        try:
            self._write_joints(paths[0])
            self._write_camera(paths[1], self.wrist, "camera")
            self._write_camera(paths[2], self.base, "global_camera")
        except Exception:
            for path in paths:
                path.unlink(missing_ok=True)
            self.episode -= 1
            raise
        finally:
            self.clear()
        print(f"Correction saved: {count} samples to {base.name}")

    def _attrs(self, file, stream):
        file.attrs["episode"] = self.episode
        file.attrs["start_wall_time"] = self.started
        file.attrs["t0_monotonic"] = self.t0
        file.attrs["task"] = self.task
        file.attrs["stream"] = stream
        file.attrs["clock"] = "time.monotonic; timestamps are seconds since t0_monotonic"

    def _write_joints(self, path):
        rows = self.joints
        kw = {"compression": H5_COMPRESSION} if H5_COMPRESSION else {}
        with h5py.File(path, "x") as file:
            file.create_dataset("timestamps", data=[row[0] for row in rows], **kw)
            leader = file.create_group("leader")
            leader.create_dataset("joint_angles", data=[row[1] for row in rows], **kw)
            leader.create_dataset("gripper", data=[row[2] for row in rows], **kw)
            follower = file.create_group("follower")
            follower.create_dataset("joint_angles", data=[row[3][:7] for row in rows], **kw)
            follower.create_dataset("gripper", data=[row[3][7] for row in rows], **kw)
            target = file.create_group("action_target")
            target.create_dataset("joint_angles", data=[row[4][:7] for row in rows], **kw)
            target.create_dataset("gripper", data=[row[4][7] for row in rows], **kw)
            self._attrs(file, "joints")
            file.attrs["target_rate_hz"] = LOOP_HZ
            file.attrs["leader_channel"] = LEADER_CHANNEL
            file.attrs["follower_channel"] = FOLLOWER_CHANNEL
            file.attrs["units"] = "joints: rad; gripper: m; timestamps: s"

    def _write_camera(self, path, rows, stream):
        with h5py.File(path, "x") as file:
            file.create_dataset("timestamps", data=[row[0] for row in rows],
                                compression=H5_COMPRESSION)
            images = file.create_dataset("color_jpeg", (len(rows),),
                                         dtype=h5py.vlen_dtype(np.uint8))
            for i, (_, jpeg) in enumerate(rows):
                images[i] = np.frombuffer(jpeg, dtype=np.uint8)
            self._attrs(file, stream)
            file.attrs["rgb_size"] = list(SAVED_IMAGE_SIZE)
            file.attrs["jpeg_quality"] = JPEG_QUALITY
            file.attrs["target_fps"] = CAMERA_FPS


class Intervention:
    def __init__(self, directory, task):
        self.recorder = CorrectionRecorder(directory, task)
        config = create_agx_arm_config(robot=ArmModel.NERO, firmeware_version=NeroFW.V121,
                                       interface="candle", channel=LEADER_CHANNEL)
        self.leader = AgxArmFactory.create_arm(config)
        self.leader.connect()
        self.gripper = self.leader.init_effector(self.leader.OPTIONS.EFFECTOR.AGX_GRIPPER)
        self.gc = False
        self.last_reassert = 0
        self.next_tick = 0

    def close(self):
        if self.gc:
            self.lock()
        self.leader.disconnect()

    def lock(self):
        if not all(self.leader.get_joints_enable_status_list()):
            self.leader.enable()
            time.sleep(0.5)
        self.leader.set_follower_mode()
        time.sleep(0.08)
        self.gc = False
        pose = self.leader.get_joint_angles()
        if pose is not None:
            self.leader.move_js(list(pose.msg))

    def align(self, follower_state):
        self.lock()
        mode_deadline = time.monotonic() + 3
        while True:
            status = self.leader.get_arm_status()
            if status is not None and status.msg.ctrl_mode == self.leader.ARM_STATUS.CtrlMode.CAN_CTRL:
                break
            if time.monotonic() >= mode_deadline:
                mode = status.msg.ctrl_mode if status is not None else "no feedback"
                raise TimeoutError(f"Leader did not enter position control: {mode}. No MoveJ sent")
            time.sleep(0.05)
        print("[align] Leader entered position control")
        pose = self.leader.get_joint_angles()
        if pose is None:
            raise RuntimeError("Cannot read locked Leader pose for alignment")
        target = follower_state[:7].tolist()
        gap = max(abs(float(a - b)) for a, b in zip(pose.msg, target))
        print(f"Aligning Leader to held follower: largest joint move {gap:.3f} rad at 50% speed.")
        print(f"[align] target={[round(x, 3) for x in target]}, "
              f"auto-mode={self.leader.get_auto_set_motion_mode_enabled()}")

        def log_feedback():
            actual = self.leader.get_joint_angles()
            status = self.leader.get_arm_status()
            if actual is None or status is None:
                print(f"[align] feedback missing: joints={actual is not None}, status={status is not None}")
                return
            remaining = max(abs(float(a - b)) for a, b in zip(actual.msg, target))
            print(f"[align] gap={remaining:.3f} rad, joints={[round(x, 3) for x in actual.msg]}, "
                  f"control={status.msg.ctrl_mode}, mode={status.msg.mode_feedback}, "
                  f"motion={status.msg.motion_status}, arm={status.msg.arm_status}, "
                  f"motors={self.leader.get_joints_enable_status_list()}")

        log_feedback()
        self.leader.set_speed_percent(50)
        self.leader.move_j(target)
        print("[align] MoveJ sent; SDK does not acknowledge target acceptance")
        deadline = time.monotonic() + 20
        next_log = time.monotonic() + 1
        while time.monotonic() < deadline:
            sample = self.leader.get_joint_angles()
            if sample is not None and max(abs(float(a - b)) for a, b in zip(sample.msg, target)) < 0.03:
                log_feedback()
                break
            if time.monotonic() >= next_log:
                log_feedback()
                next_log += 1
            time.sleep(0.05)
        else:
            log_feedback()
            raise TimeoutError("Leader did not reach follower pose")
        width = float(np.clip(follower_state[7] / GRIP_GAIN, FOLLOWER_MIN_M, FOLLOWER_MAX_M))
        print(f"Aligning empty Leader gripper to {width:.3f} m at {FOLLOWER_GRIP_FORCE:.1f} N; keep fingers clear.")
        self.gripper.move_gripper_m(value=width, force=FOLLOWER_GRIP_FORCE)
        deadline = time.monotonic() + 3
        last_width = None
        while time.monotonic() < deadline:
            actual = self.gripper.get_gripper_status()
            if actual is not None:
                last_width = actual.msg.value
            if last_width is not None and abs(last_width - width) < 0.005:
                break
            time.sleep(0.05)
        else:
            raise TimeoutError(f"Leader gripper did not align: target {width:.3f} m, "
                               f"measured {last_width} m")
        print("Leader aligned and locked. Support it, then press r to take control and record.")

    def start(self):
        self.leader.set_leader_mode()
        time.sleep(0.08)
        self.gc = True
        self.next_tick = time.monotonic()
        self.recorder.start()

    def step(self, follower, follower_gripper):
        time.sleep(max(0, self.next_tick - time.monotonic()))
        self.next_tick = time.monotonic() + 1 / LOOP_HZ
        leader = self.leader.get_leader_joint_angles()
        grip = self.gripper.get_gripper_ctrl_states()
        if leader is None or grip is None:
            raise RuntimeError("Leader feedback missing during correction")
        joints = np.asarray(leader.msg, dtype=np.float32)
        width = float(grip.msg.value)
        if joints.shape != (7,) or not np.isfinite(joints).all() or not np.isfinite(width):
            raise RuntimeError("Invalid Leader feedback during correction")
        follower_joints = follower.get_joint_angles()
        follower_width = follower_gripper.get_gripper_status()
        if follower_joints is None or follower_width is None:
            raise RuntimeError("Follower feedback missing during correction")
        state = np.asarray([*follower_joints.msg, follower_width.msg.value], dtype=np.float32)
        target_width = float(np.clip(width * GRIP_GAIN, FOLLOWER_MIN_M, FOLLOWER_MAX_M))
        target = np.asarray([*joints, target_width], dtype=np.float32)
        if follower.move_js(joints.tolist()) is False:
            raise RuntimeError("Follower rejected Leader joint target")
        if follower_gripper.move_gripper_m(value=target_width, force=FOLLOWER_GRIP_FORCE) is False:
            raise RuntimeError("Follower rejected Leader gripper target")
        if time.monotonic() - self.last_reassert >= GRIP_REASSERT_S:
            self.gripper.move_gripper_m(value=LEADER_GRIP_TARGET, force=LEADER_GRIP_FORCE)
            self.last_reassert = time.monotonic()
        self.recorder.sample(joints, width, state, target)

    def home(self, follower_home):
        self.lock()
        if not follower_home():
            return False
        self.leader.set_speed_percent(50)
        self.leader.move_j(list(HOME_POSE))
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            pose = self.leader.get_joint_angles()
            if pose is not None and max(abs(float(a - b)) for a, b in zip(pose.msg, HOME_POSE)) < 0.03:
                self.leader.move_js(list(pose.msg))
                self.gripper.move_gripper_m(value=0.0, force=FOLLOWER_GRIP_FORCE)
                print("Both arms at HOME with grippers commanded closed.")
                return True
            time.sleep(0.05)
        raise TimeoutError("Leader did not reach HOME")
