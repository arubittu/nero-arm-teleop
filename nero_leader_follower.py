"""Control two NERO arms through separate USB-CAN adapters.

Run ``python nero_leader_follower.py --help`` for commands and options.
Keep raised arms powered and supported; power loss removes motor support.
"""

import argparse
from contextlib import contextmanager
import math
import time

import can
from pyAgxArm import AgxArmFactory, ArmModel, NeroFW, create_agx_arm_config


HOME = [-0.056, -0.329, 0.164, 1.730, -0.001, 0.091, 0.281]


FOLLOWER_CHANNEL = "003800294148570A20343133:0"  # Original arm
LEADER_CHANNEL = "004900334148570D20343133:0"  # New arm on USB hub
SETUP_SPEED_PERCENT = 10
HOME_SPEED_PERCENT = 50
ALIGN_SPEED_PERCENT = 50
FOLLOW_SPEED_PERCENT = 70
MAX_FOLLOW_SPEED_PERCENT = 80
GRIPPER_FORCE_N = 3.0
GRIPPER_DEADBAND_M = 0.001  # Ignore sub-millimeter Leader gripper jitter.
NERO_JOINT_LIMITS = create_agx_arm_config(robot=ArmModel.NERO, firmeware_version=NeroFW.V121)["joint_limits"]
START_GAP = 0.10  # rad, largest allowed joint difference before following
MAX_TARGET_STEP = 0.15  # rad per joint from measured follower pose
PERIOD = 0.02  # Target 50 follower updates per second
ALIGN_STEP = 0.10  # rad, maximum joint change per automatic alignment waypoint
ALIGN_MIN_Z = 0.30  # m, minimum planned flange height for raised alignment
ALIGN_LEADER_TOLERANCE = 0.05  # rad, Leader must stay near its previewed pose


def connect(channel):
    found = {item["channel"] for item in can.detect_available_configs("candle")}
    if channel not in found:
        raise SystemExit(f"USB-CAN adapter {channel} is not connected.")
    config = create_agx_arm_config(
        robot=ArmModel.NERO,
        firmeware_version=NeroFW.V121,
        interface="candle",
        channel=channel,
    )
    arm = AgxArmFactory.create_arm(config)
    arm.connect()
    return arm


@contextmanager
def connected_pair():
    leader = connect(LEADER_CHANNEL)
    try:
        follower = connect(FOLLOWER_CHANNEL)
        try:
            yield leader, follower
        finally:
            follower.disconnect()
    finally:
        leader.disconnect()


def joint_gap(a, b):
    return max(abs(x - y) for x, y in zip(a, b))


def within_limits(angles, limits):
    return all(
        limit.msg.min_angle_limit <= angle <= limit.msg.max_angle_limit
        for angle, limit in zip(angles, limits)
    )


def stepped_target(desired, current, max_step):
    return [
        actual + max(-max_step, min(max_step, target - actual))
        for target, actual in zip(desired, current)
    ]


def read(arm):
    time.sleep(0.6)  # Joint feedback arrives in several CAN frames.
    status = arm.get_arm_status()
    if status is not None and status.msg.ctrl_mode == arm.ARM_STATUS.CtrlMode.CAN_CTRL:
        return "position", arm.get_joint_angles(), status
    leader_angles = arm.get_leader_joint_angles()
    if leader_angles is not None:
        return "leader", leader_angles, status
    return "unknown", None, status


def show_status():
    poses = {}
    for name, channel in (("follower", FOLLOWER_CHANNEL), ("leader", LEADER_CHANNEL)):
        arm = connect(channel)
        try:
            mode, angles, status = read(arm)
            print(f"{name}: {channel}")
            print("  mode:", "gravity (Leader mode)" if mode == "leader" else mode)
            if status is not None:
                print("  arm status:", status.msg.arm_status)
            if mode == "position":
                print("  motors enabled:", arm.get_joints_enable_status_list())
            elif mode == "leader":
                print("  motors enabled: unavailable in Leader mode")
            print("  joints (rad):", [round(x, 3) for x in angles.msg] if angles else None)
            if angles is not None:
                poses[name] = list(angles.msg)
        finally:
            arm.disconnect()
    if len(poses) == 2:
        print(f"Largest joint difference: {joint_gap(poses['leader'], poses['follower']):.3f} rad")


def move_to_pose(arm, name, target, speed_percent=SETUP_SPEED_PERCENT):
    mode, angles, status = read(arm)
    if mode != "position" or angles is None or status.msg.arm_status != arm.ARM_STATUS.ArmStatus.NORMAL:
        raise SystemExit(f"{name} is not ready for position control. No move sent.")
    if not all(arm.get_joints_enable_status_list()):
        raise SystemExit(f"{name} motors are not all enabled. No move sent.")
    for joint, angle in enumerate(target, start=1):
        limit = arm.get_joint_angle_vel_limits(joint)
        if limit is None or not limit.msg.min_angle_limit <= angle <= limit.msg.max_angle_limit:
            raise SystemExit(f"{name} joint {joint} target is outside its limits.")

    arm.set_speed_percent(speed_percent)
    arm.move_j(target)
    best_error = joint_gap(angles.msg, target)
    last_progress = time.monotonic()
    retried = False
    for second in range(90):
        time.sleep(1)
        actual = arm.get_joint_angles()
        status = arm.get_arm_status()
        if status is not None and status.msg.arm_status != arm.ARM_STATUS.ArmStatus.NORMAL:
            raise SystemExit(f"{name} reports {status.msg.arm_status}. Use the E-stop if unsafe.")
        if actual is None:
            continue
        error = joint_gap(actual.msg, target)
        if second % 2 == 0:
            print(f"{name} largest joint error: {error:.3f} rad")
        if error < 0.03:
            print(f"{name} reached its target.")
            return
        if error < best_error - 0.001:
            best_error = error
            last_progress = time.monotonic()
        elif time.monotonic() - last_progress >= 5:
            motion_done = (
                status is not None
                and status.msg.motion_status == arm.ARM_STATUS.MotionStatus.REACH_TARGET_POS_SUCCESSFULLY
            )
            if retried or not motion_done:
                raise SystemExit(f"{name} stopped progressing. No further waypoints sent.")
            print(f"{name} target was not reached; retrying this waypoint once.")
            arm.move_j(target)
            retried = True
            last_progress = time.monotonic()
    raise SystemExit(f"{name} did not reach its target in 90 seconds. Keep power on.")


def plan_alignment(arm, start, target):
    """Check a raised joint path before sending any alignment command."""
    limits = [arm.get_joint_angle_vel_limits(i) for i in range(1, 8)]
    if any(limit is None for limit in limits):
        raise SystemExit("Could not read follower joint limits. No move sent.")
    count = max(1, math.ceil(joint_gap(start, target) / ALIGN_STEP))
    waypoints = [
        [a + (b - a) * step / count for a, b in zip(start, target)]
        for step in range(count + 1)
    ]
    minimum_z = float("inf")
    for pose in waypoints:
        if not within_limits(pose, limits):
            raise SystemExit("Alignment path exceeds a follower joint limit. No move sent.")
        z = arm.fk(pose)[2]
        if not math.isfinite(z) or z < ALIGN_MIN_Z:
            raise SystemExit("Alignment flange path is too low or unknown. No move sent.")
        minimum_z = min(minimum_z, z)
    return waypoints[1:], minimum_z


def gripper_alignment_state(leader, follower):
    leader_gripper = leader.init_effector(leader.OPTIONS.EFFECTOR.AGX_GRIPPER)
    follower_gripper = follower.init_effector(follower.OPTIONS.EFFECTOR.AGX_GRIPPER)
    time.sleep(0.2)
    leader_grip = leader_gripper.get_gripper_ctrl_states()
    follower_grip = follower_gripper.get_gripper_status()
    config = follower_gripper.get_gripper_teaching_pendant_param()
    if leader_grip is None or follower_grip is None or config is None:
        raise SystemExit("Gripper feedback is missing. No move sent.")
    if leader_grip.msg.status_code != 1 or follower_grip.msg.mode != "width":
        raise SystemExit("Grippers are not both in width mode. No move sent.")
    if not follower_grip.msg.foc_status.driver_enable_status:
        raise SystemExit("Follower gripper is disabled. No move sent.")
    target = leader_grip.msg.value
    if not 0 <= target <= config.msg.max_range_config:
        raise SystemExit("Leader gripper opening exceeds follower range. No move sent.")
    return leader_gripper, follower_gripper, target, follower_grip.msg.value


def align_gripper(leader_gripper, follower_gripper, target):
    current = leader_gripper.get_gripper_ctrl_states()
    if current is None or current.msg.status_code != 1 or abs(current.msg.value - target) > 0.005:
        raise SystemExit("Leader gripper changed during alignment. No gripper move sent.")
    input("Clear both grippers of fingers and objects. Press Enter to match their opening: ")
    follower_gripper.move_gripper_m(target, force=GRIPPER_FORCE_N)
    for _ in range(50):
        time.sleep(0.1)
        actual = follower_gripper.get_gripper_status()
        if actual is None or actual.msg.mode != "width" or not actual.msg.foc_status.driver_enable_status:
            raise SystemExit("Follower gripper feedback stopped. Check the gripper.")
        if abs(actual.msg.value - target) <= 0.005:
            print(f"Grippers aligned at {actual.msg.value:.3f} m.")
            return
    raise SystemExit("Follower gripper did not reach the Leader opening in 5 seconds.")


def align_to_leader(preview=False):
    """Match the raised follower and its gripper to the still Leader."""
    with connected_pair() as (leader, follower):
        leader_mode, leader_msg, _ = read(leader)
        follower_mode, follower_msg, follower_status = read(follower)
        if leader_mode != "leader" or leader_msg is None or follower_mode != "position" or follower_msg is None:
            raise SystemExit("Run 'home' to prepare the Leader and follower modes first.")
        if follower_status.msg.arm_status != follower.ARM_STATUS.ArmStatus.NORMAL:
            raise SystemExit("Follower is not in normal status. No move sent.")
        if not all(follower.get_joints_enable_status_list()):
            raise SystemExit("Follower motors are not all enabled. No move sent.")

        leader_gripper, follower_gripper, grip_target, grip_actual = gripper_alignment_state(leader, follower)
        start, target = list(follower_msg.msg), list(leader_msg.msg)
        waypoints, minimum_z = plan_alignment(follower, start, target)
        print("Follower:", [round(x, 3) for x in start])
        print("Leader:  ", [round(x, 3) for x in target])
        print(f"Gripper: follower {grip_actual:.3f} m -> Leader {grip_target:.3f} m")
        print(f"Plan: {len(waypoints)} follower steps at {ALIGN_SPEED_PERCENT}% speed; "
              f"lowest modeled flange height {minimum_z:.3f} m.")
        if preview:
            return

        input("Check the full arm, tool and cable sweep; keep the Leader still and E-stop close. Press Enter to align: ")
        current = leader.get_leader_joint_angles()
        if current is None or joint_gap(current.msg, target) > ALIGN_LEADER_TOLERANCE:
            raise SystemExit("Leader moved since preview. No move sent.")
        for waypoint in waypoints:
            current = leader.get_leader_joint_angles()
            if current is None or joint_gap(current.msg, target) > ALIGN_LEADER_TOLERANCE:
                raise SystemExit("Leader moved. Alignment stopped; follower may finish its last small step.")
            move_to_pose(follower, "Follower", waypoint, ALIGN_SPEED_PERCENT)
        actual = follower.get_joint_angles()
        leader_actual = leader.get_leader_joint_angles()
        if actual is None or leader_actual is None or joint_gap(actual.msg, leader_actual.msg) > START_GAP:
            raise SystemExit("Arms did not finish aligned. No gripper move sent.")
        align_gripper(leader_gripper, follower_gripper, grip_target)
        print("Arms and grippers aligned. Run 'run' to follow.")


def home_both(execute=False):
    """Move both raised arms to HOME, then restore the new arm to Leader mode."""
    with connected_pair() as (leader, follower):
        leader_mode, leader_msg, leader_status = read(leader)
        follower_mode, follower_msg, follower_status = read(follower)
        if leader_mode not in ("leader", "position") or leader_msg is None or follower_mode != "position" or follower_msg is None:
            raise SystemExit("Leader or follower mode is unknown. No move sent.")
        if leader_mode == "position":
            if leader_status.msg.arm_status != leader.ARM_STATUS.ArmStatus.NORMAL:
                raise SystemExit("Leader is not in normal status. No move sent.")
            if not all(leader.get_joints_enable_status_list()):
                raise SystemExit("Leader motors are off. No move sent.")
        if follower_status.msg.arm_status != follower.ARM_STATUS.ArmStatus.NORMAL:
            raise SystemExit("Follower is not in normal status. No move sent.")
        if not all(follower.get_joints_enable_status_list()):
            raise SystemExit("Follower motors are off. No move sent.")

        leader_start = list(leader_msg.msg)
        follower_start = list(follower_msg.msg)
        leader_path, leader_z = plan_alignment(follower, leader_start, HOME)
        follower_path, follower_z = plan_alignment(follower, follower_start, HOME)
        print("Leader:  ", [round(x, 3) for x in leader_start])
        print("Follower:", [round(x, 3) for x in follower_start])
        print("HOME:    ", HOME)
        print(f"Leader first: {len(leader_path)} path samples checked, minimum modeled flange height {leader_z:.3f} m")
        print(f"Follower next: {len(follower_path)} path samples checked, minimum modeled flange height {follower_z:.3f} m")
        print(f"One HOME target per arm at {HOME_SPEED_PERCENT}% speed.")
        if not execute:
            print("Preview only. Run 'home' without --preview to move.")
            return
        action = "switch Leader to position control" if leader_mode == "leader" else "move Leader toward HOME"
        input(f"Support the Leader and check both full sweeps. Keep the E-stop close. Press Enter to {action}: ")

        current = leader.get_leader_joint_angles() if leader_mode == "leader" else leader.get_joint_angles()
        if current is None or joint_gap(current.msg, leader_start) > ALIGN_LEADER_TOLERANCE:
            raise SystemExit("Leader moved since preview. No mode switch sent.")
        if leader_mode == "leader":
            leader.set_follower_mode()
            if not leader.enable():
                raise SystemExit("Leader did not enable in position control. Keep it supported.")
            leader.set_speed_percent(SETUP_SPEED_PERCENT)
            leader.move_j(leader_start)
            mode, actual, status = read(leader)
            if mode != "position" or actual is None or status.msg.arm_status != leader.ARM_STATUS.ArmStatus.NORMAL:
                raise SystemExit("Leader position control was not verified. Keep it supported.")
            if joint_gap(actual.msg, leader_start) > START_GAP:
                raise SystemExit("Leader shifted during its mode switch. No HOME move sent.")
        else:
            actual = current
        plan_alignment(leader, list(actual.msg), HOME)
        move_to_pose(leader, "Leader", HOME, HOME_SPEED_PERCENT)

        input("Leader reached HOME. Keep the follower's full sweep clear. Press Enter to move the follower: ")
        actual = follower.get_joint_angles()
        if actual is None or joint_gap(actual.msg, follower_start) > ALIGN_LEADER_TOLERANCE:
            raise SystemExit("Follower shifted since preview. No follower HOME move sent.")
        plan_alignment(follower, list(actual.msg), HOME)
        move_to_pose(follower, "Follower", HOME, HOME_SPEED_PERCENT)

        input("Support the raised Leader. Press Enter to return it to Leader mode: ")
        leader.set_leader_mode()
        for _ in range(20):
            time.sleep(0.1)
            leader_msg = leader.get_leader_joint_angles()
            if leader_msg is not None:
                break
        follower_msg = follower.get_joint_angles()
        if leader_msg is None or follower_msg is None or joint_gap(leader_msg.msg, follower_msg.msg) > START_GAP:
            raise SystemExit("Arms are not aligned after the mode switch. Do not start following.")
        try:
            leader_gripper, follower_gripper, grip_target, grip_actual = gripper_alignment_state(leader, follower)
        except SystemExit as exc:
            print(f"Both arms are at HOME, but gripper alignment is unavailable: {exc}")
            return
        if abs(grip_target - grip_actual) > 0.005:
            align_gripper(leader_gripper, follower_gripper, grip_target)
        print("Both arms at HOME and aligned. Run 'run' to follow.")


def plan_restore_path(arm, start):
    """Plan a rising path from a supported low resting pose."""
    limits = [arm.get_joint_angle_vel_limits(i) for i in range(1, 8)]
    if any(limit is None for limit in limits):
        raise SystemExit("Could not read joint limits. No move sent.")
    if any(
        not limit.msg.min_angle_limit <= angle <= limit.msg.max_angle_limit
        for index, (angle, limit) in enumerate(zip(start, limits)) if index != 1
    ):
        raise SystemExit("A resting joint other than J2 is outside its limit. No move sent.")
    j2_max = limits[1].msg.max_angle_limit
    if not 1.5 <= start[1] <= j2_max + 0.15:
        raise SystemExit("Resting J2 is outside the supported low-pose range. No move sent.")
    first = start.copy()
    first[1] = min(first[1], j2_max - 0.02)
    count = max(1, math.ceil(joint_gap(first, HOME) / ALIGN_STEP))
    path = [first] + [
        [a + (b - a) * step / count for a, b in zip(first, HOME)]
        for step in range(1, count + 1)
    ]
    start_z = arm.fk(start)[2]
    if not math.isfinite(start_z) or not -0.02 <= start_z <= 0.15:
        raise SystemExit("Arm is not at a low resting flange height. No move sent.")
    minimum_z = start_z
    for pose in path:
        if not within_limits(pose, limits):
            raise SystemExit("Restoration path exceeds a joint limit. No move sent.")
        z = arm.fk(pose)[2]
        if not math.isfinite(z) or z < start_z - 0.005:
            raise SystemExit("Restoration path dips below its starting height. No move sent.")
        minimum_z = min(minimum_z, z)
    return path, minimum_z


def restore_home(execute=False, leader_at_home=False):
    """Raise disabled arms from supported low table rests to HOME."""
    with connected_pair() as (leader, follower):
        plans = []
        for name, arm in (
            ("Leader", leader),
            ("Follower", follower),
        ):
            mode, angles, status = read(arm)
            if mode != "position" or angles is None or status is None:
                raise SystemExit(f"{name} is not reporting position mode. No move sent.")
            if name == "Leader" and leader_at_home:
                if (status.msg.arm_status != arm.ARM_STATUS.ArmStatus.NORMAL
                        or not all(arm.get_joints_enable_status_list())
                        or joint_gap(angles.msg, HOME) > 0.05):
                    raise SystemExit("Leader is no longer enabled at HOME. No move sent.")
                print("Leader is already enabled at HOME.")
                continue
            if any(arm.get_joints_enable_status_list()):
                raise SystemExit(f"{name} motors are already on. No move sent.")
            if status.msg.arm_status != arm.ARM_STATUS.ArmStatus.JOINT_BRAKE_NOT_RELEASED:
                raise SystemExit(f"{name} reports {status.msg.arm_status}. No move sent.")
            start = list(angles.msg)
            try:
                path, minimum_z = plan_restore_path(arm, start)
            except SystemExit as exc:
                raise SystemExit(f"{name}: {exc}") from exc
            print(f"{name}: {[round(x, 3) for x in start]}")
            print(f"  {len(path)} path samples checked; minimum modeled flange height {minimum_z:.3f} m")
            plans.append((name, arm, start, path))
        print("First target brings any over-limit J2 back inside its limit at 10% speed.")
        print(f"Then one HOME target per moving arm at {HOME_SPEED_PERCENT}% speed.")
        if not execute:
            print("Preview only. Run 'home' without --preview to enable and move the arms.")
            return

        for name, arm, start, path in plans:
            input(f"Support the resting {name}; clear its full arm, gripper, and cable sweep. Keep the E-stop close. Press Enter to enable and raise it: ")
            current = arm.get_joint_angles()
            if current is None or joint_gap(current.msg, start) > 0.05:
                raise SystemExit(f"{name} shifted since preview. No enable sent.")
            arm.set_speed_percent(SETUP_SPEED_PERCENT)
            arm.move_j(path[0])  # Load an in-limit, rising target before enabling motors.
            if not arm.enable() or not all(arm.get_joints_enable_status_list()):
                raise SystemExit(f"{name} did not enable. Keep it supported.")
            mode, actual, status = read(arm)
            if mode != "position" or actual is None or status.msg.arm_status != arm.ARM_STATUS.ArmStatus.NORMAL:
                raise SystemExit(f"{name} position control is not ready. Keep it supported.")
            if joint_gap(actual.msg, start) > 0.15:
                raise SystemExit(f"{name} shifted unexpectedly while enabling. No HOME targets sent.")
            move_to_pose(arm, name, path[0], SETUP_SPEED_PERCENT)
            current = arm.get_joint_angles()
            if current is None:
                raise SystemExit(f"{name} feedback stopped. No HOME target sent.")
            plan_restore_path(arm, list(current.msg))
            move_to_pose(arm, name, HOME, HOME_SPEED_PERCENT)

        leader_actual = leader.get_joint_angles()
        if (leader_actual is None or joint_gap(leader_actual.msg, HOME) > 0.05
                or not all(leader.get_joints_enable_status_list())):
            raise SystemExit("Leader is not holding HOME. No mode switch sent.")
        input("Support the raised Leader. Press Enter to switch it to Leader mode: ")
        leader.set_leader_mode()
        leader_mode, leader_msg, _ = read(leader)
        follower_mode, follower_msg, _ = read(follower)
        if (leader_mode != "leader" or leader_msg is None or follower_mode != "position"
                or follower_msg is None or joint_gap(leader_msg.msg, follower_msg.msg) > START_GAP):
            raise SystemExit("Arms did not finish aligned. Keep them powered and supported.")
        print("Both arms are aligned at HOME.")
        try:
            leader_gripper, follower_gripper, grip_target, grip_actual = gripper_alignment_state(leader, follower)
        except SystemExit as exc:
            print(f"Gripper alignment unavailable: {exc}")
            print("Check gripper feedback before running 'run'.")
            return
        if abs(grip_target - grip_actual) > 0.005:
            align_gripper(leader_gripper, follower_gripper, grip_target)
        print("Arms and grippers are aligned at HOME. Run 'run' to follow.")


def home(preview=False):
    """Select the saved-rest or raised route to HOME from live feedback."""
    with connected_pair() as (leader, follower):
        leader_mode, leader_msg, _ = read(leader)
        follower_mode, follower_msg, _ = read(follower)
        if leader_msg is None or follower_msg is None or follower_mode != "position":
            raise SystemExit("Arm feedback or follower mode is unavailable. No move sent.")
        follower_enabled = all(follower.get_joints_enable_status_list())
        leader_enabled = all(leader.get_joints_enable_status_list()) if leader_mode == "position" else None

    if leader_mode == "position" and leader_enabled is False and not follower_enabled:
        restore_home(execute=not preview)
    elif (leader_mode == "position" and leader_enabled and not follower_enabled
            and joint_gap(leader_msg.msg, HOME) <= 0.05):
        restore_home(execute=not preview, leader_at_home=True)
    elif leader_mode in ("leader", "position") and follower_enabled:
        home_both(execute=not preview)
    else:
        raise SystemExit("Unsupported mix of modes or motor states. Run 'status'; no move sent.")


def set_arm_mode(arm_name, target_mode):
    """Switch one enabled arm between hand-guided and position control."""
    channel = LEADER_CHANNEL if arm_name == "leader" else FOLLOWER_CHANNEL
    arm = connect(channel)
    try:
        mode, angles, status = read(arm)
        if mode not in ("leader", "position") or angles is None:
            raise SystemExit(f"{arm_name} feedback is unavailable. No mode change sent.")
        if mode == "position":
            if not all(arm.get_joints_enable_status_list()):
                raise SystemExit(f"{arm_name} motors are off. Run 'home' to enable from rest.")
            if status is None or status.msg.arm_status != arm.ARM_STATUS.ArmStatus.NORMAL:
                raise SystemExit(f"{arm_name} is not in normal status. No mode change sent.")
        if (mode == "leader" and target_mode == "gravity") or (mode == "position" and target_mode == "position"):
            print(f"{arm_name} is already in {target_mode} control.")
            return

        target = list(angles.msg)
        if target_mode == "position":
            for joint, angle in enumerate(target, 1):
                low, high = NERO_JOINT_LIMITS[f"joint{joint}"]
                if not low <= angle <= high:
                    raise SystemExit(
                        f"{arm_name} J{joint} is {angle:.3f} rad, outside {low:.3f} to {high:.3f} rad. "
                        "Guide it inside before position control."
                    )
        print(f"{arm_name} joints (rad): {[round(x, 3) for x in target]}")
        input(f"Support the {arm_name} and keep its path clear. Press Enter for {target_mode} control: ")
        current = arm.get_leader_joint_angles() if mode == "leader" else arm.get_joint_angles()
        if current is None or joint_gap(current.msg, target) > 0.05:
            raise SystemExit(f"{arm_name} shifted. No mode change sent.")

        if target_mode == "gravity":
            arm.set_leader_mode()
            for _ in range(20):
                time.sleep(0.1)
                actual = arm.get_leader_joint_angles()
                if actual is not None:
                    break
            if actual is None:
                raise SystemExit(f"{arm_name} gravity feedback did not start. Keep it supported.")
            if joint_gap(actual.msg, target) > 0.05:
                raise SystemExit(f"{arm_name} shifted during the mode switch. Keep it supported.")
        else:
            arm.set_follower_mode()
            arm.set_speed_percent(SETUP_SPEED_PERCENT)
            arm.move_j(target)
            if not all(arm.get_joints_enable_status_list()) and not arm.enable():
                raise SystemExit(f"{arm_name} motors did not enable in position control. Keep it supported.")
            new_mode, actual, new_status = read(arm)
            if (new_mode != "position" or actual is None or new_status is None
                    or new_status.msg.arm_status != arm.ARM_STATUS.ArmStatus.NORMAL
                    or not all(arm.get_joints_enable_status_list())):
                raise SystemExit(f"{arm_name} position control was not verified. Keep it supported.")
            if joint_gap(actual.msg, target) > 0.05:
                raise SystemExit(f"{arm_name} shifted during the mode switch. Keep it supported.")
        print(f"{arm_name} is in {target_mode} control.")
    finally:
        arm.disconnect()


def release():
    """Turn off motors only after the operator confirms full support."""
    with connected_pair() as (leader, follower):
        leader_mode, leader_msg, _ = read(leader)
        follower_mode, follower_msg, _ = read(follower)
        if leader_mode not in ("leader", "position") or leader_msg is None or follower_mode != "position" or follower_msg is None:
            raise SystemExit("Arm feedback or modes are unavailable. No motors disabled.")
        follower_flags = follower.get_joints_enable_status_list()
        leader_flags = leader.get_joints_enable_status_list() if leader_mode == "position" else None
        if leader_flags == [False] * 7 and follower_flags == [False] * 7:
            print("Both arms are already disabled.")
            return
        if (any(follower_flags) and not all(follower_flags)) or (leader_flags is not None and any(leader_flags) and not all(leader_flags)):
            raise SystemExit("A motor is only partly enabled. No motors disabled.")
        for name, arm, angles in (("Leader", leader, leader_msg), ("Follower", follower, follower_msg)):
            z = arm.fk(list(angles.msg))[2]
            print(f"{name}: {[round(x, 3) for x in angles.msg]}, modeled flange height {z:.3f} m")
            if not math.isfinite(z) or z > 0.15:
                raise SystemExit(f"{name} is not at a low resting pose. No motors disabled.")
        input("Both arms fully supported? Press Enter to release the follower: ")

        current = follower.get_joint_angles()
        if current is None or joint_gap(current.msg, follower_msg.msg) > 0.05:
            raise SystemExit("Follower shifted. No motors disabled.")
        if any(follower_flags):
            follower.disable()
            time.sleep(0.6)
            if any(follower.get_joints_enable_status_list()):
                raise SystemExit("Follower motors did not all disable. Keep it supported.")
            print("Follower motors disabled.")
        input("Follower settled? Press Enter to release the Leader: ")

        if leader_mode == "leader":
            current = leader.get_leader_joint_angles()
            if current is None or joint_gap(current.msg, leader_msg.msg) > 0.05:
                raise SystemExit("Leader shifted. No Leader motor change sent.")
            target = list(current.msg)
            leader.set_follower_mode()
            if not all(leader.get_joints_enable_status_list()) and not leader.enable():
                raise SystemExit("Leader did not enter position control. Keep it supported.")
            leader.set_speed_percent(SETUP_SPEED_PERCENT)
            leader.move_j(target)
            mode, actual, status = read(leader)
            if mode != "position" or actual is None or status.msg.arm_status != leader.ARM_STATUS.ArmStatus.NORMAL:
                raise SystemExit("Leader position control was not verified. Keep it supported.")
            if not all(leader.get_joints_enable_status_list()):
                raise SystemExit("Leader motors are not all enabled in position control. Keep it supported.")
            if joint_gap(actual.msg, target) > 0.05:
                raise SystemExit("Leader shifted during mode switch. Keep it supported.")
        elif any(leader_flags):
            current = leader.get_joint_angles()
            if current is None or joint_gap(current.msg, leader_msg.msg) > 0.05:
                raise SystemExit("Leader shifted. No Leader motors disabled.")

        if leader_mode == "leader" or any(leader_flags):
            leader.disable()
            time.sleep(0.6)
            if any(leader.get_joints_enable_status_list()):
                raise SystemExit("Leader motors did not all disable. Keep it supported.")
            print("Leader motors disabled.")
        input("Leader settled? Press Enter to finish: ")
        if any(follower.get_joints_enable_status_list()) or any(leader.get_joints_enable_status_list()):
            raise SystemExit("A motor still reports enabled. Keep both arms supported.")
        print("Both arms are supported and disabled. You can switch off their power.")


def check_follow_start(leader, follower, leader_gripper, follower_gripper):
    leader_mode, leader_msg, _ = read(leader)
    follower_mode, follower_msg, status = read(follower)
    if leader_mode != "leader" or leader_msg is None:
        raise SystemExit("New arm is not in Leader mode. Run 'home' first.")
    if follower_mode != "position" or follower_msg is None:
        raise SystemExit("Original arm is not in position control. Run 'home' first.")
    if status.msg.arm_status != follower.ARM_STATUS.ArmStatus.NORMAL:
        raise SystemExit(f"Follower reports {status.msg.arm_status}. No motion sent.")
    if not all(follower.get_joints_enable_status_list()):
        raise SystemExit("Follower motors are not all enabled. No motion sent.")

    limits = [follower.get_joint_angle_vel_limits(i) for i in range(1, 8)]
    if any(limit is None for limit in limits):
        raise SystemExit("Could not read all follower joint limits. No motion sent.")
    if joint_gap(leader_msg.msg, follower_msg.msg) > START_GAP:
        raise SystemExit(
            f"Starting poses differ by more than {START_GAP} rad. "
            "Guide the new Leader closer to the follower first; no motion sent."
        )
    leader_grip = leader_gripper.get_gripper_ctrl_states()
    follower_grip = follower_gripper.get_gripper_status()
    gripper_config = follower_gripper.get_gripper_teaching_pendant_param()
    if any(x is None for x in (leader_grip, follower_grip, gripper_config)):
        raise SystemExit("Gripper feedback is missing. No motion sent.")
    if leader_grip.msg.status_code != 1 or follower_grip.msg.mode != "width":
        raise SystemExit("Grippers are not both in width mode. No motion sent.")
    if not follower_grip.msg.foc_status.driver_enable_status:
        raise SystemExit("Follower gripper is disabled. No motion sent.")
    max_width = gripper_config.msg.max_range_config
    if not 0 <= leader_grip.msg.value <= max_width:
        raise SystemExit("Leader gripper width exceeds the follower limit. No motion sent.")
    return leader_msg, follower_msg, leader_grip, follower_grip, limits, max_width


def follow(speed_percent=FOLLOW_SPEED_PERCENT, max_target_step=MAX_TARGET_STEP):
    leader = connect(LEADER_CHANNEL)
    try:
        follower = connect(FOLLOWER_CHANNEL)
        try:
            leader_gripper = leader.init_effector(leader.OPTIONS.EFFECTOR.AGX_GRIPPER)
            follower_gripper = follower.init_effector(follower.OPTIONS.EFFECTOR.AGX_GRIPPER)
            leader_msg, follower_msg, leader_grip, follower_grip, limits, max_width = check_follow_start(
                leader, follower, leader_gripper, follower_gripper
            )

            print("Leader:  ", [round(x, 3) for x in leader_msg.msg])
            print("Follower:", [round(x, 3) for x in follower_msg.msg])
            print(f"Grippers: leader {leader_grip.msg.value:.3f} m, follower {follower_grip.msg.value:.3f} m")
            print(f"J control: {speed_percent}% speed, at most {max_target_step:.3f} rad target error per joint")
            input("Clear both paths and grippers. Keep the physical E-stop close. Press Enter to follow: ")
            follower.set_speed_percent(speed_percent)

            last_leader_stamp = leader_msg.timestamp
            last_follower_stamp = follower_msg.timestamp
            last_gripper_stamp = leader_grip.timestamp
            last_leader_update = last_follower_update = time.monotonic()
            last_gripper_update = time.monotonic()
            last_gripper_command = None
            last_print = next_tick = time.perf_counter()
            cycles = late_ticks = 0
            worst_cycle_ms = 0.0
            while True:
                cycle_start = time.perf_counter()
                leader_msg = leader.get_leader_joint_angles()
                follower_msg = follower.get_joint_angles()
                leader_grip = leader_gripper.get_gripper_ctrl_states()
                follower_grip = follower_gripper.get_gripper_status()
                status = follower.get_arm_status()
                now = time.monotonic()
                if any(x is None for x in (leader_msg, follower_msg, leader_grip, follower_grip, status)):
                    raise SystemExit("Feedback lost. Following stopped; support the follower if raised.")
                if leader_msg.timestamp != last_leader_stamp:
                    last_leader_stamp = leader_msg.timestamp
                    last_leader_update = now
                if follower_msg.timestamp != last_follower_stamp:
                    last_follower_stamp = follower_msg.timestamp
                    last_follower_update = now
                if leader_grip.timestamp != last_gripper_stamp:
                    last_gripper_stamp = leader_grip.timestamp
                    last_gripper_update = now
                if now - min(last_leader_update, last_follower_update, last_gripper_update) > 0.5:
                    raise SystemExit("Feedback is stale. Following stopped; support the follower if raised.")
                if status.msg.arm_status != follower.ARM_STATUS.ArmStatus.NORMAL:
                    raise SystemExit(f"Follower reports {status.msg.arm_status}. Following stopped.")
                if not follower_grip.msg.foc_status.driver_enable_status:
                    raise SystemExit("Follower gripper is disabled. Following stopped.")
                gap = joint_gap(leader_msg.msg, follower_msg.msg)
                if not within_limits(leader_msg.msg, limits):
                    raise SystemExit("Leader angle exceeds a follower joint limit. Following stopped.")
                if leader_grip.msg.status_code != 1 or not 0 <= leader_grip.msg.value <= max_width:
                    raise SystemExit("Leader gripper command is outside the follower width range. Following stopped.")

                follower.move_j(stepped_target(leader_msg.msg, follower_msg.msg, max_target_step))
                grip_target = leader_grip.msg.value
                if last_gripper_command is None or abs(grip_target - last_gripper_command) >= GRIPPER_DEADBAND_M:
                    follower_gripper.move_gripper_m(grip_target, force=GRIPPER_FORCE_N)
                    last_gripper_command = grip_target
                cycles += 1
                cycle_end = time.perf_counter()
                worst_cycle_ms = max(worst_cycle_ms, (cycle_end - cycle_start) * 1000)
                if cycle_end - last_print >= 2:
                    oldest_feedback_ms = (now - min(last_leader_update, last_follower_update, last_gripper_update)) * 1000
                    print(
                        f"Following; joint gap: {gap:.3f} rad, "
                        f"gripper: {follower_grip.msg.value:.3f}/{leader_grip.msg.value:.3f} m, "
                        f"rate: {cycles / (cycle_end - last_print):.1f} Hz, "
                        f"late: {late_ticks}, worst cycle: {worst_cycle_ms:.1f} ms, "
                        f"oldest feedback: {oldest_feedback_ms:.0f} ms"
                    )
                    last_print = cycle_end
                    cycles = late_ticks = 0
                    worst_cycle_ms = 0.0
                next_tick += PERIOD
                slack = next_tick - time.perf_counter()
                if slack > 0:
                    time.sleep(slack)
                else:
                    late_ticks += 1
                    next_tick = time.perf_counter()
        finally:
            follower.disconnect()
    finally:
        leader.disconnect()


def main():
    parser = argparse.ArgumentParser(description="Control two NERO arms through separate USB-CAN adapters.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("status", help="Read both modes, motor states, and joint angles")
    home_parser = subparsers.add_parser("home", help="Align both arms at HOME")
    home_parser.add_argument("--preview", action="store_true", help="check the route without moving")
    align_parser = subparsers.add_parser("align", help="Match the raised follower and gripper to the Leader")
    align_parser.add_argument("--preview", action="store_true", help="check the route without moving")
    mode_parser = subparsers.add_parser("mode", help="Switch an enabled arm between gravity and position control")
    mode_parser.add_argument("arm", choices=("leader", "follower"))
    mode_parser.add_argument("control", choices=("gravity", "position"))
    run_parser = subparsers.add_parser("run", help="Follow the Leader joints and gripper")
    run_parser.add_argument(
        "--speed", type=int, default=FOLLOW_SPEED_PERCENT, metavar="PERCENT",
        help=f"follower speed percent (default: {FOLLOW_SPEED_PERCENT}; max: {MAX_FOLLOW_SPEED_PERCENT})",
    )
    subparsers.add_parser("release", help="Disable both arms after confirming full support")
    args = parser.parse_args()
    if args.command == "status":
        show_status()
    elif args.command == "home":
        home(args.preview)
    elif args.command == "align":
        align_to_leader(args.preview)
    elif args.command == "mode":
        set_arm_mode(args.arm, args.control)
    elif args.command == "run":
        if not 1 <= args.speed <= MAX_FOLLOW_SPEED_PERCENT:
            parser.error(f"--speed must be between 1 and {MAX_FOLLOW_SPEED_PERCENT}")
        try:
            follow(speed_percent=args.speed)
        except KeyboardInterrupt:
            print("\nFollowing stopped.")
    else:
        release()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped. Motors remain powered; the follower may finish its last target.")
