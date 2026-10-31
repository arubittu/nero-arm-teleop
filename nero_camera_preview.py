"""Separate RGB preview process for nero_apanda_mac.py."""

import mmap
import os
import re
import signal
import subprocess
import sys
import time

import cv2
import numpy as np


WRIST_SHAPE = (400, 640, 3)
FRAME_BYTES = int(np.prod(WRIST_SHAPE))
WINDOW = "NERO cameras"


def real_sense_rgb_index():
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
            capture_output=True, text=True, timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        raise SystemExit("Camera preview needs ffmpeg: brew install ffmpeg") from exc
    match = re.search(r"\[(\d+)\] Intel\(R\) RealSense.* RGB\s*$", result.stderr, re.M)
    if match is None:
        raise SystemExit("RealSense RGB is not listed by macOS. Reconnect its USB cable.")
    return int(match.group(1))


def panel(frame, label):
    out = np.zeros(WRIST_SHAPE, dtype=np.uint8)
    if frame is not None:
        h, w = frame.shape[:2]
        scale = min(640 / w, 400 / h)
        image = cv2.resize(frame, (round(w * scale), round(h * scale)),
                           interpolation=cv2.INTER_AREA)
        y = (400 - image.shape[0]) // 2
        x = (640 - image.shape[1]) // 2
        out[y:y + image.shape[0], x:x + image.shape[1]] = image
    cv2.putText(out, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(out, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def show(path):
    index = real_sense_rgb_index()
    # OpenCV's AVFoundation capture can return black frames from this D435
    # even while ffmpeg reads the RGB stream correctly.
    camera = subprocess.Popen(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "avfoundation",
         "-framerate", "30", "-video_size", "640x480", "-i", f"{index}:none",
         "-an", "-vf", "fps=15", "-pix_fmt", "bgr24", "-f", "rawvideo", "pipe:1"],
        stdout=subprocess.PIPE,
    )
    try:
        with open(path, "r+b") as frame_file:
            with mmap.mmap(frame_file.fileno(), 0, access=mmap.ACCESS_READ) as shared:
                cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(WINDOW, 1280, 400)
                while True:
                    started = time.perf_counter()
                    raw = camera.stdout.read(640 * 480 * 3)
                    if len(raw) != 640 * 480 * 3:
                        raise SystemExit("RealSense RGB stopped returning frames")
                    global_frame = np.frombuffer(raw, dtype=np.uint8).reshape(480, 640, 3)
                    slot = shared[0]
                    offset = 1 + slot * FRAME_BYTES
                    wrist = np.ndarray(WRIST_SHAPE, dtype=np.uint8,
                                       buffer=shared, offset=offset).copy()
                    view = np.hstack((panel(global_frame, "GLOBAL RGB"),
                                      panel(wrist, "WRIST RGB")))
                    cv2.imshow(WINDOW, view)
                    key = cv2.waitKey(1)
                    if 0 <= key < 256:
                        os.write(sys.stdout.fileno(), bytes([key]))
                    if key == ord("q") or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                        break
                    time.sleep(max(0, 1 / 15 - (time.perf_counter() - started)))
    finally:
        camera.terminate()
        camera.wait()
        camera.stdout.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--check":
        if not cv2.currentUIFramework():
            raise SystemExit("Camera preview needs OpenCV with window support")
        real_sense_rgb_index()
    elif len(sys.argv) == 2:
        signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
        show(sys.argv[1])
    else:
        raise SystemExit("Use --check or pass the preview buffer path")
