#!/usr/bin/env python3
"""π0.5 inference for the Mac NERO runner, with optional RTC guidance."""

import argparse
import base64
import io
import json
import os
import threading
import time
from collections import Counter
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, required=True, help="CUDA GPU index, such as 1")
    parser.add_argument("--checkpoint", type=Path, required=True, help="local pretrained_model directory")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    return parser.parse_args()


RUN_ARGS = parse_args() if __name__ == "__main__" else None

os.environ["HF_HOME"] = "/mnt/storage/aryan/nero_pi05/cache/huggingface"
os.environ["HF_HUB_OFFLINE"] = "1"
print("[startup 1/4] Importing PyTorch and LeRobot...", flush=True)
_imports_start = time.perf_counter()
import numpy as np
import torch
from PIL import Image
from safetensors import safe_open

from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.configs import PreTrainedConfig, RTCAttentionSchedule
print(f"[startup 1/4] Imports ready in {time.perf_counter() - _imports_start:.1f}s", flush=True)

RTC_EXECUTION_HORIZON = 35


def load_step(label, operation):
    print(label, flush=True)
    start = time.perf_counter()
    done = threading.Event()

    def heartbeat():
        while not done.wait(10):
            print(f"  Still loading ({time.perf_counter() - start:.0f}s elapsed)...", flush=True)

    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    try:
        result = operation()
    finally:
        done.set()
        thread.join()
    print(f"  Done in {time.perf_counter() - start:.1f}s", flush=True)
    return result


def decode_image(value):
    image = Image.open(io.BytesIO(base64.b64decode(value, validate=True))).convert("RGB")
    if image.size != (224, 224):
        raise ValueError(f"Expected 224x224 RGB, got {image.size}")
    return np.asarray(image, dtype=np.uint8).copy()


class Predictor:
    def __init__(self, checkpoint):
        weights = checkpoint / "model.safetensors"
        if not weights.is_file():
            raise FileNotFoundError(f"Local checkpoint weights missing: {weights}")
        print(f"[startup] Local checkpoint: {weights} ({weights.stat().st_size / 1e9:.1f} GB). Downloads off.", flush=True)
        config = PreTrainedConfig.from_pretrained(checkpoint, local_files_only=True)
        config.rtc_config = RTCConfig(
            enabled=True, execution_horizon=RTC_EXECUTION_HORIZON, max_guidance_weight=10.0,
            prefix_attention_schedule=RTCAttentionSchedule.EXP,
        )
        self.policy = load_step(
            "[startup 2/4] Building π0.5 and loading checkpoint weights...",
            lambda: PI05Policy.from_pretrained(checkpoint, config=config, local_files_only=True),
        )
        key = "model.action_in_proj.bias"
        with safe_open(weights, framework="pt", device="cpu") as saved:
            reference = saved.get_tensor(key)
        loaded = self.policy.state_dict()[key].cpu()
        if not torch.equal(loaded, reference.to(dtype=loaded.dtype)):
            raise RuntimeError("Checkpoint weights did not load correctly; refusing to serve")
        self.preprocessor, self.postprocessor = load_step(
            "[startup 3/4] Loading saved normalization and tokenizer...",
            lambda: make_pre_post_processors(self.policy.config, pretrained_path=str(checkpoint)),
        )
        load_step("[startup 4/4] Moving model to GPU...", lambda: self.policy.to("cuda").eval())
        if self.policy.config.chunk_size != 50 or self.policy.config.output_features["action"].shape != (8,):
            raise ValueError("Checkpoint must predict 50 steps of 8 NERO action values")
        counts = Counter()
        for parameter in self.policy.parameters():
            counts[str(parameter.dtype)] += parameter.numel()
        print(f"[startup] GPU weight types (parameters): {dict(counts)}; no int8/4-bit quantization.", flush=True)

    def predict(self, request):
        state = np.asarray(request["state"], dtype=np.float32)
        if state.shape != (8,) or not np.isfinite(state).all():
            raise ValueError("state must contain eight finite values")
        task = request["task"]
        if not isinstance(task, str) or not task.strip():
            raise ValueError("task must be nonempty text")
        observation = prepare_observation_for_inference(
            {
                "observation.state": state,
                "observation.images.base_0_rgb": decode_image(request["base_rgb"]),
                "observation.images.left_wrist_0_rgb": decode_image(request["wrist_rgb"]),
            },
            device=torch.device("cpu"),
            task=task,
        )
        start = time.perf_counter()
        rtc = request.get("rtc", False)
        prefix = request.get("prev_chunk_left_over")
        delay = request.get("inference_delay", 0)
        if type(rtc) is not bool or type(delay) is not int or not 0 <= delay <= RTC_EXECUTION_HORIZON:
            raise ValueError("Invalid RTC request")
        if prefix is not None:
            prefix = np.asarray(prefix, dtype=np.float32)
            if not rtc or prefix.ndim != 2 or prefix.shape[1] != 8 or not 1 <= len(prefix) <= 50 or not np.isfinite(prefix).all():
                raise ValueError("Invalid RTC action prefix")
            prefix = torch.from_numpy(prefix).to("cuda")
        elif delay:
            raise ValueError("RTC delay requires an action prefix")
        with torch.no_grad():
            batch = self.preprocessor(observation)
            original = self.policy.predict_action_chunk(
                batch, inference_delay=delay if rtc else None,
                prev_chunk_left_over=prefix,
            )
            actions = self.postprocessor(original.reshape(-1, 8)).reshape(50, 8)
        actions = actions.detach().cpu().float().numpy()
        original = original.squeeze(0).detach().cpu().float().numpy()
        if not np.isfinite(actions).all() or not np.isfinite(original).all():
            raise ValueError("Model returned a non-finite action")
        return {"cycle": request["cycle"], "actions": actions.tolist(),
                "raw_actions": original.tolist() if rtc else None,
                "inference_ms": round((time.perf_counter() - start) * 1000, 1)}


def handler_for(predictor):
    class Handler(BaseHTTPRequestHandler):
        def reply(self, code, payload):
            data = json.dumps(payload, separators=(",", ":")).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/health":
                self.reply(200, {"ready": True, "chunk_size": 50, "action_dim": 8,
                                 "rtc": True, "rtc_execution_horizon": RTC_EXECUTION_HORIZON})
            else:
                self.reply(404, {"error": "unknown path"})

        def do_POST(self):
            if self.path != "/predict":
                self.reply(404, {"error": "unknown path"})
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 1_000_000:
                    raise ValueError("Request size out of range")
                request = json.loads(self.rfile.read(size))
                response = predictor.predict(request)
                self.reply(200, response)
                print(f"cycle={response['cycle']} inference={response['inference_ms']} ms", flush=True)
            except (KeyError, TypeError, ValueError) as exc:
                self.reply(400, {"error": str(exc)})

    return Handler


def main(args):
    if not 0 <= args.gpu < torch.cuda.device_count():
        raise SystemExit(f"GPU {args.gpu} is unavailable; visible GPUs: {torch.cuda.device_count()}")
    torch.cuda.set_device(args.gpu)
    checkpoint = args.checkpoint.expanduser().resolve()
    print(f"[startup] Using GPU {args.gpu}: {torch.cuda.get_device_name(args.gpu)}", flush=True)
    predictor = Predictor(checkpoint)
    print(f"Ready: {checkpoint} on {args.host}:{args.port}", flush=True)
    HTTPServer((args.host, args.port), handler_for(predictor)).serve_forever()


if __name__ == "__main__":
    main(RUN_ARGS)
