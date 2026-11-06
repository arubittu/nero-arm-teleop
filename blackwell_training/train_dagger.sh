#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
GPU=${1:-0}
DATASET="$ROOT/dataset/lerobot_nero_pi05_dagger_3x_trainstats"
START_MODEL="$ROOT/outputs/pi05_76_color_gpu1_bs4/checkpoints/last/pretrained_model"
OUT="$ROOT/outputs/pi05_dagger_76plus50_gpu${GPU}_bs32"
STEPS=6000
EVAL_FREQ=1000

if [[ ! -f "$DATASET/meta/train_split.json" || ! -f "$DATASET/meta/stats.json" ]]; then
    echo "Prepared DAgger dataset is missing: $DATASET" >&2
    exit 1
fi
if [[ ! -f "$START_MODEL/model.safetensors" ]]; then
    echo "76-demo starting checkpoint is missing: $START_MODEL" >&2
    exit 1
fi
if [[ -e "$OUT" ]]; then
    echo "Output already exists; refusing to overwrite: $OUT" >&2
    exit 1
fi

mapfile -t SETTINGS < <(python - "$DATASET" <<'PY'
import json, sys
from pathlib import Path

dataset = Path(sys.argv[1])
split = json.loads((dataset / "meta/train_split.json").read_text())
info = json.loads((dataset / "meta/info.json").read_text())
train, val, test = (split[f"{name}_episodes"] for name in ("train", "validation", "test"))
if (info["total_tasks"] != 1 or len(train) != 210 or len(val) != 8 or len(test) != 8
        or set(train) & set(val) or set(train) & set(test) or set(val) & set(test)):
    raise SystemExit("Unexpected task count or DAgger split")
print(split["repo_id"])
print(json.dumps(train + val))
print(len(val) / (len(train) + len(val)))
print(split["frames"]["train"])
PY
)
if (( ${#SETTINGS[@]} != 4 )); then
    echo "Could not read the DAgger split" >&2
    exit 1
fi
REPO_ID=${SETTINGS[0]}
EPISODES=${SETTINGS[1]}
EVAL_SPLIT=${SETTINGS[2]}
TRAIN_FRAMES=${SETTINGS[3]}

export HF_HOME="$ROOT/cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export TORCH_HOME="$ROOT/cache/torch"
source /home/manthan/anaconda3/etc/profile.d/conda.sh
conda activate "$ROOT/env"
GPU_UUID=$(nvidia-smi -i "$GPU" --query-gpu=uuid --format=csv,noheader)
export CUDA_VISIBLE_DEVICES="$GPU_UUID"

FREE_MIB=$(nvidia-smi -i "$GPU" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
if (( FREE_MIB < 45000 )); then
    echo "GPU $GPU has only ${FREE_MIB} MiB free; batch 32 needs at least 45000 MiB free." >&2
    exit 1
fi

mkdir -p "$(dirname "$OUT")"
echo "Starting from: $START_MODEL"
echo "Training: $TRAIN_FRAMES frames, batch 32, $STEPS updates, GPU $GPU"
echo "Loss every step; validation and checkpoint every $EVAL_FREQ updates."
echo "Output: $OUT"

lerobot-train \
    --dataset.repo_id="$REPO_ID" \
    --dataset.root="$DATASET" \
    --dataset.episodes="$EPISODES" \
    --dataset.eval_split="$EVAL_SPLIT" \
    --dataset.image_transforms.enable=true \
    --dataset.image_transforms.max_num_transforms=3 \
    --dataset.image_transforms.tfs='{"brightness":{"weight":1,"type":"ColorJitter","kwargs":{"brightness":[0.9,1.1]}},"contrast":{"weight":1,"type":"ColorJitter","kwargs":{"contrast":[0.9,1.1]}},"saturation":{"weight":1,"type":"ColorJitter","kwargs":{"saturation":[0.95,1.05]}}}' \
    --policy.type=pi05 \
    --policy.pretrained_path="$START_MODEL" \
    --policy.chunk_size=50 \
    --policy.n_action_steps=10 \
    --policy.empty_cameras=1 \
    --policy.train_expert_only=false \
    --policy.freeze_vision_encoder=false \
    --policy.gradient_checkpointing=true \
    --policy.dtype=bfloat16 \
    --policy.device=cuda \
    --policy.optimizer_lr=2.5e-5 \
    --policy.scheduler_warmup_steps=200 \
    --policy.scheduler_decay_steps="$STEPS" \
    --policy.scheduler_decay_lr=2.5e-6 \
    --policy.push_to_hub=false \
    --save_checkpoint_to_hub=false \
    --output_dir="$OUT" \
    --batch_size=32 \
    --num_workers=4 \
    --steps="$STEPS" \
    --log_freq=1 \
    --eval_steps="$EVAL_FREQ" \
    --env_eval_freq=0 \
    --save_freq="$EVAL_FREQ" \
    --wandb.enable=false \
    --seed=1000 2>&1 | tee "$OUT.train.log"

python "$ROOT/scripts/eval_final.py" "$OUT/checkpoints/last" "$DATASET" \
    2>&1 | tee "$OUT.eval.log"
