#!/usr/bin/env bash
set -Eeuo pipefail

if [[ ${1:-} == resume ]]; then
    MODE=resume
    OUT=${2:?Usage: train_one_epoch.sh resume OUTPUT_DIR [GPU]}
    GPU=${3:-1}
else
    MODE=start
    DATASET=${1:?Usage: train_one_epoch.sh STAGED_DATASET OUTPUT_DIR [BATCH_SIZE] [GPU]}
    OUT=${2:?Usage: train_one_epoch.sh STAGED_DATASET OUTPUT_DIR [BATCH_SIZE] [GPU]}
    BATCH_SIZE=${3:-4}
    GPU=${4:-1}
fi
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export HF_HOME="$ROOT/cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export TORCH_HOME="$ROOT/cache/torch"
export CUDA_VISIBLE_DEVICES="$GPU"
source /home/manthan/anaconda3/etc/profile.d/conda.sh
conda activate "$ROOT/env"

if ! hf auth whoami >/dev/null 2>&1; then
    echo 'Hugging Face login is missing. Run: hf auth login' >&2
    exit 1
fi
FREE_MIB=$(nvidia-smi -i "$GPU" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
if (( FREE_MIB < 40000 )); then
    echo "GPU $GPU has only ${FREE_MIB} MiB free; wait until other jobs finish." >&2
    exit 1
fi
if [[ $MODE == resume ]]; then
    CHECKPOINT="$OUT/checkpoints/last"
    CONFIG="$CHECKPOINT/pretrained_model/train_config.json"
    STATE="$CHECKPOINT/training_state/training_step.json"
    if [[ ! -f "$CONFIG" || ! -f "$STATE" ]]; then
        echo "No complete checkpoint to resume in: $OUT" >&2
        exit 1
    fi
    mapfile -t RESUME_SETTINGS < <(python - "$CONFIG" "$STATE" <<'PY'
import json, sys
from pathlib import Path

config = json.loads(Path(sys.argv[1]).read_text())
state = json.loads(Path(sys.argv[2]).read_text())
print(config["dataset"]["root"])
print(state["step"])
print(config["steps"])
PY
)
    if (( ${#RESUME_SETTINGS[@]} != 3 )); then
        echo 'Could not read checkpoint settings.' >&2
        exit 1
    fi
    DATASET=${RESUME_SETTINGS[0]}
    if (( RESUME_SETTINGS[1] >= RESUME_SETTINGS[2] )); then
        echo "Run already reached its target of ${RESUME_SETTINGS[2]} steps." >&2
        exit 1
    fi
    echo "Resuming step ${RESUME_SETTINGS[1]} of ${RESUME_SETTINGS[2]} on GPU $GPU."
    lerobot-train --resume=true --config_path="$CONFIG" --output_dir="$OUT" \
        2>&1 | tee -a "$OUT.train.log"
    python "$ROOT/scripts/eval_final.py" "$OUT/checkpoints/last" "$DATASET" \
        2>&1 | tee "$OUT.eval.log"
    exit 0
fi
if [[ ! -f "$DATASET/meta/train_split.json" || ! -f "$DATASET/meta/stats.json" ]]; then
    echo "Prepared train/validation/test dataset is missing: $DATASET" >&2
    exit 1
fi
if [[ -e "$OUT" ]]; then
    echo "Output already exists; refusing to overwrite: $OUT" >&2
    exit 1
fi
if [[ ! "$BATCH_SIZE" =~ ^[1-9][0-9]*$ ]]; then
    echo 'BATCH_SIZE must be a positive integer.' >&2
    exit 1
fi

mapfile -t SETTINGS < <(python - "$DATASET" "$BATCH_SIZE" <<'PY'
import json, math, sys
from pathlib import Path

dataset, batch_size = Path(sys.argv[1]), int(sys.argv[2])
split = json.loads((dataset / 'meta/train_split.json').read_text())
info = json.loads((dataset / 'meta/info.json').read_text())
if info['total_tasks'] != 1:
    raise SystemExit('This trainer requires a single task for its episode split.')
train, val = split['train_episodes'], split['validation_episodes']
print(split['repo_id'])
print(json.dumps(train + val))
print(len(val) / (len(train) + len(val)))
print(math.ceil(split['frames']['train'] / batch_size))
print(split['frames']['train'])
PY
)
if (( ${#SETTINGS[@]} != 5 )); then
    echo 'Could not read the split manifest.' >&2
    exit 1
fi
REPO_ID=${SETTINGS[0]}
EPISODES=${SETTINGS[1]}
EVAL_SPLIT=${SETTINGS[2]}
STEPS=${SETTINGS[3]}
TRAIN_FRAMES=${SETTINGS[4]}
EVAL_FREQ=${EVAL_FREQ:-4000}
SAVE_FREQ=${SAVE_FREQ:-4000}

mkdir -p "$(dirname "$OUT")"
echo "Training $STEPS updates for one pass over $TRAIN_FRAMES frames, batch $BATCH_SIZE, GPU $GPU."
echo 'Color augmentation: brightness 0.9-1.1, contrast 0.9-1.1, saturation 0.95-1.05; no geometry.'
echo "Training loss every step; validation every $EVAL_FREQ steps; checkpoints every $SAVE_FREQ steps and at the end."
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
    --policy.pretrained_path=lerobot/pi05_base \
    --policy.chunk_size=50 \
    --policy.n_action_steps=10 \
    --policy.empty_cameras=1 \
    --policy.train_expert_only=false \
    --policy.freeze_vision_encoder=false \
    --policy.gradient_checkpointing=true \
    --policy.dtype=bfloat16 \
    --policy.device=cuda \
    --policy.optimizer_lr=2.5e-5 \
    --policy.push_to_hub=false \
    --save_checkpoint_to_hub=false \
    --output_dir="$OUT" \
    --batch_size="$BATCH_SIZE" \
    --num_workers=4 \
    --steps="$STEPS" \
    --log_freq=1 \
    --eval_steps="$EVAL_FREQ" \
    --env_eval_freq=0 \
    --save_freq="$SAVE_FREQ" \
    --wandb.enable=false \
    --seed=1000 2>&1 | tee "$OUT.train.log"

python "$ROOT/scripts/eval_final.py" "$OUT/checkpoints/last" "$DATASET" \
    2>&1 | tee "$OUT.eval.log"
echo "Finished. Checkpoints and logs are under $(dirname "$OUT")."
