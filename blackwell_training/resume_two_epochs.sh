#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/mnt/storage/aryan/nero_pi05
export HF_HOME="$ROOT/cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export TORCH_HOME="$ROOT/cache/torch"
export CUDA_VISIBLE_DEVICES=1
source /home/manthan/anaconda3/etc/profile.d/conda.sh
conda activate "$ROOT/env"

CHECKPOINT="$ROOT/outputs/pi05_one_epoch_gpu1_bs4/checkpoints/001237/pretrained_model/train_config.json"
OUT="$ROOT/outputs/pi05_three_epochs_gpu1_bs4"
if [[ ! -f "$CHECKPOINT" ]]; then
    echo "Finish the first run before resuming: $CHECKPOINT" >&2
    exit 1
fi
if [[ -e "$OUT" ]]; then
    echo "Output already exists; refusing to overwrite: $OUT" >&2
    exit 1
fi
FREE_MIB=$(nvidia-smi -i 1 --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
if (( FREE_MIB < 40000 )); then
    echo "GPU 1 has only ${FREE_MIB} MiB free; wait until other jobs finish." >&2
    exit 1
fi

echo 'Resuming from step 1237 to step 3711 (two more passes at batch size 4).'
lerobot-train \
    --resume=true \
    --config_path="$CHECKPOINT" \
    --steps=3711 \
    --output_dir="$OUT" \
    2>&1 | tee "$ROOT/outputs/pi05_three_epochs_gpu1_bs4.train.log"

python "$ROOT/scripts/eval_final.py" "$OUT/checkpoints/last" \
    2>&1 | tee "$ROOT/outputs/pi05_three_epochs_gpu1_bs4.eval.log"
