#!/usr/bin/env bash
set -euo pipefail
LEROBOT_DIR="${LEROBOT_DIR:-$HOME/projects/lerobot}"
cd "$LEROBOT_DIR"
if [[ ! -x .venv/bin/python ]]; then
  echo "未找到 $LEROBOT_DIR/.venv，请先按 README 安装环境。" >&2
  exit 1
fi
source .venv/bin/activate

CONFIG="${RESUME_CONFIG:-outputs/train/act_pusht_10k/checkpoints/last/pretrained_model/train_config.json}"
OUT="${RUN_OUTPUT_DIR:-outputs/train/act_pusht_50k}"
[[ -f "$CONFIG" ]] || { echo "缺少续训配置：$CONFIG" >&2; exit 1; }
[[ ! -e "$OUT" ]] || { echo "输出已存在：$OUT，请使用新的 RUN_OUTPUT_DIR。" >&2; exit 1; }
lerobot-train \
  --config_path="$CONFIG" --resume=true --steps=50000 \
  --save_freq=10000 --log_freq=500 --eval_freq=0 \
  --output_dir="$OUT" --job_name=act_pusht_50k --wandb.enable=false
