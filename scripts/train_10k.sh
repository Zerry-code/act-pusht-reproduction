#!/usr/bin/env bash
set -euo pipefail
LEROBOT_DIR="${LEROBOT_DIR:-$HOME/projects/lerobot}"
cd "$LEROBOT_DIR"
if [[ ! -x .venv/bin/python ]]; then
  echo "未找到 $LEROBOT_DIR/.venv，请先按 README 安装环境。" >&2
  exit 1
fi
source .venv/bin/activate

OUT="${RUN_OUTPUT_DIR:-outputs/train/act_pusht_10k}"
[[ ! -e "$OUT" ]] || { echo "输出已存在：$OUT，请使用新的 RUN_OUTPUT_DIR。" >&2; exit 1; }
lerobot-train \
  --dataset.repo_id=lerobot/pusht \
  --dataset.video_backend=pyav \
  --policy.type=act \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --batch_size=4 --num_workers=0 --steps=10000 \
  --eval_freq=0 --save_freq=2000 --log_freq=100 --seed=42 \
  --output_dir="$OUT" --job_name=act_pusht_10k --wandb.enable=false
