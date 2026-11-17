#!/usr/bin/env bash
set -euo pipefail
LEROBOT_DIR="${LEROBOT_DIR:-$HOME/projects/lerobot}"
cd "$LEROBOT_DIR"
if [[ ! -x .venv/bin/python ]]; then
  echo "未找到 $LEROBOT_DIR/.venv，请先按 README 安装环境。" >&2
  exit 1
fi
source .venv/bin/activate

SEED="${1:-42}"
EPISODES="${2:-50}"
ACTION_STEPS="${3:-10}"
OUT="${4:-outputs/eval/act_pusht_50k_action10_n50}"
POLICY="${POLICY_PATH:-outputs/train/act_pusht_50k/checkpoints/last/pretrained_model}"
[[ -f "$POLICY/model.safetensors" ]] || { echo "缺少模型权重：$POLICY" >&2; exit 1; }
[[ ! -e "$OUT" ]] || { echo "输出已存在：$OUT，请通过第四个参数指定新目录。" >&2; exit 1; }
SDL_VIDEODRIVER=dummy lerobot-eval \
  --policy.path="$POLICY" --policy.device=cuda \
  --policy.n_action_steps="$ACTION_STEPS" --env.type=pusht \
  --eval.n_episodes="$EPISODES" --eval.batch_size=1 --seed="$SEED" \
  --output_dir="$OUT"
