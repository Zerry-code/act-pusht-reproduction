#!/usr/bin/env bash
set -euo pipefail

export MUJOCO_GL=egl
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

RUN_NAME="act_task34_10k_eval_$(date +%Y%m%d_%H%M%S)"
mkdir -p outputs/logs

lerobot-eval \
  --policy.path=outputs/train/act_libero_task34_10k_20261125_181140/pretrained_model \
  --policy.device=cuda \
  --policy.n_action_steps=10 \
  --env.type=libero \
  --env.task=libero_spatial \
  --env.task_ids='[0]' \
  --env.control_mode=relative \
  --env.init_states=true \
  --env.observation_height=256 \
  --env.observation_width=256 \
  --eval.batch_size=1 \
  --eval.n_episodes=10 \
  --seed=42 \
  --output_dir="outputs/eval/$RUN_NAME" \
  2>&1 | tee "outputs/logs/$RUN_NAME.log"
