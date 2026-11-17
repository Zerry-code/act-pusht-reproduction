#!/usr/bin/env bash
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LEROBOT_DIR="${LEROBOT_DIR:-$HOME/projects/lerobot}"
if [[ ! -x "$LEROBOT_DIR/.venv/bin/python" ]]; then
  echo "找不到原项目虚拟环境：$LEROBOT_DIR/.venv" >&2
  exit 1
fi
"$LEROBOT_DIR/.venv/bin/python" "$REPO_DIR/scripts/collect_local.py" "$LEROBOT_DIR" "$REPO_DIR"
