#!/usr/bin/env bash
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LEROBOT_DIR="${LEROBOT_DIR:-$HOME/projects/lerobot}"
mkdir -p "$REPO_DIR/media"
for spec in '44 success' '42 failure'; do
  read -r seed label <<< "$spec"
  input="$LEROBOT_DIR/outputs/eval/demo_seed_${seed}/videos/pusht_0/eval_episode_0.mp4"
  output="$REPO_DIR/media/act_pusht_${label}_seed${seed}.gif"
  if [[ -e "$output" && "${REBUILD_GIFS:-0}" != "1" ]]; then
    echo "保留已有 GIF：$output"
    continue
  fi
  [[ -f "$input" ]] || { echo "缺少视频：$input" >&2; exit 1; }
  overwrite_flag=-n
  if [[ "${REBUILD_GIFS:-0}" == "1" ]]; then overwrite_flag=-y; fi
  ffmpeg "$overwrite_flag" -i "$input" \
    -filter_complex '[0:v]fps=10,scale=384:-1:flags=lanczos,split[a][b];[a]palettegen[p];[b][p]paletteuse' \
    -loop 0 "$output"
done
