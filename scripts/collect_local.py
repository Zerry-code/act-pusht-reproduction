"""Copy original run evidence, never weights or credentials. Python 3.10+."""
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

src = Path(sys.argv[1]).expanduser().resolve()
repo = Path(sys.argv[2]).expanduser().resolve()
dst = repo / "evidence"
dst.mkdir(parents=True, exist_ok=True)
manifest = {"collected_at_utc": datetime.now(timezone.utc).isoformat(), "copied": [], "missing": [], "errors": []}

def copy(relative):
    source = src / relative
    target = dst / relative
    if not source.is_file():
        manifest["missing"].append(relative)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    manifest["copied"].append(relative)

for run in ("act_pusht_smoke", "act_pusht_10k", "act_pusht_10k_action10", "act_pusht_50k_action10", "act_pusht_50k_action10_n50", *(f"demo_seed_{s}" for s in range(42, 47))):
    copy(f"outputs/eval/{run}/eval_info.json")
for run in ("act_pusht_smoke", "act_pusht_10k", "act_pusht_50k"):
    for name in ("config.json", "train_config.json", "policy_preprocessor.json", "policy_postprocessor.json"):
        copy(f"outputs/train/{run}/checkpoints/last/pretrained_model/{name}")
    copy(f"outputs/train/{run}/checkpoints/last/training_state/training_step.json")
copy("pyproject.toml")

# A true environment snapshot; no proxy environment variables or tokens are exported.
for name, cmd in (
    ("pip-freeze.txt", [sys.executable, "-m", "pip", "freeze"]),
    ("lerobot-commit.txt", ["git", "-C", str(src), "rev-parse", "HEAD"]),
    ("lerobot-working-tree.txt", ["git", "-C", str(src), "status", "--short"]),
    ("ffmpeg-version.txt", ["ffmpeg", "-version"]),
):
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        (dst / name).write_text(result.stdout, encoding="utf-8")
        manifest["copied"].append(name)
    except (OSError, subprocess.CalledProcessError) as exc:
        manifest["errors"].append({"file": name, "error": str(exc)})

weights = src / "outputs/train/act_pusht_50k/checkpoints/last/pretrained_model/model.safetensors"
if weights.is_file():
    digest = hashlib.sha256()
    with weights.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    (dst / "model-sha256.txt").write_text(digest.hexdigest() + "  act_pusht_50k/model.safetensors\n", encoding="utf-8")
else:
    manifest["missing"].append("final model.safetensors (hash only)")
(dst / "collection_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"已采集 {len(manifest['copied'])} 项；缺失 {len(manifest['missing'])} 项；错误 {len(manifest['errors'])} 项。")
print(f"查看清单：{dst / 'collection_manifest.json'}")
