import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors

models = {
    "100_steps": (
        "outputs/train/act_libero_task34_smoke_20261125_175530"
        "/pretrained_model"
    ),
    "10000_steps": (
        "outputs/train/act_libero_task34_10k_20261125_181140"
        "/pretrained_model"
    ),
}

selection = json.loads(
    Path("libero_task34_split.json").read_text(encoding="utf-8")
)

dataset = LeRobotDataset(
    selection["repo_id"],
    revision=selection["revision"],
    episodes=selection["val_episodes"],
    video_backend="pyav",
)
loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)

rows = []

for name, model_dir in models.items():
    print("开始验证：", name, flush=True)

    policy = ACTPolicy.from_pretrained(model_dir)
    policy = policy.to(policy.config.device)
    policy.eval()

    preprocessor, _ = make_pre_post_processors(
        policy.config, model_dir
    )

    total_error = 0.0
    total_values = 0

    with torch.no_grad():
        for raw_batch in loader:
            prepared = preprocessor(raw_batch)
            policy.reset()
            prediction = policy.select_action(prepared)

            error = (prediction - prepared["action"]).abs()
            total_error += error.sum().item()
            total_values += error.numel()

    score = total_error / total_values
    rows.append({
        "model": name,
        "validation_episodes": len(selection["val_episodes"]),
        "validation_frames": len(dataset),
        "normalized_first_action_mae": score,
        "model_path": model_dir,
    })

    print(name, "验证集首步动作 MAE：", score, flush=True)
    del policy

Path("results").mkdir(exist_ok=True)

with open(
    "results/act_task34_validation.csv",
    "w", newline="", encoding="utf-8",
) as file:
    writer = csv.DictWriter(file, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

print("比较表已保存：results/act_task34_validation.csv")
