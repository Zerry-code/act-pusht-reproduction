import json
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

selection = json.loads(
    Path("libero_task34_split.json").read_text(encoding="utf-8")
)
train_ids = selection["train_episodes"]

files = HfApi().list_repo_files(
    repo_id=selection["repo_id"],
    repo_type="dataset",
    revision=selection["revision"],
)

parts = []

for name in files:
    if not (name.startswith("data/") and name.endswith(".parquet")):
        continue

    path = hf_hub_download(
        repo_id=selection["repo_id"],
        repo_type="dataset",
        filename=name,
        revision=selection["revision"],
    )

    table = pd.read_parquet(
        path,
        columns=[
            "episode_index", "task_index",
            "observation.state", "action",
        ],
    )

    selected = table[table["episode_index"].isin(train_ids)]
    if not selected.empty:
        parts.append(selected)

train = pd.concat(parts, ignore_index=True)

assert set(train["episode_index"].astype(int)) == set(train_ids)
assert (train["task_index"] == selection["task_index"]).all()

stats = {}

for key in ["observation.state", "action"]:
    values = np.stack(train[key].to_numpy()).astype(np.float64)
    stats[key] = {
        "mean": values.mean(axis=0).tolist(),
        "std": values.std(axis=0).clip(min=1e-6).tolist(),
    }
    print(key)
    print("均值：", stats[key]["mean"])
    print("标准差：", stats[key]["std"])

Path("libero_task34_train_stats.json").write_text(
    json.dumps(stats, indent=2) + "\n",
    encoding="utf-8",
)

print("参与统计的训练示范数：", train["episode_index"].nunique())
print("参与统计的时间步数：", len(train))
print("统计量已保存：libero_task34_train_stats.json")
