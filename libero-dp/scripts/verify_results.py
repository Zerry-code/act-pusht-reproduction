"""Recompute published DP metrics from raw records; no ML packages required."""

import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(name):
    return json.loads((ROOT / "results" / name).read_text(encoding="utf-8"))


def check(name, count, base_seed, expected_successes):
    data = read(name)
    rows = sorted(data["episodes"], key=lambda row: row["init_state_id"])
    assert [row["init_state_id"] for row in rows] == list(range(count))
    assert [row["seed"] for row in rows] == list(range(base_seed, base_seed + count))
    assert all(isinstance(row["success"], bool) for row in rows)
    successes = sum(row["success"] for row in rows)
    assert successes == expected_successes == data["overall"]["successes"]
    assert count == data["overall"]["n_episodes"]
    assert abs(data["overall"]["pc_success"] - 100 * successes / count) < 1e-9
    assert all(1 <= row["steps"] <= 280 for row in rows)
    assert all(0 <= row["gripper_sign_changes"] < row["steps"] for row in rows)
    flips = statistics.median(100 * row["gripper_sign_changes"] / row["steps"] for row in rows)
    print(f"{name}: {successes}/{count}; median flips/100 steps={flips:.4f}")
    return rows


def main():
    old = check("baseline50_raw.json", 50, 1042, 0)
    check("screen10_raw.json", 10, 42, 10)
    new = check("confirm50_raw.json", 50, 1042, 42)
    assert sum(row["success"] for row in old[10:]) == 0
    assert sum(row["success"] for row in new[10:]) == 33
    assert [row["init_state_id"] for row in new if not row["success"]] == [3, 10, 18, 23, 24, 26, 47, 49]
    print("Unscreened initial states: old 0/40; new 33/40 (82.5%)")
    audit = read("data_audit_raw.json")
    split = json.loads((ROOT / "config/libero_task34_split.json").read_text())
    assert set(split["train_episodes"]).isdisjoint(split["val_episodes"])
    assert set(audit["train"]["episode_ids"]) == set(split["train_episodes"])
    assert set(audit["val"]["episode_ids"]) == set(split["val_episodes"])
    assert audit["train"]["frames"] == 3573
    assert audit["val"]["frames"] == 914
    validation = read("validation_raw.json")
    assert validation["val_episodes"] == split["val_episodes"]
    assert audit["val"]["gripper_labels_counts"]["1.0"] == 438
    for model, metrics in validation["models"].items():
        assert metrics["frames"] == 914
        numerator = 418 if "grasp_20261126" in model else 289
        assert abs(metrics["gripper_close_recall"] - numerator / 438) < 1e-6
    print("Split, frame counts, reported close recalls: consistent")
    print("Checks verify saved evidence consistency; they do not rerun training or simulation.")


if __name__ == "__main__":
    main()
