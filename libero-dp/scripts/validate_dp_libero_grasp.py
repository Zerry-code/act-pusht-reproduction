"""Compare first-action prediction on all nine held-out task 34 demonstrations."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch

from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.factory import make_pre_post_processors

from train_dp_libero_grasp import BASE, SPLIT, load_frames


@torch.inference_mode()
def score(model, data, seed):
    policy = DiffusionPolicy.from_pretrained(model, strict=True).eval()
    pre, post = make_pre_post_processors(policy.config, pretrained_path=str(model))
    torch.manual_seed(seed)
    predictions = []
    targets = data["action"][:, 0]
    for start in range(0, len(targets), 16):
        batch = {
            key: data[key][start:start + 16] for key in policy.config.input_features
        }
        for key in policy.config.image_features:
            batch[key] = batch[key].float().div_(255)
        batch["observation.state"] = batch["observation.state"][:, 0]
        policy.reset()
        predictions.append(post(policy.select_action(pre(batch))).cpu())
    predictions = torch.cat(predictions)
    matches = predictions[:, -1].sign() == targets[:, -1].sign()
    close = targets[:, -1] > 0
    approaching_switch = (
        (data["action"][:, 1:5, -1] != targets[:, -1:]) & ~data["action_is_pad"][:, 1:5]
    ).any(dim=1)
    return {
        "frames": len(targets),
        "first_action_mae": (predictions - targets).abs().mean().item(),
        "per_dimension_mae": (predictions - targets).abs().mean(dim=0).tolist(),
        "gripper_sign_accuracy": matches.float().mean().item(),
        "gripper_close_recall": matches[close].float().mean().item(),
        "gripper_open_recall": matches[~close].float().mean().item(),
        "frames_before_switch": int(approaching_switch.sum()),
        "gripper_sign_accuracy_before_switch": matches[approaching_switch].float().mean().item(),
    }


if __name__ == "__main__":
    from lerobot.configs.policies import PreTrainedConfig

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", type=Path, nargs="+")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=137)
    args = parser.parse_args()
    selection = json.loads(SPLIT.read_text())
    assert len(set(selection["val_episodes"])) == 9
    assert set(selection["train_episodes"]).isdisjoint(selection["val_episodes"])
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    base = PreTrainedConfig.from_pretrained(BASE / "pretrained_model")
    data = load_frames(selection, "val", replace(base, resize_shape=(256, 256)))
    results = {str(model): score(model, data, args.seed) for model in args.models}
    report = {"seed": args.seed, "val_episodes": selection["val_episodes"], "models": results}
    with args.output.open("x") as file:
        json.dump(report, file, indent=2)
    print(json.dumps(report, indent=2))
