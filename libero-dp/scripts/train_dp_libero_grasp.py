"""Task 34 experiment using a short, clean-action-prediction Diffusion Policy."""

import argparse
import json
import random
import shutil
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch.optim.swa_utils import AveragedModel
from torchvision.transforms.functional import resize

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.factory import make_pre_post_processors

BASE = Path("outputs/train/dp_libero_task34_minmax_10k_20261125_225332")
SPLIT = Path("libero_task34_split.json")


def load_frames(selection, split, config):
    episodes = selection[f"{split}_episodes"]
    ds = LeRobotDataset(selection["repo_id"], revision=selection["revision"],
                         episodes=episodes, video_backend="pyav")
    assert ds.fps == 10
    rows = ds.hf_dataset
    episode_ids = torch.tensor(list(rows["episode_index"]))
    assert set(episode_ids.tolist()) == set(episodes)
    assert set(torch.tensor(list(rows["task_index"])).tolist()) == {34}
    frames = {
        key: torch.empty((len(rows), 3, *config.resize_shape), dtype=torch.uint8)
        for key in config.image_features
    }
    action_indices = torch.empty((len(rows), config.horizon), dtype=torch.long)
    padding = torch.empty_like(action_indices, dtype=torch.bool)
    for episode in episodes:
        indices = torch.where(episode_ids == episode)[0]
        assert torch.equal(indices, torch.arange(indices[0], indices[-1] + 1))
        timestamps = torch.stack(rows[indices.tolist()]["timestamp"]).tolist()
        images = ds._query_videos({key: timestamps for key in frames}, episode)
        for key, images_for_camera in images.items():
            frames[key][indices] = resize(images_for_camera, config.resize_shape).mul_(255).round_().to(torch.uint8)
        indices_window = indices[:, None] + torch.arange(config.horizon)
        padding[indices] = indices_window > indices[-1]
        action_indices[indices] = indices_window.clamp(max=indices[-1])
    action = torch.stack(list(rows["action"]))
    print(f"{split}: {len(episodes)} episodes, {len(action)} frames", flush=True)
    return {
        **frames,
        "observation.state": torch.stack(list(rows["observation.state"])).unsqueeze(1),
        "action": action[action_indices],
        "action_is_pad": padding,
    }


def train(args):
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(SPLIT, args.output)
    shutil.copy2(__file__, args.output)
    (args.output / "run.json").write_text(json.dumps(vars(args), default=str, indent=2))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    assert torch.cuda.is_available()
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.allow_tf32 = True

    selection = json.loads(SPLIT.read_text())
    assert len(set(selection["train_episodes"])) == 36
    assert len(set(selection["val_episodes"])) == 9
    assert set(selection["train_episodes"]).isdisjoint(selection["val_episodes"])
    config = replace(
        PreTrainedConfig.from_pretrained(BASE / "pretrained_model"),
        horizon=16, n_action_steps=4, resize_shape=(128, 128),
        down_dims=(128, 256, 512), use_group_norm=True,
        pretrained_backbone_weights=None, prediction_type="sample",
        num_inference_steps=10,
    )
    config.save_pretrained(args.output)
    data = load_frames(selection, "train", config)
    saved_stats = json.loads((BASE / "dp_train_stats.json").read_text())
    stats = {key: {name: torch.tensor(value) for name, value in values.items()}
             for key, values in saved_stats.items()}
    for key in ("observation.state", "action"):
        values = data[key][:, 0]
        stats[key].update(min=values.amin(0), max=values.amax(0))
    pre, post = make_pre_post_processors(config, dataset_stats=stats)
    policy = DiffusionPolicy(config).to("cuda").train()
    ema = AveragedModel(policy, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.995))
    ema.module.eval()
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-4, betas=(0.95, 0.999),
                                  weight_decay=1e-6, fused=True)
    count = len(data["action"])
    order = torch.empty(0, dtype=torch.long)
    running_loss = 0.0
    start = time.monotonic()
    for step in range(1, args.steps + 1):
        if len(order) < args.batch_size:
            order = torch.cat((order, torch.randperm(count)))
        indices, order = order[:args.batch_size], order[args.batch_size:]
        batch = {key: value[indices] for key, value in data.items()}
        for key in config.image_features:
            batch[key] = batch[key].float().div_(255)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, _ = policy(pre(batch))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        ema.update_parameters(policy)
        running_loss += loss.item()
        if step % 100 == 0 or step == 1:
            print(f"step={step} loss={running_loss / (1 if step == 1 else 99 if step == 100 else 100):.5f} "
                  f"elapsed={time.monotonic() - start:.1f}s "
                  f"peak_GiB={torch.cuda.max_memory_allocated() / 1024**3:.2f}", flush=True)
            running_loss = 0.0
        if step in {args.steps, args.screen_step}:
            model = args.output / f"step_{step:06d}" / "pretrained_model"
            ema.module.save_pretrained(model)
            pre.save_pretrained(model, config_filename="policy_preprocessor.json")
            post.save_pretrained(model, config_filename="policy_postprocessor.json")
            print(f"saved {model}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--screen-step", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    train(parser.parse_args())
