"""Evaluate a DP checkpoint on explicitly enumerated LIBERO spatial task 0 states."""

import argparse
import json
import time
from pathlib import Path

import torch

from lerobot.envs.configs import LiberoEnv as LiberoConfig
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.scripts.lerobot_eval import rollout
from lerobot.utils.random_utils import set_seed


def evaluate(args):
    if args.episodes % args.batch_size:
        raise ValueError("episodes must be a multiple of batch size")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    assert torch.cuda.is_available()
    policy = DiffusionPolicy.from_pretrained(args.model, strict=True).eval()
    pre, post = make_pre_post_processors(policy.config, pretrained_path=str(args.model))
    env_cfg = LiberoConfig(task="libero_spatial", task_ids=[0], init_states=True,
                           control_mode="relative", observation_width=256, observation_height=256)
    env_pre, env_post = make_env_pre_post_processors(env_cfg, policy.config)
    env = make_env(env_cfg, n_envs=args.batch_size)["libero_spatial"][0]
    end = args.start + args.episodes
    try:
        assert 0 <= args.start < end <= len(env.envs[0]._init_states)
        results = []
        start_time = time.monotonic()
        report = {"config": {key: str(value) if isinstance(value, Path) else value
                             for key, value in vars(args).items()}, "episodes": results}
        for first in range(args.start, end, args.batch_size):
            ids = list(range(first, first + args.batch_size))
            for subenv, state_id in zip(env.envs, ids, strict=True):
                subenv.init_state_id = state_id
            seeds = [args.seed + i for i in ids]
            set_seed(seeds[0])
            data = rollout(env, policy, env_pre, env_post, pre, post, seeds=seeds)
            for row, state_id in enumerate(ids):
                stop = int(torch.where(data["done"][row])[0][0]) + 1
                action = data["action"][row, :stop, -1]
                success = bool(data["success"][row, :stop].any())
                results.append({
                    "init_state_id": state_id, "seed": seeds[row], "success": success,
                    "steps": stop, "gripper_positive_steps": int((action > 0).sum()),
                    "gripper_sign_changes": int((action[1:].sign() != action[:-1].sign()).sum()),
                })
            count = sum(row["success"] for row in results)
            report["overall"] = {
                "successes": count, "n_episodes": len(results),
                "pc_success": 100 * count / len(results),
                "eval_s": time.monotonic() - start_time,
            }
            (args.output / "eval_info.json").write_text(json.dumps(report, indent=2))
            print(f"init states {ids}: {count}/{len(results)} cumulative", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--episodes", required=True, type=int)
    parser.add_argument("--start", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--batch-size", type=int, default=5)
    evaluate(parser.parse_args())
