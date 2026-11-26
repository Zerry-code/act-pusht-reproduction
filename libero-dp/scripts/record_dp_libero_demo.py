"""Record supplemental demos through the experiment's unchanged evaluation path."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("SLURM_JOB_ID", "libero-demo")

MODEL = Path("outputs/train/dp_task34_grasp_20261126/step_004000/pretrained_model")


def make_gif(video, gif):
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
        "-i", str(video), "-filter_complex",
        "[0:v]fps=10,scale=384:-1:flags=lanczos,split[a][b];"
        "[a]palettegen[p];[b][p]paletteuse",
        "-loop", "0", str(gif),
    ], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--states", type=int, nargs="+", default=[0, 3])
    parser.add_argument("--seed", type=int, default=1042,
                        help="Environment seed is this base plus the state ID.")
    args = parser.parse_args()
    if not (Path.cwd() / "eval_dp_libero_grasp.py").is_file():
        raise SystemExit("请在 ~/projects/lerobot 根目录运行，那里应有 eval_dp_libero_grasp.py。")
    if not args.states or len(set(args.states)) != len(args.states):
        raise SystemExit("状态ID必须非空且不能重复。")
    if not all(0 <= state < 50 for state in args.states):
        raise SystemExit("本次演示使用0至49中的状态ID。")
    if shutil.which("ffmpeg") is None:
        raise SystemExit("找不到ffmpeg，请先检查现有FFmpeg安装。")
    for name in ["model.safetensors", "config.json", "policy_preprocessor.json", "policy_postprocessor.json"]:
        if not (args.model / name).is_file():
            raise SystemExit(f"最终模型目录缺少：{name}")

    # Import the actual experiment entry point from the user's LeRobot root.
    sys.path.insert(0, str(Path.cwd()))
    import imageio.v2 as imageio
    import numpy as np
    import eval_dp_libero_grasp as experiment

    original_rollout = experiment.rollout
    run = "dp_task34_demo_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output = Path("outputs/eval") / run
    output.mkdir(parents=True, exist_ok=False)
    media = output / "media"
    media.mkdir()
    config = json.loads((args.model / "config.json").read_text())
    with (args.model / "model.safetensors").open("rb") as file:
        digest = hashlib.sha256()
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    report = {
        "kind": "supplemental_demo_not_original_50_episode_evaluation",
        "model": str(args.model), "model_sha256": digest.hexdigest(),
        "playback_fps": 10, "playback_note": "10 frames/s; not wall-clock inference speed",
        "policy_config": config, "episodes": [],
    }
    shutil.copy2(__file__, output / "record_dp_libero_demo.py")
    (output / "model_config.json").write_text(json.dumps(config, indent=2))

    for state in args.states:
        frames = []

        def render_frame(env):
            assert env.num_envs == 1
            frame = np.asarray(env.envs[0].render())
            assert frame.ndim == 3 and frame.shape[-1] == 3
            assert frame.dtype == np.uint8
            frames.append(frame.copy())

        def recorded_rollout(*positional, **keywords):
            keywords["render_callback"] = render_frame
            return original_rollout(*positional, **keywords)

        episode_output = output / f"state_{state:03d}"
        print(f"正在录制状态 {state}，seed={args.seed + state}……", flush=True)
        experiment.rollout = recorded_rollout
        try:
            experiment.evaluate(argparse.Namespace(
                model=args.model, output=episode_output, episodes=1,
                start=state, seed=args.seed, batch_size=1,
            ))
        finally:
            experiment.rollout = original_rollout

        result = json.loads((episode_output / "eval_info.json").read_text())["episodes"][0]
        assert result["init_state_id"] == state
        assert result["seed"] == args.seed + state
        frames = frames[:result["steps"] + 1]
        assert len(frames) == result["steps"] + 1
        label = "success" if result["success"] else "failure"
        stem = f"state_{state:03d}_seed_{args.seed + state}_{label}"
        video, gif = media / f"{stem}.mp4", media / f"{stem}.gif"
        imageio.mimwrite(video, frames, fps=10, codec="libx264", quality=8,
                         pixelformat="yuv420p", ffmpeg_params=["-movflags", "+faststart"])
        make_gif(video, gif)
        row = {**result, "frames": len(frames), "mp4": str(video), "gif": str(gif)}
        report["episodes"].append(row)
        (output / "demo_info.json").write_text(json.dumps(report, indent=2))
        print(f"状态 {state}：{label}，步数 {result['steps']}，MP4和GIF已保存。", flush=True)

    print("\n补录完成；这批回合不计入原来的42/50。")
    print("视频与GIF目录：", media.resolve())
    print("原始补录记录：", (output / "demo_info.json").resolve())


if __name__ == "__main__":
    main()
