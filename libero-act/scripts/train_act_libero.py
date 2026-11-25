import json
import torch
from torch.utils.data import DataLoader
from lerobot.datasets.lerobot_dataset import LeRobotDataset

#读取固定名单，只选择训练示范
with open("libero_task34_split.json", encoding="utf-8") as file:
    selection = json.load(file)

train_ids = selection["train_episodes"]
val_ids = selection["val_episodes"]

assert set(train_ids).isdisjoint(val_ids)

chunk_size = 50
fps = 10.0

print("开始加载 36 条训练示范……", flush=True)

#创建数据集和 DataLoader，组织动作块、组成 batch
train_dataset = LeRobotDataset(
    selection["repo_id"],
    revision=selection["revision"],
    episodes=train_ids,
    video_backend="pyav",
    delta_timestamps={
        "action": [i / fps for i in range(chunk_size)],
    },
)

assert train_dataset.meta.fps == fps

loader = DataLoader(
    train_dataset,
    batch_size=2,
    shuffle=True,
#shuffle=True 会改变下一轮样本的取样顺序，不会打乱动作块内部的时间顺序。
    num_workers=0,
)
#---------------------------

batch = next(iter(loader))

print("训练示范数：", len(train_ids))
print("训练时间步数：", len(train_dataset))
print("本批次的任务编号：", batch["task_index"].unique().tolist())

for key in [
    "observation.images.image",
    "observation.images.image2",
    "observation.state",
    "action",
    "action_is_pad",
]:
    print(key, tuple(batch[key].shape))

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors

torch.manual_seed(42)

# 1. 模型接收什么、预测什么
image_keys = [
    "observation.images.image",
    "observation.images.image2",
]

input_features = {
    key: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256))
    for key in image_keys
}
input_features["observation.state"] = PolicyFeature(
    type=FeatureType.STATE, shape=(8,)
)

#配置 ACT 的输入、输出和预测长度
config = ACTConfig(
    input_features=input_features,
    output_features={
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(7,))
    },
    chunk_size=chunk_size,
    n_action_steps=10,
    device="cuda",
)

# 2. 使用训练集统计量和固定的图像统计量
#加载训练集统计量，创建预处理器
with open("libero_task34_train_stats.json", encoding="utf-8") as file:
    stats_data = json.load(file)

for key in image_keys:
    stats_data[key] = {
        "mean": [[[0.485]], [[0.456]], [[0.406]]],
        "std": [[[0.229]], [[0.224]], [[0.225]]],
    }

stats = {
    key: {
        name: torch.tensor(value, dtype=torch.float32)
        for name, value in values.items()
    }
    for key, values in stats_data.items()
}

# 3. 创建模型与配套处理器
policy = ACTPolicy(config).to(config.device)

#加载训练集统计量，创建预处理器
preprocessor, postprocessor = make_pre_post_processors(
    config,
    dataset_stats=stats,
)

prepared_batch = preprocessor(batch)

policy.train()
with torch.no_grad():
    loss, loss_terms = policy.forward(prepared_batch)
#policy.forward(...)：计算预测与损失；这行调用 ACT，返回：
#loss        用于反向传播的总损失
#loss_terms  动作误差、KL 损失等记录

print("训练脚本中的 ACT 前向计算成功")
print("当前批次损失：", loss.item())
print("损失分项：", loss_terms)

import time
import shutil
from pathlib import Path
from datetime import datetime

steps = 10000
learning_rate = 1e-5

optimizer = torch.optim.AdamW(
    policy.parameters(),
    lr=learning_rate,
    weight_decay=1e-4,
)

run_dir = Path("outputs/train") / (
    "act_libero_task34_10k_" + datetime.now().strftime("%Y%m%d_%H%M%S")
)
run_dir.mkdir(parents=True)

iterator = iter(loader)
policy.train()

torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()
start_time = time.perf_counter()
running_loss = 0.0

print(f"开始训练，共 {steps} 次更新……", flush=True)

#-----------------------（重点）循环取样、计算损失、更新参数
for step in range(1, steps + 1):
    try:
        raw_batch = next(iterator)
#next(iterator)：拿下一批教材
#这里通常取出两份样本。每份包含：
#起点图像：主相机＋腕部相机
#起点状态：8 个数
#参考答案：从起点开始的 50 个七维动作
#填充标记：哪些参考动作有效
    except StopIteration:
#StopIteration 表示这一轮数据取完了。代码重新开始取数据，但没有重新创建模型，因此之前学到的参数保留着。
        iterator = iter(loader)
        raw_batch = next(iterator)

    training_batch = preprocessor(raw_batch)
#preprocessor(raw_batch)：把数据转成模型使用的尺度，输入数据仍表达同一件事，只是完成了归一化、设备转换等处理

    optimizer.zero_grad(set_to_none=True)
#先清空之前的梯度。默认情况下，新计算的梯度会累积，不能每批都不清理。
    loss, loss_terms = policy.forward(training_batch)
#shuffle=True 会改变下一轮样本的取样顺序，不会打乱动作块内部的时间顺序。
    loss.backward()
#计算当前损失对各个可训练参数的梯度，此时还没执行优化器更新。

    torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
#clip_grad_norm_(...)限制梯度整体大小，避免一次更新过猛。它不代表“每个权重最多变化 1”。
    optimizer.step()
#AdamW 根据梯度、学习率和自身保存的历史状态，实际调整参数。
#-------------------------------------------

    if step % 2000 == 0:
        checkpoint = run_dir / "checkpoints" / f"step_{step:06d}"
        checkpoint.mkdir(parents=True, exist_ok=True)
        model_folder = checkpoint / "pretrained_model"

        config.push_to_hub = False
        policy.save_pretrained(model_folder)
        preprocessor.save_pretrained(
            model_folder, config_filename="policy_preprocessor.json"
        )
        postprocessor.save_pretrained(
            model_folder, config_filename="policy_postprocessor.json"
        )
        torch.save(
            {"step": step, "optimizer": optimizer.state_dict()},
            checkpoint / "optimizer_state.pt",
        )
        print("checkpoint 已保存：", checkpoint, flush=True)

    running_loss += loss.item()

    if step % 20 == 0:
        print(
            f"step={step}，最近20步平均loss={running_loss / 20:.4f}",
            flush=True,
        )
        running_loss = 0.0

torch.cuda.synchronize()
elapsed = time.perf_counter() - start_time
peak_gib = torch.cuda.max_memory_allocated() / 1024**3

model_dir = run_dir / "pretrained_model"
config.push_to_hub = False

policy.save_pretrained(model_dir)
preprocessor.save_pretrained(
    model_dir, config_filename="policy_preprocessor.json"
)
postprocessor.save_pretrained(
    model_dir, config_filename="policy_postprocessor.json"
)

torch.save(
    {
        "step": steps,
        "optimizer": optimizer.state_dict(),
    },
    run_dir / "optimizer_state.pt",
)

shutil.copy2("libero_task34_split.json", run_dir)
shutil.copy2("libero_task34_train_stats.json", run_dir)
shutil.copy2("train_act_libero.py", run_dir)

print("训练完成，更新次数：", steps)
print("训练耗时：", round(elapsed, 2), "秒")
print("平均每次更新：", round(elapsed / steps, 3), "秒")
print("PyTorch峰值张量显存：", round(peak_gib, 2), "GiB")
print("结果目录：", run_dir)
