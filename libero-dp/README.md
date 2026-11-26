# Diffusion Policy：LIBERO 单任务训练与失败排查

在与 [LIBERO ACT 实验](../libero-act/README.md)相同的任务和示范划分上，训练并改进 LeRobot Diffusion Policy。最终组合配置在 50 个官方初始状态上成功 **42/50（84%）**；其中未参加小规模筛选的 40 个初始状态成功 **33/40（82.5%）**。

这是一项现有算法的工程复现与调试实验，没有提出新算法，未进行实机部署，也未覆盖整个 LIBERO-Spatial 套件。

## 从哪里开始看

- **[完整排查记录：现象 → 假设 → 验证 → 结论 → 下一步](docs/DEBUGGING.md)**：包含无收益的尝试、方案修正、最终改动以及因果解释的边界。
- [最终策略配置](config/final_policy_config.json)、[训练运行参数](config/train_run.json)、[固定示范划分与数据版本](config/libero_task34_split.json)。
- [旧模型正式对照](results/baseline50_raw.json)、[新模型筛选](results/screen10_raw.json)、[新模型正式确认](results/confirm50_raw.json)。
- [留出示范验证](results/validation_raw.json)、[数据审计](results/data_audit_raw.json)、[原始文件来源](results/source_manifest.json)。

## 任务、数据与结果

任务为拿起盘子与烤碗之间的黑碗，放到盘子上。数据集任务编号为 **34**，对应仿真 `libero_spatial` 的任务 **0**。

数据来自 `lerobot/libero`，revision 固定在划分文件中。45 条示范按 episode 划分为 36 条训练示范（3573 帧）与 9 条验证示范（914 帧）。两路 RGB 图像与 8 维机械臂状态作为输入，输出 7 维相对控制动作；没有用验证示范更新权重。

| 本次评测 | 状态与环境种子 | 成功回合 |
|---|---|---:|
| 旧 MIN_MAX DP 10k 正式对照 | 状态0–49，seed=1042+状态ID | 0/50（0%） |
| 新 DP 4k 筛选 | 状态0–9，seed=42+状态ID | 10/10（100%） |
| 固定新模型正式确认 | 状态0–49，seed=1042+状态ID | **42/50（84%）** |
| 正式确认中的未筛选状态子集 | 状态10–49，同上 | **33/40（82.5%）** |

40 个未筛选状态是正式确认的子集，不是另外新增的 40 回合；也不能称为与训练场景完全独立的泛化测试。旧模型评测批量为5，新模型为1，扩散噪声的分组不同。

ACT 的历史参考结果为48/50（96%），但其评测入口与种子序列不同，尚未按本目录协议重测；不能把96%与84%当作严格的算法消融比较。

## 最终配置

- 单步观察，动作预测窗口16，每次执行4步后重新规划。
- 双相机原始观测256×256，编码器缩放至128×128；ResNet18随机初始化、GroupNorm。
- U-Net通道128/256/512；扩散网络预测干净动作 `prediction_type="sample"`，仍通过扩散采样生成动作。
- DDIM，100个训练加噪时间步，推理10次去噪；MIN_MAX状态/动作归一化，开启样本裁剪。
- 4000次更新，batch=8；AdamW，lr=1e-4，betas=(0.95,0.999)，weight_decay=1e-6，梯度裁剪1.0。
- EMA衰减0.995，最终保存EMA权重；训练使用BF16 autocast。

保存配置中的 `scheduler_name` / `scheduler_warmup_steps` 是类的预设字段；本次训练脚本没有创建或调用学习率调度器，不能写成实际使用了cosine或warmup。

## 代码与复现状态

[scripts/](scripts/) 保留用户实际运行并上传的训练、验证和评测脚本原文。它们在已配置好的 LeRobot v0.4.4 根目录运行，不是独立软件包；输入划分文件也需位于该根目录。

训练脚本的 `BASE` 仍引用旧10k实验目录，以读取配置与统计量；新模型由 `DiffusionPolicy(config)` 重新初始化，不加载旧策略权重。旧BASE资料、最终权重、完整依赖锁和模拟资源没有随本目录发布，所以这不是下载后即可独立运行的完整分发包。迁移到新机器时需要先补齐这些依赖并调整路径。

本机最终权重位置（相对 LeRobot 根目录）：

```text
outputs/train/dp_task34_grasp_20261126/step_004000/pretrained_model
```

该模型目录需要完整保留模型、config、前处理器、后处理器及其归一化统计，不能只复制 `model.safetensors`。

核对已发布证据中的数字只需Python标准库，无需GPU：

```bash
python3 libero-dp/scripts/verify_results.py
```

## 演示状态

最终84%模型的正式确认没有录制视频。本目录暂不展示GIF，历史旧模型的视频不能冒充最终模型成果；后续录制应固定最终权重与配置，并注明是补录演示，不属于原50回合录像。

## 工作来源

使用上游 LeRobot / Diffusion Policy 算法。环境、训练与评测在实验者本机完成；前期通过AI讲解与协作排查，后期经明确授权由本地Codex自动执行组合改进、评测和报告整理。本目录对上传原始记录做了静态核验，没有由网页会话重新运行训练或仿真。不将AI辅助工作描述为独立提出算法或独立手写全部实现。
