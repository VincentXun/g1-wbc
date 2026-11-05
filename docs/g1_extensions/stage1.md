# 阶段一：官方 G1 基线的运行与验收

当前交付：启动与记录脚本、任务/控制链核对、无仿真依赖的脚本测试。
AutoDL 上的任务运行、显存/吞吐测量和训练更新尚待实跑，不能据此宣称阶段一仿真验收通过。
原始需求见 [plan-v0.md](plan-v0.md)。本阶段复用官方任务，不创建自定义任务或改动 `pick_place/g1/`。
后续有控制改动时，再在约定的 `pick_place/g1_extensions/` 中注册独立任务 ID。

## 版本与环境

2026-11-05 核对：[官方最新发布](https://github.com/nvidia-isaac/WBC-AGILE/releases) 为 v1.3.2，
本仓库基线提交为 `f5c388ebc7c1b056de891e3edb8f47693a50d5d3`。
按仓库 `pyproject.toml` 和 `uv.lock` 安装：Linux x86_64、Python 3.12、
Isaac Lab 3.0.0b2、Isaac Sim wheel 6.0.0.1、PyTorch 2.11.0 + CUDA 12.8、RSL-RL 5.4.1 + AGILE patch。
参见 [官方安装说明](https://nvidia-isaac.github.io/WBC-AGILE/getting-started.html)。

AutoDL 选择单卡 RTX 4090 的 Linux 环境，先检查驱动和系统兼容性；平台镜像自带的旧 Isaac Lab/Sim 不能代替本仓库依赖。
不要在当前 Windows 开发机上运行仓库的 `uv sync`；锁文件仅支持 Linux x86_64。
本机可做代码编辑、任务清单生成和命令预览。未验证 Windows/WSL 的 Isaac Sim 运行支持。

在 **AutoDL 的仓库根目录**执行（修改后的脚本需要一并同步到云端）：

```bash
git describe --tags --always
git rev-parse HEAD
nvidia-smi
git lfs pull
uv sync --frozen
uv run --frozen scripts/g1_extensions/stage1.py preflight
uv run --frozen scripts/g1_extensions/stage1.py inventory
```

`preflight` 检查平台、解释器、关键安装包版本、CUDA 张量运算、RSL-RL patch、策略文件和参考轨迹；
占位的 Git LFS pointer 不算有效资源。失败返回非零状态并写明原因。
预检不会自动安装/升级依赖；USD 模型、桌子和物体资产的网络可达性，仍需启动仿真验证。
若 Isaac Sim 首次启动提示许可协议，按官方流程完成后重新执行。

## 执行顺序

### 1. 官方预训练行走

```bash
uv run --frozen scripts/g1_extensions/stage1.py walk
```

调用 `scripts/eval.py`，任务 `Velocity-G1-History-v0`，加载仓库自带的匹配 state-dict checkpoint。
默认 2 个环境、2000 控制步（约 40 秒仿真时间），使用官方随机指令调度器切换前后、侧向和转向速度，随机种子 42。
它验证的是预训练策略闭环行走，不是从零训练，也不是 `play.py` 的正弦测试动作。

有 Linux 图形桌面时加 `--viewer`；无桌面时先完成 headless 运行，再录像：

```bash
uv run --frozen scripts/g1_extensions/stage1.py walk --video
```

官方评估脚本将视频写到 checkpoint 所在目录的 `videos/play/`，也可能生成 `exported/`。
重复录像前另行保留上一轮视频。这些目录位置遵循官方入口，启动脚本不会移动产物。

### 2. 官方速度＋身体高度控制

```bash
uv run --frozen scripts/g1_extensions/stage1.py height
```

任务 `Velocity-Height-G1-History-v0`，加载匹配的 history checkpoint，同时切换速度和身体高度命令。
这是后续 PickPlace 使用的下肢控制能力的基础验证；PickPlace 实际冻结加载的是 teacher TorchScript，
不是这里的 history checkpoint。
本阶段仅做运行检查；固定命令序列、速度/高度跟踪误差、跌倒率等定量对照留到阶段二。

### 3. PickPlace 场景与接口检查

```bash
uv run --frozen scripts/g1_extensions/stage1.py pickplace-scene --steps 100
```

调用官方 `scripts/play.py --task PickPlace-G1-v0`，用正弦动作检查带手 G1、桌面、物体、参考轨迹和冻结下肢策略是否能加载/步进。
这里的上层动作不是已训练策略，不能把摔倒或抓取失败当作训练结果，也不能把该回放当作成功抓取 demo。
阶段一不使用固定根链接的 `PickPlace-G1-Debug-v0` 代替真实自由基座测试。

### 4. 至少一次 PPO 更新

```bash
uv run --frozen scripts/g1_extensions/stage1.py pickplace-train
```

默认 32 个环境、2 次 PPO 迭代、每环境每次 24 步，沿用官方奖励、控制和 PPO 配置，
仅覆盖环境数量、训练次数、实验目录、保存间隔与 logger。使用 TensorBoard，无需 W&B 账号。
保存间隔设为 1，并要求日志含预期迭代、checkpoint 和解析后的 `env.yaml`、`agent.yaml`，否则记为 `evidence_missing`。
两次更新用于验证采样、反向传播和模型保存链路，不意味着策略收敛或已学会抓取。

默认命令不恢复任何旧 checkpoint。资源不足时先降到 8 个环境；跑通后可测量 32/64/128：

```bash
uv run --frozen scripts/g1_extensions/stage1.py pickplace-train --num-envs 8
uv run --frozen scripts/g1_extensions/stage1.py pickplace-train --num-envs 64 --iterations 20
```

这些数量是待测起点，不是已验证的 4090 容量结论。正式测量吞吐应增加迭代数，排除初始化和前几次更新的影响。

### 5. 训练产物回放

训练脚本结束会打印下一条带实际 checkpoint 路径的命令。也可从该次 `manifest.json` 的 `checkpoint` 字段获取：

```bash
uv run --frozen scripts/g1_extensions/stage1.py pickplace-eval --checkpoint /absolute/path/to/model_2.pt --steps 500
```

使用实际产物路径，示例中的文件名不保证等于该次保存名称。脚本不自动选取“最近一次”实验，避免加载错误模型。
仓库没有随附可直接用于该任务的 PickPlace 上层训练权重，不能拿下肢 checkpoint 替代。
短训模型回放只验证载入与推理；阶段二再进行足够训练并评估跟踪/操作表现。

## 核对后的任务与控制链

| 任务 | 本阶段用途 | PPO/策略控制范围 | 状态 |
| --- | --- | --- | --- |
| `Velocity-G1-History-v0` | 最先运行的行走基线 | 腿 12 关节＋腰 roll/pitch；5 帧历史 | 注册、随附权重已静态核对；待实跑 |
| `Velocity-Height-G1-History-v0` | 速度/高度控制 | 腿 12 关节；5 帧历史；其他关节由环境中的辅助动作处理 | 注册、随附权重已静态核对；待实跑 |
| `Velocity-Height-G1-Teacher-v0` | PickPlace 冻结下肢的策略来源 | 特权观测；速度与高度命令 | 随附 teacher 权重；待在 PickPlace 中验证 |
| `PickPlace-G1-v0` | 场景、PPO 小训与 checkpoint 回放 | 右臂/右手/腰增量关节目标＋4 维下肢命令 | 训练入口、参考轨迹已静态核对；待实跑 |
| `PickPlace-G1-Debug-v0` | 可选 GUI 检查 | 固定根、手动调关节/物体 | 不作为稳定性基线 |

`inventory` 另导出当前源码的全部 Gym 注册项（当前 17 项），每项标记 `runtime_verified=false`。
注册存在不等于资源齐全或可用任务验收通过。

PickPlace 的代码入口：

- `agile/rl_env/tasks/pick_place/g1/__init__.py`：官方任务注册与 PPO 配置关联。
- `agile/rl_env/tasks/pick_place/g1/g1_pick_place_tracking_env_cfg.py`：G1 带手模型与参考轨迹。
- `agile/rl_env/tasks/pick_place/pick_place_tracking_env_cfg.py`：场景、观测、动作、奖励、终止条件。
- `agile/rl_env/tasks/pick_place/g1/agents/rsl_rl_ppo_cfg.py`：PPO；每环境每次 24 步，默认完整训练上限 20000 次。
- `agile/rl_env/assets/motion_data/object_pick_and_place_retarget_motion_g1_3finger_hands.yaml`：既有参考轨迹。
- `agile/rl_env/mdp/actions/policy_actions.py`：冻结下肢 TorchScript 推理。

上层 `policy` 观测包括参考基座/关节偏差、物体与手的相对位置、物体跟踪误差、轨迹进度、
基座速度、关节位置/速度及上一动作；它输出右侧操作关节和腰的增量目标，左臂/手被保持在默认目标，
并通过 `lower_body_joint_pos` 输出 `[vx, vy, wz, height]`。下肢模块在 `torch.inference_mode()` 内调用 teacher，
仅把其关节输出施加到双腿；当前 PPO 不更新 teacher 权重。
确切关节数量、观测维数和动作维数以运行时 manager 输出/保存配置为准，不照抄文档中的简化数量。
仿真步长 0.005 秒，decimation=4，控制频率 50Hz；PickPlace episode 为 8 秒。

## 记录与验收

每次执行生成独立目录 `logs/g1_extensions/stage1/<UTC时间>_<唯一编号>_<动作>/`：

| 文件/字段 | 含义 |
| --- | --- |
| `manifest.json` | Git 提交/工作区状态、锁文件 SHA256、解释器、参数、实际子进程命令、状态和退出码 |
| `preflight.json` | 包版本、平台、GPU、CUDA/patch 检查和失败原因 |
| `tasks.json` | 仅 `inventory` 生成的源码注册清单 |
| `stdout.log` | 实际运行的官方入口完整输出与错误 |
| `gpu.csv` | 约每秒一次的逐 GPU 显存使用和利用率采样 |
| `throughput_env_steps_per_s` | 从 RSL-RL 更新日志提取的采样/学习吞吐；推理时可能为空 |
| `checkpoint` / `configs` | 仅本次训练目录中的模型与解析配置路径 |

`sampled_whole_gpu_peak_mib` 是采样到的**整卡**显存最高值，包含其他进程，且可能漏掉短时峰值；
它不是 PyTorch allocator 峰值。`wall_seconds_including_startup` 包含启动、资产加载和关闭耗时，不能直接当稳态吞吐。
`gpu_sample_count=0` 表示遥测缺失，不能完成显存验收；需修复 `nvidia-smi` 后重跑。
默认 1800 秒超时；可用 `--timeout 3600` 调整。超时或 Ctrl+C 会停止子进程并记录状态。
运行时以 `stdout.log` 查看进度；包装脚本每 30 秒打印运行心跳。

| 验收项 | 通过标准 | 当前状态 |
| --- | --- | --- |
| 环境 | Linux 4090 的 preflight 通过，记录锁文件与版本 | 待 AutoDL |
| 下肢 | walk、height 实际运行；保留日志并观察动作/视频 | 待 AutoDL |
| PickPlace | 场景加载步进，无依赖/维数错误 | 待 AutoDL |
| 训练 | 至少一次真实 PPO 更新，有 checkpoint、解析配置与吞吐记录 | 待 AutoDL |
| 回放 | 载入本次上层 checkpoint 后运行，无维数/载入错误 | 待 AutoDL |
| 资源 | 上述运行有 GPU 样本、运行时间、训练吞吐 | 待 AutoDL |

`passed` 只表示该命令运行成功；不代表稳定行走、抓取成功或收敛。阶段一最终验收需逐项检查上述证据。
本包装器不启用官方通用 motion evaluator：部分 PickPlace 配置没有 `eval` 观测组，
也不将缺少任务专用成功判据的通用指标当成抓取成功率。

本地无仿真依赖验证（已有独立 Python 时）：

```powershell
uv run --no-project --python 3.15 python scripts/g1_extensions/test_stage1.py
uv run --no-project --python 3.15 python scripts/g1_extensions/stage1.py walk --dry-run
```

这里的 3.15 仅用于当前开发机的标准库工具测试；AutoDL 仿真始终使用仓库锁定的 3.12。
`--dry-run` 不做环境检查、模型载入或训练，状态固定为 `dry_run`，不能充当验收记录。

完成阶段一后：固定命令做速度/高度误差与跌倒统计 → PickPlace 正式训练与轨迹误差评估 →
梳理冻结/训练模块和上下肢接口 → 开始 FK/IK/Jacobian 与腰臂协调扩展。
下肢从零 PPO 训练放在主线跑通以后，不阻塞 PickPlace。

## 本地验证记录（2026-11-05）

- 静态解析出 17 个官方任务注册项；五个启动动作的命令预览完成。
- 标准库测试覆盖命令与官方 CLI/任务注册的对应、锁文件版本匹配、LFS 检查、GPU 样本解析、
  更新日志解析、子进程失败/超时、训练产物隔离、预检拦截与训练证据缺失，共 11 项。
- Ruff 代码检查与格式检查通过；未启动 Isaac Sim、未进行策略训练。
- 本机预检记录：`logs/g1_extensions/stage1/20261105T023259Z_611802f0_preflight/`。
  本地是 Windows 11、Python 3.15.2、RTX 5060 Laptop 8151 MiB，未安装目标仿真依赖；
  状态为 `preflight_failed`。三个所需策略文件和参考 YAML 已检查存在，且不是 LFS pointer。
- 运行日志位于被 Git 忽略的 `logs/` 中；AutoDL 实跑后需另行保留/下载证据，并补充上述验收表。
