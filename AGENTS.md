# G1 二开目录约定

复用仓库现有结构，统一用 `g1_extensions/` 子目录存放二开内容：

- `docs/g1_extensions/`：项目计划、设计说明和实验记录。
- `scripts/g1_extensions/`：云端训练、评估和本地推理的启动脚本，复用现有入口。
- `agile/rl_env/tasks/pick_place/g1_extensions/`：自定义 PickPlace 任务及其配置和控制代码。

为自定义任务注册独立的任务 ID，保留 `pick_place/g1/` 作为官方基线。各子目录按需创建。
