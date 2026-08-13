# 架构边界

策略适配器把模型输入输出转换为带名称和单位的观测/动作；执行合同检查关节映射和单位，后端负责实际观测与目标发送并返回回执；任务评价器依据任务定义生成阶段状态和运行记录。拒绝的候选也须记录，不能记成已发送目标。

主要实现位于 `src/cross_backend/`：`execution_contract.py` 定义后端中立的动作请求与映射，`so101_lerobot_backend_adapter.py` 和 `so101_dispatch_backend.py` 接入真机，`tape_sim_backend.py` 接入 MuJoCo，`tape_task.py`、`completion_record.py`、`run_record.py` 负责评价与记录。SO101 真机开发入口为 `scripts/run_so101_legacy30_audited_pilot.py`，需新的现场方案和明确批准才可发送动作。

共用边界已在一个真实 SO101 实例中验证。ALOHA ACT/DOT 比较的两个模型仍经不同评测入口运行；任意新机器人、模型或任务都需要适配及独立验证。
