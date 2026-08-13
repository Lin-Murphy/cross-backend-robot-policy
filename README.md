# Cross-Backend Robot Policy Deployment and Evaluation

用于研究策略、任务评价与记录如何跨机器人后端复用。当前实现包含 MuJoCo 仿真和 SO101 真机适配实例；ACT、DOT、SmolVLA 与 ALOHA 是验证用的策略或任务，不代表任意组合都已支持。

## 已验证到哪里

- SO101/MuJoCo 共用观测、动作请求、执行回执和任务阶段记录接口。SO101/SmolVLA 有一次经现场确认的胶带放置成功开发回合，共 494 个六关节目标；这不是成功率估计，也不是同策略、同初态的真仿性能比较。
- SO101/ACT 三次真机开发回合各发送 50 个目标，验证第二策略可进入相同的发送、拒绝和记录路径；任务结果未知。
- 独立的 ALOHA TransferCube 任务中，ACT 与 DOT 在配对的 30 个初态上分别成功 11/30、26/30。两者仍使用不同评测入口，之后按统一判据汇总。
- MuJoCo 方块示例的固定轨迹在 799 步完成任务；固定轨迹不是学习策略。

结论、证据范围和未完成条件见[结果与边界](docs/RESULTS.md)。接口划分见[架构](docs/ARCHITECTURE.md)。Git 中仅保留源码、测试、必要配置与可运行的仿真场景/网格。原始视频、相机画面、试跑日志、权重与长篇开发记录保留在本机工作区，不进入 Git；因此 Git 副本不能独立重验历史真机视频或重新推理模型。

## 无机器人仿真自检

需要 Linux、Python 3.12、MuJoCo 3.3.7、NumPy 2.5.3、FFmpeg 和可用的 EGL 渲染环境。准备本地 Python 环境后运行：

```bash
bash scripts/run_sim_cube_v1.sh
```

默认使用 `.venv-sim/bin/python`；可用 `SIM_V1_PYTHON` 指定其他已安装依赖的解释器。该命令只执行仿真固定轨迹，结果写入被 Git 忽略的 `artifacts/sim-cube-v1-demo-*`。依赖列表见[requirements-sim-v1.txt](requirements-sim-v1.txt)。已有本地模型权重和相应环境时，可用 `bash scripts/run_aloha_3d_comparison.sh` 跑 ALOHA 对比；权重不随 Git 分发。真机入口需要单独的现场方案与确认，不能把历史开发参数直接当作新设备安全参数。

原创代码和文档采用 [MIT](LICENSE)；随附 SO101 网格及 DOT 兼容代码保留各自许可证，见[第三方说明](THIRD_PARTY_NOTICES.md)。
