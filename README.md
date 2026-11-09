# RECAP on Cobot Magic

在 AgileX Cobot Magic（双 Piper，ROS + 四路 CAN）上完整复现 RECAP（π\*0.6 式离线强化学习）的工程：
示教采集 → LeRobot 转换 → π0.5 微调 → 真机推理 → 带人工接管采集 → RECAP（Value、优势标注、CFG）→ 部署迭代。

> 本仓库只包含代码与配置。**数据集、微调权重和 Python 环境都不上传**；环境由 `env/install_all.sh` 安装，任务数据需要按文档自行采集。

## 快速开始

参考环境：Ubuntu 20.04、ROS Noetic、NVIDIA 驱动 ≥ 535（CUDA 12）、RTX 4090。π0.5 全参微调和 RECAP 训练建议放到多卡服务器上。

```bash
git clone https://github.com/ROI-su7/RECAP-CobotMagic.git ~/RECAP
cd ~/RECAP
bash env/install_all.sh     # apt/ROS（需要 sudo）+ 两个 Python 环境 + 基座权重
bash env/verify.sh          # 环境、ROS 工作空间与单元测试自检，不会动机械臂
```

**换一台机器时，先改 CAN 的 USB 映射和相机序列号**（[docs/FOUR_CAN.md 第 1.3 节](docs/FOUR_CAN.md)）。之后按 [docs/PIPELINE.md](docs/PIPELINE.md) 一步步执行。

## 目录

| 路径 | 内容 |
|---|---|
| `docs/PIPELINE.md` | **完整流程**，从环境到 RECAP 的每一步命令 |
| `docs/FOUR_CAN.md` | 四路 CAN、示教采集、人工接管、停机与故障处理 |
| `env/` | 一键安装、固定版本依赖、自检 |
| `openpi/` | openpi fork：π0.5 训练、数据配置、原生 JAX 服务、Piper 真机客户端 |
| `openpi/RLinf/` | RLinf fork：RECAP 训练（Value / advantages / CFG）、`rlinf/recap_piper`（记录器、导出、服务）、`slurm/` 训练脚本 |
| `openpi/RLinf/README_PIPER_RECAP.md` | RECAP 打分规则、参数含义、训练细节和问题记录 |
| `cobot_magic/Piper_ros_private-ros-noetic/` | Piper ROS 驱动：四路 CAN、主从接管 mux |
| `cobot_magic/lingbot_piper/` | 只用两个从臂时的修复版 mux |
| `cobot_magic/camera_ws/`、`collect_data/`、`tools/` | Astra 相机、示教录制与 LeRobot 转换、CAN/相机工具 |

## 安全

- 三种 Piper 会话互斥：示教采集、两从臂推理、监督推理/人工接管。
- launch 运行时不要重配 CAN。
- PiPER 没有关节抱闸，失能或急停时机械臂会下落。第一次测试 H（回原点）或任何失能操作时，请有人扶着机械臂。
- 软件 HOLD 不是安全认证急停。

## 许可

openpi、RLinf、lerobot 均为 Apache-2.0，其余组件保留各自许可证，见 [NOTICE](NOTICE)。
