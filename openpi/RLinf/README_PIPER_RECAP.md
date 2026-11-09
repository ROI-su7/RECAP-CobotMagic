# Cobot Magic / Piper：独立 RECAP 适配与真机流程

本文件说明 RLinf 官方 RECAP 离线流程在 Piper 上的完整使用方法：

```text
JAX 基础策略 / 上一轮 RECAP 策略 + 原有 T 人工接管
       ↓ 采集图像、状态、实际仲裁动作，C/X 标记整条回合
独立 raw episodes → 新的 LeRobot 快照（rollout）
       ↓ + 原 350 条示教（sft）
计算 returns → 训练 Value Model → 计算 advantage → CFG 微调 π₀.₅
       ↓
独立 websocket 服务 → 原 OpenPI 真机客户端 → 原 ROS mux → 下一轮采集
```

RECAP 的 Value Model 只用于
给离线数据计算 advantage；最终更新的是 π₀.₅ 动作专家上的 LoRA 适配器，基础权重冻结。

参考：仓库内 [官方 RECAP 说明](docs/source-en/rst_source/examples/embodied/recap.rst)、
[中文说明](docs/source-zh/rst_source/examples/embodied/recap.rst)，以及本机
[docs/FOUR_CAN.md](../../docs/FOUR_CAN.md)。人工接管的 CAN 角色切换、延时、HOLD 和 HOME 都由该文档中的
原 ROS launch/mux 完成，新代码只发送相同请求。

## 0. 总流程一览

| 步骤 | 在哪做 | 本文章节 |
| --- | --- | --- |
| 开机准备：roscore、CAN、相机 | 本机 | 第 3 节 |
| 用原生 JAX π₀.₅ 采集成功/失败/接管回合 | 本机 | 第 4 节 |
| 导出 LeRobot 快照 | 本机 | 第 5 节 |
| 准备 350 条示教的隔离副本 | 本机或服务器 | 第 6 节 |
| JAX→PyTorch 转换、SigLIP2/Gemma3 权重 | 服务器（推荐）或本机 | 第 7 节 |
| returns → value → advantages → cfg | 服务器（推荐，第 9 节）或本机 4090（第 8 节） | 第 8、9 节 |
| 对照基线：SFT + 人工接管片段 | 本机 4090 或服务器 | 第 8.5 节 |
| 部署 CFG 策略、评估并采集下一轮 | 本机 | 第 11 节 |

第一轮不需要任何转换，直接用已训练好的 JAX checkpoint 采集。

## 1. 数据组成：RECAP 到底需要什么

### 1.1 三类回合及其作用

| 回合类型 | 怎么得到 | 标记 | 在 RECAP 中的作用 |
| --- | --- | --- | --- |
| 自主成功 | `I → C` | `is_success=true` | 告诉 Value 哪些状态离完成近 |
| 自主失败 | `I → X` | `is_success=false` | **RECAP 最依赖的数据**：Value 只有见过失败才能区分好坏动作 |
| 人工接管 | `I → T → 人工 → I → … → C/X` | 接管帧 `intervention=true` | 在策略容易出错的阶段提供纠正动作 |

接管回合也必须按最终结果打 C 或 X，人工救回来的回合标 C。接管帧中保存的是主臂经
mux 仲裁后的实际命令，即人工动作。

### 1.1.1 按键与 RECAP 打分机制的对应

RECAP（π\*0.6 式 5 与 IV-B/V-C/V-D 节）只有三个打分量：每回合一个**成功标签**、由它推出的
**稀疏奖励**，以及人工纠正的**改进指示 Iₜ**。每个按键对应其中一项：

| 按键 / 事件 | 记录的字段 | RECAP 中的量 | 训练时的取值（默认配置） |
| --- | --- | --- | --- |
| **I**（推理确认后自动开回合）之后的每个记录帧 | 一帧 | 普通步奖励 | rₜ = −1 |
| **C**（三根都插好、灯亮） | 终止帧 `is_success=1` | 成功标签 = 1 | r_T = 0 |
| **X** | 终止帧 `is_success=0`，`failed_insertion` | 成功标签 = 0 | r_T = −C_fail，`failure_reward: -3500` |
| `--time-limit-s` 到时 | 同 X | "在时限内完成"才算成功（论文 200/500/600 s） | 同 X |
| **T** … **I** | 人工帧 `intervention=1` | 人工纠正：**强制 Iₜ = True** | positive，与回合成败无关（`intervention_success_only: false`） |
| 其余帧 | `intervention=0` | Iₜ = 1[A > ε]，ε 取 30% 分位 | 由 Value 计算的 advantage 决定 |
| 示教（`type: sft`） | 无按键 | 全部成功 | r 同上；RLinf 约定全部 positive |
| N / U（可选，**不需要按**） | `insertions_done` | **不进入奖励**，只用于按根统计 | — |
| 空格 / H / Q | 不写帧 / 回合作废 | 不打分 | 不进入训练 |

γ = 1，回报 Rₜ = Σ r，即"−剩余步数"（失败再加 −C_fail）。C_fail 按论文要求"使失败状态的价值足够低"取
3500，大于最长成功回合（示教最长 3336 步）。上游默认值 −300 只适合约 300 步的任务。

下面两项**不属于 RECAP 原文**，单独列出，可分别关闭：

| 扩展 | 配置键（默认值） | 来源 | 关闭后 |
| --- | --- | --- | --- |
| 接管惩罚 | `data.takeover_penalty: 300.0` | Evo-RL：接管前一步 rₜ = −1 − C_int | `=0` 即严格的 RECAP 奖励 |
| 按剩余根数加重失败 | `data.stage_failure_cost: 0.0`（默认已关闭） | 本适配的可选项 | — |

`intervention_success_only=true` 是 Evo-RL 的变体，只对成功回合的人工帧强制为正。C_int=300 是经验起点
（Evo-RL 未公开取值），大约相当于"一次接管 = 多走 300 步"。

采集建议（经验值，不是官方要求）：

- 第一轮先采 30–50 条，失败回合不少于 1/3。不要只保留成功回合，也不要中途删掉失败回合。
- 不要刻意制造失败：按平常的初始摆放让策略自己执行，自然失败就按 X。
- 策略在精细阶段明显要失败时再按 T 接管，接管前的帧仍然是策略自己的动作。
- 每次只改一个条件（摆放、光照、线材位置），以便判断哪个阶段在改进。

### 1.2 要与原来的 350 条示教融合吗

**要，推荐融合。** 这也是官方 RECAP 的做法：示教作为 `type: sft`（全部视为成功），
真机采集作为 `type: rollout`（含成功、失败和接管）。各阶段的用法：

| 阶段 | 是否加入 350 条 | 原因 |
| --- | --- | --- |
| returns | 两份分别计算 | sft 每条都按成功计算回报；rollout 读取 C/X |
| Value 训练 | **加入** | 350 条提供大量"离完成多远"的信号，少量 rollout 提供失败信号 |
| advantages | **两份都算** | 全局统一回报范围和 30% 阈值；CFG 训练要求每份数据都有 advantage。RLinf 约定 `type: sft` 的**所有帧**强制为 positive，阈值只作用于 rollout |
| CFG 训练 | **加入** | 保持原有示教分布，避免只在几十条 rollout 上微调导致退化；`balance_dataset_weights: true` 按帧数比例采样（见第 8 节第四步） |

已核对的兼容性：

- 原示教 `all_wire_data`：350 条、562k 帧，三路相机名与 14 维 state/action 与导出快照一致，
  任务提示词与 `pi05_connect_strings` 相同。
- 示教数据为 30 fps，记录器默认 15 fps。原客户端以 `--max-hz 15` 执行，每个动作步
  1/15 s，因此**按"步"计算**，自主回合与示教一致。回报以步计，CFG 动作块按各数据集
  自身 fps 取连续 50 帧。**不要**把示教下采样到 15 fps，否则动作块每步位移翻倍、执行速度变快。
- 人工接管段是按真实时间以 15 fps 记录的，每步位移约为示教的 2 倍。为免把接管段的
  速度学进去，接管时主臂动作宜慢而平稳，评估时注意接近段速度是否变化。
- 示教回合长度为 1166–3336 步，平均 1606 步。上游默认 `failure_reward=-300` 是按较短回合
  设定的，用在这里会使"失败前的状态"看起来比"刚开始的成功状态"更好。Piper 配置因此默认
  `failure_reward: -3500`（大于最长成功回合的步数），保证任何失败状态的回报都低于任何
  成功状态。同一轮 returns、value、advantages 必须使用同一套 returns 结果。
- `compute_returns` 会**原地改写**数据集的 `meta/stats.json` 和 `meta/info.json`。因此不能
  直接指向 pi05 原训练数据目录，要先按第 6 节建立副本。
- 示教为 mp4 视频格式，导出快照为 PNG。视频解码由首次 2-step GPU 冒烟验证。

## 2. 隔离与当前验证边界

| 项目 | RECAP 使用位置 |
| --- | --- |
| 原始回合（可持续追加） | `recap_data/raw/episode_<uuid>/` |
| 每轮不可覆盖的 LeRobot 快照 | `recap_data/lerobot_v1/`、`lerobot_v2/` 等 |
| 350 条示教的隔离副本 | `recap_data/sft_all_wire_data/` |
| 转换后的基础策略、Value 预训练权重 | `recap_artifacts/` |
| HF 数据缓存 | `recap_artifacts/datasets_cache/` |
| 训练日志、checkpoint | `recap_runs/<时间>/value/`、`cfg/` |
| websocket | `127.0.0.1:6016` |
| 离线训练 Ray head | `127.0.0.1:6385`，dashboard 可用 8266 |

脚本不执行 `ray stop`，不自动启动机械臂或修改 CAN。离线训练只连接显式指定的
RECAP Ray head。**有其他模型服务或训练运行时，不要在同一张 4090 上启动
RECAP 模型服务、转换或训练；端口分开并不隔离显存。** 脚本的 GPU 空闲检查会拒绝启动。

解释器默认是 `openpi/RLinf/.venv-recap/bin/python`（`env/install_all.sh` 创建），可通过 `RECAP_PYTHON` 指定另一个兼容环境。

已在 CPU 上验证：原始回合写入、成功/失败回报、LeRobot 导出、三相机 Value 输入、
advantage 标签保留、LoRA 参数选择、四步配置解析（含第 8 节的示教+rollout 融合覆盖）。
尚未验证：实际预训练权重加载、显存峰值、示教视频解码，以及 CFG 的真机效果。CPU
测试通过不代表真机成功率。推荐先用 `runner.max_steps=2` 做完整 GPU 冒烟。

## 3. 每个新终端与每次开机的准备

### 3.1 每个新终端先加载独立环境

```bash
cd $RECAP_ROOT/openpi/RLinf
source examples/recap_piper/env.sh
bash examples/recap_piper/run.sh check
```

默认使用本机已经存在的原生基础策略：

```text
config: pi05_connect_strings
JAX: $RECAP_ROOT/openpi/checkpoints/pi05_connect_strings/pi05_connect_strings_repro/30000
stats: <上述路径>/assets/all_wire_data/norm_stats.json
```

`RECAP_TASK` 通过 AST 读取原 OpenPI config 的 `default_prompt`，不会导入并初始化
JAX。当前应输出：

```text
connect the light strings and the battery box to the splitter cable so the lights turn on.
```

不要将另一篇 README 的 `insert the wire` 提示词直接套到 connect-strings checkpoint。
若换基础策略，在 `source` **之前**设置匹配的配置名、JAX 路径、stats，并让 task 重新解析：

```bash
export RECAP_SOURCE_CONFIG=pi05_wire_all_data_full
export RECAP_PI05_JAX_DIR=$RECAP_ROOT/openpi/checkpoints/pi05_wire_all_data_full/pi05_wire_all_data_full_20260818/30000
# 从该 checkpoint 的 assets 中选择与它训练时一致的 norm_stats.json，不能沿用另一模型的统计。
export RECAP_NORM_STATS=/绝对路径/该模型的/assets/数据集名/norm_stats.json
unset RECAP_TASK
source examples/recap_piper/env.sh
```

同一轮采集、转换、CFG 训练和部署必须使用同一套 source/task/stats。

### 3.2 开机准备（四 CAN README 第 4 节）

先确认 GPU 上没有其他模型服务，且没有旧的 Piper/客户端进程：

```bash
nvidia-smi
ps -ef | grep -E "piper|roslaunch|examples.piper_real" | grep -v grep
```

终端 1，ROS Master：

```bash
source /opt/ros/noetic/setup.bash
roscore
```

配置 CAN。只能在全部 Piper 节点都停止时执行；launch 运行期间严禁重新执行：

```bash
cd $RECAP_ROOT/cobot_magic/Piper_ros_private-ros-noetic
sudo bash can_config.sh
ip -br link show type can
# 必须看到 can_l_leader、can_l_follower、can_r_leader、can_r_follower 均为 UP
```

终端 2，相机：

```bash
cd $RECAP_ROOT/cobot_magic/camera_ws
source /opt/ros/noetic/setup.bash
source devel/setup.bash
roslaunch astra_camera multi_camera.launch
```

## 4. 真机采集：复用现有主从接管

不运行独立 `piper_mode_keyboard.py`，也不同时运行另一个 policy 客户端。

### 终端 A：四臂 launch

```bash
cd $RECAP_ROOT/cobot_magic/Piper_ros_private-ros-noetic
source /opt/ros/noetic/setup.bash
source devel/setup.bash
roslaunch piper start_inference_takeover.launch \
  master_left_can:=can_l_leader \
  master_right_can:=can_r_leader \
  puppet_left_can:=can_l_follower \
  puppet_right_can:=can_r_follower \
  publish_hz:=200
```

### 终端 B：初始基础策略服务

```bash
cd $RECAP_ROOT/openpi/RLinf
bash examples/recap_piper/run_native_server.sh
```

它只在 RECAP 的 6016 端口启动原生 JAX 基础策略。
后续已经训练好 RECAP 时，用第 11 节的 CFG 服务替代此服务，不能两者同占 6016。

### 终端 C：唯一键盘和记录器

```bash
cd $RECAP_ROOT/openpi/RLinf
bash examples/recap_piper/run.sh collect --profile takeover
```

### 终端 D：原 OpenPI 真机客户端

```bash
cd $RECAP_ROOT/openpi/RLinf
bash examples/recap_piper/run_client.sh takeover
```

这个脚本复用原 OpenPI 客户端，使用 6016 端口、关闭视频。
`takeover` 和 `followers` 都采用桌面《Piper双从臂推理问题与解决方案》的 Pi05
抓取时序参数：`--action-horizon 10 --rtc-prefix-len 4 --rtc-boundary-blend-steps 5
--action-smoothing-alpha 0.5`。`takeover` 按现场要求保留原来的手臂速度上限
`--max-arm-velocity-rad-s 0.4`；`followers` 保持文档参数 `0.6`。
旧 takeover 的 20/15 设置会让约 85 ms 的推理结果在约 16 步（1.07 秒）后才执行，
可能使夹爪在手臂到达前闭合。新设置缩短旧观测预测的执行时间，并减小手臂滞后。
运行 `bash examples/recap_piper/run_client.sh takeover --dry-run` 可查看命令而不启动机器人
（未设置 `RECAP_TASK` 时仍会读取任务配置）。启动日志也记录完整命令。
修改后需在 HOLD 状态停止旧客户端并重新启动终端 D 才生效；抓取效果仍需实机复测。
客户端使用本机 `pi0` conda Python，默认会使能并开始推理；不要另外发布 `/enable_flag=true`。
原客户端仍负责 200 Hz 发布、RTC 和 HOLD ACK，mux 负责主从控制权。
RECAP 入口 `rlinf.recap_piper.client` 只对交接位置偏差做 5 步平滑衰减，
保留新预测内部的运动增量，避免旧轨迹慢速时反复压低互插阶段的小幅推进。
每臂交接修正量不超过 `0.002 rad` 时直接采用新预测，由客户端原有 EMA 平滑；
更大偏差的过渡限制在上一目标与新预测窗口范围内，夹爪指令不参与融合。
启动后的 ROS 日志包含 `RTC offset-decay handover` 及最大修正量，便于核查。
**当前默认（2026-11-02 起）：全程执行 10 步，不再扩展。** round_001 策略的失败集中在插入前最后
约 1 mm 的同心对准，30 步（约 2 s）只重规划一次相当于最后对准时不看新图像；改为 10 步让最后对准
每约 0.67 s 用新观测修正一次。代价是下面离线分析指出的推进段可能变慢，需实机对比。
`RECAP_STABLE_HOLD_HORIZON=30 bash examples/recap_piper/run_client.sh takeover` 恢复下述扩展：

两个夹爪都经历过张开（>35 mm），随后连续 3 次观测
均闭合到 ≤30 mm，且新预测直到将执行窗口结束都保持 ≤30 mm 时，执行长度扩展为
30 步，以执行对齐计划的后续推进段；这只是持续夹持的调度判据，不保证已抓住物体。
预取仍为 4 步，不改变速度上限；新预测包含松爪、观测条件不满足或预测剩余长度不足时，
下次动作块交接恢复 10 步。只在交接处切换长度，保持预取触发点正确。
启用扩展时，ROS 日志中的 `RTC execution horizon 10 -> 30` 表示进入持续夹持调度；默认 10 步时不出现该日志。
2026-10-29 最后一条 `episode_a19e24e0eeba48b9b07f21025dbac441` 的第 650/850 帧
离线重复推理显示，右臂预测第 5–15 步净位移约 0.002–0.007 rad，而第 30–45 步
约 0.017–0.033 rad；固定短动作块会反复重规划，截断后续推进。该对比是离线预测，
不能等同于实机提速倍数。
推理等待导致 HOLD 时不沿用旧速度，键盘 HOLD/reset 仍走原客户端的清队列流程。
HOLD/reset 同时清除持续夹持计数和张开历史，恢复后先采用短动作块。
该修正保留当前速度上限和抓取时序参数；需要重启终端 D，实机效果待复测。

### 按键（全部在终端 C）

| 按键 | 行为 |
| --- | --- |
| I | **没有打开的回合时**：请求 INFERENCE，mux 确认后**自动开始新回合**，从此每步 rₜ = −1。**回合进行中**：恢复推理，仍是同一回合。接管（T）之后直接按 I 即可，记录器会自动先请求 HOLD 再请求 INFERENCE（原 mux 只允许从 HOLD 恢复推理），**不需要再先按空格**。上一回合 C 成功后，必须先 H 回原点才接受 I |
| B | 可选：手动开始回合（例如回合从 TAKEOVER 开始）。一般不需要 |
| T | 原 TAKEOVER 请求；看到原 launch 的 ACTIVE 确认后再动主臂。**不结束回合**，按 I 回到推理仍是同一回合 |
| C | 成功：三根都插好、灯亮时按。保存 `is_success=true`，请求 HOLD |
| N / U | **可选，正常采集不需要按**。每插好一根按 N（U 撤销），只用于 `run.sh score` 的按根统计。加 `--require-marks` 启动时，C 必须先有 3 次 N |
| X | 失败：保存 `is_success=false`，并记下失败发生在第几根，请求 HOLD |
| 空格 | 原软件 HOLD，保持使能；**回合不结束**，暂停帧不写；I 恢复 |
| H | 已确认 HOLD 后请求 HOME：从臂夹爪先原地张开 1.5 s 放下物体，张着回原点，到位后合上（超时也会合上）。**回合进行中按 H 会作废整条回合**，只在 C/X 之后用 |
| Q / Ctrl-C | 请求 HOLD 并退出记录器；没有 C/X 的当前回合标记 aborted，不入训练 |

正常序列（三次插接）：

- 自主回合：`I →（策略插三根）→ 灯亮按 C → 等待 HOLD → H → 重置三根线 → I`
- 失败后两种做法，按现场决定（失败回合在按 X 时已经保存，后面按什么都不会丢）：
  - 现场还能继续：`X → I`，新回合从当前现场开始，插完剩下的根数后灯亮即可按 C。
  - 现场无法继续（线扯坏、插头掉落、姿态卡住等）：`X → 等待 HOLD → H → 重置三根线 → I`，
    新回合从头开始。

**推理中途机械臂自己停下：** 原 mux 在策略命令超过 `policy_input_timeout_s`（1 s）没有更新时会自动
HOLD。记录器会打印"`The mux entered HOLD by itself`"，回合不结束，直接按 I 恢复。客户端和服务端的
输出会自动保存到 `recap_runs/robot_logs/client_*.log`、`server_*.log`，出现停顿时用来排查原因。

**某一根策略插不上、人工接管插好之后怎么做：** 直接按 I 让策略继续插下一根。
**不要按空格 + H**：空格只是暂停，H 会作废整条回合。三根都适用同样的做法，一个回合里可以接管多次：

```text
I                      策略开始，回合开始（每步 −1）
  第 1 根策略插不上  →  T（等 ACTIVE）→ 主臂插好 → I（策略继续，同一回合）
  第 2 根策略插好    →  不用按键
  第 3 根又插不上    →  T → 主臂插好 →（灯亮可直接 C）
  灯亮               →  C   成功回合：is_success=1，所有人工帧强制 positive
  等待 HOLD → H → 重置三根线 → I 开始下一回合

  救不回来           →  X   失败回合：is_success=0
  现场能继续 → I；不能继续 → 等待 HOLD → H → 重置三根线 → I
```

只有判定"这一回合救不回来"（线扯坏、超时、不打算接管）时才按 X。接管把这一根救回来不算失败。

H 的释放动作由原 ROS mux（`piper_four_state_mux.py`）执行，参数在 `start_inference_takeover.launch` 中
默认开启：`home_release_gripper_m:=0.06`（张开宽度，米）、`home_release_s:=1.5`（原地张开时长）。
设 `home_release_gripper_m:=0` 恢复原 HOME。其他 launch（含只用从臂的 `lingbot_piper` launch）
不受影响。修改前的文件备份在 `backups/ros_home_release_20261028/`。

### 4.1 三次插接的打分规则（RECAP 专用）

打分实现在 `rlinf/recap_piper/scoring.py`。

| 规则 | 说明 |
| --- | --- |
| 回合 = 完整的三次插接 | 与 350 条示教一致；不要按单根分回合，否则同一状态的回报目标会与示教冲突 |
| 成功 | 三根都插好且灯亮时按 C |
| 失败 | 判定救不回来时立即按 X（越接近出错时刻越好）；记录 `failed_insertion` |
| 超时 | RECAP 的成功是"在时限内完成"。`--time-limit-s` 到时自动按 X 记录并 HOLD |
| 接管 | 是纠正而不是失败；救回来、最终三根都插好就按 C |
| 从中途开始的回合 | 有效数据（Value 估计"从这里到三根全部插好还要多远"），但不计入自主成功率 |

时间限制只计记录的帧（HOLD 暂停不计），默认关闭。示教平均 1606 步，在 15 Hz 执行约 107 s，
最长 3336 步约 222 s。建议先取 240 s，同一轮采集保持不变：

```bash
bash examples/recap_piper/run.sh collect --profile takeover --time-limit-s 240
```

每个回合的 `episode.json` 中保存 `score`。只按 C/X 时只有成功/失败；按了 N 才有每根的帧号和
"失败在第几根"。导出时生成逐帧的 `insertions_done` 和 `insertion_event` 列，**不改变 RECAP 奖励**。

可选扩展（不属于 RECAP，默认关闭）：按剩余根数加重失败惩罚。

```text
失败终止奖励 = failure_reward − stage_failure_cost × (3 − 已完成根数)
data.stage_failure_cost=535     # ≈ 示教平均 1606 步 / 3；默认 0
```

统计采集结果（只读 raw 回合，不导出）：

```bash
bash examples/recap_piper/run.sh score
```

输出：回合数、成功数、从头开始的回合数、**无接管的自主成功数**（只按 C/X 即可得到）。
每一根的尝试次数、完成率、接管率和平均用时只统计按了 N 的回合（`per_insertion_episodes`）；
需要这部分时用 `--require-marks` 启动记录器。
旧版记录器采的、没有 `score` 的回合不计入统计，导出时 `insertions_done` 为 0。

`T` 两侧角色切换、实际等待时间、HOME 的路径均由当前 ROS 代码决定，记录器不会复写。
H 不会猜测成功/失败；按 H 前忘了 C/X，该条回合保留为 aborted，不会自动当成功。
记录中缺图像、反馈/命令过期或磁盘来不及写入，会 HOLD 并排除这条不完整回合。

每帧保存三路 RGB PNG、从臂状态、`/master/joint_left/right` 的最终仲裁目标、policy
目标、leader 位姿及人工标记。`intervention=true` 只在 `TAKEOVER/ACTIVE` 确认后记录。
policy 在人工阶段可能暂停，其新鲜性由 `policy_valid` 表示；没有真实 leader 流时
`leader_valid=false`。动作是 rad/米的命令目标，不是电机已经达到的位置。

仅从臂采集/推理见第 11.4 节：`bash examples/recap_piper/run.sh launch followers`。
记录器和客户端都选择 `followers` profile，T 被禁用。

停机：先在终端 C 按 Q，再停终端 D、B、A，最后停相机与 roscore（四 CAN README 第 9 节）。

## 5. 导出新的 LeRobot 数据快照

停止记录后执行；导出期间不要再往同一批待导出的 raw 回合写入。

```bash
cd $RECAP_ROOT/openpi/RLinf
source examples/recap_piper/env.sh
bash examples/recap_piper/run.sh export
bash examples/recap_piper/run.sh check --stage dataset
```

输出 `recap_data/lerobot_v1`，显示 episodes、successes、failures、frames。无 C/X 的
回合不会计入。导出保存本次的原始回合映射 `meta/piper_recap_manifest.json`。

以后继续采集时重复运行 collect 即可，raw 目录按 UUID 追加，不会覆盖旧回合。
下一轮导出选择新路径：

```bash
export RECAP_DATASET=$PWD/recap_data/lerobot_v2
bash examples/recap_piper/run.sh export
```

已有快照一律拒绝覆盖；不要删除正在训练的 dataset。四步使用同一个 `RECAP_DATASET`。
评估时按整条 episode 留出数据。

## 6. 准备 350 条示教的隔离副本

`compute_returns` 会改写数据集的 `meta/stats.json` 与 `meta/info.json`，因此建立副本：
`data/`、`videos/` 用硬链接（不占额外空间），`meta/` 完整复制（避免改写原文件）。

```bash
cd $RECAP_ROOT/openpi/RLinf
SRC=$RECAP_ROOT/openpi/lerobot_data/all_wire_data
export RECAP_SFT_DATASET=$PWD/recap_data/sft_all_wire_data
test ! -e "$RECAP_SFT_DATASET"            # 已存在就不要覆盖
mkdir -p "$RECAP_SFT_DATASET"
cp -al "$SRC/data" "$SRC/videos" "$RECAP_SFT_DATASET/"
cp -a  "$SRC/meta" "$RECAP_SFT_DATASET/"
```

硬链接要求在同一文件系统；服务器上用服务器的 `~/RECAP/openpi/lerobot_data/all_wire_data`
重复同样步骤。每轮可以复用这份副本，但换 `failure_reward` 后要重新计算 returns。

## 7. 准备模型资产

### 7.1 基础策略转换

```bash
cd $RECAP_ROOT/openpi/RLinf
source examples/recap_piper/env.sh
bash examples/recap_piper/run.sh convert
```

转换以 CPU 运行，输出 `recap_artifacts/pi05_base`，需要较多 CPU RAM 和磁盘。本机只有
31 GB 内存，推荐在服务器的 CPU 节点上转换后再传回本机。源模型是完整参数时保留完整
参数；源模型含原生 JAX LoRA 时保留其独立 A/B 分支。不写入源 checkpoint，已存在的
转换输出拒绝覆盖。

CFG 默认对动作专家做 LoRA 微调：新加 LoRA 的 A 随机初始化、B 初始化为零，因此初始输出
等同基础模型。保留源模型 50-step、32-D padded attention 布局；真机仍输出绝对 14-D 动作。

### 7.2 Value Model 的预训练权重

需要本地完整的 SigLIP2-so400m 与 Gemma3-270M（含 tokenizer）。两者都是 Google gated 模型：先在 Hugging Face 网页接受许可，再在本机登录并下载
（服务器无外网，下载后用 rsync 传过去）：

```bash
source examples/recap_piper/env.sh
"$(dirname "$RECAP_PYTHON")/huggingface-cli" login
"$(dirname "$RECAP_PYTHON")/huggingface-cli" download \
  google/siglip2-so400m-patch14-224 --local-dir "$RECAP_SIGLIP_DIR"
"$(dirname "$RECAP_PYTHON")/huggingface-cli" download \
  google/gemma-3-270m --local-dir "$RECAP_GEMMA_DIR"
```

不要将 token 写进 README 或 shell 脚本。

## 8. 四步离线训练（示教 + rollout 融合）

先退出模型服务与真机客户端，使 4090 空闲。离线训练不连接 ROS/CAN，也不需要
机械臂通电。脚本会拒绝在已有大显存模型进程运行时再启动模型任务。

新终端中设置本轮固定目录与数据，并启动**独立的** RECAP Ray：

```bash
cd $RECAP_ROOT/openpi/RLinf
source examples/recap_piper/env.sh
export RECAP_RUN_DIR=$PWD/recap_runs/round_001
export RECAP_DATASET=$PWD/recap_data/lerobot_v1
export RECAP_SFT_DATASET=$PWD/recap_data/sft_all_wire_data
export CUDA_VISIBLE_DEVICES=0
export RLINF_NODE_RANK=0
export RLINF_COMM_NET_DEVICES=eno1

# 融合数据的 Hydra 覆盖；同一轮四步必须一致。
RET_DATA="data.train_data_paths=[{dataset_path:$RECAP_SFT_DATASET,type:sft},{dataset_path:$RECAP_DATASET,type:rollout}]"
VAL_DATA="data.train_data_paths=[{dataset_path:$RECAP_SFT_DATASET,type:sft,weight:1.0,robot_type:piper,model_type:pi05},{dataset_path:$RECAP_DATASET,type:rollout,weight:1.0,robot_type:piper,model_type:pi05}]"
ADV_DATA="data.train_data_paths=[{dataset_path:$RECAP_SFT_DATASET,type:sft,weight:1.0,robot_type:piper},{dataset_path:$RECAP_DATASET,type:rollout,weight:1.0,robot_type:piper}]"
CFG_DATA="data.train_data_paths=[{dataset_path:$RECAP_SFT_DATASET,type:sft,weight:1.0},{dataset_path:$RECAP_DATASET,type:rollout,weight:1.0}]"

# 不执行 ray stop。
env -u RAY_ADDRESS "$(dirname "$RECAP_PYTHON")/ray" start --head \
  --port=6385 --dashboard-port=8266 --node-ip-address=127.0.0.1 \
  --num-cpus=8 --num-gpus=1 \
  --temp-dir=/tmp/rlinf_piper_recap_ray --disable-usage-stats
```

只用 rollout、不融合时，去掉各步的 `*_DATA` 参数即可（使用 yaml 默认值）。打分参数（C_fail=3500、
接管惩罚、强制正样本）已是 Piper 配置的默认值，见第 1.1.1 节。
每一步都可以先加 `--dry-run` 只解析配置，不建立 worker、不占显存。

第一步，计算回报：

```bash
bash examples/recap_piper/run.sh returns "$RET_DATA"
```

在两个数据集的 `meta/` 下分别写出 `returns_piper_recap_v1.parquet`，并更新各自的 stats/info。
回报按 RECAP 构造：普通步 -1，成功终止步 0，失败终止步为 `failure_reward`（-3500），gamma=1；
扩展项：每次接管发生前的最后一个策略帧再扣 `takeover_penalty`（默认 300，第 1.1.1 节）。
sft 数据集每条都按成功计算，没有 `intervention` 列，不受接管惩罚影响。这个负回报不回写 raw 数据。

第二步，训练 Value Model：

```bash
bash examples/recap_piper/run.sh check --stage value
# 首次先做 2-step GPU 冒烟（同时验证示教视频解码）；不要把它当可用的 value model。
bash examples/recap_piper/run.sh value "$VAL_DATA" runner.max_steps=2 runner.save_interval=1
# 正式训练（换新的运行目录，避免覆盖冒烟 checkpoint）：
export RECAP_RUN_DIR=$PWD/recap_runs/round_001_full
bash examples/recap_piper/run.sh value "$VAL_DATA"
```

4090 默认 micro batch=1、global batch=8，冻结 SigLIP 视觉编码器，训练 Gemma3、
新建的图像投影和轻量 Critic Expert。输出 201-bin 的归一化 V 值，回报范围取所有
数据集 stats 的全局最小/最大值。训练上限默认 2000 step、每 100 step 保存。正式步数应
按留出回合表现决定，固定步数不保证收敛。

第三步，给两份数据计算 advantage：

```bash
# 替换为第二步实际保存的 checkpoint，不要指向其他实验的 global_step。
export RECAP_VALUE_CHECKPOINT=$RECAP_RUN_DIR/value/piper_recap_value/checkpoints/global_step_2000/actor/model_state_dict
bash examples/recap_piper/run.sh advantages "$ADV_DATA"
```

使用 N=50 帧 lookahead（= 策略的 50 步动作块，`Advantage: positive` 条件约束的正是这一整块；
π*0.6 也是 N=H=50。rollout 15 fps 下是 3.3 s，示教 30 fps 下是 1.67 s），并先在回合内把 V(o_t)
做 1 秒的居中滑动平均（`advantage.value_smoothing_s: 1.0`）再做 N 步差分，结果写到每个数据集
`meta/` 下的 `advantages_piper_recap_adv_v1.parquet`。

为什么不是 round_001 用的 N=10：相邻帧（67 ms）的 V 预测中位相差 0.019，而 10 帧的真实奖励只有
-10/6481 = 0.0015，A 几乎全是 V 的帧间噪声。用 round_001 的逐帧 V 离线重算（62 个 rollout）：

| 设置 | 相邻帧标签翻转 | 成功/失败回合正样本率差 | 接管前 3 s 被标正 |
| --- | --- | --- | --- |
| N=10（round_001） | 28% | +6.4 个百分点 | 30%（未识别） |
| N=50 | 23% | +10.2 | 26% |
| N=50 + V 平滑 1 s（round_002 起） | 3% | +13.7 | 22% |

`value_smoothing_s: 0` 并改回 N=10 即可复现 round_001（已核对与 round_001 sidecar 一致到 float32 舍入）。
阈值（round_002 起）只在 rollout 帧上计算，取前 40% 为正（`positive_quantile: 0.4`、
`threshold_dataset_types: [rollout]`，对应 π*0.6 微调阶段"约 40% 的评估 rollout 为正"），再用同一阈值
标注所有数据集；示教帧仍全部强制为正，接管帧强制为正后 rollout 的实际正样本比例会略高于 40%。
round_001 是在含示教的全部帧上取前 30%，rollout 实际 48% 为正（其中强制正样本 7%）。

与论文、Evo-RL 的主要参数差异：π*0.6 为 50 Hz、H=N=50（1 s），C_fail 未公开，无接管惩罚，Value
为 670M 并联合网络数据训练、按任务最长回合归一化，每轮从预训练权重全量微调，β 默认 1（可用 1.5–2.5）。
Evo-RL 为 N=50 帧、按任务前 30%、Value 输入含离散化状态、C_fail=任务最长回合长度、可选接管惩罚 150。
V 平滑是本项目根据 Value 帧间噪声加的（二者都没有）。待后续评估：示教 30 fps 与 rollout 15 fps
的每帧时间成本不一致；Value 输入加入机器人状态；β>1 推理（需预取 7–8 步）。
阈值在合并后的全部帧上统一计算，rollout 中最高 30% 标为 positive；sft 示教的所有帧按 RLinf
约定强制为 positive；所有人工接管帧（不论回合成败，RECAP 原文做法）也强制为 positive（日志打印强制了多少帧、
最终正样本比例，sidecar 中有 `forced_positive` 列）。rollout 的 positive 不等同于"成功回合的
所有帧"：失败回合中也可能有相对更好的动作。新采集改变数据和回报范围后，应重新计算
returns、训练 Value、计算 advantage，不能复制旧 sidecar。

第四步，CFG 微调 π₀.₅：

```bash
bash examples/recap_piper/run.sh check --stage cfg
bash examples/recap_piper/run.sh cfg "$CFG_DATA"
```

正 advantage 帧使用 `Advantage: positive`，负帧使用 `Advantage: negative`，任一帧有 30% 概率去掉
条件、走无条件分支（`positive_only_conditional: false`、`unconditional_prob: 0.3`，与 π*0.6 和
Evo-RL 一致，round_002 起）。round_001 用的是"负帧只走无条件分支、正帧 10% 无条件"；部署时 serve
按 checkpoint 的 `piper_recap_policy.json` 记录的方式加载，两种 checkpoint 都能服务。`balance_dataset_weights: true`（RLinf 原语义）把每个数据集的 `weight` 再乘以
它的帧数，即在全部帧上均匀采样：round_001 中示教约 87.7%、rollout 约 12.3%（人工接管帧约 0.9%）。
要提高 rollout 比例，调大 `CFG_DATA` 中 rollout 条目的 `weight`（如 3.0 → rollout 约 30%）。使用原始物理动作、Piper/Aloha 适配、源统计归一化和 50-step flow-matching
训练。

默认只训练动作专家 LoRA，冻结其余基础参数，micro batch=4、global batch=4（单卡 4090；CFG 不能用梯度累积，必须保持 global = micro × GPU 数，否则报 `Cannot writeback when the parameter shape changes`），保存完整 FSDP
权重和 `piper_recap_policy.json`，用于严格恢复以及校验基础模型、任务和归一化统计。训练
上限默认 2000 step，每 100 step 保存；增大步数时同时调整优化器 total_training_steps。

### 8.5 对照基线：SFT + 人工接管（Evo-RL 的 SFT+Int）

只做普通 LoRA 微调：数据是 350 条示教，加上**成功回合中的人工接管片段**（每段连续的人工帧
导出为一个独立回合，自主帧和失败回合全部丢弃）。不需要 Value、advantage 和 CFG，是最便宜
的对照。Evo-RL 论文在 PiperX 插装上的结果：SFT 22.7% → SFT+Int 78.7%，而他们实现的 RECAP 为
57.8%。该结论单次实验、样本少，需要在本机任务上自行验证。

```bash
# 与 RECAP 使用同一批 raw 回合；默认输出 recap_data/interventions_v1，至少 10 帧的片段才保留
bash examples/recap_piper/run.sh export-interventions
bash examples/recap_piper/run.sh check --stage sftint     # 同时检查示教副本和接管片段
bash examples/recap_piper/run.sh sftint runner.max_steps=2 runner.save_interval=1   # GPU 冒烟
export RECAP_RUN_DIR=$PWD/recap_runs/round_001_sftint
bash examples/recap_piper/run.sh sftint
```

配置 `piper_recap_sft_int.yaml` 与 `piper_recap_cfg.yaml` 使用相同的基础模型、LoRA 参数、
优化器和步数（单元测试保证两者一致），区别只在：`data.plain_sft: true`（所有帧走原任务提示词，
不加 `Advantage: positive`），以及训练数据为 `RECAP_SFT_DATASET` + `RECAP_INT_DATASET`。
`export-interventions` 可加 `--min-segment-frames N` 调整最短片段。

部署时 `serve` 会读取 checkpoint 中 `piper_recap_policy.json` 的 `policy_mode` 自动选择配置；
基线只走无条件分支，不要传 `--guidance-scale`：

```bash
bash examples/recap_piper/run.sh serve \
  --checkpoint /绝对路径/sftint/piper_recap_sft_int/checkpoints/global_step_2000
```

建议对比方式：同一批初始摆放，交替部署 RECAP 和 SFT+Int（每个至少 20 次），统计无接管成功率、
完成用时和接近段速度。两者训练步数相同，才是公平对比。

训练结束后不要执行 `ray stop`：它会停止本机**所有** Ray 进程。
只有确认本机没有其他 Ray 集群时才可以停。

## 9. 在服务器上训练（推荐）

训练服务器（下文写作 `<server>`）是 Slurm 集群：登录节点只有 CPU，可以提交作业、
看日志、开 TensorBoard，但不能跑模型。可用分区：`3090`（finai01-16，每节点 8×RTX 3090 24 GB）、
`gpu`（hgx，8×A100-40G，经常要排队）；本账号**不能用 `cpu` 分区**。服务器无外网。代码镜像在
`~/RECAP/openpi/RLinf`，`.venv-recap` 可用；服务器上的 openpi `config.py` 是训练版本，
不要用本机文件覆盖。

### 9.1 一轮的总流程

```text
本机：collect → export（lerobot_vN）→ check --stage dataset
  ↓ rsync 代码 + 数据（9.3）→ touch .inputs_ready
服务器 prep 作业（3090 单卡，CPU 活）：示教副本 → JAX→PyTorch 转换 → returns
  ↓ afterok 依赖自动开始
服务器 train 作业（8 卡一个节点）：Value（8000 步）→ advantages（8 卡推理）→ CFG（30000 步）
  ↓ rsync 回本机（9.6）
本机：run.sh serve → 真机评估 / 下一轮采集（第 11 节）
```

| 脚本 | 作用 |
| --- | --- |
| `slurm/recap_server_env.sh` | 服务器路径（`RECAP_RUN_DIR`、`RECAP_DATASET`、示教副本）和四步的 `*_DATA` 覆盖 |
| `slurm/recap_round_prep.sbatch` | 等 `.inputs_ready` 标记（最多 8 h）→ 示教副本 → 转换 → returns → preflight → `prep.done` |
| `slurm/recap_round_train_8xa100.sbatch` | Value → advantages → CFG；每步写 `<stage>.done`，重交跳过已完成的步，并从最新 checkpoint 续训 |
| `slurm/recap_status.sh` | 一屏查看作业、阶段、步数/剩余时间、最新 checkpoint 和报错（只读） |

训练超参（与原 `pi05_connect_strings` 一致）：CFG global batch 128、30000 步、cosine
（warmup 1000，峰值 2.5e-5 → 2.5e-6）、AdamW β2=0.95、clip 1.0；Value global batch 256、8000 步。
正式训练前，每一步先用 3 个 step 的冒烟测试从大到小试 micro batch，取第一个能跑通的；不足的部分
用梯度累积补齐，global batch 不变。冒烟测试单次超过 45 分钟（`SMOKE_TIMEOUT`）判为失败，防止卡死。

### 9.2 3090 与 A100 的区别

脚本默认值按 A100-40G 设置（`no_shard`，micro batch Value 32/16/8、CFG 16/8/4/2）。round_001 在 A100
上实际选中 Value 16（32 OOM），CFG 的选择见 9.5 的 `recap_status.sh`。

**Value 和 CFG 都只能用 `no_shard`**（原因见 9.7 第 7、11 条）：openpi 的 PaliGemmaWithExpert 绕过解码层的
forward 直接调用层内投影，Value 模型绕过 FSDP 单元直接调用 embedding，按层分片的 `full_shard` 都会出错。
`RECAP_FSDP_SHARDING` 开关还在脚本里，但不要设成 `full_shard`。

3090 只有 24 GB，`no_shard` 下每张卡要放下整个 pi05 和动作专家的优化器状态，CFG 能否放下**尚未验证**；
在 3090 上跑时只调小 micro batch：

| 变量 | A100 默认 | 3090 |
| --- | --- | --- |
| `VALUE_MICRO_CANDIDATES` | `32 16 8` | `16 8 4 2` |
| `CFG_MICRO_CANDIDATES` | `16 8 4 2` | `8 4 2 1` |

checkpoint 都保存完整的 `full_weights.pt`，部署方式相同。3090 单步更慢，30000 步的总时长以
`recap_status.sh` 显示的剩余时间为准。

### 9.3 同步代码和数据

同步代码（不删服务器文件，被覆盖的旧文件备份到 `backups/sync_<日期>/`）：

```bash
cd $RECAP_ROOT/openpi/RLinf
rsync -a --backup --backup-dir=$HOME/RECAP_backups/sync_$(date +%Y%m%d) +%Y%m%d) \
  --exclude .git/ --exclude '.venv*/' --exclude logs/ --exclude outputs/ --exclude backups/ \
  --exclude checkpoints/ --exclude recap_data/ --exclude recap_artifacts/ --exclude recap_runs/ \
  --exclude __pycache__/ --exclude HOLD \
  ./ <server>:RECAP/openpi/RLinf/
```

需要传到服务器的数据：

| 内容 | 说明 |
| --- | --- |
| `30000/params` 与 `30000/assets`（约 12 GB） | 放在 `~/RECAP/openpi/checkpoints/pi05_connect_strings/pi05_connect_strings_repro/30000` |
| `recap_artifacts/siglip2-so400m-patch14-224`、`gemma-3-270m` | 一次性传完 |
| `recap_data/lerobot_v<N>` | **每轮导出后传**（AV1 视频，约 2 GB/62 回合） |

```bash
N=1   # 本轮快照编号
rsync -a recap_data/lerobot_v$N/ <server>:RECAP/openpi/RLinf/recap_data/lerobot_v$N/
# 校验：输出 0 才算一致
rsync -an --checksum --out-format='%i %n' recap_data/lerobot_v$N/ \
  <server>:RECAP/openpi/RLinf/recap_data/lerobot_v$N/ | grep -c '^<f'
```

### 9.4 提交作业

在登录节点执行（`ssh <server>` 后）。新一轮换 `RECAP_ROUND`，并用 `RECAP_DATASET` 指向本轮快照：

```bash
cd ~/RECAP/openpi/RLinf
export RECAP_ROUND=round_001                                   # 新一轮改成 round_002 …
export RECAP_DATASET=$PWD/recap_data/lerobot_v1                # 默认就是 lerobot_v1
mkdir -p recap_runs/$RECAP_ROUND && touch recap_runs/$RECAP_ROUND/.inputs_ready   # 数据已同步并校验后再打

P=$(sbatch --parsable --export=ALL slurm/recap_round_prep.sbatch)

# 在 3090 上训练：
export VALUE_MICRO_CANDIDATES="16 8 4 2" CFG_MICRO_CANDIDATES="8 4 2 1"   # CFG 在 24 GB 上未验证
T=$(sbatch --parsable -p 3090 --job-name=recap_train_3090 --export=ALL \
      --dependency=afterok:$P slurm/recap_round_train_8xa100.sbatch)
# 3090 与 A100 谁先空用谁（候选覆盖两种卡，冒烟测试自动选）：
#   export VALUE_MICRO_CANDIDATES="32 16 8 4 2" CFG_MICRO_CANDIDATES="16 8 4 2 1"
#   T=$(sbatch --parsable -p 3090,gpu --job-name=recap_train --export=ALL \
#         --dependency=afterok:$P slurm/recap_round_train_8xa100.sbatch)
# 或在 A100 上训练（用脚本默认值，不要导出上面的变量；round_001 实际用的就是这种）：
#   T=$(sbatch --parsable --export=ALL --dependency=afterok:$P slurm/recap_round_train_8xa100.sbatch)
echo "prep=$P train=$T"
```

- 3090 节点偶尔被别人的纯 CPU 作业占满 64 核（GPU 空着，`sinfo` 显示 allocated），此时作业显示
  `Pending (Resources)`，只能等或改用 `-p 3090,gpu`。
- prep 成功后 train 才开始；prep 失败时 train 会显示 `DependencyNeverSatisfied`，要 `scancel` 掉再重交。
- 停止训练：`scancel <作业号>`。已完成的阶段保留，重交同一条 train 命令（去掉 `--dependency`）会跳过
  已完成阶段并从最新 checkpoint 续训。
- 想重做某一步：删掉对应的 `recap_runs/<轮次>/<stage>.done`（以及它之后各步的 `.done`）再重交。

### 9.5 查看训练情况

**一屏总览**（本机直接运行，最常用）：

```bash
ssh <server> 'bash ~/RECAP/openpi/RLinf/slurm/recap_status.sh'         # 加 --gpu 看显卡占用
ssh <server> 'RECAP_ROUND=round_002 bash ~/RECAP/openpi/RLinf/slurm/recap_status.sh'   # 其他轮次
```

输出依次是：Slurm 作业状态、四个阶段是否完成、micro batch 冒烟结果、Value/CFG 当前步数
（`3600/30000 [已用<剩余, 每步耗时]`）、最新 checkpoint、prep/train 日志最后几行和报错。

**实时日志**：

```bash
ssh <server> 'tail -f ~/RECAP/openpi/RLinf/logs/recap_train_<作业号>.out'      # 作业总日志
ssh <server> "tail -f ~/RECAP/openpi/RLinf/recap_runs/round_001/cfg.log | tr '\r' '\n'"   # 某一阶段（value.log / advantages.log / cfg.log）
```

**TensorBoard 曲线**（loss、学习率、每步耗时）：在本机开一个终端，保持不关：

```bash
ssh -L 6106:127.0.0.1:6106 <server> \
  'cd ~/RECAP/openpi/RLinf && .venv-recap/bin/tensorboard --logdir recap_runs/round_001 --host 127.0.0.1 --port 6106'
```

然后在本机浏览器打开 <http://127.0.0.1:6106>。端口被占用就换一个（两处一起改）。不要用 6006/6016，
那是本机的策略服务端口。

**怎么判断是否正常**：

| 现象 | 含义 / 处理 |
| --- | --- |
| prep 日志停在 `Merged 0 OpenPI LoRA tensors` 一段时间 | 正在写转换后的权重，正常 |
| `smoke_cfg_mb8.log` FAILED（OOM），`mb4` ok | 正常，正式训练用 4 |
| 所有候选都 FAILED | 作业退出；看 `recap_runs/<轮次>/smoke_*.log` 末尾的报错。不是 OOM 的错误不会再试更小的 micro batch |
| `Could not load libtorchcodec` / `libavutil.so.5x` | 服务器没有 torchcodec 能用的 FFmpeg；`recap_server_env.sh` 已默认 `RECAP_VIDEO_BACKEND=pyav`（PyAV 自带解码器，单帧约 0.06–0.14 s） |
| `train/` loss 前 1000 步（warmup）明显下降后趋平 | 正常 |
| loss 变成 nan，或每步耗时突然变长很多 | `scancel`，看日志；从上一个 checkpoint 续训 |

Value 每 1000 步、CFG 每 2500 步保存一次 checkpoint，位置在
`recap_runs/<轮次>/{value,cfg}/<实验名>/checkpoints/global_step_<N>/`。

### 9.6 取回结果并部署

训练结束时 train 日志最后一行是 `RECAP ROUND DONE: <checkpoint 路径>`。只取部署需要的文件
（不取优化器状态）：

```bash
cd $RECAP_ROOT/openpi/RLinf
R=round_001; STEP=30000
SRC=RECAP/openpi/RLinf/recap_runs/$R/cfg/piper_recap_cfg/checkpoints/global_step_$STEP
rsync -am --include='*/' --include='full_weights.pt' --include='piper_recap_policy.json' --exclude='*' \
  <server>:$SRC/ recap_runs/$R/cfg/piper_recap_cfg/checkpoints/global_step_$STEP/
# 基础模型转换结果只需取一次
rsync -a <server>:RECAP/openpi/RLinf/recap_artifacts/pi05_base/ recap_artifacts/pi05_base/
```

然后按第 11 节 `run.sh serve --checkpoint $PWD/recap_runs/$R/cfg/piper_recap_cfg/checkpoints/global_step_$STEP`。
中间 checkpoint（如 15000、25000）也可以用同样方式取回做对比。

### 9.7 round_001 遇到的问题与解决方案

2026-10-30 第一次在服务器上跑通四步时遇到的全部问题，按出现顺序。前三类是流程/环境问题，后面是
RLinf × openpi × FSDP 的兼容问题；修复都已在代码或配置里，下一轮不需要再处理，这里留作排错参考。

**流程与环境**

| # | 现象 | 原因 | 解决 |
| --- | --- | --- | --- |
| 1 | prep 作业 8 h 后 `Timed out waiting for .../.inputs_ready` | 本机等待导出完成的后台脚本用 `pgrep -f "...cli export"`，匹配到了它自己的命令行，永远不结束，所以没同步数据、没打标记 | 不要用自匹配的 `pgrep -f` 等待；导出后按 9.3 手动同步、校验再 `touch .inputs_ready` |
| 2 | 3090 作业 `Pending (Resources)`，`sinfo` 显示 allocated 但 GPU 空着 | 其他用户的不限时纯 CPU 作业占满了节点的 64 核 | `-p 3090,gpu` 同时排队，或改用 A100（9.4） |
| 3 | `Could not load libtorchcodec`、`libavutil.so.5x: cannot open shared object file` | 集群没有系统 FFmpeg；torchcodec 0.2 只支持 FFmpeg 4–7，PyAV 18 自带的是 FFmpeg 8，不能借用 | Value 数据集、advantages、CFG worker 读取 `RECAP_VIDEO_BACKEND`；`recap_server_env.sh` 默认设为 `pyav`（本机不设，仍用 torchcodec） |
| 4 | `TypeError: 'numpy.bool' object cannot be interpreted as an integer` | numpy 2.4 + torch 2.6：`torch.tensor` / `default_collate` 不再接受 `np.bool_` 标量（本机也能复现） | 先转成 Python `bool` 或 numpy 数组：Value collator（2 处）、`modeling_critic` mask（2 处）、`cfg_model` advantage、CFG worker 的 DataLoader 加 `_collate` |
| 5 | 一个与 batch 无关的错误被 4 个 micro batch 依次重试 | 冒烟测试把所有失败都当成 OOM | 只有 OOM 或超时（`SMOKE_TIMEOUT`，默认 45 分钟）才试更小的 micro batch，其他错误直接停并指出日志 |
| 6 | 提交的作业用的是旧代码 | 新会话里 rsync 的排除列表文件不存在，同步失败，但后面的 `sbatch` 照样执行 | 每次提交前确认 `rsync -an --checksum ... \| grep -c '^<f'` 输出 0，再 `sbatch` |

**Value（ValueCriticModel：SigLIP2 + Gemma3-270M）**

| # | 现象 | 原因 | 解决 |
| --- | --- | --- | --- |
| 7 | `full_shard` 下 `RuntimeError: 'weight' must be 2-D`（`embed_language_tokens`） | embedding 在 FSDP 单元的 forward 之外被直接调用，`full_shard` 下拿到的是一维分片 | Value 固定用配置里的 `no_shard`，`RECAP_FSDP_SHARDING` 不作用于 Value |
| 8 | A100 上 micro batch 32 OOM | 显存不够 | 冒烟测试自动退到 16；global batch 256 用梯度累积补齐 |

**CFG（pi05 + 原生 JAX LoRA，只训练动作专家）**

| # | 现象 | 原因 | 解决 |
| --- | --- | --- | --- |
| 9 | `mat1 and mat2 must have the same dtype, Float and BFloat16`（`action_in_proj`） | openpi 刻意把归一化层、视觉 patch/position embedding、动作/时间投影保留为 FP32（`to_bfloat16_for_selected_params`），配置里的 FSDP 混合精度 `param_dtype: bf16` 又把它们转成 BF16 | `piper_recap_cfg.yaml`、`piper_recap_sft_int.yaml` 的 `mixed_precision` 全部设为 `null`，与 openpi 原生 PyTorch 训练的数值完全一致 |
| 10 | 反向 `Found dtype Float but expected BFloat16`（`mse_loss`） | 第 8 条曾用 autocast 打补丁，投影输出变成 BF16，与 FP32 目标做 `mse_loss` 时反向梯度类型不对 | 随第 8 条一起撤掉所有 autocast 补丁，`openpi_cfg_action_model.py` 恢复为 git 原版（补丁版本备份在 `backups/video_backend_20261030/`） |
| 11 | `full_shard` 下先是 `'weight' must be 2-D`，去掉 `lm_head` 单元后又是 `size mismatch ... vec (0)` | (a) Gemma 的 `lm_head` 与 `embed_tokens` 共用权重（checkpoint 里只有 `lm_head.weight`），默认切分把 `lm_head` 单独成单元，分片了共享权重；(b) openpi `compute_layer_complete` 直接调用每层的 `q_proj` 等，不走 `GemmaDecoderLayer.forward`，按层分片的单元永远不会展开 | 配置里写明 `wrap_policy`：默认切分目标去掉 `lm_head`；CFG 也只能用 `no_shard` |
| 12 | 第 1 步正常，第 2 步开头 `Cannot writeback when the parameter shape changes`（`Expects [1327104] but got [1152, 1152]`） | `use_orig_params` 下 FSDP 前向时把参数换成 Tensor 视图，靠该单元自己的 post-backward 钩子换回 Parameter。根单元只含冻结的 SigLIP 编码器、投影和 embedding，扁平参数不需要梯度，钩子不触发，438 个参数留在 Tensor 状态 | CFG worker 每次优化器更新后调用 `_restore_fsdp_param_views()`：对模块属性已不是原 Parameter 的单元执行 FSDP 本该执行的 `_use_sharded_views()`（`no_shard` 下把同一存储上的 Parameter 挂回，数据不动）。日志出现一次 `Re-registered parameters of 1 frozen FSDP unit(s)` 是正常的 |
| 13 | README 说 `balance_dataset_weights` "平衡"示教与 rollout | RLinf 原语义是 weight × 帧数，即按帧均匀采样 | 文档已改正（第 8 节第四步）；round_001 保持原样（示教约 88%、rollout 约 12%） |

**排错工具**（以后遇到新问题先用这些，比反复提交整轮作业快得多）

- 只用 CPU 检查数据管线：`slurm/recap_check_loaders.py`（用法见文件开头）。在登录节点用正式配置构建
  Value / CFG 的 dataloader 并取几个 batch，提前发现数据、collate、dtype 问题，不占 GPU。第 4 条中
  CFG 的那处就是这样在提交前发现的。
- 只跑 CFG 的几步：`sbatch -p gpu slurm/recap_cfg_debug.sbatch`，日志在 `recap_runs/<轮次>/debug_cfg.log`。
  它设置 `RECAP_DEBUG_ANOMALY=1`：打开 `torch.autograd.set_detect_anomaly`（反向报错时给出对应的前向
  算子），并在每步前后打印 `[fsdp-debug] ... detached params`（脱离 FSDP 存储或被替换的参数名、形状）。
  该开关很慢，正式训练不要设置。
- 冒烟测试日志：`recap_runs/<轮次>/smoke_<stage>_mb<N>.log`；报错时 `recap_status.sh` 会列出第一条错误。

**round_002（2026-11-03）**

| # | 现象 | 原因 | 解决 |
| --- | --- | --- | --- |
| 14 | A100 作业停在开头的 `nvidia-smi`，该进程处于 D 状态，`scancel` 后仍然卡着 | hgx001 的 GPU 驱动故障；它长期显示空闲正是因为这个 | `recap_submit_full_cfg.sh` 默认 `--exclude=hgx001`（用 `RECAP_FULL_EXCLUDE` 修改）。`SBATCH_EXCLUDE` 环境变量无效 |
| 15 | CFG 冒烟测试报 `_load_advantages_lookup() takes from 1 to 2 positional arguments but 3 were given`，提示 not an OOM | 新增 `model_provider_func` 时插在了 `@staticmethod` 和 `_load_advantages_lookup` 之间，装饰器被它"抢走"。之前的冒烟测试用的是 `plain_sft`，不走这个函数，所以没发现 | 恢复装饰器位置；新增单元测试 `test_cfg_worker_method_kinds`；修改 CFG 代码后，先用 `slurm/recap_check_loaders.py cfg "$CFG_DATA"`（不加 `plain_sft`）在 CPU 上加载真实 advantage |
| 16 | 磁盘只剩 118 GB | 全量 CFG 的 checkpoint 约 116 GB/个，LoRA 约 33 GB/个，每 2500 步保存一次 | 删除测试产物和 round_001 的中间 checkpoint，后台运行 `slurm/recap_prune_ckpts.sh` |

### 9.8 整个模型全量微调（与 LoRA 对照）

默认的 CFG 训练是 LoRA（`recap_full_finetune: false`）。全量微调训练全部 36.2 亿参数，包括 SigLIP、
PaliGemma（含词嵌入）和动作专家。全量版在 A100 上独立跑完整流程：Value → advantage → 全量 CFG。
它和 4090 上的 LoRA 版互不依赖，各用各的目录和数据副本。

| 项目 | LoRA（round_002，4090） | 全量（round_002_full，A100） |
|---|---|---|
| Value | 从头训练 8000 步 | 从 LoRA 版 Value 的 6000 步 checkpoint 接着训练到 8000 步，之后独立运行 |
| 训练参数 | LoRA | 全部参数，`recap_lora_only=false`、`train_expert_only=false`、`recap_full_finetune=true` |
| 精度 | 原设置 | fp32 主权重，`amp_autocast` bf16 计算（不用 FSDP mixed_precision） |
| FSDP | `no_shard` | `full_shard`，包装单元 `GemmaMLP, SiglipEncoderLayer, SiglipVisionEmbeddings, GemmaRMSNorm, GemmaRotaryEmbedding` |
| global batch | 128 | **64**（A100-40G 显存只够这么多） |
| micro batch | 16/8/4/2 中自动选 | 2/1 中自动选（mb=1 实测峰值 39.4 GB / 40 GB） |
| 步数、学习率 | 30000 步，cosine 2.5e-5 → 2.5e-6，warmup 1000 | 相同 |
| 预计时长 | 约 17 小时 | mb=1 时每步约 8 秒，约 2.5–3 天 |

数据：`recap_data/round_002_full/` 是 `recap_data/round_002/` 的副本。`data/` 和 `videos/` 用硬链接，
不占额外空间；`meta/` 是真正复制的文件。这是因为计算 advantage 时会写入
`meta/advantages_piper_recap_adv_v1.parquet` 和 `meta/mixture_config.yaml`，两个流程不能写到同一个文件。

提交（登录节点，几秒就能完成）：脚本会建好数据副本，把 Value checkpoint 硬链接过来，跑一遍 preflight，
然后向 `gpu` 分区提交 `recap_train_full`：

```bash
ssh <server>
cd ~/RECAP/openpi/RLinf
bash slurm/recap_submit_full_cfg.sh
RECAP_ROUND=round_002_full bash slurm/recap_status.sh   # 训练进度
```

磁盘：一个全量 CFG checkpoint 约 116 GB，一个 LoRA checkpoint 约 33 GB，一个 Value checkpoint 约 9 GB。
登录节点后台运行着清理脚本，每个阶段只保留最新的一个完整 checkpoint
（最新的那个 15 分钟没有写入后，才删除更旧的）。
`RECAP_PRUNE_KEEP` 里列出的 checkpoint 不会被删除，默认保留 LoRA 25000 和全参 20000（两者的 30000 作为最新的会自动保留）。
修改保留列表：先 `pkill -f recap_prune_ckpts.sh`，再带着 `RECAP_PRUNE_KEEP="round_002/cfg/global_step_25000 ..."` 重新启动。

```bash
nohup setsid bash slurm/recap_prune_ckpts.sh > logs/recap_prune.out 2>&1 < /dev/null &
cat logs/recap_prune.out                                 # 删除记录
```

注意：
- 显存余量只有约 1.5 GB，因此设置了 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`，用来减少碎片。
  如果中途 OOM，重新提交会从最近的 checkpoint（每 2500 步保存一次）继续训练。
- checkpoint 里的 `piper_recap_policy.json` 记录了 `full_finetune: true`，`serve` 会按这个设置加载，
  启动命令和 LoRA 版一样（11.2 节）。
- 3090（24 GB）放不下全量微调。

## 10. 断点恢复

raw 采集重启：重复 `run.sh collect`，已 C/X 提交的 episode 均保留；退出时尚未打分
的片段不会猜测标签。

Value 或 CFG 恢复使用**相应阶段**的完整 checkpoint，并带上与原运行相同的数据覆盖：

```bash
bash examples/recap_piper/run.sh value "$VAL_DATA" \
  runner.resume_dir=/绝对路径/value/piper_recap_value/checkpoints/global_step_100

bash examples/recap_piper/run.sh cfg "$CFG_DATA" \
  runner.resume_dir=/绝对路径/cfg/piper_recap_cfg/checkpoints/global_step_100
```

新一轮数据集属于另一轮离线训练，先重新计算标签。

## 11. 部署 RECAP 策略并迭代

### 11.1 第二轮：用 round_001 的 CFG 策略采集与评估

前提：本机已有 `recap_artifacts/pi05_base` 和 CFG checkpoint（9.6 取回，SHA256 已与服务器核对）。
round_001 的最终 checkpoint 是
`recap_runs/round_001/cfg/piper_recap_cfg/checkpoints/global_step_30000`。

第二轮的 raw 回合写到**独立目录** `recap_data/raw_round002`（已建好），不要写进第一轮的
`recap_data/raw`：collect 不记录回合由哪个策略产生，分开存放才能单独统计 RECAP 策略的成功率，
下一轮也能按轮次组合数据。与第 4 节相比只有终端 B 换成 CFG 服务、终端 C 换目录；按键和打分规则
（第 4 节「按键」、4.1）完全不变。

**终端 A：四臂 launch**（与第 4 节相同）

```bash
cd $RECAP_ROOT/cobot_magic/Piper_ros_private-ros-noetic
source /opt/ros/noetic/setup.bash
source devel/setup.bash
roslaunch piper start_inference_takeover.launch \
  master_left_can:=can_l_leader \
  master_right_can:=can_r_leader \
  puppet_left_can:=can_l_follower \
  puppet_right_can:=can_r_follower \
  publish_hz:=200
```

**终端 B：RECAP CFG 策略服务**（替代 `run_native_server.sh`，两者不能同占 6016）

```bash
cd $RECAP_ROOT/openpi/RLinf
source examples/recap_piper/env.sh
bash examples/recap_piper/run.sh serve \
  --checkpoint $PWD/recap_runs/round_002/cfg/piper_recap_cfg/checkpoints/global_step_30000 \
  --guidance-scale 1.0
```

启动前会检查 GPU 空闲（`--gpu-idle`）：先停掉原 JAX 服务或其他占用 4090 的进程。看到 websocket
服务在 6016 监听后再启动终端 D。

**终端 C：键盘和记录器，写入第二轮目录**

```bash
cd $RECAP_ROOT/openpi/RLinf
export RECAP_RAW_DIR=$PWD/recap_data/raw_round003
bash examples/recap_piper/run.sh collect --profile takeover
```

`RECAP_RAW_DIR` 只在当前终端有效；每次新开终端 C 都要先 export。按 I 开始/继续回合即开始记录，
C/X 判成功/失败后回合才写入；与第一轮相同，未打分的回合不会保留。

**终端 D：原 OpenPI 真机客户端**（仍连 6016；CFG 服务要把预取改为 5 步）

```bash
cd $RECAP_ROOT/openpi/RLinf
bash examples/recap_piper/run_client.sh takeover --rtc-prefix-len 5
```

`--rtc-prefix-len 5` 覆盖 profile 里的 4（后出现的参数生效）。原因见下方「推理延迟与断续」；
换回原 JAX 服务时去掉该参数。

**推理延迟与断续（2026-11-02 实测与修复）**

现象：推理时手臂每隔约 0.67 s 出现 mm 级的停顿再追上，运动不连续。客户端 ROS 日志
（`~/.ros/log/openpi_piper_runtime_*.log`）显示 751 个动作块中 747 个出现
`RTC prefetch did not finish before the action chunk ended; holding the final command`，
推理中位 443 ms（原 JAX 82 ms）。客户端在剩 4 步（267 ms）时预取下一块，结果没到就保持最后指令
约 170 ms，新块又按约半秒前的观测从第 7 步接上；每 10 步交接一次，所以周期性断续。

| 设置 | 推理 p50 / p90（4090，`slurm/recap_bench_serve.py`） | 预取窗口 |
| --- | --- | --- |
| 修复前：guidance 1.0 仍计算无条件分支 | 433 / 438 ms | 4 步 = 267 ms（不够） |
| 跳过无条件分支（输出逐位相同） | 225 / 227 ms | 4 步 = 267 ms（真机多约 10–40 ms 传输，余量太小） |
| 再加 `--rtc-prefix-len 5` | 同上 | 5 步 = 333 ms |

修复：`sample_actions` 在 `guidance_scale == 1.0` 时只算条件分支（无条件项权重恰为 0，同一噪声下
输出与修复前逐位相同，已用固定噪声比对）；客户端预取改为 5 步。guidance 不为 1.0 时两个分支都要算，
延迟回到约 430 ms，需要更大的预取（7–8 步）。复测基准：

```bash
source examples/recap_piper/env.sh   # 先停掉终端 B，GPU 空闲时运行
$RECAP_PYTHON slurm/recap_bench_serve.py \
  recap_runs/round_001/cfg/piper_recap_cfg/checkpoints/global_step_30000 \
  recap_data/raw_round002/<任一回合目录> 20
```

运行后检查：

```bash
f=$(ls -t ~/.ros/log/openpi_piper_runtime_*.log | head -1)
grep -c 'did not finish before the action chunk ended' $f   # 应接近 0
grep -c 'prefetch completed' $f                              # 动作块总数
```

**评估前先核对**

1. 上面的预取告警接近 0，手臂运动连续；终端 B / D 没有 policy-stale HOLD。
2. 接近段速度与基础策略一致后，再统计成功率。
3. 成功率按**无人工接管的 C** 计，每个设置至少 20 次，初始摆放与第一轮尽量一致，才能与基础策略比较。

查看第二轮已记录的回合与打分：

```bash
export RECAP_RAW_DIR=$PWD/recap_data/raw_round002
bash examples/recap_piper/run.sh score
```

换回基础策略对比时：终端 B 停掉 CFG 服务，改运行 `bash examples/recap_piper/run_native_server.sh`，
并给终端 C 另设目录（如 `recap_data/raw_round002_base`），两种策略的回合不要混在一个目录。

### 11.2 CFG 服务说明

服务不打开 ROS/CAN，不会使能机器人。只接受完整 RECAP FSDP 权重，缺失键时直接报错，
不会把其他类型的 checkpoint 当成 RECAP；启动时还会核对 checkpoint 里 `piper_recap_policy.json`
记录的基础模型、归一化统计和任务与本机 `pi05_base` 一致。端口 6006 被入口明确拒绝，默认 6016。

CFG 在每一步流积分中组合速度场：

```text
v = (1 - guidance_scale) * v_unconditional + guidance_scale * v_positive
```

`1.0` 使用 positive 条件速度场；更大的 guidance 不是更小的物理动作，也不是 Actor
纠正限幅。第一次保持 1.0，不要把它当速度旋钮。

训练后服务与初始服务都返回 `[50,14]` 的绝对关节/夹爪计划，客户端负责 RTC、平滑和实际执行。
中间 checkpoint（`global_step_2500` … `27500`）留在服务器上，需要对比时用 9.6 的命令换 `STEP` 取回。

### 11.3 下一轮训练

```bash
export RECAP_RAW_DIR=$PWD/recap_data/raw_round002
export RECAP_DATASET=$PWD/recap_data/lerobot_v2
bash examples/recap_piper/run.sh export
bash examples/recap_piper/run.sh check --stage dataset
```

然后按第 9 节同步 `lerobot_v2` 并以 `RECAP_ROUND=round_002` 提交。示教副本可复用，但 returns、
Value、advantage 都要重新计算；把 `lerobot_v1` 作为另一个 rollout 条目加入 `*_DATA`（修改
`slurm/recap_server_env.sh` 中的四个 `*_DATA`），两轮数据一起训练。

### 11.4 仅两个从臂推理（修复版 follower mux）

两个主臂可以断电，**两个从臂保持通电**。与第 11.1 节相比，只换终端 A，终端 C/D 改用 `followers`。

只用从臂时的 mux 是 `cobot_magic/lingbot_piper`（`roslaunch lingbot_piper start_inference_followers.launch`，由下面的命令启动）。
出厂的 follower mux 按 Space/H/I 时会把新鲜反馈误判为过期并失能双臂，机械臂会下落，所以本仓库没有包含它。
修复版在反馈短暂中断时保持力矩，HOME 途中按 I 会排队到位后再恢复，从臂固件故障时拒绝使能，并发布
`/teleop/follower_mux_impl` 标识；`rlinf.recap_piper.mux_guard` 在启动 launch、记录器和客户端时检查这个标识。

**终端 A：两从臂驱动 + 修复版 mux**（不打开主臂 CAN）

```bash
cd $RECAP_ROOT/openpi/RLinf
bash examples/recap_piper/run.sh launch followers
```

末尾可追加 roslaunch 参数，例如 `gripper_squeeze_margin_m:=0.008`。默认 0：RECAP 的训练数据都在四臂
takeover mux 下采集，那里没有额外夹紧，保持 0 才与训练时的下发指令一致（Pi05 在 lingbot 文档中用 0.008）。
可选确认：`rostopic echo -n 1 /teleop/follower_mux_impl` 应输出上面的标识。

**终端 B：CFG 策略服务**（同第 11.1 节，把 checkpoint 换成要评估的轮次）

```bash
cd $RECAP_ROOT/openpi/RLinf
source examples/recap_piper/env.sh
bash examples/recap_piper/run.sh serve \
  --checkpoint $PWD/recap_runs/round_003/cfg/piper_recap_cfg/checkpoints/global_step_30000 \
  --guidance-scale 1.0
```

**终端 C：键盘和记录器**（评估回合存独立目录）

```bash
cd $RECAP_ROOT/openpi/RLinf
export RECAP_RAW_DIR=$PWD/recap_data/raw_round003_eval
bash examples/recap_piper/run.sh collect --profile followers
```

**终端 D：客户端**

```bash
cd $RECAP_ROOT/openpi/RLinf
bash examples/recap_piper/run_client.sh followers --rtc-prefix-len 5
```

`followers` profile 的手臂速度上限是 0.6 rad/s（Pi05 两从臂文档参数）；与四臂采集时的 0.4 一致时追加
`--max-arm-velocity-rad-s 0.4`（后出现的参数生效）。

按键与第 4 节相同，只是没有 T：

| 按键 | 行为 |
| --- | --- |
| I | 开始/继续回合；修复版 mux 在推理中按 I 会忽略（回合照常开始），HOME 途中按 I 排队到位后再推理 |
| Space | 软件 HOLD，保持力矩和当前位置，夹住的线不松 |
| H | HOLD 后限速 0.25 rad/s 回到本次 launch 捕获的姿态；C 成功后必须先 H |
| C / X | 成功 / 失败，随后自动 HOLD |
| T | 无效（没有主臂） |

修复版 mux 在推理时报告 `INFERENCE_FOLLOWERS`，记录器按 `INFERENCE` 处理（`collect.normalize_mode`）；
等新 policy 的 `WAITING_POLICY_RESYNC` 期间不记录。没有人工接管，`intervention` 全为 0，
统计结果就是无接管成功率：`bash examples/recap_piper/run.sh score`。

停机：Space → 终端 C 按 Q → 终端 D Ctrl+C → **支撑两个从臂后**
`rostopic pub -1 /enable_flag std_msgs/Bool "data: false"` → 停终端 A、B。
从臂固件锁存急停（`rostopic echo -n 1 /puppet/arm_status_left` 为 1）时重启 launch 不能清除，
需两个从臂断电再上电（LingBot 文档第 4 节）。

## 12. 代码与检查入口

| 文件 | 作用 |
| --- | --- |
| `examples/recap_piper/env.sh`、`run.sh` | 本机独立环境与阶段入口 |
| `run_native_server.sh`、`run_client.sh` | 复用原基础服务和已验证真机客户端 |
| `examples/offline_rl/config/piper_recap_*.yaml` | 四步单机配置，以及 `piper_recap_sft_int.yaml` 对照基线 |
| `rlinf/recap_piper/collect.py` | 原 ROS 请求 + 图像/动作/标签记录 |
| `rlinf/recap_piper/scoring.py` | 三次插接打分（N/U/C 校验）、逐帧进度、按根统计 |
| `rlinf/recap_piper/data.py` | 原始回合事务、不可覆盖 LeRobot 导出 |
| `rlinf/recap_piper/source.py`、`cli.py` | 原配置 prompt/架构解析、转换、CFG 服务 |
| `rlinf/recap_piper/policy.py` | 原生 LoRA、源统计、14-D CFG 动作适配 |
| `rlinf/recap_piper/preflight.py` | 不启动 ROS/CUDA/Ray 的数据与资产检查 |
| `tests/unit_tests/test_piper_recap.py` | 无硬件数据、标签、变换和部署契约测试 |

对既有 RECAP 代码的补充包括：ValueDataset 的 Piper 三相机、Value 推理映射、
advantage 的 Piper key mapping、CFG 的 openpi_data 传递和 Piper 模型加载分支、
`[False]` 成功标记的解析修复，以及三个默认关闭、由 Piper 配置开启的选项：
`compute_returns` 的 `takeover_penalty`、`compute_advantages` 的 `force_intervention_positive`、
`FSDPCfgWorker` 的 `data.plain_sft`。未改 Piper ROS 驱动。

```bash
.venv-recap/bin/python -m pytest -q tests/unit_tests/test_piper_recap.py
```

本文件为独立中文 README，未新增 Sphinx 页面。
