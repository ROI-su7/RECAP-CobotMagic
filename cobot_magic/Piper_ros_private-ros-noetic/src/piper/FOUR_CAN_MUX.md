# Piper 四独立 CAN 最终控制结构

本目录的最终方案是实现手册中的“方案一：主臂模式 + ROS 话题切换”。
左右主臂和左右从臂各占用一条独立 SocketCAN；出厂双臂共用一条 CAN 时的
主臂控制帧不再直接到达从臂，因此由 ROS reader/driver、mux 和从臂 driver
完成应用层转发。

## 三个互斥命令行配置

| launch | 打开的 CAN | 用途 |
| --- | --- | --- |
| `start_four_can_collect.launch` | 两主臂 + 两从臂 | 预置并验证后的固定 `0xFA MASTER` 示范采集 |
| `start_inference_followers.launch` | 仅两从臂 | 默认模型推理，主臂不跟随 |
| `start_inference_takeover.launch` | 两主臂 + 两从臂 | 四臂推理，`T` 后人工纠偏 |

任意时刻只能运行一个配置，也不能与旧 `start_ms_piper.launch`、role 配置工具
或其他直连 CAN 程序同时运行。

### 采集配置

`start_four_can_collect.launch` 不再包含可切换角色的 leader driver，而是固定包含：

```text
piper_fixed_master_left_reader
piper_fixed_master_right_reader
piper_collect_puppet_left_driver
piper_collect_puppet_right_driver
piper_fixed_collect_mux
```

启动前必须在全部 Piper ROS/SDK 节点停止时，将两只主臂设置为 `0xFA`。最终
默认采用官方 SDK demo 的最小序列：正常初始化后只保存 `0xFA`，完整断开两只
主臂控制器电源 10 秒再上电；不再把已经失败的运行时“`0xFA -> 待机`”当作
默认路径。配置后再用只读
`piper_verify_fixed_master.py` 做独立复核。采集 launch 本身只读主臂 CAN，绝不发送
`0xFA/0xFC`、使能或位置目标，因此不会再出现运行时切换造成的“抬起后失能”。

收到 `/enable_flag=true` 且两路固定 MASTER 与两路 follower 反馈都新鲜后，mux
进入 `COLLECT`。主臂到从臂默认关节跟随限速为 `2.5 rad/s`；200 Hz、
`queue_size=1` 与 TCP_NODELAY 避免旧命令排队。

### 普通推理配置

`start_inference_followers.launch` 只创建：

```text
piper_inference_left_follower
piper_inference_right_follower
piper_policy_follower_mux
```

它不会创建 `C_PiperInterface` leader 实例，也不会打开 `can_l_leader` 或
`can_r_leader`。policy 必须发布到 `/teleop/policy/joint_left/right`。公开的
`/enable_flag=true` 只武装 mux；mux 等待双 policy 与双 follower feedback
齐全后，才通过私有 `/teleop/follower_enable` 使能从臂。模型首帧超过 10 秒时
只打印缺失流并继续安全等待，不再提前锁存急停。

### 监督推理和接管配置

`start_inference_takeover.launch` 固定：

```text
session_profile=supervised
auto_start_inference=true
auto_start_collect=false
```

policy、leader 和 follower 反馈齐全后自动进入 `INFERENCE`。主臂以 `0xFC`
OUTPUT 低速对齐对应从臂，然后四臂同时跟随 policy。`T` 一经 mux 接收就立即
进入带使能的 `HOLD`：冻结两个从臂目标、发布 `inference_active=false` 和
`hold_active=true`，让 OpenPI 丢弃排队动作。OpenPI 只有在最后一次 ROS publish
返回且插值/动作缓存清空后才发布 `/teleop/policy_hold_ack=true`。一秒扶臂期内
主臂仍持续接收当前位姿 OUTPUT HOLD，避免在延迟中漂移并丢失主从对齐；一秒到期
且对齐确认后才锁存该对齐结果并关闭 `/teleop/leader_tx_enable`。左右主臂 driver
各自在当前 CAN 发送完成后发布
`/teleop/leader/tx_quiet_left/right=true`，再等待 0.20 秒 SocketCAN 静默期。
静默期间不再重复要求 OUTPUT 对齐（主臂正在退出 OUTPUT）；两个 TX ACK 和静默期
完成后立即发送运行时 `0x470/0xFA`，使两只主臂进入
MASTER 零力输入；只有左右两侧都产生完整、新鲜的 `0x155–0x159` 后，mux 才原子地
把两个从臂 command owner 改为 leader。一秒是扶稳延迟，不是接管已生效的保证；
须以终端的 `TAKEOVER active` 为准。首次无缝 `0xFA` 完全没有主臂帧时，驱动约一秒后
按协议执行一次 `0xFA -> STANDBY` 激活并保留完整确认窗口；mux 最多等待 15 秒，期间从臂始终 HOLD。确认失败时恢复当前位姿的带使能
OUTPUT HOLD，不自动回到 policy；必须由操作者按 `I` 明确恢复。

## 唯一话题所有权

```text
OpenPI:
  /teleop/policy/joint_left
  /teleop/policy/joint_right

leader feedback:
  /teleop/leader/joint_left
  /teleop/leader/joint_right

follower feedback:
  /puppet/joint_left
  /puppet/joint_right

mux only outputs:
  /teleop/command/joint_left
  /teleop/command/joint_right
```

每个从臂 driver 只订阅对应 `/teleop/command/*`。`/master/joint_left/right` 是
mux 发布的兼容 action 记录，不是另一个控制入口。

## 主臂角色握手

官方 SDK 定义：

```text
0xFA -> teaching-input/master
0xFC -> motion-output
MASTER control      -> 0x155/156/157/159，或按 0x470 偏移为 0x16x/0x17x
GetArmJointMsgs()   -> 0x2A5/0x2A6/0x2A7
GetArmGripperMsgs() -> 0x2A8
```

采集 launch 不使用运行时 `0xFA`。前置维护工具默认执行
`--master-activation restart`：先确认主臂仍有完整 OUTPUT 反馈，再使用官方 SDK
初始化和单次 `MasterSlaveConfig(0xFA, 0, 0, 0)` 保存角色；它不调用使能/失能，
也不发送关节或夹爪目标。断电重启后，`piper_verify_fixed_master.py` 直接接收原始
CAN，同时检查协议允许的 `0x15x/0x16x/0x17x` 三组控制 ID，并输出总线上实际
出现的最高频 ID。固定 reader 同样使用原始 SocketCAN 自动识别三组合法偏移，
不依赖 SDK 只识别原生 `0x15x` 的映射。启动时每只主臂必须至少收到一组完整
三条关节帧；之后固件在主臂静止时可能暂停发送，reader 会重复发布最后姿态并
进入 `MASTER_HOLD`，不会把正常静止误判为错误。对应夹爪帧单独报告，兼容静止
夹爪不持续发送的固件。

监督推理的 `T` 接管使用运行时 OUTPUT→MASTER：先尝试新版 AgileX
`pyAgxArm.set_leader_mode()` 的单帧 `0x470/0xFA` 无缝切换；若一秒内完全无
`0x155-0x157`，再按 0x470 协议要求执行一次 `0xFA -> MotionCtrl_2(STANDBY)`
激活。两个路径都不调用 `DisableArm`。2026-08-27 实机日志证明
`0x150` 的“示教轨迹记录”在当前两只主臂上始终回报 `ctrl_mode=0x01`，它不是
这套主从臂的零力拖动切换入口。T 在一秒带使能 HOLD 后写入 `0xFA`，只有左右
两侧都产生完整、新鲜的 `0x155/0x156/0x157` 后才把 follower 的命令源切到
leader；失败则整体保留/恢复带使能 HOLD，
不执行失能回退。

监督 driver 在 T 开始后同时打开一条**只接收**的原始 SocketCAN 观察通道，支持
`0x155-0x157`、`0x165-0x167`、`0x175-0x177` 三组合法控制偏移。OUTPUT 阶段该
观察器关闭，避免把本机发给主臂的 JointCtrl 误判成 MASTER；只有 policy 已停止、
从臂已 HOLD 后才清零计数并开始接收。MASTER 激活和 leader 姿态优先采用这条已由
固定采集验证过的原始帧路径，避免 SDK `GetArmJointCtrl().Hz` 偶发为 0 造成错误回退。
原始总线也没有完整关节组时仍拒绝 `TAKEOVER active`，不会伪造成功。

同一观察通道还会为每次运行时 `0xFA` 输出一条 `MASTER handshake ... bus
timeline`：`0x470_TX_ECHO` 是独立 SocketCAN socket 看到的 SDK 发送回读，
`0x151_STANDBY_TX_ECHO` 是第二阶段待机命令回读，`0x2A1` 按连续相同状态压缩并
保留 `ctrl_mode/arm/teach/motion/error`，`CAN_ERROR` 记录切换窗口内的 SocketCAN
错误帧。没有 `0x470_TX_ECHO` 时先查主机/USB-CAN 发送链路；有回读且无错误、但
`ctrl_mode` 始终未进入 `0x06` 时，判定控制器忽略/拒绝了这次角色激活；进入
`0x06` 却仍无完整 `0x155-0x157` 时，判定角色已切换但 MASTER 控制流缺失。
这里的 `TX_ECHO` 证明帧已进入 Linux SocketCAN/USB-CAN 发送路径，但不是机械臂
应用层 ACK；必须与 `CAN_ERROR`、随后出现的 `0x2A1` 和 `0x155-0x157` 一起解释。
第二次尝试会明确执行协议附录要求的 `0x470/0xFA -> 0x151/STANDBY`。如果控制器
随后持续停在 `ctrl_mode=0x00`，监督 driver 会继续发送 SDK
`ReqMasterArmMoveToHome(0)` 对应的 `0x191`“恢复主从模式”命令；mode 0 不是主臂或
主从臂回原点（回原点分别是 mode 1/2），也不会失能。日志中的
`0x191_RESTORE_TX_ECHO` 会确认这一步是否进入发送路径。如果三条发送回读都存在、
没有 CAN 错误且 `ctrl_mode` 仍未进入 `0x06`，就不是代码漏掉待机/恢复步骤，而是
该控制器/固件没有接受这次运行时激活。

当前 USB 固定映射为 `can_l_leader=1-13.1:1.0`、
`can_l_follower=1-13.4:1.0`、`can_r_leader=1-13.3:1.0`、
`can_r_follower=1-4.2:1.0`（底盘 `can0=1-4.1:1.0`）。右从臂从扩展坞
`1-13.2:1.0` 移到新口是为隔离历史上的 `gs_usb` 僵死和
`SEND_MESSAGE_FAILED`；两只 leader 的 USB 路径未改变，所以该换口不能单独解释
左右 leader 随机接受 `0xFA` 的现象。`piper_log_snapshot.sh` 会保存当次 sysfs
设备路径和 RX/TX 错误、丢包计数，便于后续按时间关联。

从 MASTER 返回推理时发送 `0xFC`。AgileX 已说明固件 v1.8-5～v1.8-8 在
Leader→Follower 后可能残留力矩控制状态，因此 driver 在获得 `0x2Ax` 实测姿态
后，连续发送若干次该“当前位置”的 MOVE J，再允许 policy 恢复。固定 MASTER
数据采集仍使用独立只读 launch，两者不能同时运行。

`0xFC` OUTPUT 重新使能前，driver 必须先读取实测 J1..J6/夹爪，以该姿态预装
目标，再进入 MOVE J 和使能。相同的“实测姿态先于使能”规则也用于两个从臂，
防止控制器中残留目标导致上抬、回零或夹爪突然闭合。

`--role output` 每次都会明确写入 `0xFC`，即使当前已经能看到 `0x2Ax` 反馈；
否则只恢复运行态、不清除 Flash 中的 `0xFA`，下一次主臂掉电仍可能重新进入
重力补偿 MASTER 状态。

## 状态和按键

采集与普通推理由命令行配置自动进入，不使用模式按键。监督推理键盘必须显式
使用 `_profile:=inference`，避免先前 follower-only 会话遗留的 ROS 私有参数复用：

```bash
rosrun piper piper_mode_keyboard.py _profile:=inference
```

监督推理键盘：

```text
T      立即停止 policy 并进入带使能 HOLD；一秒且 policy 静默后请求 0xFA MASTER，看到 TAKEOVER active 后接管
I      从 HOLD 恢复推理，或人工纠偏后恢复推理
Space  软件 HOLD：冻结实测姿态，保持电机使能和位置闭环；OpenPI 清空动作块
H      从 HOLD 限速回到本次 launch 捕获的启动姿态，到位后继续 HOLD
Q      先进入 HOLD，再只退出键盘；机械臂 launch 不退出
```

空格不再发送协议 `0x150` 快速急停。HOLD 发布 `/teleop/hold_active=true`，
Piper OpenPI 客户端会停止策略执行并清空 RTC/插补目标；`I` 后 mux 只接受晚于
本次恢复请求的左右 policy 消息，防止补执行暂停期间的旧轨迹。附录明确说明快速急停会让全部关节进入
阻尼，PiPER 没有关节抱闸，机械臂会下落，因此操作者暂停不能使用它。当前空格
是软件急停/保持：mux 忽略后续 policy，持续以 200 Hz 发布冻结姿态且不撤销
enable。这不是安全认证急停；CAN、电源或驱动器真实故障导致闭环不可用时，
软件无法保证 HOLD。普通 follower-only 推理同样支持 `Space/H/I/Q`；`T` 仅
监督推理支持。

## MASTER 0 Hz 最终判定流程

在已经验证的 `0xFC` OUTPUT 左右 200 Hz 状态下，停止全部 Piper ROS/SDK
客户端，然后执行：

```bash
rosrun piper piper_configure_leader_role.py \
  --role master \
  --master-activation restart \
  --left-can can_l_leader \
  --right-can can_r_leader \
  --verify-timeout-s 8
```

该命令应打印两只主臂的固件版本和 `saved 0xFA`。然后关闭的是**两只主臂控制器
电源**，不是只拔 CAN 航空插头；等待 10 秒再上电。从臂无需断电。上电后不要
启动任何 Piper ROS/SDK 节点，直接执行：

```bash
rosrun piper piper_verify_fixed_master.py \
  --left-can can_l_leader \
  --right-can can_r_leader \
  --duration-s 5 \
  --min-joint-hz 10
```

工具会分别显示 `offset 0x00/0x10/0x20`。任一完整组通过都可由固定 reader 自动
使用。如果三组全为 0 且 `observed no CAN frame at all`，结合 `0xFC` 能恢复
200 Hz，可确定 SocketCAN、CAN 命名、波特率和接收线路正常，而控制器没有完成
协议要求的 `0xFA -> 控制模式 0x06`。此时继续改 ROS 超时、重复使能或伪造
MASTER 话题都不能解决；必须凭工具打印的固件版本向 AgileX 确认/升级支持
0x470 联动示教输入的控制器固件。未出现真实 MASTER 控制帧前，固定 MASTER
采集和 T 接管都必须禁止；普通 follower-only 推理不受影响。

## Fail-closed 条件

- CAN 接口不存在、DOWN、bus-off 或发送失败；
- JointState 少于 7 维、非有限值或输入过期；
- policy、左右 leader 或 follower 任一侧不完整；
- 固定采集的 `0xFA/0xFC` 完整流未持续通过验证；
- T 接管未收到左右两侧完整且新鲜的 `0x155/0x156/0x157` MASTER 输入；
- 角色切换前后 J1..J6/夹爪映射连续性超差；
- 固件报告低压、过流、过温、碰撞、驱动错误或堵转；
- 使能/失能状态不能由六个电机反馈确认。

以上故障不能通过增大超时、关闭验证或重复按键规避。

## 最终与历史文件

最终流程使用：

- `launch/start_four_can_collect.launch`
- `launch/start_inference_followers.launch`
- `launch/start_inference_takeover.launch`
- `launch/start_four_can_mux.launch`
- `scripts/piper_leader_mode_driver.py`
- `scripts/piper_four_state_mux.py`
- `scripts/piper_policy_follower_mux.py`
- `scripts/piper_fixed_master_reader.py`
- `scripts/piper_fixed_collect_mux.py`
- `scripts/piper_configure_leader_role.py`
- `scripts/piper_verify_fixed_master.py`
- `scripts/piper_log_snapshot.sh`
- `scripts/piper_mode_keyboard.py`
- `scripts/piper_start_slave_node.py`

`piper_leader_mode_driver.py` 和 `piper_four_state_mux.py` 只供监督推理/T 接管；
固定 MASTER 采集不引用它们。`piper_configure_leader_role.py` 是采集前置维护命令，
只能在全部 Piper ROS/SDK 节点停止时执行。

## 离线验证

这些命令不会打开 CAN：

```bash
python3 -m unittest discover -s src/piper/test -p 'test_*.py' -v

source /opt/ros/noetic/setup.bash
catkin_make
source devel/setup.bash

roslaunch --nodes piper start_four_can_collect.launch
roslaunch --nodes piper start_inference_followers.launch
roslaunch --nodes piper start_inference_takeover.launch
```

预期单测验证角色流连续性、采集维护用固定 `0xFA` 序列、T 接管用新版官方
单帧 `0xFA` 无失能序列、`0xFC` 当前位姿恢复、采集 action 话题及三个配置的
CAN 隔离；launch 节点数依次为
5、3、5。
