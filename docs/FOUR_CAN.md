# 四路 CAN：硬件、采集、接管与故障处理

本文档只讲机械臂这一侧：四路 CAN、三种互斥会话、示教采集、人工接管和常见故障。完整流程见 [PIPELINE.md](PIPELINE.md)。

命令里的 `$RECAP_ROOT` 是仓库根目录，每个新终端先 `export RECAP_ROOT=~/RECAP`（按实际路径改）。

三种会话互斥，同一时间只能运行一个：

| 会话 | 启动命令 | 主臂 | 从臂命令源 |
| --- | --- | --- | --- |
| 示教采集 | `roslaunch piper start_four_can_collect.launch` | 固定 MASTER，launch 只读 | 左右主臂 |
| 两从臂推理 | `bash openpi/RLinf/examples/recap_piper/run.sh launch followers` | 不打开主臂 CAN，可断电 | 策略 |
| 监督推理 / 人工接管 | `bash openpi/RLinf/examples/recap_piper/run.sh launch takeover` | 推理时 `0xFC OUTPUT` 跟随；`T` 后 `0xFA MASTER` | 策略或左右主臂 |

> 两从臂推理用的是 `cobot_magic/lingbot_piper` 里的修复版 mux。出厂的 follower mux 按 Space/H/I 时会误判反馈过期、失能双臂，机械臂会下落，所以本仓库没有包含它。

---

## 1. 从出厂双 CAN 到四独立 CAN

### 1.1 硬件接口

```text
can0             底盘，500000 bit/s
can_l_leader     左主臂，1000000 bit/s
can_r_leader     右主臂，1000000 bit/s
can_l_follower   左从臂，1000000 bit/s
can_r_follower   右从臂，1000000 bit/s
```

Linux 接口名最多 15 个字符，必须使用这些短名；`follower` 不能再拼成
`folower`。

### 1.2 为什么四 CAN 必须增加 ROS 仲裁

出厂双臂主从方案中，每一对主从臂共用一条 CAN，主臂在 `0xFA` 模式主动发送
`0x151/0x155/0x156/0x157/0x159`，从臂在同一物理总线上直接接收。改成四条
独立 CAN 后，这些帧不会跨 CAN 自动转发，因此需要：

```text
leader CAN -> leader driver -> ROS leader topic -> mux
policy -------------------------------------------> mux
follower feedback -------------------------------> mux
mux -> 唯一 follower command topic -> follower driver -> follower CAN
```

任何时刻，左右从臂的最终命令只能由当前 mux 发布：

```text
/teleop/command/joint_left
/teleop/command/joint_right
```

OpenPI 只能向仲裁输入发布：

```text
/teleop/policy/joint_left
/teleop/policy/joint_right
```

不能让 OpenPI 直接绕过 mux 控制从臂。

### 1.3 换一台 Cobot Magic 时必须改的两处

CAN 与相机的映射绑定在具体的 USB 口和序列号上，**不能照搬本仓库的值**。

1. **CAN**：编辑 `cobot_magic/Piper_ros_private-ros-noetic/can_config.sh` 里的 `USB_PORTS`（`EXPECTED_CAN_COUNT=4` 对应四条机械臂 CAN；底盘的 `can0` 由脚本另外处理）。逐个插入 CAN 模块，用 `sudo ethtool -i <接口名> | grep bus` 读出 `bus-info`，再填入：

```bash
USB_PORTS["1-13:1.0"]="can_l_leader:1000000"     # 把 1-13:1.0 换成你读到的 bus-info
USB_PORTS["1-12:1.0"]="can_l_follower:1000000"
USB_PORTS["1-1:1.0"]="can_r_leader:1000000"
USB_PORTS["1-4:1.0"]="can_r_follower:1000000"
```

2. **相机**：`cd cobot_magic && bash tools/camera_serial.sh` 查看三台 Astra 的序列号（需要先编译好 camera_ws），填入 `cobot_magic/camera_ws/src/ros_astra_camera/launch/multi_camera.launch` 的 `camera1_serila_number`（右腕）、`camera2_serila_number`（左腕）、`camera3_serila_number`（顶部），改完重新 `catkin_make`。

---

## 2. 按键（推理键盘）

```text
T       冻结模型；两秒后进入人工接管（仅监督推理）
I       从 HOLD 恢复推理，或人工纠正结束后恢复模型推理
Space   软件 HOLD：冻结实测姿态，保持使能和位置闭环
H       从 HOLD 限速返回本次 launch 捕获的启动姿态，到位后继续 HOLD
Q       先 HOLD，再只退出键盘；机械臂 launch 不退出
```

RECAP 采集时这些按键都在 `run.sh collect` 的终端里（多了 C/X 打分），不要同时运行 `piper_mode_keyboard.py`。

协议 `0x150` 快速急停会进入阻尼；PiPER 没有关节抱闸，因此可能下落。键盘空格不发送这个命令，而是屏蔽 policy 并以 200 Hz 持续保持位置。软件 HOLD 不是安全认证急停，电源、CAN 或驱动故障导致闭环不可用时必须使用物理支撑和现场硬件安全措施。

---

## 4. 每次开机的公共准备

### 4.1 上电前安全条件

1. 四只机械臂周围无人、无障碍物，主臂已有物理支撑。
2. 电机过热时先断电并完全冷却，不能用反复使能绕过保护。
3. 确认没有旧 Piper launch、SDK 脚本或 OpenPI 真机客户端占用 CAN/话题。

```bash
ps -ef | grep -E "piper|roslaunch|examples.piper_real" | grep -v grep
```

### 4.2 终端 1：ROS Master

```bash
source /opt/ros/noetic/setup.bash
roscore
```

### 4.3 配置五条 CAN

只能在全部 Piper 节点都停止时执行：

```bash
cd $RECAP_ROOT/cobot_magic/Piper_ros_private-ros-noetic

sudo bash can_config.sh
ip -br link show type can
```

必须看到：

```text
can0             UP
can_l_leader     UP
can_l_follower   UP
can_r_leader     UP
can_r_follower   UP
```

任一接口 DOWN 时不要启动 Piper。launch 运行期间严禁重新执行 `can_config.sh`，
否则会出现 `SEND_MESSAGE_FAILED (100017)`、接口 not UP、发送队列或 bus-off
问题。

### 4.4 终端 3：启动相机

```bash
cd $RECAP_ROOT/cobot_magic/camera_ws

source /opt/ros/noetic/setup.bash
source devel/setup.bash

roslaunch astra_camera multi_camera.launch
```

`Can't open /dev/bus/usb/: Busy (-6)` 表示同一相机已被另一进程占用。不要同时
重复启动相机 launch 和预览程序；先检查：

```bash
ps -ef | grep -E "astra|camera|preview_cameras" | grep -v grep
```

---

## 5. 数据采集完整流程

数据采集是独立命令行会话，不使用 `C`，最终采用“预置固定 MASTER + ROS
话题转发”。采集 launch 不再把主臂从 OUTPUT 动态切到 MASTER，也不会向主臂
发送使能、失能、位置保持、`0xFA` 或 `0xFC`。

要点：

- 配置工具默认执行官方最小序列：SDK 正常初始化后只保存一次 `0xFA`，不使能
  主臂，也不再默认执行已经反复失败的运行时 OUTPUT→MASTER 切换；
- 保存后彻底断开两只主臂控制器电源 10 秒，再上电并只读复核；
- 启动 ROS 前用原始 SocketCAN 同时统计协议允许的 `0x15x/0x16x/0x17x` 三组；
- 固定 reader 直接读原始 CAN，自动识别三种偏移，以持续完整的三条关节帧作为
  MASTER 必要条件，对应夹爪帧单独告警；
- 预检通过后，采集 launch 全程只读主臂 CAN。

如果使用官方最小 `0xFA` 保存序列并彻底断电重启后，三组合法 ID 仍没有 MASTER 关节控制
帧，那么问题在主臂控制器/固件没有进入 MASTER。受“只能用主臂模式 + 话题
切换”约束，ROS 端没有第二种算法可以凭空生成主臂输入；此时必须升级/修复两只
主臂固件或由 AgileX 确认对应固件的固定 MASTER 配置方式，不能退回反复动态
切换或伪造 MASTER 成功。

### 5.1 无 Piper 节点时：配置并验证固定 MASTER

停止采集、两种推理、OpenPI 真机客户端和所有直连 Piper SDK 程序。`roscore`
和相机可以保留，但四个 Piper CAN 不能有其他 owner。然后执行：

```bash
cd $RECAP_ROOT/cobot_magic/Piper_ros_private-ros-noetic

source /opt/ros/noetic/setup.bash
source devel/setup.bash

rosrun piper piper_configure_leader_role.py \
  --role master \
  --master-activation restart \
  --left-can can_l_leader \
  --right-can can_r_leader \
  --verify-timeout-s 8
```

配置工具先确认两只主臂仍有 OUTPUT 反馈，再按官方 demo 初始化 SDK、打印固件
版本并只保存 `0xFA`。它不调用 `EnableArm/DisableArm`，也不发送关节/夹爪
目标。正确输出必须包含左右两侧 `saved 0xFA`。随后关闭的是左右主臂控制器
电源，不是只拔 CAN 航空插头；等待 10 秒后再上电。从臂无需断电。

命令成功后立即执行独立只读预检：

```bash
ip -br link show type can

rosrun piper piper_verify_fixed_master.py \
  --left-can can_l_leader \
  --right-can can_r_leader \
  --duration-s 5 \
  --min-joint-hz 10
```

工具会分别打印 offset `0x00/0x10/0x20`。每侧至少一组的三条关节帧必须明显
大于 `10 Hz`，最后显示：

```text
MASTER preflight PASSED for both leaders; fixed collection may be started.
```

对应的夹爪帧会单独显示。若其为 0，关节摇操仍可验证，但正式录制前必须小幅操作
夹爪并确认 `/teleop/leader/joint_*` 的第 7 维能够更新；否则该固件下的夹爪 action
不完整，不能开始采集。

如果三组都为 0 且工具显示 `observed no CAN frame at all`，但恢复 `0xFC` 又能
得到约 200 Hz，则 CAN 名称、波特率和接收线路已经排除，失败点是控制器没有从
`0xFA` 进入协议控制模式 `0x06`。不要启动采集 launch，也不要重复发送
`/enable_flag=true`；记录配置工具打印的固件版本并由 AgileX 确认/升级支持
0x470 联动示教输入的固件。ROS 不能在不伪造主臂输入的情况下补出这些帧。
保存一次诊断快照：

```bash
rosrun piper piper_log_snapshot.sh
```

### 5.2 终端 2：启动只读固定 MASTER 采集配置

```bash
cd $RECAP_ROOT/cobot_magic/Piper_ros_private-ros-noetic

source /opt/ros/noetic/setup.bash
source devel/setup.bash

roslaunch piper start_four_can_collect.launch \
  master_left_can:=can_l_leader \
  master_right_can:=can_r_leader \
  puppet_left_can:=can_l_follower \
  puppet_right_can:=can_r_follower \
  auto_enable:=false \
  publish_hz:=200 
  #max_joint_velocity_rad_s:=2.5
```

预期五个节点：

```text
/piper_fixed_master_left_reader
/piper_fixed_master_right_reader
/piper_collect_puppet_left_driver
/piper_collect_puppet_right_driver
/piper_fixed_collect_mux
```

其中两个 fixed reader 只使用原始 SocketCAN 接收，自动识别合法控制偏移。启动
后如果主臂静止不再发送新目标，reader 会保持最后姿态；首次启动仍建议在 30 秒
等待窗口内分别移动左右主臂一次，使 reader 完成角色确认。看到以下两行后，再
让操作者把主臂和
从臂放在相近、安全的姿态，执行一次显式使能：

```text
can_l_leader fixed MASTER stream verified at control offset 0x..
can_r_leader fixed MASTER stream verified at control offset 0x..
```

```bash
rostopic pub -1 /enable_flag std_msgs/Bool "data: true"
```

预期关键日志：

```text
Fixed collection enable received; seeding commands from follower feedback
Fixed collection active: verified MASTER inputs own both followers
```

只有最后一行出现后才做低幅映射验收：左只控制左、右只控制右；J1～J6 不交换
且符号一致；夹爪开合方向一致。`2.5 rad/s` 是 ROS 跟随斜率上限，不会修改
主臂重力补偿参数；200 Hz、`queue_size=1` 和 TCP_NODELAY 防止旧命令排队。

检查话题：

```bash
rostopic echo -n 1 /teleop/mode
rostopic hz /teleop/leader/joint_left
rostopic hz /teleop/leader/joint_right
rostopic hz /puppet/joint_left
rostopic hz /puppet/joint_right
```

`/teleop/mode` 必须为 `COLLECT`，四路数据必须持续更新。

### 5.3 可选采集急停键盘

```bash
source /opt/ros/noetic/setup.bash
source $RECAP_ROOT/cobot_magic/Piper_ros_private-ros-noetic/devel/setup.bash

rosrun piper piper_mode_keyboard.py _profile:=collect
```

该配置只使用空格急停和 `Q` 退出；`C/I/T/H` 都禁用。

### 5.4 终端 6：连续录制 episode

必须显式使用独立主臂话题作为 action。当前默认值也已改正为这两个话题：

```bash
cd $RECAP_ROOT/cobot_magic/collect_data

source /opt/ros/noetic/setup.bash
source $RECAP_ROOT/cobot_magic/camera_ws/devel/setup.bash
source $RECAP_ROOT/cobot_magic/Piper_ros_private-ros-noetic/devel/setup.bash

/usr/bin/python3 new_collect_data.py \
  --dataset_dir ~/recap_data_raw/hdf5 \
  --task_name connect_strings \
  --max_timesteps 3900 \
  --episode_idx 0 \
  --frame_rate 30 \
  --next_episode_delay -1 \
  --master_arm_left_topic /teleop/leader/joint_left \
  --master_arm_right_topic /teleop/leader/joint_right \
  --puppet_arm_left_topic /puppet/joint_left \
  --puppet_arm_right_topic /puppet/joint_right
```

录制逻辑：

- 到 `max_timesteps` 后自动保存当前 episode，但程序不退出。
- 出现 `Ready for next episode` 后，按右方向键、`n` 或空格开始下一条。
- 录制中按右方向键、`n` 或空格会提前保存本条；再按一次开始下一条。
- `q` 保存已有帧并退出。
- episode 编号从 `--episode_idx` 自动递增。

每条 HDF5 至少应有：

```text
/observations/qpos    (N, 14)：两个从臂状态
/observations/qvel    (N, 14)
/observations/effort  (N, 14)
/action               (N, 14)：两个主臂示教目标
三路图像              N 帧
```

若 `/action` 与 `/observations/qpos` 总是完全相同，首先检查 recorder 是否错误
订阅了 `/puppet/joint_*` 作为 master action。

### 5.5 正常结束采集

1. 在 recorder 按 `q`。
2. 在采集键盘按 `Q`（若启动了键盘）。
3. 执行失能：

```bash
rostopic pub -1 /enable_flag std_msgs/Bool "data: false"
```

4. 确认从臂停止后，再在采集 launch 终端按 `Ctrl+C`。

固定 MASTER 会保留到主动恢复 OUTPUT。切换到两从臂推理时可直接
关闭两只主臂电源；切换到监督推理前，必须在所有 Piper 节点停止时恢复：

```bash
rosrun piper piper_configure_leader_role.py \
  --role output \
  --left-can can_l_leader \
  --right-can can_r_leader \
  --verify-timeout-s 8
```

该命令即使当前已经能看到 `0x2Ax` OUTPUT 反馈，也会再次写入 `0xFC` 并进入
待机，确保 Flash 中残留的 `0xFA` 被清除；否则当前运行态虽像 OUTPUT，下一次
主臂掉电重启仍可能重新进入重力补偿 MASTER 状态。

---

## 6. 人工接管（监督推理）

四臂都必须通电，四条 CAN 都必须 UP；主臂启动姿态应有支撑。启动前让对应主从臂接近同一姿态：mux 的启用前安全门默认要求每个关节差不超过 `0.08 rad`、夹爪差不超过 `0.015 m`，超差时打印实际差值，主臂保持失能，不会用电机扭矩强行追赶。如果刚用过固定 MASTER 采集，必须先在所有 Piper 节点停止时恢复 OUTPUT（见第 5.5 节）。

```bash
bash $RECAP_ROOT/openpi/RLinf/examples/recap_piper/run.sh launch takeover
```

预期五个节点：`/piper_master_left_driver`、`/piper_master_right_driver`、`/piper_puppet_left_driver`、`/piper_puppet_right_driver`、`/piper_four_state_mux`。

### 6.1 人工接管与恢复推理

1. 推理正常且只需小幅纠正时按 `T`。
2. 两秒内扶住左右主臂。
3. 看到 `TAKEOVER active` 后再摇动主臂纠正从臂。
4. 完成纠正后按 `I`。
5. 系统将主臂恢复 OUTPUT，重新对齐并在确认后恢复 policy。

如果出现：

```text
did not start 0x155-0x157 control feedback after 0xFA
A leader role transition failed
TAKEOVER master-role transition timed out
```

说明固件没有产生可持续验证的 MASTER 流；从臂不会接受半套接管。按空格进入
HOLD，不要重复按 `T`。

### 6.2 急停和 HOME

- 异常运动立即按空格；四臂冻结实测姿态、保持使能和位置闭环；OpenPI 暂停且
  丢弃已有动作块，policy 不只是被 mux 屏蔽。
- 确认空间安全后按 `H`，系统以限速返回本次 launch 启动时捕获的四臂姿态，
  到位后继续 HOLD；按 `I` 才恢复 policy。
- `Q` 先 HOLD 再退出键盘，不会关闭 launch 或撤销使能。
- HOME 不是硬编码全零，也不会使用未验证的控制器旧目标。
- 软件 HOLD 不是安全认证急停；电源/CAN/驱动故障使闭环不可用时仍需物理支撑。

---

## 7. 两种推理的正常停机

1. 先停止 真机客户端（`Ctrl+C`；默认会发布当前反馈姿态作为 hold）。
2. 立即显式失能：

```bash
rostopic pub -1 /enable_flag std_msgs/Bool "data: false"
```

3. 监督推理键盘按 `Q`；两从臂推理键盘若启动也按 `Q`。
4. 再停止当前唯一 Piper launch、Policy Server、相机和 roscore。

不要先停止 CAN owner 后再发送失能，因为那时 SDK socket 已关闭，可能看到
shutdown 阶段的 `SEND_MESSAGE_FAILED (100017)`。

---

## 8. 三种会话之间怎样切换

这不是热切换。必须：

```text
停止动作源 -> /enable_flag=false -> 停止当前 Piper launch
-> 确认无 Piper/SDK owner -> 必要时重新执行 can_config.sh -> 启动另一套 launch
```

- 两从臂推理：可以只给从臂通电；主臂可断电。
- 数据采集：四臂都需通电，主臂必须预置、重启并预检为固定 MASTER。
- 监督推理：四臂都需通电，主臂必须先恢复为 OUTPUT。
- 从两从臂推理切到采集/监督推理前，应先关闭普通推理 launch，再给主臂上电并
  确认 leader CAN 为 UP。
- 采集与监督推理不能热切：固定 MASTER 采集结束后，必须停止所有 Piper 节点，
  用 `piper_configure_leader_role.py --role output` 恢复并验证 OUTPUT，才能启动
  监督推理。

---

## 9. 常见故障

### 9.1 CAN 不存在、DOWN、`100017`

历史日志中的：

```text
CAN socket ... does not exist
RuntimeError: CAN port ... is not UP
SendCanMessage(SEND_MESSAGE_FAILED (100017))
```

属于接口命名/UP 状态、运行中重配 CAN 或发送队列问题。处理顺序：急停、失能、
停止 owner，再检查/重配；不能在 launch 存活时执行 `can_config.sh`。

```bash
ip -details -statistics link show can_l_follower
ip -details -statistics link show can_r_follower
tc -s qdisc show dev can_l_follower
tc -s qdisc show dev can_r_follower
```

### 9.2 主臂抬起、数秒后失能或过热

最新日志证明该现象不是关节映射超差或硬件过热保护，而是运行时 `0xFA` 没有
产生持续 `0x155-0x157`，随后 mux 主动 fail-close。最终修正：

- 数据采集彻底删除运行时 OUTPUT→MASTER；
- 采集 reader 只读已经配置并验证为固定 MASTER 的主臂；
- launch 前用原始 CAN 逐项预检 `0x15x/0x16x/0x17x` 三组合法控制 ID；
- follower 仍先读当前 J1～J6/夹爪并预装同一目标，再进入 MOVE J 和使能；
- 固件报告过流、过温、碰撞、driver error、stall 或 NTC 过温时拒绝使能/运动；
- 六个电机必须全部反馈使能成功。

### 9.3 `MASTER` 假就绪或 T 后仍不能控制

固定采集要求左右两侧至少一组合法的三条关节控制帧持续刷新，对应夹爪帧单独
报告；预检不通过时 launch 根本不启动。监督推理的 T 路径仍要求运行时角色
持续流和映射连续性通过，T 期间从臂先冻结，双侧都成功后才统一换 owner。

### 9.4 CAN 发送卡死（SDK 报 `100017`，接口仍是 UP）

candleLight 适配器固件偶尔丢失回显帧，`gs_usb` 的 10 个发送槽漏光后内核发送队列永久停住；接口看起来仍是 UP / ERROR-ACTIVE，但什么都发不出去。推理和采集都可能出现。恢复（launch 不用重启，驱动会重新打开端口）：

```bash
sudo ip link set can_r_follower down && sudo ip link set can_r_follower up
```

根治要刷新 candleLight 固件或换适配器，USB 隔离器和总线终端电阻也值得检查。

