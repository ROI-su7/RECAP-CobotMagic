# agx 机械臂(noetic)

[EN](README(EN).md)

![ubuntu](https://img.shields.io/badge/Ubuntu-20.04-orange.svg)

Test:

|PYTHON |STATE|
|---|---|
|![noeti](https://img.shields.io/badge/ros-noetic-blue.svg)|![Pass](https://img.shields.io/badge/Pass-blue.svg)|

## 安装方法

### 安装依赖

```shell
pip3 install python-can
```

```shell
pip3 install piper_sdk
```

## can模块使用

### 2.1 使能can模块

首先需要设置好shell脚本参数

#### 2.1.1 单条机械臂

##### 1）pc只插入一个usb转can模块

- **此处使用`can_activate.sh`脚本**

直接执行

```bash
bash can_activate.sh can0 1000000
```

##### 2）pc插入多个usb转can模块

- **此处使用`can_activate.sh`脚本**

拔掉所有can模块

只将连接到机械臂的can模块插入PC，执行

```shell
sudo ethtool -i can0 | grep bus
```

并记录下`bus-info`的数值例如`1-2:1.0`

ps:**一般第一个插入的can模块会默认是can0，如果没有查询到can可以使用`bash find_all_can_port.sh`来查看刚才usb地址对应的can名称**

假设上面的操作记录的`bus-info`数值为`1-2:1.0`

然后执行,查看can设备是否激活成功

```bash
bash can_activate.sh can_piper 1000000 "1-2:1.0"
```

ps:**此处的意思是，1-2:1.0硬件编码的usb端口插入的can设备，名字被重命名为can_piper,波特率为1000000，并激活**

然后执行`ifconfig`查看是否有`can_piper`，如果有则can模块设置成功

### 运行节点

#### 多个机械臂

##### 读取主从臂消息，不控制从臂

`start_ms_piper.launch` 文件如下：

```xml
<launch>
  <!-- 定义 mode 参数，默认值为 0,读取主从臂的消息，并转发到ROS -->
  <arg name="mode" default="0" />
  <arg name="auto_enable" default="true" />
  <!-- 启动左侧机械臂节点 -->
    <node name="$(anon piper_left)" pkg="piper" type="piper_start_ms_node.py" output="screen">
      <param name="can_port" value="can_left" />
      <param name="mode" value="$(arg mode)" />
      <param name="auto_enable" value="$(arg auto_enable)" />
      <remap from="/puppet/joint_states" to="/puppet/joint_left" />
      <remap from="/master/joint_states" to="/master/joint_left" />
    </node>

  <!-- 启动右侧机械臂节点 -->
    <node name="$(anon piper_right)" pkg="piper" type="piper_start_ms_node.py" output="screen">
      <param name="can_port" value="can_right" />
      <param name="mode" value="$(arg mode)" />
      <param name="auto_enable" value="$(arg auto_enable)" />
      <remap from="/puppet/joint_states" to="/puppet/joint_right" />
      <remap from="/master/joint_states" to="/master/joint_right" />
    </node>
</launch>
```

启动节点，主要使用launch启动左右侧臂

```shell
# 采集数据时，读取主从臂的消息，此处auto_enable参数设定可去掉
roslaunch piper start_ms_piper.launch mode:=0 auto_enable:=true
```

然后执行

```shell
rostopic list
```

有如下几个topic：

```shell
/arm_status
/master/joint_left
/master/joint_right
/puppet/joint_left
/puppet/joint_right
```

其中`/master/joint_left`、`/master/joint_right`、`/puppet/joint_left`、`/puppet/joint_right`可以读取到主臂和从臂的消息

##### 通过节点控制从臂

**先将主臂的航插断开！！！**

```shell
# 控制两条从臂，此处auto_enable参数设定可去掉
roslaunch piper start_ms_piper.launch mode:=1 auto_enable:=true
```

然后执行

```shell
rostopic list
```

有如下几个topic：

```shell
/arm_status
/enable_flag
/master/joint_left
/master/joint_right
/puppet/joint_left
/puppet/joint_right
```

`/enable_flag`用来使能机械臂，launch里默认打开自动使能，如果关闭，给这个topic 发送true，令其使能

其中，`/master/joint_left`、`/master/joint_right`是用来外部控制的topic，进而控制从臂

***如何不通过终端打印判断使能: 确认发送了使能消息后，用手摇动机械臂，如果无法运动关节，说明关节上电，使能成功**

## 注意事项

- 需要先激活can设备，并且设置正确的波特率，才可以读取机械臂消息或者控制机械臂
- **想要用ros控制两条从机械臂，需要先将主臂的航插拔掉！！！**
- 在节点参数mode为1的情况下`/master/joint_left`、`/master/joint_right`、`/puppet/joint_left`、`/puppet/joint_right`仅仅为数据topic，只需要读取，一般这种模式下，主臂和从臂的can是连通的
- 在节点参数mode为0的情况下，`/master/joint_left`、`/master/joint_right`为控制从臂的topic，`/puppet/joint_left`、`/puppet/joint_right`为从臂反馈，想要给`/master/joint_left`、`/master/joint_right`发送数据运动从臂之前，必须要给`/enable_flag`topic 发送true

## 有关piper的控制

### （1）令主从机械臂回零

- 必须要主从机械臂连接

```shell
rosservice call /can_left/go_zero_master
rosservice call /can_right/go_zero_master
```

注意执行后不能直接遥操作，此时进入的是受控的回零模式，要回到遥操作模式需要再执行下面的指令

### （2）令主从机械臂从回零模式回到主从控制模式

```shell
rosservice call /can_left/restore_ms_mode
rosservice call /can_right/restore_ms_mode
```

该指令会令主从机械臂回到遥操作模式

### piper自定义消息

ros功能包`piper_msgs`

机械臂自身状态反馈消息，对应can协议中`id=0x2A1`的反馈消息

`PiperStatusMsg.msg`

```c
uint8 ctrl_mode
uint8 arm_status
uint8 mode_feedback
uint8 teach_status
uint8 motion_status
uint8 trajectory_num
int64 err_code
bool joint_1_angle_limit
bool joint_2_angle_limit
bool joint_3_angle_limit
bool joint_4_angle_limit
bool joint_5_angle_limit
bool joint_6_angle_limit
bool communication_status_joint_1
bool communication_status_joint_2
bool communication_status_joint_3
bool communication_status_joint_4
bool communication_status_joint_5
bool communication_status_joint_6
```

机械臂末端位姿控制，注意:有些奇异点无法到达

`PosCmd.msg`

```c
float64 x
float64 y
float64 z
float64 roll
float64 pitch
float64 yaw
float64 gripper
int32 mode1
int32 mode2
```
