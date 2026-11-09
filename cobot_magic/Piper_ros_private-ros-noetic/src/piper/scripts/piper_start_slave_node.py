#!/usr/bin/env python3
# -*-coding:utf8-*-
# 本文件为打开从臂控制
# 默认认为从臂有夹爪
# mode为0时为发送从臂消息
# mode为1时为控制从臂，此时如果要控制从臂，需要给主臂的topic发送消息
import rospy
import rosnode
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, String
import time
import threading
import math
from piper_sdk import *
from piper_sdk import C_PiperInterface
from piper_can_tx_guard import CanLinkWatch, CanTxGuard, reset_command
from piper_msgs.msg import PiperStatusMsg, PosCmd
from geometry_msgs.msg import PoseStamped
from tf.transformations import quaternion_from_euler  # 用于欧拉角到四元数的转换

def check_ros_master():
    try:
        rosnode.rosnode_ping('rosout', max_count=1, verbose=False)
        rospy.loginfo("ROS Master is running.")
    except rosnode.ROSNodeIOException:
        rospy.logerr("ROS Master is not running.")
        raise RuntimeError("ROS Master is not running.")

class C_PiperRosNode():
    """机械臂ros节点
    """
    def __init__(self) -> None:
        check_ros_master()
        rospy.init_node('piper_start_all_node', anonymous=True)

        self.can_port = "can0"
        if rospy.has_param('~can_port'):
            self.can_port = rospy.get_param("~can_port")
            rospy.loginfo("%s is %s", rospy.resolve_name('~can_port'), self.can_port)
        else: 
            rospy.loginfo("未找到can_port参数,请输入 _can_port:=can0 类似的格式")
            exit(0)
        # 模式，模式为1的时候，才能够控制从臂
        self.mode = 0
        if rospy.has_param('~mode'):
            self.mode = rospy.get_param("~mode")
            rospy.loginfo("%s is %s", rospy.resolve_name('~mode'), self.mode)
        else:
            rospy.loginfo("未找到mode参数,请输入 _mode:=0 类似的格式")
            exit(0)

        # 是否自动使能，默认不自动使能，只有模式为1的时候才能够被设置为自动使能
        self.auto_enable = False
        if rospy.has_param('~auto_enable'):
            if(rospy.get_param("~auto_enable") and self.mode == 1):
                self.auto_enable = True
        rospy.loginfo("%s is %s", rospy.resolve_name('~auto_enable'), self.auto_enable)
        # Joint commands can arrive at 100-200 Hz. Logging every joint of every
        # command floods the terminal and ~/.ros logs, so keep it opt-in and
        # throttle enabled summaries.
        self.log_joint_commands = bool(
            rospy.get_param('~log_joint_commands', False)
        )
        self.log_joint_commands_period = float(
            rospy.get_param('~log_joint_commands_period_s', 1.0)
        )
        # Piper SDK initialization sends 12 motor-limit queries and one
        # firmware-version query. They are optional for feedback, enable and
        # joint control, and some gs_usb adapters reject this startup burst.
        # Preserve the legacy default; the four-CAN launch disables only these
        # metadata queries.
        self.sdk_piper_init = bool(
            rospy.get_param('~sdk_piper_init', True)
        )
        self.enable_timeout = float(rospy.get_param('~enable_timeout_s', 5.0))
        self.enable_retry_period = float(
            rospy.get_param('~enable_retry_period_s', 0.1)
        )
        self.feedback_seed_timeout = float(
            rospy.get_param('~feedback_seed_timeout_s', 2.0)
        )
        # The enable watchdog reads a cached SDK frame.  Anything older than
        # this is reported as lost confirmation rather than trusted.
        self.feedback_stale_timeout = float(
            rospy.get_param('~feedback_stale_s', 0.5)
        )
        if (
            self.enable_timeout <= 0.0
            or self.enable_retry_period <= 0.0
            or self.feedback_seed_timeout <= 0.0
            or self.feedback_stale_timeout <= 0.0
        ):
            raise ValueError("enable and feedback timeouts must be positive")
        rospy.loginfo(
            "%s is %s", rospy.resolve_name('~sdk_piper_init'), self.sdk_piper_init
        )
        # publish
        self.joint_std_pub_puppet = rospy.Publisher('/puppet/joint_states', JointState, queue_size=1, tcp_nodelay=True)
        self.arm_status_pub = rospy.Publisher('/puppet/arm_status', PiperStatusMsg, queue_size=1, tcp_nodelay=True)
        self.end_pose_pub = rospy.Publisher('/puppet/end_pose', PoseStamped, queue_size=1, tcp_nodelay=True)
        
        self.__enable_flag = False
        self._enable_requested = self.auto_enable
        self._emergency_stop_active = False
        self._shutting_down = False
        self._enable_lock = threading.Lock()
        self._command_lock = threading.Lock()
        rospy.on_shutdown(self._on_shutdown)
        # 从臂消息
        self.joint_state_slave = JointState()
        self.joint_state_slave.name = ['joint0', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
        self.joint_state_slave.position = [0.0] * 7
        self.joint_state_slave.velocity = [0.0] * 7
        self.joint_state_slave.effort = [0.0] * 7

        # gs_usb/candleLight adapters can stop transmitting while the netdev
        # stays UP (kernel: "Unexpected unused echo id"). The arm then holds
        # its last target and silently ignores every command. The guard
        # notices commands being sent while tx_packets is frozen and the mux
        # enters HOLD; a down/up of the interface (no relaunch) recovers it.
        self.tx_stall_timeout = float(rospy.get_param('~tx_stall_timeout_s', 0.5))
        # MOVE J mode (MotionCtrl_2) only needs re-asserting, not one frame per
        # target: it was the fifth frame of every command. 0 = every command.
        self.motion_ctrl_period = float(rospy.get_param('~motion_ctrl_period_s', 0.1))
        self._last_motion_ctrl = 0.0
        self._tx_guard = CanTxGuard(self.can_port, stall_timeout=self.tx_stall_timeout)
        self._link_watch = CanLinkWatch(self.can_port)
        self.tx_stall_pub = rospy.Publisher(
            '/teleop/can_tx_stalled', String, queue_size=10
        )
        self.piper = C_PiperInterface(can_name=self.can_port)
        self.piper.ConnectPort(piper_init=self.sdk_piper_init)
        # 模式为1的时候，订阅控制消息
        if(self.mode == 1):
            sub_pos_th = threading.Thread(target=self.SubPosThread)
            sub_joint_th = threading.Thread(target=self.SubJointThread)
            sub_enable_th = threading.Thread(target=self.SubEnableThread)
            
            sub_pos_th.daemon = True
            sub_joint_th.daemon = True
            sub_enable_th.daemon = True
            
            sub_pos_th.start()
            sub_joint_th.start()
            sub_enable_th.start()

    def GetEnableFlag(self):
        return self.__enable_flag

    def _on_shutdown(self):
        # Subscribers may still receive the mux's final message while roslaunch
        # is closing the SDK socket. Stop new transmissions, wait for any
        # in-flight command batch, then close the SDK port explicitly.
        self._shutting_down = True
        self.__enable_flag = False
        if not hasattr(self, 'piper'):
            return
        with self._command_lock:
            try:
                self.piper.DisconnectPort(thread_timeout=0.2)
            except Exception as exc:
                rospy.logwarn("Piper CAN close during shutdown failed: %s", exc)

    # foc_status bit meanings, in the SDK's own bit order.  Only
    # driver_enable_status was ever consulted, so a firmware-side protection
    # trip used to disable an arm without leaving any trace in the logs.
    FOC_FAULT_FIELDS = (
        ("voltage_too_low", "欠压"),
        ("motor_overheating", "电机过温"),
        ("driver_overcurrent", "驱动过流"),
        ("driver_overheating", "驱动过温"),
        ("collision_status", "碰撞保护"),
        ("driver_error_status", "驱动错误"),
        ("stall_status", "堵转保护"),
    )

    def _low_speed_motors(self, low_speed):
        return [
            getattr(low_speed, "motor_{}".format(index)) for index in range(1, 7)
        ]

    def _driver_enable_states(self, low_speed=None):
        if low_speed is None:
            low_speed = self.piper.GetArmLowSpdInfoMsgs()
        return [
            bool(motor.foc_status.driver_enable_status)
            for motor in self._low_speed_motors(low_speed)
        ]

    def _feedback_age(self, low_speed):
        """Seconds since the last low-speed frame, or None if never seen.

        GetArmLowSpdInfoMsgs returns the SDK's cached frame with no freshness
        guarantee.  When CAN RX stops, the cache keeps reporting the last
        known-good state forever, so a stale read must never be mistaken for
        a healthy arm.
        """
        stamp = float(low_speed.time_stamp)
        if stamp <= 0.0:
            return None
        return max(0.0, time.time() - stamp)

    def _driver_faults(self, low_speed):
        """Latched protection bits per motor, as ['J5 驱动过流', ...]."""
        faults = []
        for index, motor in enumerate(self._low_speed_motors(low_speed), start=1):
            for field, label in self.FOC_FAULT_FIELDS:
                if bool(getattr(motor.foc_status, field, False)):
                    faults.append("J{} {}".format(index, label))
        return faults

    def _driver_telemetry(self, low_speed):
        return " ".join(
            "J{}:{:.1f}V/{}°C/{}°C/{:.2f}A".format(
                index,
                motor.vol / 10.0,
                motor.foc_temp,
                motor.motor_temp,
                motor.bus_current / 1000.0,
            )
            for index, motor in enumerate(self._low_speed_motors(low_speed), start=1)
        )

    def _supervised_enable_states(self):
        """Enable states for the watchdog, with staleness and fault reporting.

        Raises when the feedback is too old to trust so the caller treats it
        as lost confirmation instead of silently reusing the cached frame.
        """
        low_speed = self.piper.GetArmLowSpdInfoMsgs()
        age = self._feedback_age(low_speed)
        if age is None:
            raise RuntimeError("no low-speed feedback received yet")
        if age > self.feedback_stale_timeout:
            raise RuntimeError(
                "low-speed feedback is {:.2f}s old (limit {:.2f}s); "
                "CAN feedback from {} has stopped".format(
                    age, self.feedback_stale_timeout, self.can_port
                )
            )
        faults = self._driver_faults(low_speed)
        if faults:
            rospy.logerr_throttle(
                1.0,
                "%s driver protection tripped: %s | %s",
                self.can_port,
                ", ".join(faults),
                self._driver_telemetry(low_speed),
            )
        return self._driver_enable_states(low_speed)

    def _wait_for_seed_feedback(self):
        deadline = time.monotonic() + self.feedback_seed_timeout
        joint_message = self.piper.GetArmJointMsgs()
        gripper_message = self.piper.GetArmGripperMsgs()
        while not self._shutting_down and not rospy.is_shutdown():
            joint_message = self.piper.GetArmJointMsgs()
            gripper_message = self.piper.GetArmGripperMsgs()
            if (
                float(joint_message.time_stamp) > 0.0
                and float(gripper_message.time_stamp) > 0.0
            ):
                return joint_message, gripper_message
            if time.monotonic() >= deadline:
                break
            time.sleep(0.02)
        rospy.logerr(
            "Cannot seed %s before enable: joint/gripper feedback is unavailable "
            "(joint_stamp=%.3f gripper_stamp=%.3f)",
            self.can_port,
            float(joint_message.time_stamp),
            float(gripper_message.time_stamp),
        )
        return None, None

    def _seed_current_joint_target(self):
        """Preload measured J1..J6/gripper before any mode-1 motor enable."""
        joint_message, gripper_message = self._wait_for_seed_feedback()
        if joint_message is None or gripper_message is None:
            return False
        joints = joint_message.joint_state
        values = (
            int(joints.joint_1),
            int(joints.joint_2),
            int(joints.joint_3),
            int(joints.joint_4),
            int(joints.joint_5),
            int(joints.joint_6),
        )
        gripper = max(
            0,
            min(80000, int(gripper_message.gripper_state.grippers_angle)),
        )
        # Standby, target preload, CAN MOVE J, repeat the same measured target.
        # The gripper position loop remains disabled until enable is confirmed.
        self.piper.MotionCtrl_2(0x00, 0x01, 0)
        self.piper.JointCtrl(*values)
        self.piper.GripperCtrl(gripper, 1000, 0x00, 0)
        self.piper.MotionCtrl_2(0x01, 0x01, 100)
        self.piper.JointCtrl(*values)
        self.piper.GripperCtrl(gripper, 1000, 0x00, 0)
        return True

    def _enable_and_wait(self):
        """Request enable until all six motor drivers confirm it."""
        with self._enable_lock:
            self.__enable_flag = False
            with self._command_lock:
                if self._shutting_down or rospy.is_shutdown():
                    return False
                if not self._seed_current_joint_target():
                    return False
            deadline = time.monotonic() + self.enable_timeout
            states = self._driver_enable_states()
            while (
                not self._shutting_down
                and not rospy.is_shutdown()
                and not self._emergency_stop_active
            ):
                states = self._driver_enable_states()
                if all(states):
                    self.__enable_flag = True
                    with self._command_lock:
                        if self._shutting_down or rospy.is_shutdown():
                            self.__enable_flag = False
                            return False
                        # Hold the measured gripper opening on enable. Sending
                        # an invented zero target closes the gripper abruptly.
                        gripper = int(
                            self.piper.GetArmGripperMsgs()
                            .gripper_state.grippers_angle
                        )
                        gripper = max(0, min(80000, gripper))
                        self.piper.GripperCtrl(gripper, 1000, 0x01, 0)
                    rospy.loginfo("All six Piper motor drivers are enabled")
                    return True
                if time.monotonic() >= deadline:
                    break
                with self._command_lock:
                    if self._shutting_down or rospy.is_shutdown():
                        break
                    self.piper.EnableArm(7)
                time.sleep(self.enable_retry_period)
            rospy.logerr("Piper enable timed out; motor states=%s", states)
            return False

    def _disable_and_wait(self):
        """Request disable until all six motor drivers confirm it.

        A single CAN disable frame is not sufficient on a busy or recovering
        SocketCAN interface.  Keep the software command gate closed first,
        then retry the physical disable and verify low-speed motor feedback.
        """
        with self._enable_lock:
            self.__enable_flag = False
            deadline = time.monotonic() + self.enable_timeout
            states = self._driver_enable_states()
            while not self._shutting_down and not rospy.is_shutdown():
                states = self._driver_enable_states()
                if not any(states):
                    rospy.loginfo("All six Piper motor drivers are disabled")
                    return True
                if time.monotonic() >= deadline:
                    break
                with self._command_lock:
                    if self._shutting_down or rospy.is_shutdown():
                        break
                    self.piper.DisableArm(7)
                    gripper = int(
                        self.piper.GetArmGripperMsgs()
                        .gripper_state.grippers_angle
                    )
                    self.piper.GripperCtrl(
                        max(0, min(80000, gripper)), 1000, 0x00, 0
                    )
                time.sleep(self.enable_retry_period)
            rospy.logerr("Piper disable timed out; motor states=%s", states)
            return False

    def Pubilsh(self):
        """机械臂消息发布
        """
        rate = rospy.Rate(200)  # 200 Hz
        auto_enable_attempted = False
        while not rospy.is_shutdown():
            if self.auto_enable and self.mode == 1 and not auto_enable_attempted:
                auto_enable_attempted = True
                if not self._enable_and_wait():
                    rospy.logerr("Automatic Piper enable failed; control remains inhibited")
            # 发布消息
            self.PublishSlaveArmJointAndGripper()
            self.PublishSlaveArmState()
            self.PublishSlaveArmEndPose()
            self._check_can_link()
            self._check_tx_stall()

            rate.sleep()

    def _check_tx_stall(self):
        """Report once when commands are sent but the CAN adapter sends none."""
        event = self._tx_guard.poll()
        if event == "resumed":
            rospy.logwarn(
                "%s CAN transmission resumed; press I to leave HOLD", self.can_port
            )
        elif event == "stalled":
            rospy.logerr(
                "%s CAN TX STALLED: joint commands are sent but the adapter has "
                "transmitted nothing for %.1fs (tx_packets=%s); this arm only "
                "holds its last target (dmesg: gs_usb 'echo id'). Keep this "
                "launch running and reset the adapter: `%s`",
                self.can_port,
                self._tx_guard.stalled_for(),
                self._tx_guard.last_value,
                reset_command(self.can_port),
            )
            self.tx_stall_pub.publish(String(data=self.can_port))

    def _check_can_link(self):
        """Reopen the SDK port after the interface was reset or re-created.

        piper_sdk's read thread exits on the ENETDOWN of a down/up, so
        feedback would stay frozen without a fresh socket.
        """
        if not self._link_watch.poll():
            return
        with self._command_lock:
            self.piper.DisconnectPort()
            self.piper.ConnectPort(can_init=True, piper_init=False)
            # The next command re-asserts MOVE J mode.
            self._last_motion_ctrl = 0.0
        rospy.logwarn("%s CAN link was reset; SDK port reopened", self.can_port)

    def PublishSlaveArmState(self):
        arm_status = PiperStatusMsg()
        arm_status.ctrl_mode = self.piper.GetArmStatus().arm_status.ctrl_mode
        arm_status.arm_status = self.piper.GetArmStatus().arm_status.arm_status
        arm_status.mode_feedback = self.piper.GetArmStatus().arm_status.mode_feed
        arm_status.teach_status = self.piper.GetArmStatus().arm_status.teach_status
        arm_status.motion_status = self.piper.GetArmStatus().arm_status.motion_status
        arm_status.trajectory_num = self.piper.GetArmStatus().arm_status.trajectory_num
        arm_status.err_code = self.piper.GetArmStatus().arm_status.err_code
        arm_status.joint_1_angle_limit = self.piper.GetArmStatus().arm_status.err_status.joint_1_angle_limit
        arm_status.joint_2_angle_limit = self.piper.GetArmStatus().arm_status.err_status.joint_2_angle_limit
        arm_status.joint_3_angle_limit = self.piper.GetArmStatus().arm_status.err_status.joint_3_angle_limit
        arm_status.joint_4_angle_limit = self.piper.GetArmStatus().arm_status.err_status.joint_4_angle_limit
        arm_status.joint_5_angle_limit = self.piper.GetArmStatus().arm_status.err_status.joint_5_angle_limit
        arm_status.joint_6_angle_limit = self.piper.GetArmStatus().arm_status.err_status.joint_6_angle_limit
        arm_status.communication_status_joint_1 = self.piper.GetArmStatus().arm_status.err_status.communication_status_joint_1
        arm_status.communication_status_joint_2 = self.piper.GetArmStatus().arm_status.err_status.communication_status_joint_2
        arm_status.communication_status_joint_3 = self.piper.GetArmStatus().arm_status.err_status.communication_status_joint_3
        arm_status.communication_status_joint_4 = self.piper.GetArmStatus().arm_status.err_status.communication_status_joint_4
        arm_status.communication_status_joint_5 = self.piper.GetArmStatus().arm_status.err_status.communication_status_joint_5
        arm_status.communication_status_joint_6 = self.piper.GetArmStatus().arm_status.err_status.communication_status_joint_6
        self.arm_status_pub.publish(arm_status)
    
    def PublishSlaveArmEndPose(self):
        # 末端位姿
        endpos = PoseStamped()
        endpos.pose.position.x = self.piper.GetArmEndPoseMsgs().end_pose.X_axis/1000000
        endpos.pose.position.y = self.piper.GetArmEndPoseMsgs().end_pose.Y_axis/1000000
        endpos.pose.position.z = self.piper.GetArmEndPoseMsgs().end_pose.Z_axis/1000000
        roll = self.piper.GetArmEndPoseMsgs().end_pose.RX_axis/1000
        pitch = self.piper.GetArmEndPoseMsgs().end_pose.RY_axis/1000
        yaw = self.piper.GetArmEndPoseMsgs().end_pose.RZ_axis/1000
        roll = math.radians(roll)
        pitch = math.radians(pitch)
        yaw = math.radians(yaw)
        quaternion = quaternion_from_euler(roll, pitch, yaw)
        endpos.pose.orientation.x = quaternion[0]
        endpos.pose.orientation.y = quaternion[1]
        endpos.pose.orientation.z = quaternion[2]
        endpos.pose.orientation.w = quaternion[3]
        # 为末端位姿增加时间戳
        endpos.header.stamp = rospy.Time.now()
        self.end_pose_pub.publish(endpos)
    
    def PublishSlaveArmJointAndGripper(self):
        # 从臂反馈消息
        joint_message = self.piper.GetArmJointMsgs()
        gripper_message = self.piper.GetArmGripperMsgs()
        if (
            float(joint_message.time_stamp) <= 0.0
            or float(gripper_message.time_stamp) <= 0.0
        ):
            rospy.logwarn_throttle(
                2.0,
                "Skipping %s follower publish until real joint/gripper feedback arrives",
                self.can_port,
            )
            return
        self.joint_state_slave.header.stamp = rospy.Time.now()
        joint_0:float = (joint_message.joint_state.joint_1/1000) * 0.017444
        joint_1:float = (joint_message.joint_state.joint_2/1000) * 0.017444
        joint_2:float = (joint_message.joint_state.joint_3/1000) * 0.017444
        joint_3:float = (joint_message.joint_state.joint_4/1000) * 0.017444
        joint_4:float = (joint_message.joint_state.joint_5/1000) * 0.017444
        joint_5:float = (joint_message.joint_state.joint_6/1000) * 0.017444
        joint_6:float = gripper_message.gripper_state.grippers_angle/1000000
        vel_0:float = self.piper.GetArmHighSpdInfoMsgs().motor_1.motor_speed/1000
        vel_1:float = self.piper.GetArmHighSpdInfoMsgs().motor_2.motor_speed/1000
        vel_2:float = self.piper.GetArmHighSpdInfoMsgs().motor_3.motor_speed/1000
        vel_3:float = self.piper.GetArmHighSpdInfoMsgs().motor_4.motor_speed/1000
        vel_4:float = self.piper.GetArmHighSpdInfoMsgs().motor_5.motor_speed/1000
        vel_5:float = self.piper.GetArmHighSpdInfoMsgs().motor_6.motor_speed/1000
        effort_6:float = self.piper.GetArmGripperMsgs().gripper_state.grippers_effort/1000
        self.joint_state_slave.position = [joint_0,joint_1, joint_2, joint_3, joint_4, joint_5,joint_6]  # Example values
        self.joint_state_slave.velocity = [vel_0, vel_1, vel_2, vel_3, vel_4, vel_5, 0.0]  # Example values
        self.joint_state_slave.effort[6] = effort_6
        self.joint_std_pub_puppet.publish(self.joint_state_slave)
    
    def SubPosThread(self):
        """机械臂末端位姿订阅
        
        """
        rospy.Subscriber('pos_cmd', PosCmd, self.pos_callback,queue_size=1, tcp_nodelay=True)
        rospy.spin()
    
    def SubJointThread(self):
        """机械臂关节订阅
        
        """
        rospy.Subscriber('/master/joint_states', JointState, self.joint_callback,queue_size=1, tcp_nodelay=True)
        rospy.spin()
    
    def SubEnableThread(self):
        """机械臂使能
        
        """
        rospy.Subscriber('/enable_flag', Bool, self.enable_callback,queue_size=1, tcp_nodelay=True)
        rospy.Subscriber(
            '/teleop/emergency_stop',
            Bool,
            self.emergency_stop_callback,
            queue_size=1,
            tcp_nodelay=True,
        )
        rospy.spin()

    def pos_callback(self, pos_data):
        """机械臂末端位姿订阅回调函数

        Args:
            pos_data (): 
        """
        factor = 180 / 3.1415926
        x = round(pos_data.x*1000) * 1000
        y = round(pos_data.y*1000) * 1000
        z = round(pos_data.z*1000) * 1000
        rx = round(pos_data.roll*1000*factor) 
        ry = round(pos_data.pitch*1000*factor)
        rz = round(pos_data.yaw*1000*factor)
        rospy.loginfo("Received PosCmd:")
        rospy.loginfo("x: %f", x)
        rospy.loginfo("y: %f", y)
        rospy.loginfo("z: %f", z)
        rospy.loginfo("roll: %f", rx)
        rospy.loginfo("pitch: %f", ry)
        rospy.loginfo("yaw: %f", rz)
        rospy.loginfo("gripper: %f", pos_data.gripper)
        rospy.loginfo("mode1: %d", pos_data.mode1)
        rospy.loginfo("mode2: %d", pos_data.mode2)
        if(self.GetEnableFlag() and not self._emergency_stop_active):
            self.piper.MotionCtrl_1(0x00, 0x00, 0x00)
            self.piper.MotionCtrl_2(0x01, 0x00, 50)
            self.piper.EndPoseCtrl(x, y, z, 
                                    rx, ry, rz)
            gripper = round(pos_data.gripper*1000*1000)
            if(pos_data.gripper>80000): gripper = 80000
            if(pos_data.gripper<0): gripper = 0
            if(self.girpper_exist):
                self.piper.GripperCtrl(abs(gripper), 1000, 0x01, 0)
            self.piper.MotionCtrl_2(0x01, 0x00, 50)
    
    def joint_callback(self, joint_data):
        """机械臂关节角回调函数

        Args:
            joint_data (): 
        """
        if self._shutting_down or rospy.is_shutdown():
            return
        if len(joint_data.position) < 7:
            rospy.logwarn_throttle(2.0, "Ignoring joint command with fewer than 7 positions")
            return
        factor = 57324.840764 #1000*180/3.14
        factor1 = 57.32484
        if self.log_joint_commands:
            rospy.loginfo_throttle(
                self.log_joint_commands_period,
                "Joint command: [%.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f]",
                joint_data.position[0],
                joint_data.position[1],
                joint_data.position[2],
                joint_data.position[3],
                joint_data.position[4],
                joint_data.position[5],
                joint_data.position[6],
            )
        joint_0 = round(joint_data.position[0]*factor)
        joint_1 = round(joint_data.position[1]*factor)
        joint_2 = round(joint_data.position[2]*factor)
        joint_3 = round(joint_data.position[3]*factor)
        joint_4 = round(joint_data.position[4]*factor)
        joint_5 = round(joint_data.position[5]*factor)
        joint_6 = round(joint_data.position[6]*1000*1000)
        if(joint_6>80000): joint_6 = 80000
        if(joint_6<0): joint_6 = 0
        if self.GetEnableFlag() and not self._emergency_stop_active:
            with self._command_lock:
                if (
                    self._shutting_down
                    or rospy.is_shutdown()
                    or not self.GetEnableFlag()
                    or self._emergency_stop_active
                ):
                    return
                now = time.monotonic()
                if now - self._last_motion_ctrl >= self.motion_ctrl_period:
                    self.piper.MotionCtrl_2(0x01, 0x01, 100)
                    self._last_motion_ctrl = now
                self.piper.JointCtrl(joint_0, joint_1, joint_2,
                                     joint_3, joint_4, joint_5)
                self.piper.GripperCtrl(abs(joint_6), 1000, 0x01, 0)
                self._tx_guard.note_command_sent()
    
    def enable_callback(self, enable_flag:Bool):
        """机械臂使能回调函数

        Args:
            enable_flag (): 
        """
        rospy.loginfo_throttle(2.0, "Received enable flag: %s", enable_flag.data)
        self._enable_requested = bool(enable_flag.data)
        # Re-assert MOVE J mode with the first command after any enable change.
        self._last_motion_ctrl = 0.0
        if(enable_flag.data):
            if self._emergency_stop_active:
                rospy.logwarn(
                    "Piper enable deferred while QUICK EMERGENCY STOP is active"
                )
                return
            if self.GetEnableFlag():
                return
            if not self._enable_and_wait():
                rospy.logerr("Manual Piper enable failed; control remains inhibited")
        else:
            if not self._disable_and_wait():
                rospy.logerr("Manual Piper disable was not confirmed")

    def emergency_stop_callback(self, stop_flag: Bool):
        """Apply SDK quick E-stop; recovery re-enables before HOME commands."""
        requested = bool(stop_flag.data)
        if requested:
            if self._emergency_stop_active:
                return
            self._emergency_stop_active = True
            self.__enable_flag = False
            with self._command_lock:
                if not self._shutting_down and not rospy.is_shutdown():
                    self.piper.MotionCtrl_1(0x01, 0x00, 0x00)
            rospy.logerr("Piper QUICK EMERGENCY STOP active on %s", self.can_port)
            return
        if not self._emergency_stop_active:
            return
        with self._command_lock:
            if self._shutting_down or rospy.is_shutdown():
                return
            self.piper.MotionCtrl_1(0x02, 0x00, 0x00)
        self._emergency_stop_active = False
        rospy.logwarn(
            "Piper emergency stop reset on %s; measured-pose enable follows",
            self.can_port,
        )
        if self._enable_requested and not self._enable_and_wait():
            rospy.logerr(
                "Piper re-enable after emergency stop failed on %s", self.can_port
            )

if __name__ == '__main__':
    try:
        piper_ms = C_PiperRosNode()
        piper_ms.Pubilsh()
    except rospy.ROSInterruptException:
        pass
