import pyaubo_sdk
import time
import math
import numpy as np

TaskFrameType = pyaubo_sdk.TaskFrameType
RobotModeType = pyaubo_sdk.RobotModeType

def print_red(text):
    print(f"\033[31m{text}\033[0m")

def print_green(text):
    print(f"\033[32m{text}\033[0m")

def print_yellow(text):
    print(f"\033[33m{text}\033[0m")

def wait_arrival(robot_interface):
    max_retry_count = 5
    cnt = 0

    # 接口调用: 获取当前的运动指令 ID
    exec_id = robot_interface.getMotionControl().getExecId()

    # 等待机械臂开始运动
    while exec_id == -1:
        if cnt > max_retry_count:
            return -1
        time.sleep(0.05)
        cnt += 1
        exec_id = robot_interface.getMotionControl().getExecId()

    # 等待机械臂运动完成
    while robot_interface.getMotionControl().getExecId() != -1:
        time.sleep(0.02)

    return 0

# 机械臂上电
def start_up(robot_interface):
    if 0 == robot_interface.getRobotManage().poweron():  # 接口调用: 发起机器人上电请求
        print("[aubo] The robot is requesting power-on!")
        if 0 == robot_interface.getRobotManage().startup():  # 接口调用: 发起机器人启动请求
            print("[aubo] The robot is requesting startup!")
            # 循环直至机械臂松刹车成功
            while 1:
                robot_mode = robot_interface \
                    .getRobotState().getRobotModeType()  # 接口调用: 获取机器人的模式类型
                print("[aubo] Robot current mode: %s" % (robot_mode.name))
                if robot_mode == RobotModeType.Running:
                    break
                time.sleep(1)


class AuboArm:
    def __init__(self,
        robot_ip,
        robot_port=30004
        ):
        self.robot_ip = robot_ip
        self.robot_port = robot_port
        # 注意：需要保持RpcClient类实例
        self.robot_rpc_client = pyaubo_sdk.RpcClient()
        self.connect()

        robot_name = self.robot_rpc_client.getRobotNames()[0]
        self.interface = self.robot_rpc_client.getRobotInterface(robot_name)

        start_up(self.interface)  # 上电、启动
        self.set_global_speed(0.6)

    def __del__(self):
        self.disconnect()

    def connect(self):
        self.robot_rpc_client.setRequestTimeout(1000)  # 设置 RPC 超时，ms
        self.robot_rpc_client.connect(self.robot_ip, self.robot_port)  # 连接 RPC 服务
        if not self.robot_rpc_client.hasConnected():
            raise ConnectionError("Failed to connect to RPC service")
        print_yellow("[aubo] RPC客户端连接成功!")
        self.robot_rpc_client.login("admin", "1")  # 登录
        if not self.robot_rpc_client.hasLogined():
            raise RuntimeError("Failed to login to RPC service")
        print_yellow("[aubo] RPC客户端登录成功!")

    def disconnect(self):
        if self.robot_rpc_client.hasConnected():
            self.robot_rpc_client.logout()  # 退出登录
            self.robot_rpc_client.disconnect()  # 断开连接

    def set_global_speed(self, speed: float=0.5):
        """ 设置全局运动速度比例 0~1.0 """
        self.interface.getMotionControl().setSpeedFraction(speed)

    def set_tcp_offset(self, pose:list):
        """ 设置TCP偏移 """
        return self.interface.getRobotConfig().setTcpOffset(pose)   

    def get_tcp_offset(self):
        """ 获取TCP偏移"""
        return self.interface.getRobotConfig().getTcpOffset()

    def move_joint(self, joint_pos, joint_acc = 60, joint_vel = 40, wait = True, is_rad = True):
        """ 执行关节运动，返回是否成功 """
        if not is_rad:
            joint_pos = [math.radians(joint_pos[0]), math.radians(joint_pos[1]), math.radians(joint_pos[2]),
                        math.radians(joint_pos[3]), math.radians(joint_pos[4]), math.radians(joint_pos[5])] 

        flag = self.interface.getMotionControl(). \
                        moveJoint(joint_pos, joint_acc * (math.pi / 180), joint_vel * (math.pi / 180), 0, 0)
        
        if wait:
            wait_arrival(self.interface)

        return (flag == 0)
            
    def get_flange_pose(self):
        '''获取法兰盘位姿'''
        return self.interface.getRobotState().getToolPose()

    def get_tcp_pose(self):
        '''获取TCP位姿'''
        return self.interface.getRobotState().getTcpPose()
    
    def get_tcp_vel(self):
        '''获取TCP速度'''
        return self.interface.getRobotState().getTcpSpeed()
    
    def get_joint_pos(self):
        '''获取关节位置'''
        return self.interface.getRobotState().getJointPositions()    
    
    def get_joint_vel(self):
        '''获取关节速度'''
        return self.interface.getRobotState().getJointSpeeds()

    def get_joint_cur(self):
        '''获取关节电流'''
        return self.interface.getRobotState().getJointCurrents()

    def get_joint_max_vel(self):
        """ 获取关节最大速度 """
        return self.interface.getRobotConfig().getJointMaxSpeeds()

    def get_joint_max_acc(self):
        """ 获取关节最大加速度 """
        return self.interface.getRobotConfig().getJointMaxAccelerations()

    def move_line(self, line_pos, line_acc = 1, line_vel = 0.5, wait = True):
        """ 执行笛卡尔运动，返回是否成功 """
        flag = self.interface.getMotionControl(). \
            moveLine(line_pos, line_acc, line_vel, 0, 0)
        if wait:
            wait_arrival(self.interface)

        return (flag == 0)

    def set_servo_pose(self, pose, cycle_t):
        """发送伺服指令"""      
        a = 0.8
        v = 0.25
        t = 1.0 * cycle_t
        lookahead_time = 0.2
        gain = 200
        flag = self.interface.getMotionControl().servoCartesian(pose,a,v,t,lookahead_time,gain)
        return flag

    def set_servo_joint(self, joints, cycle_t):
        """发送伺服指令"""      
        a = 0.8
        v = 0.25
        t = 1.0 * cycle_t
        lookahead_time = 0.2
        gain = 200
        flag = self.interface.getMotionControl().servoJoint(joints,a,v,t,lookahead_time,gain)
        return flag

    def set_servo_mode(self, mode=1):
        """设置伺服模式"""
        #flag = self.interface.getMotionControl().setServoMode(True)
        flag = self.interface.getMotionControl().setServoModeSelect(mode)  #0-退出伺服模式 1-(截断式)规划伺服模式 2-透传模式(直接下发) 3-透传模式(缓存) 4-1ms透传模式(缓存) 5-规划伺服模式
        return flag

    def get_servo_mode(self):
        """获取当前伺服模式"""
        mode = self.interface.getMotionControl().getServoModeSelect()
        return mode
    
    def stop_move(self):
        """停止运动"""
        flag = self.interface.getMotionControl().stopMove(True,True)
        return flag

    def move_spiral(self, param, blend_radius=0, v=0.002, a=0.1, t=0):
        """螺旋运动"""
        self.interface.getMotionControl().moveSpiral(param, blend_radius, v, a, t)
    
    def move_spline(self, q, a=1.4, v=1, duration=0):
        """样条插补运动"""
        self.interface.getMotionControl().moveSpline(q, a, v, duration)
    
    def wait_spline_finished(self):
        """等待样条插补运动完成"""
        # 等待样条运动开始
        while self.interface.getMotionControl().getExecId() == -1 :
            time.sleep(0.01)
        
        print("[aubo] 样条运动开始")

        while True:
            if self.interface.getMotionControl().getExecId() == -1 :
                break
            time.sleep(0.05)

        print("[aubo] 样条运动结束")
    
    def inverse_kinematics(self, qnear, pose, tcp_offset):
        """逆运动学求解"""
        return self.interface.getRobotAlgorithm().inverseKinematics1(qnear, pose, tcp_offset)


#=======================================================================================================================================================
# 力控模式相关
    def set_payload(self, mass:float, cog:list, aom:list, inertia:list):
        """ 设置负载 """
        return self.interface.getRobotConfig().setPayload(mass, cog, aom, inertia) 

    def has_force_sensor(self):
        ''' 是否安装了力传感器 '''
        return self.interface.getRobotConfig().hasBaseForceSensor()#hasTcpForceSensor()

    def set_force_sensor(self, sensor: str):
        """ 设置tcp力传感器类型 """
        return self.interface.getRobotConfig().selectTcpForceSensor(sensor)
    
    def set_force_sensor_pose(self, pose:list):
        """ 设置tcp力传感器位置 """
        return self.interface.getRobotConfig().setTcpForceSensorPose(pose)

    def get_force_sensor(self):
        ''' 获取可用力传感器 '''
        return self.interface.getRobotConfig().getTcpForceSensorNames()
    
    def get_payload(self):
        """ 获取负载信息 """
        return self.interface.getRobotConfig().getPayload()

    def fc_enalbe(self) -> bool:
        """ 尝试启用力控模式 """
        if self.interface.getForceControl().isFcEnabled():
            print_yellow(f"[aubo] The robot {self.robot_ip} has already been in force control mode.")
            return True
        else:
            self.interface.getForceControl().fcEnable()
            # 等待启动力控模式，有限次等待
            cnt = 1
            max_retry_count = 5
            while not self.interface.getForceControl().isFcEnabled():
                if cnt > max_retry_count:
                    print_red(f"[aubo] The robot {self.robot_ip} Enable force control mode failed, try {cnt} times")
                    return False
                time.sleep(0.04)
                cnt += 1
            print_green(f"[aubo] The robot {self.robot_ip} Enable force control mode success, try {cnt} times")
            return True
        
    def fc_disable(self) -> bool:
        """ 尝试关闭力控模式 """
        if not self.interface.getForceControl().isFcEnabled():
            print_yellow(f"[aubo] The robot {self.robot_ip} has already quit force control mode.")
            return True
        else:
            self.interface.getForceControl().fcDisable()
            # 等待关闭力控模式，有限次等待
            cnt = 1
            max_retry_count = 5
            while self.interface.getForceControl().isFcEnabled():
                if cnt > max_retry_count:
                    print_red(f"[aubo] Disable force control mode failed, try {cnt} times")
                    return False
                time.sleep(0.04)
                cnt += 1
            print_green(f"[aubo] Disable force control mode success, try {cnt} times")
            return True

    def is_fc_enabled(self) -> bool:
        """ 力控模式是否被启用 """
        return self.interface.getForceControl().isFcEnabled()

    def get_tcp_force(self):
        """ 获取 TCP 力矩，矫零后的 """
        return self.interface.getRobotState().getTcpForce()
    
    def get_sensor_force(self):
        """ 获取 传感器 力矩，也即原始信息 """
        return self.interface.getRobotState().getTcpForceSensors()

    def set_dynamic_model(self, m, d, k):
        """ 设置力控参数 """
        return self.interface.getForceControl().setDynamicModel(m, d, k)

    def set_target_force(self, feature, compliance, target_wrench, speed_limits, 
                            type = TaskFrameType.TOOL_FORCE) -> bool:
        """ 设置目标力矩, 返回成功与否 """
        flag = self.interface.getForceControl().setTargetForce(feature, compliance, target_wrench, speed_limits, type)
        return (flag == 0)

    def get_tcp_target_force(self):
        """ 获取 TCP 目标力矩 """
        return self.interface.getRobotState().getTcpTargetForce()

    def get_sensor_status(self, name):
        ''' 获取传感器状态 '''
        return self.interface.getRobotState().getTcpForceSensorStatus(name)
    
    def get_tcp_force_sensor_pose(self):
        ''' 获取传感器位置偏移 '''
        return self.interface.getRobotConfig().getTcpForceSensorPose()
    
    def get_dynamic_model(self):
        """ 获取力控参数 """
        return self.interface.getForceControl().getDynamicModel()
    
    def set_lp_filter(self, frequency):
        """ 设置低通滤波器 """
        ret = self.interface.getForceControl().setLpFilter(frequency)
        if ret==0:
            print_green(f"[aubo] Set low pass filter frequency to {frequency} successfully.")
        else:
            print_red(f"[aubo] Set low pass filter frequency to {frequency} failed.")
    
    def reset_lp_filter(self):
        """ 重置低通滤波器 """
        ret = self.interface.getForceControl().resetLpFilter()
        if ret==0:
            print_green(f"[aubo] Reset low pass filter frequency successfully.")
        else:
            print_red(f"[aubo] Reset low pass filter frequency failed.")

    def set_sensor_limits(self, limits):
        """ 设置最大力限制 """
        self.interface.getForceControl().fcSetSensorLimits(limits)

    def get_sensor_limits(self):
        """ 获取最大力限制  """
        return self.interface.getForceControl().getFcSensorLimits()
    
    def set_sensor_thresholds(self, thresholds):
        """ 设置力控阈值 """
        self.interface.getForceControl().fcSetSensorThresholds(thresholds)

    def get_sensor_thresholds(self):
        """ 获取力控阈值  """
        return self.interface.getForceControl().getFcSensorThresholds()

    def set_tcp_force_offset(self, offset):
        """ 设置末端力矩偏移 """
        return self.interface.getRobotConfig().setTcpForceOffset(offset)

    def get_tcp_force_offset(self):
        """ 获取末端力矩偏移 """
        return self.interface.getRobotConfig().getTcpForceOffset()

    def calibrate_force_sensor(self, force, pose, mass, cog):
        """ 标定力传 """
        return self.interface.getRobotAlgorithm().calibrateTcpForceSensor3(force, pose, mass, cog)
