#!/usr/bin/env python3
"""
机械臂位姿录制脚本：只保存 robot pose，保存频率 20Hz
"""

import threading
import time
import json
import os
import numpy as np
from datetime import datetime

# ======== Robot Config ========
ROBOT_IP = "192.168.1.20"
ROBOT_PORT = 30004

# ======== Recording Config ========
OUTPUT_DIR = "dataset/pose_data"
SAVE_FREQ_HZ = 20
MAX_FRAMES = -1  # -1=无限

# ======== 全局变量 ========
latest_pose = None
latest_pose_ts = 0
pose_lock = threading.Lock()
recording = True
frame_count = 0

# robot params
robot_rpc_client = None
robot_name = None

# ===================== 工具函数 =====================
def get_robot_pose(robot_rpc_client, robot_name):
    """从 RPC 获取机器人 TCP 位姿，转为 4x4 矩阵 (mm)"""
    from scipy.spatial.transform import Rotation
    tcp_pose = robot_rpc_client.getRobotInterface(robot_name).getRobotState().getTcpPose()
    r = Rotation.from_euler('xyz', tcp_pose[3:])
    t = np.array(tcp_pose[:3]).reshape((3, 1))
    pose = np.eye(4)
    pose[:3, :3] = r.as_matrix()
    pose[:3, 3] = t.flatten() * 1000  # mm
    return pose, tcp_pose

def euler_from_pose(pose):
    """从 4x4 位姿提取 euler (rad) + t (mm)"""
    from scipy.spatial.transform import Rotation
    R = pose[:3, :3]
    t = pose[:3, 3]
    euler = Rotation.from_matrix(R).as_euler('xyz')
    return euler.tolist(), t.tolist()

# ===================== 数据采集线程 =====================
def read_robot():
    global latest_pose, latest_pose_ts
    global robot_rpc_client, robot_name
    interval = 1.0 / SAVE_FREQ_HZ
    while recording:
        if robot_rpc_client is None or robot_name is None:
            time.sleep(0.01)
            continue
        try:
            pose, raw_tcp = get_robot_pose(robot_rpc_client, robot_name)
            with pose_lock:
                latest_pose = pose
                latest_pose_ts = time.perf_counter_ns()
        except Exception as e:
            print(f"[Robot] read error: {e}")
            time.sleep(0.01)
            continue
        time.sleep(interval)

# ===================== 主程序 =====================
def main():
    global recording, robot_rpc_client, robot_name, frame_count

    import pyaubo_sdk

    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = os.path.join(OUTPUT_DIR, timestamp)
    os.makedirs(session_dir, exist_ok=True)
    print(f"[Recording] Output: {session_dir}")

    # 连接机械臂
    robot_rpc_client = pyaubo_sdk.RpcClient()
    robot_rpc_client.connect(ROBOT_IP, ROBOT_PORT)
    if not robot_rpc_client.hasConnected():
        print("[Error] Cannot connect to robot")
        return
    robot_rpc_client.login("aubo", "123456")
    if not robot_rpc_client.hasLogined():
        print("[Error] Cannot login to robot")
        return
    robot_name = robot_rpc_client.getRobotNames()[0]
    print(f"[Robot] Connected: {robot_name}")

    # 启动采集线程
    robot_thread = threading.Thread(target=read_robot, daemon=True)
    robot_thread.start()
    time.sleep(0.5)

    # 录制循环
    pose_records = []
    interval = 1.0 / SAVE_FREQ_HZ

    print(f"[Recording] Saving pose at {SAVE_FREQ_HZ}Hz, press Ctrl+C to quit")

    try:
        while recording:
            with pose_lock:
                pose = latest_pose
                pose_ts = latest_pose_ts

            if pose is not None:
                euler, tvec = euler_from_pose(pose)
                record = {
                    "frame_id": frame_count,
                    "pose_timestamp_ns": int(pose_ts),
                    "pose_euler_xyz": euler,
                    "pose_translation_mm": tvec,
                    "pose_matrix_4x4": pose.flatten().tolist()
                }
                pose_records.append(record)
                frame_count += 1

                if frame_count % 100 == 0:
                    print(f"[Status] recorded={frame_count}")

            if MAX_FRAMES > 0 and frame_count >= MAX_FRAMES:
                print(f"[Recording] Reached max frames ({MAX_FRAMES})")
                break

            time.sleep(interval)
    except KeyboardInterrupt:
        print("[Recording] Interrupted by user")
    finally:
        recording = False
        robot_thread.join(timeout=2)

        # 保存 JSON
        meta_path = os.path.join(session_dir, "pose_data.json")
        with open(meta_path, 'w') as f:
            json.dump({
                "session": timestamp,
                "total_frames": frame_count,
                "save_freq_hz": SAVE_FREQ_HZ,
                "records": pose_records
            }, f, indent=2)

        print(f"[Done] Saved {frame_count} poses to {meta_path}")

if __name__ == '__main__':
    main()
