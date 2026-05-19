#!/usr/bin/env python3
"""
数据集录制脚本：同步采集相机帧 + 机械臂位姿 + 时间戳
采集后按时间戳对齐，保存为图像 + JSON 元数据
"""

import cv2
import threading
import time
import json
import os
import numpy as np
from datetime import datetime

# ======== Sensor Config =========
ROBOT_IP = "192.168.1.20"
ROBOT_PORT = 30004
CAMERA_ID = 0

# ======== Recording Config =========
OUTPUT_DIR = "dataset/save_data3"
SYNC_TOLERANCE_NS = 20_000_000  # 20ms 时间差容忍
SAVE_INVALID = True  # True=保存所有帧(含时间差大的), False=只保存同步帧
MAX_FRAMES = 5000    # 最大录制帧数，-1=无限

# ======== 全局变量 =========
latest_frame = None
latest_frame_ts = 0
latest_pose = None
latest_pose_ts = 0
frame_lock = threading.Lock()
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

def pose_to_list(pose):
    """4x4 矩阵转 16 元素列表，便于 JSON 序列化"""
    return pose.flatten().tolist()

def euler_from_pose(pose):
    """从 4x4 位姿提取 euler (rad) + t (mm)"""
    from scipy.spatial.transform import Rotation
    R = pose[:3, :3]
    t = pose[:3, 3]
    euler = Rotation.from_matrix(R).as_euler('xyz')
    return euler.tolist(), t.tolist()

# ===================== 数据采集线程 =====================
def read_camera():
    global latest_frame, latest_frame_ts
    cap = cv2.VideoCapture(CAMERA_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1440)
    cap.set(cv2.CAP_PROP_BRIGHTNESS, 128)

    while recording:
        ret, frame = cap.read()
        if ret:
            with frame_lock:
                latest_frame = frame.copy()
                latest_frame_ts = time.perf_counter_ns()
        
        time.sleep(0.05)  # 20Hz 采样

    cap.release()

def read_robot():
    global latest_pose, latest_pose_ts
    global robot_rpc_client, robot_name
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
            # 捕获 RPC 超时等异常，打印并短暂退避，线程不退出
            print(f"[Robot] read error: {e}")
            time.sleep(0.01)
            continue
        time.sleep(0.02)  # 50Hz 采样

def get_synced_pair():
    """返回 (frame, pose, frame_ts_ns, pose_ts_ns, time_diff_ns) 五元组，
       任意 frame/pose 为 None 则同步失败"""
    with frame_lock:
        frame = latest_frame
        frame_ts = latest_frame_ts
    with pose_lock:
        pose = latest_pose
        pose_ts = latest_pose_ts

    if frame is None or pose is None:
        return None, None, None, None, -1

    return frame, pose, frame_ts, pose_ts, abs(frame_ts - pose_ts)

# ===================== 主程序 =====================
def main():
    global recording, robot_rpc_client, robot_name, frame_count

    import pyaubo_sdk

    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = os.path.join(OUTPUT_DIR, timestamp)
    img_dir = os.path.join(session_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
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
    cam_thread = threading.Thread(target=read_camera, daemon=True)
    robot_thread = threading.Thread(target=read_robot, daemon=True)
    cam_thread.start()
    robot_thread.start()
    time.sleep(0.5)  # 等待线程初始化

    # 录制循环
    metadata_list = []
    sync_error_count = 0
    synced_count = 0
    last_print_time = time.time()

    print("[Recording] Press 'q' to quit, 'p' to pause/resume")
    paused = False

    try:
        while recording:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                paused = not paused
                print(f"[Recording] {'Paused' if paused else 'Resumed'}")

            if paused:
                time.sleep(0.1)
                continue

            frame, pose, frame_ts, pose_ts, diff_ns = get_synced_pair()
            if frame is None:
                time.sleep(0.01)
                continue

            # 同步判定
            synced = diff_ns <= SYNC_TOLERANCE_NS

            if synced:
                synced_count += 1
            elif not SAVE_INVALID:
                sync_error_count += 1
                continue

            # 保存图像
            img_path = os.path.join(img_dir, f"frame_{frame_count:06d}.jpg")
            cv2.imwrite(img_path, frame)

            # 保存元数据
            euler, tvec = euler_from_pose(pose)
            meta = {
                "frame_id": frame_count,
                "image_path": f"images/frame_{frame_count:06d}.jpg",
                "camera_timestamp_ns": int(frame_ts) if frame_ts is not None else 0,
                "pose_timestamp_ns": int(pose_ts) if pose_ts is not None else 0,
                "time_diff_ns": int(diff_ns),
                "synced": bool(synced),
                "pose_euler_xyz": euler,
                "pose_translation_mm": tvec,
                "pose_matrix_4x4": pose_to_list(pose)
            }
            metadata_list.append(meta)

            frame_count += 1

            # 定期打印状态
            if time.time() - last_print_time >= 2.0:
                sync_rate = (synced_count / frame_count * 100) if frame_count > 0 else 0
                print(f"[Status] frames={frame_count}, synced={sync_rate:.1f}%, diff={diff_ns/1e6:.2f}ms")
                last_print_time = time.time()

            # 限制最大帧数
            if MAX_FRAMES > 0 and frame_count >= MAX_FRAMES:
                print(f"[Recording] Reached max frames ({MAX_FRAMES})")
                break
    except KeyboardInterrupt:
        print("[Recording] Interrupted by user")
    finally:
        # 停止采集线程
        recording = False
        cam_thread.join(timeout=2)
        robot_thread.join(timeout=2)

        # 保存元数据 JSON
        meta_path = os.path.join(session_dir, "metadata.json")
        with open(meta_path, 'w') as f:
            json.dump({
                "session": timestamp,
                "total_frames": frame_count,
                "sync_tolerance_ns": SYNC_TOLERANCE_NS,
                "camera_setting": {
                    "width": 2560, "height": 1440, "brightness": 128
                },
                "records": metadata_list
            }, f, indent=2)

        print(f"[Done] Saved {frame_count} frames to {session_dir}")
        print(f"[Done] Metadata: {meta_path}")

if __name__ == '__main__':
    main()