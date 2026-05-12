#!/usr/bin/env python3
"""
位姿轨迹可视化脚本
自动识别 pose_data.json 和 metadata.json 格式
"""

import json
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def load_json(path):
    """加载并自动识别格式"""
    with open(path) as f:
        data = json.load(f)
    # pose_data.json: {"records": [...]} 或 metadata.json: {"records": [...]}
    if "records" in data:
        return data["records"], data.get("save_freq_hz", 0), data.get("session", os.path.dirname(path))
    raise ValueError(f"Unknown JSON format in {path}")

def extract_positions(records):
    """从记录中提取 translation_mm"""
    positions = []
    for r in records:
        if "pose_translation_mm" in r:
            positions.append(r["pose_translation_mm"])
        elif "pose_matrix_4x4" in r:
            m = r["pose_matrix_4x4"]
            positions.append([m[12], m[13], m[14]])  # 4x4 column-major
    return np.array(positions)

def extract_eulers(records):
    """从记录中提取 euler xyz"""
    eulers = []
    for r in records:
        if "pose_euler_xyz" in r:
            eulers.append(r["pose_euler_xyz"])
    return np.array(eulers) if eulers else None

def plot_trajectory(positions, session, freq):
    fig = plt.figure(figsize=(14, 5))

    # 3D 轨迹
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot(positions[:,0], positions[:,1], positions[:,2], linewidth=0.5)
    ax1.set_xlabel('X (mm)')
    ax1.set_ylabel('Y (mm)')
    ax1.set_zlabel('Z (mm)')
    ax1.set_title(f'3D Trajectory\n{session}')
    ax1.set_aspect('equal')

    # XY 平面
    ax2 = fig.add_subplot(132)
    ax2.plot(positions[:,0], positions[:,1], linewidth=0.5)
    ax2.set_xlabel('X (mm)')
    ax2.set_ylabel('Y (mm)')
    ax2.set_title('XY Plane')
    ax2.set_aspect('equal')
    ax2.grid(True)

    # XZ 平面
    ax3 = fig.add_subplot(133)
    ax3.plot(positions[:,0], positions[:,2], linewidth=0.5)
    ax3.set_xlabel('X (mm)')
    ax3.set_ylabel('Z (mm)')
    ax3.set_title('XZ Plane')
    ax3.set_aspect('equal')
    ax3.grid(True)

    fig.suptitle(f'Trajectory ({len(positions)} points @ {freq}Hz)', fontsize=12)
    plt.tight_layout()
    plt.show()

def plot_euler(eulers, session):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    labels = ['Rx', 'Ry', 'Rz']
    colors = ['r', 'g', 'b']
    for i, (ax, label, color) in enumerate(zip(axes, labels, colors)):
        ax.plot(eulers[:, i], color=color, linewidth=0.5)
        ax.set_xlabel('Frame')
        ax.set_ylabel('rad')
        ax.set_title(f'Euler {label}')
        ax.grid(True)
    fig.suptitle(f'Euler Angles - {session}', fontsize=12)
    plt.tight_layout()
    plt.show()

def plot_xyz_vs_frame(positions, session):
    """Plot X, Y, Z position components vs frame index."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    labels = ['X', 'Y', 'Z']
    colors = ['r', 'g', 'b']
    frames = np.arange(len(positions))

    for i, (ax, label, color) in enumerate(zip(axes, labels, colors)):
        ax.plot(frames, positions[:, i], color=color, linewidth=0.5)
        ax.set_ylabel(f'{label} (mm)')
        ax.set_title(f'Position {label} vs Frame')
        ax.grid(True)

    axes[-1].set_xlabel('Frame')
    fig.suptitle(f'Position Components vs Frame - {session}', fontsize=12)
    plt.tight_layout()
    plt.show()

def main():
    if len(sys.argv) < 2:
        print("Usage: python visualize_pose.py <pose_data.json or metadata.json>")
        return

    json_path = sys.argv[1]
    records, freq, session = load_json(json_path)
    positions = extract_positions(records)

    print(f"[Info] Loaded {len(records)} records from {session}")
    print(f"[Info] Position range: X[{positions[:,0].min():.1f}, {positions[:,0].max():.1f}], "
          f"Y[{positions[:,1].min():.1f}, {positions[:,1].max():.1f}], "
          f"Z[{positions[:,2].min():.1f}, {positions[:,2].max():.1f}]")

    plot_trajectory(positions, session, freq)

    # X, Y, Z vs Frame
    plot_xyz_vs_frame(positions, session)

    eulers = extract_eulers(records)
    if eulers is not None:
        plot_euler(eulers, session)

if __name__ == '__main__':
    main()
