#!/usr/bin/env python3
"""
Simple PnP pose estimation without Kalman filter.
Records cMo, bMo per frame and plots Euler angles vs frame.
"""

import cv2 as cv
cv.setNumThreads(0)
from ultralytics import YOLO
import os
import numpy as np
from scipy.spatial.transform import Rotation
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from gemiEd import *
from static_pose_optimizer import StaticPoseOptimizer

model = YOLO("checkpoint/best.pt")

# ============ Config ============
# DATA_DIR = "dataset/save_data3/20260511_120244"
# RESULT_DIR = "result/save_data3/20260511_120244/pnp_only"
# BEGIN_FRAME_ID = 1180
# PROCESSED_INTERVAL = 1.0 # seconds, minimum time gap between processed frames

DATA_DIR = "dataset/multi_static/combined"
RESULT_DIR = "result/multi_static/combined/pnp_only"
BEGIN_FRAME_ID = -1
PROCESSED_INTERVAL = 0.0
MAX_FRAMES = -1


# Frame rejection
FIXED_ERROR_THRESHOLD = 0.8
FRAME_REJECT_THRESHOLD = 1.0
ADAPTIVE_MULTIPLIER = 2.0

# Camera on robot end-effector (eye-to-hand extrinsic)
eMc = np.array([
    [-7.2267956e-01,  6.9102561e-01, -1.4759262e-02, -5.1758522e+01],
    [-6.9116789e-01, -7.2264087e-01,  8.7790741e-03,  6.0040222e+01],
    [-4.5990809e-03,  1.6545586e-02,  9.9985254e-01,  9.7955963e+01],
    [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  1.0000000e+00]
], dtype=np.float64)

# Camera intrinsics
K = np.array([
    [2674.7629874104787, 0., 1279.5],
    [0., 2674.7629874104787, 719.5],
    [0., 0., 1.]
], dtype=np.float64)

dist = np.array([-0.11744968686298927, 0.27089153364253454, 0.0012180578884344092,
                 0.00067320963008635703, -0.078845410108757258], dtype=np.float64)

# 3D object points
obj_pts = np.array([
    [-8.0, 11.2, 0.0], [8.0, 11.2, 0.0],
    [-16.0, 0.0, 0.0], [0.0, 0.0, 0.0], [16.0, 0.0, 0.0],
    [-8.0, -13.9, 0.0], [8.0, -13.9, 0.0]
], dtype=np.float64)


def solvePnP_IPPE(pts2d, pts3d, K, dist):
    success, rvec, tvec = cv.solvePnP(pts3d, pts2d, K, dist, flags=cv.SOLVEPNP_IPPE)
    if not success:
        return None, None, False
    return rvec.flatten(), tvec.flatten(), True


def compute_per_point_errors(pts3d, rvec, tvec, pts2d, K, dist):
    proj, _ = cv.projectPoints(pts3d, rvec, tvec, K, dist)
    return np.linalg.norm(proj.reshape(-1, 2) - pts2d, axis=1)


def two_round_pnp(pts2d, pts3d, K, dist):
    rvec1, tvec1, valid1 = solvePnP_IPPE(pts2d, pts3d, K, dist)
    if not valid1:
        return None, None, False, None, None, None

    per_point_errors = compute_per_point_errors(pts3d, rvec1, tvec1, pts2d, K, dist)
    threshold = FIXED_ERROR_THRESHOLD

    inlier_mask = per_point_errors < threshold
    n_inliers = inlier_mask.sum()

    if n_inliers < 4:
        threshold = threshold * 2
        inlier_mask = per_point_errors < threshold
        n_inliers = inlier_mask.sum()

    if n_inliers >= 4:
        pts3d_inlier = pts3d[inlier_mask]
        pts2d_inlier = pts2d[inlier_mask]
        if pts3d_inlier.shape[0] >= 4 and pts2d_inlier.shape[0] >= 4:
            rvec2, tvec2, valid2 = solvePnP_IPPE(pts2d_inlier, pts3d_inlier, K, dist)
            if valid2:
                per_point_errors = compute_per_point_errors(pts3d, rvec2, tvec2, pts2d, K, dist)
                return rvec2, tvec2, True, inlier_mask, per_point_errors, threshold
        return rvec1, tvec1, True, inlier_mask, per_point_errors, threshold
    else:
        return rvec1, tvec1, True, inlier_mask, per_point_errors, threshold


def pose_to_euler_tvec(pose, unit='deg'):
    if pose is None:
        return None, None
    euler = Rotation.from_matrix(pose[:3, :3]).as_euler('xyz')
    tvec = pose[:3, 3]
    if unit == 'deg':
        euler = np.degrees(euler)
    return euler, tvec


def getInferResult(model, img):
    results = model(img)
    if len(results) == 0:
        return np.array([])
    return results[0].boxes.xyxy.cpu().numpy()


if __name__ == '__main__':
    os.makedirs(RESULT_DIR, exist_ok=True)

    # Load metadata
    meta_path = os.path.join(DATA_DIR, "metadata.json")
    with open(meta_path, 'r') as f:
        metadata = json.load(f)

    records = metadata['records']
    print(f"Loaded {len(records)} frames from {meta_path}")

    # Storage
    frame_ids = []
    cMo_list = []
    bMo_list = []
    bMe_list = []
    pnp_errors = []

    frame_id = 0
    last_timestamp_ns = None

    for record in records:
        frame_id_val = record['frame_id']
        if frame_id_val < BEGIN_FRAME_ID:
            continue

        current_timestamp_ns = record['camera_timestamp_ns']
        if last_timestamp_ns is not None:
            time_diff_s = (current_timestamp_ns - last_timestamp_ns) / 1e9
            if time_diff_s < PROCESSED_INTERVAL:
                continue
            # if time_diff_s > 2.0:
            #     print(f"[WARN] Large timestamp gap: {time_diff_s:.2f}s")
        last_timestamp_ns = current_timestamp_ns

        if MAX_FRAMES > 0 and frame_id >= MAX_FRAMES:
            print(f"\nReached max frames limit ({MAX_FRAMES})")
            break

        img_relative_path = record['image_path']
        img_path = os.path.join(DATA_DIR, img_relative_path)
        robot_pose = np.array(record['pose_matrix_4x4']).reshape(4, 4)

        print(f"\nProcessing frame {frame_id}: frame_{frame_id_val:06d}.jpg")

        img = cv.imread(img_path)
        if img is None:
            print(f"  Failed to read image")
            continue

        img_float = img.astype(np.float32)
        img_bright = np.clip(img_float - 50, 0, 255).astype(np.uint8)

        result = getInferResult(model, img_bright)
        if result.shape[0] == 0 or result.shape[1] == 0:
            print(f"  No detection, skipping")
            continue

        roi = img[int(result[0][1]):int(result[0][3]),
                  int(result[0][0]):int(result[0][2])]

        gray_roi = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)

        detector = pyced.CED(np.ascontiguousarray(roi))
        detector.run_CED()
        rotRects = detector.getEllipsesAfterCluster()

        ellipses = []
        for e in rotRects:
            ellipses.append((*e.center, e.size[0]/2, e.size[1]/2, e.angle))

        matcher = UltimateSocketMatcher()
        matcher.obj_pts = obj_pts
        matcher.K = K
        matcher.dist = dist
        matcher.eMc = eMc

        draw_ellipse(roi, ellipses)

        final_pts, status, centers = matcher.solve(ellipses, [*(result[0][:2]), *(result[0][2:]-result[0][:2])])
        print(f'  Found {len(final_pts) if final_pts is not None else 0} points')

        if final_pts is None or centers is None or centers.shape[0] < 7:
            print(f"  Not enough points for PnP")
            frame_id += 1
            continue

        pts3d = matcher.obj_pts[matcher.r_idx]
        pts2d = centers

        # Two-round PnP
        rvec, tvec, valid, inlier_mask, per_point_errors, used_threshold = two_round_pnp(
            pts2d, pts3d, K, dist)
        if not valid:
            print(f"  PnP failed, skipping")
            frame_id += 1
            continue

        # Frame rejection check
        all_exceed = np.all(per_point_errors > FRAME_REJECT_THRESHOLD)
        if all_exceed:
            print(f"  [REJECT] All points exceed threshold {FRAME_REJECT_THRESHOLD}px")
            frame_id += 1
            continue

        # Compute cMo and bMo
        cMo = np.eye(4, dtype=np.float64)
        cMo[:3, :3] = Rotation.from_rotvec(rvec).as_matrix()
        cMo[:3, 3] = tvec

        bMo = robot_pose @ eMc @ cMo

        euler_cMo, tvec_cMo = pose_to_euler_tvec(cMo)
        euler_bMo, tvec_bMo = pose_to_euler_tvec(bMo)

        mean_error = per_point_errors.mean()
        print(f"  cMo euler: {euler_cMo}, tvec: {tvec_cMo}")
        print(f"  bMo euler: {euler_bMo}, tvec: {tvec_bMo}")
        print(f"  PnP mean error: {mean_error:.4f}px")

        # ========== Save visualization image ==========
        # Crop slightly larger than ROI
        roi_x_min = int(result[0][0])
        roi_y_min = int(result[0][1])
        roi_x_max = int(result[0][2])
        roi_y_max = int(result[0][3])
        pad = 50
        x1 = max(roi_x_min - pad, 0)
        y1 = max(roi_y_min - pad, 0)
        x2 = min(roi_x_max + pad, img.shape[1])
        y2 = min(roi_y_max + pad, img.shape[0])
        img_vis = img[y1:y2, x1:x2].copy()

        # Project 3D points to 2D with distortion
        proj_pts, _ = cv.projectPoints(pts3d, rvec, tvec, K, dist)
        proj_pts = proj_pts.reshape(-1, 2)

        # Draw reprojection error lines
        for i, (pt2d, proj) in enumerate(zip(pts2d, proj_pts)):
            pt = tuple((pt2d - np.array([x1, y1])).astype(int))
            pr = tuple((proj - np.array([x1, y1])).astype(int))
            cv.line(img_vis, pt, pr, (0, 255, 0), 1)
            cv.circle(img_vis, pt, 3, (0, 0, 255), -1)
            cv.putText(img_vis, str(i), (pt[0]+5, pt[1]+5),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Draw axis at origin (50mm scale)
        axis_pts = np.array([[0, 0, 0], [50, 0, 0], [0, 50, 0], [0, 0, 50]], dtype=np.float64)
        axis_proj, _ = cv.projectPoints(axis_pts, rvec, tvec, K, dist)
        axis_proj = (axis_proj.reshape(-1, 2) - np.array([x1, y1])).astype(int)

        origin = tuple(axis_proj[0])
        cv.line(img_vis, origin, tuple(axis_proj[1]), (0, 0, 255), 2)  # X - red
        cv.line(img_vis, origin, tuple(axis_proj[2]), (0, 255, 0), 2)  # Y - green
        cv.line(img_vis, origin, tuple(axis_proj[3]), (255, 0, 0), 2)  # Z - blue

        # Save image
        out_img_path = os.path.join(RESULT_DIR, f"frame_{frame_id_val:06d}.jpg")
        cv.imwrite(out_img_path, img_vis)

        frame_ids.append(frame_id_val)
        cMo_list.append(cMo.copy())
        bMo_list.append(bMo.copy())
        bMe_list.append(robot_pose.copy())
        pnp_errors.append(mean_error)

        frame_id += 1

    # ========== Plotting ==========
    if len(frame_ids) == 0:
        print("No valid frames processed")
        exit()

    n_frames = len(frame_ids)

    # Compute oMc = cMo^-1 (object pose in camera frame, camera fixed reference)
    oMc_list = [np.linalg.inv(cmo) for cmo in cMo_list]

    # Compute bMc = bMo @ oMc (camera pose in base frame)
    bMc_list = [bmo @ omc for bmo, omc in zip(bMo_list, oMc_list)]

    # Extract euler angles
    oMc_euler = np.array([pose_to_euler_tvec(p)[0] for p in oMc_list])
    bMo_euler = np.array([pose_to_euler_tvec(p)[0] for p in bMo_list])
    bMc_euler = np.array([pose_to_euler_tvec(p)[0] for p in bMc_list])
    oMc_tvec = np.array([p[:3, 3] for p in oMc_list])
    bMo_tvec = np.array([p[:3, 3] for p in bMo_list])
    bMc_tvec = np.array([p[:3, 3] for p in bMc_list])

    fig, axes = plt.subplots(5, 3, figsize=(18, 20))

    # oMc Euler angles
    for i, label in enumerate(['rx', 'ry', 'rz']):
        axes[0, i].plot(frame_ids, oMc_euler[:, i], 'b.-')
        axes[0, i].set_xlabel('Frame')
        axes[0, i].set_ylabel(f'{label} (deg)')
        axes[0, i].set_title(f'oMc {label} (object in camera)')
        axes[0, i].grid(True)

    # oMc Translation XYZ
    for i, label in enumerate(['X', 'Y', 'Z']):
        axes[1, i].plot(frame_ids, oMc_tvec[:, i], 'b.-')
        axes[1, i].set_xlabel('Frame')
        axes[1, i].set_ylabel(f'{label} (mm)')
        axes[1, i].set_title(f'oMc translation {label} (object in camera)')
        axes[1, i].grid(True)

    # bMo Euler angles
    for i, label in enumerate(['rx', 'ry', 'rz']):
        axes[2, i].plot(frame_ids, bMo_euler[:, i], 'r.-')
        axes[2, i].set_xlabel('Frame')
        axes[2, i].set_ylabel(f'{label} (deg)')
        axes[2, i].set_title(f'bMo {label}')
        axes[2, i].grid(True)

    # bMo Translation XYZ
    for i, label in enumerate(['X', 'Y', 'Z']):
        axes[3, i].plot(frame_ids, bMo_tvec[:, i], 'g.-')
        axes[3, i].set_xlabel('Frame')
        axes[3, i].set_ylabel(f'{label} (mm)')
        axes[3, i].set_title(f'bMo translation {label}')
        axes[3, i].grid(True)

    # bMc Euler angles (camera pose in base frame)
    for i, label in enumerate(['rx', 'ry', 'rz']):
        axes[4, i].plot(frame_ids, bMc_euler[:, i], 'm.-')
        axes[4, i].set_xlabel('Frame')
        axes[4, i].set_ylabel(f'{label} (deg)')
        axes[4, i].set_title(f'bMc {label} (camera in base)')
        axes[4, i].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, 'pnp_pose_vs_frame.png'), dpi=150)
    plt.close(fig)
    print(f"\nSaved pose plot to {RESULT_DIR}/pnp_pose_vs_frame.png")

    # bMc translation and bMe comparison
    fig_bmc, axes_bmc = plt.subplots(2, 3, figsize=(18, 10))

    # bMc Translation XYZ
    for i, label in enumerate(['X', 'Y', 'Z']):
        axes_bmc[0, i].plot(frame_ids, bMc_tvec[:, i], 'm.-')
        axes_bmc[0, i].set_xlabel('Frame')
        axes_bmc[0, i].set_ylabel(f'{label} (mm)')
        axes_bmc[0, i].set_title(f'bMc translation {label} (camera in base)')
        axes_bmc[0, i].grid(True)

    # bMe Translation XYZ (robot pose from records)
    if len(bMe_list) > 0:
        bMe_tvec = np.array([p[:3, 3] for p in bMe_list])
        for i, label in enumerate(['X', 'Y', 'Z']):
            axes_bmc[1, i].plot(frame_ids, bMe_tvec[:, i], 'c.-')
            axes_bmc[1, i].set_xlabel('Frame')
            axes_bmc[1, i].set_ylabel(f'{label} (mm)')
            axes_bmc[1, i].set_title(f'bMe translation {label} (robot pose from records)')
            axes_bmc[1, i].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, 'pnp_bMc_vs_frame.png'), dpi=150)
    plt.close(fig_bmc)
    print(f"Saved bMc+bMe plot to {RESULT_DIR}/pnp_bMc_vs_frame.png")

    # bMe pose plot
    if len(bMe_list) > 0:
        bMe_euler = np.array([pose_to_euler_tvec(p)[0] for p in bMe_list])
        bMe_tvec = np.array([p[:3, 3] for p in bMe_list])

        fig_bme, axes_bme = plt.subplots(2, 3, figsize=(18, 10))

        # bMe Euler angles
        for i, label in enumerate(['rx', 'ry', 'rz']):
            axes_bme[0, i].plot(frame_ids, bMe_euler[:, i], 'm.-')
            axes_bme[0, i].set_xlabel('Frame')
            axes_bme[0, i].set_ylabel(f'{label} (deg)')
            axes_bme[0, i].set_title(f'bMe {label} (robot pose)')
            axes_bme[0, i].grid(True)

        # bMe Translation XYZ
        for i, label in enumerate(['X', 'Y', 'Z']):
            axes_bme[1, i].plot(frame_ids, bMe_tvec[:, i], 'c.-')
            axes_bme[1, i].set_xlabel('Frame')
            axes_bme[1, i].set_ylabel(f'{label} (mm)')
            axes_bme[1, i].set_title(f'bMe translation {label} (robot pose)')
            axes_bme[1, i].grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(RESULT_DIR, 'pnp_bMe_vs_frame.png'), dpi=150)
        plt.close(fig_bme)
        print(f"Saved bMe plot to {RESULT_DIR}/pnp_bMe_vs_frame.png")

    # 3D trajectory
    fig2 = plt.figure(figsize=(8, 6))
    ax = fig2.add_subplot(111, projection='3d')
    ax.plot(bMo_tvec[:, 0], bMo_tvec[:, 1], bMo_tvec[:, 2], 'b.-')
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title('bMo 3D Trajectory')
    plt.savefig(os.path.join(RESULT_DIR, 'pnp_trajectory_3d.png'), dpi=150)
    plt.close(fig2)
    print(f"Saved 3D trajectory to {RESULT_DIR}/pnp_trajectory_3d.png")

    # PnP errors
    fig3, ax = plt.subplots()
    ax.plot(frame_ids, pnp_errors, 'b.-')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Mean Reprojection Error (px)')
    ax.set_title('PnP Mean Reprojection Error')
    ax.grid(True)
    plt.savefig(os.path.join(RESULT_DIR, 'pnp_errors.png'), dpi=150)
    plt.close(fig3)
    print(f"Saved error plot to {RESULT_DIR}/pnp_errors.png")

    print(f"\nProcessed {frame_id} total, {n_frames} valid frames")
    print(f"Results saved to {RESULT_DIR}")
