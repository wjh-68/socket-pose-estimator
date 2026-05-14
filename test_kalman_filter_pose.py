#!/usr/bin/env python3
"""
Test script for Kalman Filter-based Camera Pose Estimator.

Uses dataset/save_data3 for validation.
"""

import cv2 as cv
cv.setNumThreads(0)  # Disable multithreading to avoid Qt issues
from ultralytics import YOLO
import time
import os
import numpy as np
from scipy.spatial.transform import Rotation
import json
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for matplotlib

from kalman_filter_pose_estimator import KalmanFilterPoseEstimator, RobustKalmanFilterPoseEstimator, pose_to_euler_tvec
from static_pose_optimizer import StaticPoseOptimizer
from gemiEd import *

# Load YOLO model
model = YOLO("checkpoint/best.pt")

# ============ Config ============
DATA_DIR = "dataset/save_data3/20260511_120244"
RESULT_DIR = "result/save_data3/20260511_120244/kalman_filter"
MAX_FRAMES = -1  # Limit for quick test, -1 for all
BEGIN_FRAME_ID = 1180

# PnP thresholds
ADAPTIVE_MULTIPLIER = 2.0
FIXED_ERROR_THRESHOLD = 0.8
FRAME_REJECT_THRESHOLD = 1.0

# Kalman filter config (anisotropic noise for planar scene)
# [qx, qy, qz, qrx, qry, qrz]
KALMAN_PROCESS_NOISE = (0.1, 0.1, 0.5, 0.05, 0.05, 0.01)
KALMAN_VELOCITY_NOISE = (0.05, 0.05, 0.2, 0.02, 0.02, 0.005)
KALMAN_MEASUREMENT_NOISE = 2.0
KALMAN_VELOCITY_DAMPING = 0.98
KALMAN_VELOCITY_ALPHA = 0.9
KALMAN_MAHALANOBIS_THRESHOLD = 9.21  # chi2(0.01, 2 DOF)

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
    """Solve PnP using IPPE method."""
    success, rvec, tvec = cv.solvePnP(
        pts3d, pts2d, K, dist, flags=cv.SOLVEPNP_IPPE)
    if not success:
        return None, None, False
    return rvec.flatten(), tvec.flatten(), True


def compute_per_point_errors(pts3d, rvec, tvec, pts2d, K, dist):
    """Compute reprojection error per point."""
    proj, _ = cv.projectPoints(pts3d, rvec, tvec, K, dist)
    return np.linalg.norm(proj.reshape(-1, 2) - pts2d, axis=1)


def two_round_pnp(pts2d, pts3d, K, dist):
    """Two-round PnP with adaptive threshold."""
    # Round 1: Initial estimate
    rvec1, tvec1, valid1 = solvePnP_IPPE(pts2d, pts3d, K, dist)
    if not valid1:
        return None, None, False, None, None, None

    per_point_errors = compute_per_point_errors(pts3d, rvec1, tvec1, pts2d, K, dist)
    round1_error = per_point_errors.mean()

    # Fixed threshold (as per task.txt, use fixed not adaptive)
    threshold = FIXED_ERROR_THRESHOLD

    inlier_mask = per_point_errors < threshold
    n_inliers = inlier_mask.sum()

    if n_inliers < 4:
        threshold = threshold * 2
        inlier_mask = per_point_errors < threshold
        n_inliers = inlier_mask.sum()

    # Round 2: Refine with inliers
    if n_inliers >= 4:
        pts3d_inlier = pts3d[inlier_mask]
        pts2d_inlier = pts2d[inlier_mask]
        rvec2, tvec2, valid2 = solvePnP_IPPE(pts2d_inlier, pts3d_inlier, K, dist)
        if valid2:
            per_point_errors = compute_per_point_errors(pts3d, rvec2, tvec2, pts2d, K, dist)
            return rvec2, tvec2, True, inlier_mask, per_point_errors, threshold
        else:
            return rvec1, tvec1, True, inlier_mask, per_point_errors, threshold
    else:
        return rvec1, tvec1, True, inlier_mask, per_point_errors, threshold


def getInferResult(model, img):
    """Run YOLO inference."""
    results = model(img)
    if len(results) == 0:
        return np.array([])
    return results[0].boxes.xyxy.cpu().numpy()


if __name__ == '__main__':
    os.makedirs(RESULT_DIR, exist_ok=True)

    # Initialize Kalman filter
    kalman = RobustKalmanFilterPoseEstimator(
        K, dist,
        process_noise=KALMAN_PROCESS_NOISE,
        velocity_noise=KALMAN_VELOCITY_NOISE,
        measurement_noise=KALMAN_MEASUREMENT_NOISE,
        velocity_damping=KALMAN_VELOCITY_DAMPING,
        velocity_alpha=KALMAN_VELOCITY_ALPHA,
        mahalanobis_threshold=KALMAN_MAHALANOBIS_THRESHOLD
    )

    # Load metadata
    meta_path = os.path.join(DATA_DIR, "metadata.json")
    with open(meta_path, 'r') as f:
        metadata = json.load(f)

    records = metadata['records']
    print(f"Loaded {len(records)} frames from {meta_path}")

    frame_id = 0
    last_timestamp_ns = None
    last_kalman_bMo = None

    # Trajectory storage
    trajectory_kalman = []
    frame_ids = []

    for record in records:
        frame_id_val = record['frame_id']
        if frame_id_val < BEGIN_FRAME_ID:
            continue

        current_timestamp_ns = record['camera_timestamp_ns']
        if last_timestamp_ns is not None:
            time_diff_s = (current_timestamp_ns - last_timestamp_ns) / 1e9
            if time_diff_s < 0.1:
                continue
            if time_diff_s > 1.0:
                print(f"[WARN] Large timestamp gap: {time_diff_s:.2f}s")
        last_timestamp_ns = current_timestamp_ns

        if MAX_FRAMES > 0 and frame_id >= MAX_FRAMES:
            print(f"\nReached max frames limit ({MAX_FRAMES})")
            break

        img_relative_path = record['image_path']
        img_path = os.path.join(DATA_DIR, img_relative_path)
        robot_pose = np.array(record['pose_matrix_4x4']).reshape(4, 4)

        print(f"\n==============================================")
        print(f"Processing frame {frame_id}: frame_{frame_id_val:06d}.jpg")

        img = cv.imread(img_path)
        if img is None:
            print(f"Failed to read image: {img_path}")
            continue

        # Preprocess
        img_float = img.astype(np.float32)
        img_bright = np.clip(img_float - 50, 0, 255).astype(np.uint8)

        result = getInferResult(model, img_bright)
        if result.shape[0] == 0 or result.shape[1] == 0:
            print(f"  No detection, skipping")
            continue

        roi = img[int(result[0][1]):int(result[0][3]),
                  int(result[0][0]):int(result[0][2])]
        roi_x_min = int(result[0][0])
        roi_y_min = int(result[0][1])
        roi_x_max = int(result[0][2])
        roi_y_max = int(result[0][3])

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
        print(f'Found {len(final_pts) if final_pts is not None else 0} points')

        if final_pts is None or centers is None or centers.shape[0] < 7:
            print(f"Not enough points for PnP")
            frame_id += 1
            continue

        pts3d = matcher.obj_pts[matcher.r_idx]
        pts2d = centers

        # Two-round PnP
        rvec, tvec, valid, inlier_mask, per_point_errors, used_threshold = two_round_pnp(
            pts2d, pts3d, K, dist)
        if not valid:
            print(f"PnP failed, skipping")
            continue

        # Frame rejection check
        all_exceed = np.all(per_point_errors > FRAME_REJECT_THRESHOLD)
        if all_exceed:
            print(f"  [REJECT] All points exceed threshold {FRAME_REJECT_THRESHOLD}px")
            frame_id += 1
            continue

        # Kalman Filter Update
        timestamp_ns = record['camera_timestamp_ns']
        cMo_kalman, rvec_kf, tvec_kf, errors_kalman, is_updated = kalman.update_with_observation(
            pts2d, pts3d, rvec_pnp=rvec, tvec_pnp=tvec,
            inlier_mask=inlier_mask, timestamp=timestamp_ns)

        if is_updated and cMo_kalman is not None:
            bMo_kalman = kalman.get_bMo(robot_pose, eMc)
            vel = kalman.get_velocity()

            print(f"Kalman bMo: {pose_to_euler_tvec(bMo_kalman)}")
            print(f"Kalman cMo: {pose_to_euler_tvec(cMo_kalman)}")
            print(f"Kalman mean error: {errors_kalman.mean():.4f}px")

            # Diagnostic output
            print(f"  Velocity: trans={vel[3:6]}, rot={vel[0:3]}")
            if len(kalman.mahal_history) > 0:
                print(f"  Mahalanobis dist: {kalman.mahal_history[-1]:.3f}")
            if len(kalman.cond_history) > 0:
                print(f"  Condition number: {kalman.cond_history[-1]:.2e}")
            cov_diag = kalman.get_covariance_diagonal()
            print(f"  Cov diag: pos={cov_diag[3:6]}, vel={cov_diag[9:12]}")

            # Check delta from last
            if last_kalman_bMo is not None:
                delta = bMo_kalman @ np.linalg.inv(last_kalman_bMo)
                delta_t = np.linalg.norm(delta[:3, 3])
                delta_r = np.linalg.norm(Rotation.from_matrix(delta[:3, :3]).as_rotvec())
                print(f"  Delta from last: t={delta_t:.2f}mm, r={np.degrees(delta_r):.2f}deg")

            last_kalman_bMo = bMo_kalman.copy()
            trajectory_kalman.append(bMo_kalman.copy())
            frame_ids.append(frame_id_val)

        frame_id += 1

    # ========== Trajectory Visualization ==========
    import matplotlib.pyplot as plt

    if len(trajectory_kalman) > 0:
        pos = np.array([p[:3, 3] for p in trajectory_kalman])

        fig = plt.figure(figsize=(18, 5))

        # 3D trajectory
        ax1 = fig.add_subplot(131, projection='3d')
        ax1.plot(pos[:, 0], pos[:, 1], pos[:, 2], 'b.-', label='Kalman')
        ax1.set_xlabel('X (mm)')
        ax1.set_ylabel('Y (mm)')
        ax1.set_zlabel('Z (mm)')
        ax1.set_title('3D Trajectory (bMo)')
        ax1.legend()

        # XY plane
        ax2 = fig.add_subplot(132)
        ax2.plot(pos[:, 0], pos[:, 1], 'b.-', label='Kalman')
        ax2.set_xlabel('X (mm)')
        ax2.set_ylabel('Y (mm)')
        ax2.set_title('XY Plane')
        ax2.legend()
        ax2.axis('equal')
        ax2.grid(True)

        # Position vs frame
        ax3 = fig.add_subplot(133)
        frames = list(range(len(pos)))
        ax3.plot(frames, pos[:, 0], 'r.-', label='X')
        ax3.plot(frames, pos[:, 1], 'g.-', label='Y')
        ax3.plot(frames, pos[:, 2], 'b.-', label='Z')
        ax3.set_xlabel('Frame')
        ax3.set_ylabel('Position (mm)')
        ax3.set_title('Kalman Position vs Frame')
        ax3.legend()
        ax3.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(RESULT_DIR, 'trajectory.png'), dpi=150)
        plt.close(fig)

        # Error history
        if len(kalman.error_history) > 0:
            fig2, ax = plt.subplots()
            ax.plot(kalman.error_history, 'b.-', label='Mean Error')
            if len(kalman.mahal_history) > 0:
                ax2 = ax.twinx()
                ax2.plot(kalman.mahal_history, 'r.-', alpha=0.7, label='Mahalanobis')
                ax2.set_ylabel('Mahalanobis Distance', color='r')
            ax.set_xlabel('Frame')
            ax.set_ylabel('Mean Reprojection Error (px)', color='b')
            ax.set_title('Kalman Filter Errors')
            ax.grid(True)
            plt.savefig(os.path.join(RESULT_DIR, 'error_history.png'), dpi=150)
            plt.close(fig2)

        # Velocity history
        if len(kalman.velocity_history) > 0:
            vel_hist = np.array(kalman.velocity_history)
            fig3, axes = plt.subplots(2, 3, figsize=(15, 8))
            vel_labels = ['vx', 'vy', 'vz', 'wx', 'wy', 'wz']
            for i, (ax, label) in enumerate(zip(axes.flat, vel_labels)):
                ax.plot(vel_hist[:, i], 'b.-')
                ax.set_xlabel('Frame')
                ax.set_ylabel(label)
                ax.set_title(f'Velocity {label}')
                ax.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(RESULT_DIR, 'velocity_history.png'), dpi=150)
            plt.close(fig3)

    print(f"\nResults saved to {RESULT_DIR}")
    print(f"Processed {frame_id} frames")