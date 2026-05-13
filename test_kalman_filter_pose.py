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

# Activate conda environment and source venv
# Run: conda activate cv48 && source ~/venv310/bin/activate && python test_kalman_filter_pose.py

from kalman_filter_pose_estimator import KalmanFilterPoseEstimator, RobustKalmanFilterPoseEstimator, pose_to_euler_tvec
from static_pose_optimizer import StaticPoseOptimizer
from gemiEd import *

# Load YOLO model
model = YOLO("checkpoint/best.pt")

# ============ Config ============
DATA_DIR = "dataset/save_data3/20260511_120244"
RESULT_DIR = "result/save_data3/20260511_120244/kalman_filter"
MAX_FRAMES = 50  # Limit for quick test, -1 for all
BEGIN_FRAME_ID = 1180

# PnP thresholds
ADAPTIVE_MULTIPLIER = 2.0
FIXED_ERROR_THRESHOLD = 0.4
FRAME_REJECT_THRESHOLD = 1.0

# Kalman filter config
KALMAN_PROCESS_NOISE_POS = 0.001
KALMAN_PROCESS_NOISE_VEL = 0.01
KALMAN_MEASUREMENT_NOISE = 1.5

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

    # Adaptive threshold
    median_error = np.median(per_point_errors)
    threshold = max(median_error * ADAPTIVE_MULTIPLIER, FIXED_ERROR_THRESHOLD)

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


def validate_cMo(cMo):
    """Check if cMo is physically valid."""
    R = cMo[:3, :3]
    det_R = np.linalg.det(R)
    if abs(det_R - 1.0) > 1e-6:
        return False

    tvec = cMo[:3, 3]
    if tvec[2] <= 0:
        return False

    dist_val = np.linalg.norm(tvec)
    if dist_val < 50 or dist_val > 3000:
        return False

    return True


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
        process_noise_pos=KALMAN_PROCESS_NOISE_POS,
        process_noise_vel=KALMAN_PROCESS_NOISE_VEL,
        measurement_noise=KALMAN_MEASUREMENT_NOISE,
        max_reproj_error=10.0
    )

    # Also initialize static optimizer for comparison
    static_opt = StaticPoseOptimizer(K, dist)
    static_opt.set_extrinsics(eMc)
    static_opt.set_object_pts(obj_pts)

    # Load metadata
    meta_path = os.path.join(DATA_DIR, "metadata.json")
    with open(meta_path, 'r') as f:
        metadata = json.load(f)

    records = metadata['records']
    print(f"Loaded {len(records)} frames from {meta_path}")

    frame_id = 1
    processed_frames = 0
    last_timestamp_ns = None
    last_kalman_bMo = None
    last_static_bMo = None

    # Trajectory storage for visualization
    trajectory_kalman = []
    trajectory_static = []
    trajectory_pnp = []
    frame_ids = []

    for record in records:
        frame_id_val = record['frame_id']
        if frame_id_val < BEGIN_FRAME_ID:
            continue
        if MAX_FRAMES > 0 and processed_frames >= MAX_FRAMES:
            print(f"\nReached max frames limit ({MAX_FRAMES})")
            break
        processed_frames += 1

        current_timestamp_ns = record['camera_timestamp_ns']
        if last_timestamp_ns is not None:
            time_diff_s = (current_timestamp_ns - last_timestamp_ns) / 1e9
            if time_diff_s < 0.1:
                continue
            if time_diff_s > 1.0:
                print(f"[WARN] Large timestamp gap: {time_diff_s:.2f}s")
        last_timestamp_ns = current_timestamp_ns

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
        if result.shape[0] == 0:
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
        print(f'Found {len(final_pts)} points')

        if final_pts is None or centers.shape[0] < 4:
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

        # Build cMo from PnP
        cMo = np.eye(4)
        cMo[:3, :3] = Rotation.from_rotvec(rvec).as_matrix()
        cMo[:3, 3] = tvec

        # Initial bMo
        bMo_init = robot_pose @ eMc @ cMo

        # ========== Kalman Filter Update ==========
        cMo_kalman, errors_kalman, inlier_mask_kalman, is_valid = kalman.update_with_robust(
            pts2d, pts3d, robot_pose, eMc, inlier_mask=None)

        if is_valid and cMo_kalman is not None:
            bMo_kalman = kalman.get_bMo(robot_pose, eMc)
            print(f"Kalman bMo: {pose_to_euler_tvec(bMo_kalman)}")
            print(f"Kalman cMo: {pose_to_euler_tvec(cMo_kalman)}")
            print(f"Kalman mean error: {errors_kalman.mean():.4f}px")

            # Check if consistent with last pose
            if last_kalman_bMo is not None:
                delta = bMo_kalman @ np.linalg.inv(last_kalman_bMo)
                delta_t = np.linalg.norm(delta[:3, 3])
                delta_r = np.linalg.norm(Rotation.from_matrix(delta[:3, :3]).as_rotvec())
                print(f"  Delta from last: t={delta_t:.2f}mm, r={np.degrees(delta_r):.2f}deg")

            last_kalman_bMo = bMo_kalman.copy()
            trajectory_kalman.append(bMo_kalman.copy())

        # ========== Static Optimizer Update ==========
        if not static_opt.is_initialized():
            static_opt.set_initial_pose(bMo_init)

        if len(trajectory_static) == 0 or last_static_bMo is not None:
            if static_opt.get_frame_count() >= 8:
                static_opt.remove_oldest_frame()
            static_opt.add_frame(frame_id, robot_pose, pts2d, pts3d)
            static_opt.optimize()
            bMo_static = static_opt.get_pose()
            cMo_static = static_opt.compute_cMo(robot_pose)
            last_static_bMo = bMo_static.copy()
            trajectory_static.append(bMo_static.copy())
            print(f"Static bMo: {pose_to_euler_tvec(bMo_static)}")
            print(f"Static mean error: {static_opt.get_average_error():.4f}px")

        # ========== PnP trajectory ==========
        trajectory_pnp.append(bMo_init.copy())
        frame_ids.append(frame_id_val)

        # ========== Visualization ==========
        obj_pt_names = ['L-top', 'R-top', 'L-mid', 'center', 'R-mid', 'L-bot', 'R-bot']
        print(f"Per-point errors (px), threshold={used_threshold:.3f}:")
        for i, (err, name) in enumerate(zip(per_point_errors, obj_pt_names[:len(per_point_errors)])):
            marker = '[INLIER]' if inlier_mask[i] else '[OUTLIER]'
            print(f"  Point {i} ({name}): {err:7.3f}px {marker}")

        # Draw detected centers with inlier/outlier coloring
        for i, (x, y) in enumerate(centers):
            if i < len(inlier_mask):
                if inlier_mask[i]:
                    color = (0, 255, 0)  # Green = inlier
                else:
                    color = (0, 0, 255)  # Red = outlier
            else:
                color = (255, 255, 0)  # Cyan = unknown
            cv.circle(img, (int(x), int(y)), 3, color, -1)
            cv.putText(img, str(i), (int(x)+5, int(y)-5),
                       cv.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # Draw reprojection from Kalman
        if cMo_kalman is not None:
            cv.drawFrameAxes(img, K, dist, cMo_kalman[:3, :3], cMo_kalman[:3, 3:], 10, 3)
            proj_kalman, _ = cv.projectPoints(pts3d,
                                              Rotation.from_matrix(cMo_kalman[:3, :3]).as_rotvec(),
                                              cMo_kalman[:3, 3:], K, dist)
            for i, (x, y) in enumerate(pts2d):
                x_proj, y_proj = proj_kalman[i][0]
                cv.line(img, (int(x), int(y)), (int(x_proj), int(y_proj)), (0, 255, 0), 1)
                cv.circle(img, (int(x_proj), int(y_proj)), 2, (255, 0, 0), -1)

        # Draw reprojection from PnP
        cv.drawFrameAxes(img, K, dist, rvec, tvec, 20, 1)

        # Crop and display
        vis = img[roi_y_min:roi_y_max, roi_x_min:roi_x_max]
        vis = cv.resize(vis, None, fx=2, fy=2, interpolation=cv.INTER_NEAREST)
        vis_path = os.path.join(RESULT_DIR, f"frame_{frame_id_val:06d}_vis.png")
        cv.imwrite(vis_path, vis)

        # Reprojection error visualization
        if cMo_kalman is not None:
            vis_err = img.copy()
            proj_err, _ = cv.projectPoints(pts3d,
                                           Rotation.from_matrix(cMo_kalman[:3, :3]).as_rotvec(),
                                           cMo_kalman[:3, 3:], K, dist)
            for i, (x, y) in enumerate(pts2d):
                x_proj, y_proj = proj_err[i][0]
                is_inlier = inlier_mask[i] if i < len(inlier_mask) else False
                color = (0, 255, 0) if is_inlier else (0, 0, 255)
                cv.circle(vis_err, (int(x), int(y)), 3, color, -1)
                cv.circle(vis_err, (int(x_proj), int(y_proj)), 3, (255, 0, 0), -1)
                cv.line(vis_err, (int(x), int(y)), (int(x_proj), int(y_proj)), color, 1)
            vis_err = vis_err[roi_y_min:roi_y_max, roi_x_min:roi_x_max]
            vis_err = cv.resize(vis_err, None, fx=2, fy=2, interpolation=cv.INTER_NEAREST)
            vis_err_path = os.path.join(RESULT_DIR, f"frame_{frame_id_val:06d}_reproj_err.png")
            cv.imwrite(vis_err_path, vis_err)

        frame_id += 1

    # ========== Trajectory Visualization ==========
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    if len(trajectory_kalman) > 0:
        # Extract positions
        pos_kalman = np.array([p[:3, 3] for p in trajectory_kalman])
        pos_static = np.array([p[:3, 3] for p in trajectory_static]) if len(trajectory_static) > 0 else None
        pos_pnp = np.array([p[:3, 3] for p in trajectory_pnp]) if len(trajectory_pnp) > 0 else None

        fig = plt.figure(figsize=(18, 5))

        # 3D trajectory
        ax1 = fig.add_subplot(131, projection='3d')
        ax1.plot(pos_kalman[:, 0], pos_kalman[:, 1], pos_kalman[:, 2], 'b.-', label='Kalman')
        if pos_static is not None:
            ax1.plot(pos_static[:, 0], pos_static[:, 1], pos_static[:, 2], 'g.-', label='Static')
        if pos_pnp is not None:
            ax1.plot(pos_pnp[:, 0], pos_pnp[:, 1], pos_pnp[:, 2], 'r.-', label='PnP')
        ax1.set_xlabel('X (mm)')
        ax1.set_ylabel('Y (mm)')
        ax1.set_zlabel('Z (mm)')
        ax1.set_title('3D Trajectory (bMo)')
        ax1.legend()

        # XY plane
        ax2 = fig.add_subplot(132)
        ax2.plot(pos_kalman[:, 0], pos_kalman[:, 1], 'b.-', label='Kalman')
        if pos_static is not None:
            ax2.plot(pos_static[:, 0], pos_static[:, 1], 'g.-', label='Static')
        if pos_pnp is not None:
            ax2.plot(pos_pnp[:, 0], pos_pnp[:, 1], 'r.-', label='PnP')
        ax2.set_xlabel('X (mm)')
        ax2.set_ylabel('Y (mm)')
        ax2.set_title('XY Plane')
        ax2.legend()
        ax2.axis('equal')
        ax2.grid(True)

        # Position vs frame
        ax3 = fig.add_subplot(133)
        frames = list(range(len(pos_kalman)))
        ax3.plot(frames, pos_kalman[:, 0], 'r.-', label='X')
        ax3.plot(frames, pos_kalman[:, 1], 'g.-', label='Y')
        ax3.plot(frames, pos_kalman[:, 2], 'b.-', label='Z')
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
            ax.plot(kalman.error_history, 'b.-')
            ax.set_xlabel('Frame')
            ax.set_ylabel('Mean Reprojection Error (px)')
            ax.set_title('Kalman Filter Reprojection Error History')
            ax.grid(True)
            plt.savefig(os.path.join(RESULT_DIR, 'error_history.png'), dpi=150)
            plt.close(fig2)

    print(f"\nResults saved to {RESULT_DIR}")
    print(f"Processed {processed_frames} frames")
