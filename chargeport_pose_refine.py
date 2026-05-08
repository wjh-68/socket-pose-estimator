"""
Chargeport Pose Refine - Online pose estimation with sliding window and RANSAC filtering.
Maintains a sliding window of frames and computes refined bMo using three methods:
median, weighted average, and RANSAC-filtered mean.
"""

import cv2 as cv
from ultralytics import YOLO
import time
from gemiEd import *
from scipy.spatial.transform import Rotation
import os
import re
import numpy as np
from static_pose_optimizer import StaticPoseOptimizer, pose_to_euler_tvec

# Load a model
model = YOLO("checkpoint/best.pt")

# ============ Config ============
# DATA_DIR = "dataset/save_data2"
# RESULT_DIR = "result/save_data2/pose_refine"
DATA_DIR = "dataset/save_data9"
RESULT_DIR = "result/save_data9/pose_refine"
SLIDING_WINDOW_SIZE = 8

# PnP threshold config
USE_ADAPTIVE_THRESHOLD = True  # True = adaptive, False = fixed
FIXED_ERROR_THRESHOLD = 0.4    # used when USE_ADAPTIVE_THRESHOLD = False
ADAPTIVE_MULTIPLIER = 2.0      # threshold = median_error * multiplier

# Frame rejection threshold - if mean reprojection error exceeds this, frame is rejected
FRAME_REJECT_THRESHOLD = 1.0   # pixels - if ALL points have error > this, reject frame

# Camera on robot end-effector (eye-to-hand extrinsic)
# eMc = np.array([
#     [-7.2267956e-01, 6.9102561e-01, -1.4759262e-02, -5.1758522e+01],
#     [-6.9116789e-01, -7.2264087e-01, 8.7790741e-03, 6.0040222e+01],
#     [-4.5990809e-01, 1.6545586e-02, 9.9985254e-01, 9.7955963e+01],
#     [0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 1.0000000e+00]
# ], dtype=np.float64)

# # Camera intrinsics
# K = np.array([
#     [2674.7629874104787, 0., 1279.5],
#     [0., 2674.7629874104787, 719.5],
#     [0., 0., 1.]
# ], dtype=np.float64)

# dist = np.array([-0.11744968686298927, 0.27089153364253454, 0.0012180578884344092,
#                  0.00067320963008635703, -0.078845410108757258], dtype=np.float64)

eMc = np.array([
    [-6.9855857e-01,  7.1512282e-01,  2.4804471e-02, -5.1826664e+01],
    [-7.1555281e-01, -6.9815123e-01, -2.3854841e-02,  5.5274796e+01],
    [ 2.5813223e-04, -3.4412913e-02,  9.9940765e-01,  9.5362617e+01],
    [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  1.0000000e+00]
    ], dtype=np.float64)
# camera intrinsics
K = np.array([
        [1015.445938660267, 0., 638.51741890470555],
        [0., 1015.445938660267, 386.838616473841],
        [0., 0., 1.]
        ], dtype=np.float64)
dist = np.array([
    0.11753195467413819, -0.19301774104640848,
    0.00016793575097772418, -0.00061144051421409198, 0.072260521199194336
], dtype=np.float64)


# 3D object points in object frame (charge port keypoints)
obj_pts = np.array([
    [-8.0, 11.2, 0.0], [8.0, 11.2, 0.0],
    [-16.0, 0.0, 0.0], [0.0, 0.0, 0.0], [16.0, 0.0, 0.0],
    [-8.0, -13.9, 0.0], [8.0, -13.9, 0.0]
], dtype=np.float64)

obj_pt_names = ['L-top', 'R-top', 'L-mid', 'center', 'R-mid', 'L-bot', 'R-bot']


def solvePnP_IPPE(pts2d, pts3d, K, dist):
    """Wrapper for cv2.solvePnP with IPPE method and validity checks"""
    success, rvec, tvec = cv2.solvePnP(
        pts3d, pts2d, K, dist, flags=cv2.SOLVEPNP_IPPE)
    if not success:
        return None, None, False

    rvec = rvec.flatten()
    tvec = tvec.flatten()
    cMo = np.eye(4)
    cMo[:3, :3] = Rotation.from_rotvec(rvec).as_matrix()
    cMo[:3, 3] = tvec

    valid, reason = validate_cMo(cMo)
    if not valid:
        print(f"  [WARN] PnP result invalid: {reason}")
        return None, None, False

    return rvec, tvec, True


def compute_reproj_error(pts3d, rvec, tvec, pts2d, K, dist):
    proj, _ = cv2.projectPoints(pts3d, rvec, tvec, K, dist)
    return np.linalg.norm(proj.reshape(-1, 2) - pts2d, axis=1).mean()


def compute_per_point_reproj_errors(pts3d, rvec, tvec, pts2d, K, dist):
    proj, _ = cv2.projectPoints(pts3d, rvec, tvec, K, dist)
    return np.linalg.norm(proj.reshape(-1, 2) - pts2d, axis=1)


def two_round_pnp(pts2d, pts3d, K, dist, error_threshold=0.4):
    """Two-round PnP with adaptive or fixed threshold.

    Returns:
        rvec, tvec, valid, inlier_mask, per_point_errors, round1_errors, used_threshold
    """
    # ============ Round 1: Initial estimate with all points ============
    rvec1, tvec1, valid1 = solvePnP_IPPE(pts2d, pts3d, K, dist)
    if not valid1:
        return None, None, False, None, None, None, None

    per_point_errors = compute_per_point_reproj_errors(pts3d, rvec1, tvec1, pts2d, K, dist)
    round1_error = per_point_errors.mean()

    # ============ Determine threshold ============
    if USE_ADAPTIVE_THRESHOLD:
        median_error = np.median(per_point_errors)
        current_threshold = median_error * ADAPTIVE_MULTIPLIER
        current_threshold = max(current_threshold, FIXED_ERROR_THRESHOLD)
    else:
        current_threshold = FIXED_ERROR_THRESHOLD

    # ============ Filter inliers based on error threshold ============
    inlier_mask = per_point_errors < current_threshold
    n_inliers = inlier_mask.sum()

    # If too few inliers, fall back to all points with higher threshold
    if n_inliers < 4:
        current_threshold = current_threshold * 2
        inlier_mask = per_point_errors < current_threshold
        n_inliers = inlier_mask.sum()

    # ============ Round 2: Refine with inliers only ============
    if n_inliers >= 4:
        pts3d_inlier = pts3d[inlier_mask]
        pts2d_inlier = pts2d[inlier_mask]
        rvec2, tvec2, valid2 = solvePnP_IPPE(pts2d_inlier, pts3d_inlier, K, dist)
        if valid2:
            per_point_errors_round2 = compute_per_point_reproj_errors(pts3d, rvec2, tvec2, pts2d, K, dist)
            return rvec2, tvec2, True, inlier_mask, per_point_errors_round2, round1_error, current_threshold
        else:
            return rvec1, tvec1, True, inlier_mask, per_point_errors, round1_error, current_threshold
    else:
        return rvec1, tvec1, True, inlier_mask, per_point_errors, round1_error, current_threshold


def validate_cMo(cMo):
    """Check if cMo is physically valid"""
    R = cMo[:3, :3]
    det_R = np.linalg.det(R)
    if abs(det_R - 1.0) > 1e-6:
        return False, f"Rotation det={det_R:.4f}"

    tvec = cMo[:3, 3]
    if tvec[2] <= 0:
        return False, f"Object behind camera (z={tvec[2]:.2f})"

    dist = np.linalg.norm(tvec)
    if dist < 50 or dist > 3000:
        return False, f"Object distance={dist:.1f}mm"

    return True, "ok"


def pose_to_euler_tvec(pose):
    """Convert 4x4 pose to (roll, pitch, yaw, x, y, z)"""
    R = pose[:3, :3]
    t = pose[:3, 3]
    euler = Rotation.from_matrix(R).as_euler('xyz', degrees=True)
    return f"r={euler[0]:7.2f} p={euler[1]:7.2f} y={euler[2]:7.2f}  x={t[0]:8.2f} y={t[1]:8.2f} z={t[2]:8.2f}"


def getInferResult(model, img):
    results = model(img)
    if len(results) == 0:
        return []
    return results[0].boxes.xyxy.cpu().numpy()


def rotation_mean_spd(rotvecs, weights=None):
    """Compute mean rotation using quaternion averaging with SDP.

    Properly handles the non-linear nature of rotation representations.
    """
    n = len(rotvecs)
    if weights is None:
        weights = np.ones(n)

    quats = np.array([Rotation.from_rotvec(rv).as_quat() for rv in rotvecs])

    for i in range(n):
        if quats[i, 0] < 0:
            quats[i] = -quats[i]

    Q = np.zeros((4, 4))
    for i in range(n):
        q = quats[i]
        w = weights[i]
        Q += w * np.outer(q, q)

    _, _, Vt = np.linalg.svd(Q)
    q_mean = Vt[0]
    q_mean = q_mean / np.linalg.norm(q_mean)
    return Rotation.from_quat(q_mean).as_rotvec()


def compute_bMo_statistics(window_bMo, window_rotvecs, window_errors):
    """Compute bMo using three methods: median, weighted, RANSAC.

    Returns:
        bMo_median, bMo_weighted, bMo_ransac, ransac_inlier_mask, pos_threshold, rot_threshold
    """
    if len(window_bMo) < 3:
        return None, None, None, None, None, None

    bmo_positions = np.array([b[:3, 3] for b in window_bMo])
    bmo_rotvecs = np.array(window_rotvecs)
    reproj_errors = np.array(window_errors)

    # Method 1: Median
    pos_median = np.median(bmo_positions, axis=0)
    rotvec_median = np.median(bmo_rotvecs, axis=0)

    # Method 2: Weighted average by inverse reprojection error
    weights = 1.0 / (reproj_errors ** 2 + 1e-6)
    weights /= weights.sum()
    pos_weighted = np.average(bmo_positions, weights=weights, axis=0)
    rotvec_weighted = rotation_mean_spd(bmo_rotvecs, weights=weights)

    # Method 3: RANSAC-style outlier filtering
    pos_dist = np.linalg.norm(bmo_positions - pos_median, axis=1)

    rot_median = Rotation.from_rotvec(rotvec_median)
    rot_dist = np.array([
        np.linalg.norm((rot_median * Rotation.from_rotvec(rv).inv()).as_rotvec())
        for rv in bmo_rotvecs
    ])

    # Adaptive thresholds based on distribution
    pos_threshold = np.percentile(pos_dist, 75) * 2 if len(pos_dist) > 0 else 5.0
    rot_threshold = np.percentile(rot_dist, 75) * 2 if len(rot_dist) > 0 else 0.05
    ransac_inlier_mask = (pos_dist < pos_threshold) & (rot_dist < rot_threshold)
    n_ransac_inliers = ransac_inlier_mask.sum()

    if n_ransac_inliers >= 3:
        pos_ransac = np.mean(bmo_positions[ransac_inlier_mask], axis=0)
        rotvec_ransac = rotation_mean_spd(bmo_rotvecs[ransac_inlier_mask])
    else:
        pos_ransac = pos_median
        rotvec_ransac = rotvec_median

    # Build bMo matrices
    R_median = Rotation.from_rotvec(rotvec_median).as_matrix()
    R_weighted = Rotation.from_rotvec(rotvec_weighted).as_matrix()
    R_ransac = Rotation.from_rotvec(rotvec_ransac).as_matrix()

    bMo_median = np.eye(4)
    bMo_median[:3, :3] = R_median
    bMo_median[:3, 3] = pos_median

    bMo_weighted = np.eye(4)
    bMo_weighted[:3, :3] = R_weighted
    bMo_weighted[:3, 3] = pos_weighted

    bMo_ransac = np.eye(4)
    bMo_ransac[:3, :3] = R_ransac
    bMo_ransac[:3, 3] = pos_ransac

    return bMo_median, bMo_weighted, bMo_ransac, ransac_inlier_mask, pos_threshold, rot_threshold


def draw_axes_dashed(img, K, dist, rvec, tvec, length, thickness, color):
    """Draw coordinate axes with dashed lines"""
    axes_pts = np.float32([[0, 0, 0],
                           [length, 0, 0],
                           [0, length, 0],
                           [0, 0, length]]).reshape(-1, 1, 3)

    imgpts, _ = cv2.projectPoints(axes_pts, rvec, tvec, K, dist)
    imgpts = imgpts.astype(int)

    origin = tuple(imgpts[0].ravel())
    x_pt = tuple(imgpts[1].ravel())
    y_pt = tuple(imgpts[2].ravel())
    z_pt = tuple(imgpts[3].ravel())

    draw_dashed_line(img, origin, x_pt, color[0], thickness)
    draw_dashed_line(img, origin, y_pt, color[1], thickness)
    draw_dashed_line(img, origin, z_pt, color[2], thickness)

    return img


def draw_dashed_line(img, pt1, pt2, color, thickness):
    """Draw a dashed line between two points"""
    dist = np.sqrt((pt2[0] - pt1[0]) ** 2 + (pt2[1] - pt1[1]) ** 2)
    if dist < 1:
        return
    dash_len = 10
    gap_len = 5
    dx = (pt2[0] - pt1[0]) / dist
    dy = (pt2[1] - pt1[1]) / dist

    x, y = pt1
    d = 0
    while d < dist:
        d_next = min(d + dash_len, dist)
        x_end = int(x + dx * d_next)
        y_end = int(y + dy * d_next)
        cv2.line(img, (int(x), int(y)), (x_end, y_end), color, thickness)
        d = d_next + gap_len
        x = x + dx * d
        y = y + dy * d


if __name__ == '__main__':
    os.makedirs(RESULT_DIR, exist_ok=True)

    # Get sorted image and pose files
    
    # npy_files = {f for f in os.listdir(DATA_DIR)
    #              if f.endswith('.npy')}
    # img_files = sorted(
    #         [f for f in os.listdir(DATA_DIR) if f.endswith('.jpg') and f != 'temp'],
    #         key=lambda x: int(re.search(r'(\d+)', x).group(1))
    #     )
    # ts_list = [f.replace('.jpg', '').replace('img_', '') for f in img_files]
    
    npy_files = sorted(
            [f for f in os.listdir(DATA_DIR) if f.endswith('.npy')])
    img_files = {f for f in os.listdir(DATA_DIR) if f.endswith('.png') and f != 'temp'}
    ts_list = [f.replace('.npy', '') for f in npy_files]
    
    print("=" * 80)
    print("Chargeport Pose Refine - Sliding Window Pose Estimation")
    print("=" * 80)
    print(f"\nConfig:")
    print(f"  Sliding window size: {SLIDING_WINDOW_SIZE}")
    print(f"  Adaptive threshold: {USE_ADAPTIVE_THRESHOLD}")
    print(f"  Fixed threshold: {FIXED_ERROR_THRESHOLD}")
    print(f"  Adaptive multiplier: {ADAPTIVE_MULTIPLIER}")
    print(f"  Frame reject threshold: {FRAME_REJECT_THRESHOLD} px")

    # ============ Sliding window state ============
    window_bMo = []           # list of bMo matrices
    window_rotvecs = []       # list of rotation vectors
    window_errors = []        # list of reprojection errors
    window_img_files = []     # list of corresponding jpg filenames
    window_frame_ids = []      # list of frame_ids
    accepted_frames = 0
    rejected_frames = 0
    # =============================================

    # Default bMo for visualization before any valid result
    bMo_ransac_default = np.eye(4)
    cMo_ransac_default = np.eye(4)

    frame_id = 1
    all_results = []

    for ts in ts_list:
        # img_file = f"img_{ts}.jpg"
        # npy_file = f"pose_{ts}.npy"
        img_file = f"{ts}.png"
        npy_file = f"{ts}.npy"
        if img_file not in img_files or npy_file not in npy_files:
            print(f"  Missing files for timestamp {ts}, skipping")
            continue

        print("\n" + "-" * 80)
        print(f"Frame {frame_id}: {img_file}")
        print("-" * 80)

        # Load image
        img_path = os.path.join(DATA_DIR, img_file)
        img = cv2.imread(img_path)

        # Load robot pose
        robot_pose_path = os.path.join(DATA_DIR, npy_file)
        bMe = np.load(robot_pose_path)

        # Preprocess image
        img_float = img.astype(np.float32)
        img_bright = np.clip(img_float - 50, 0, 255).astype(np.uint8)

        # Detect ROI with YOLO
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

        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        detector = pyced.CED(np.ascontiguousarray(roi))
        detector.run_CED()
        rotRects = detector.getEllipsesAfterCluster()
        ellipses_ = []
        for e in rotRects:
            ellipses_.append((*e.center, e.size[0] / 2, e.size[1] / 2, e.angle))

        matcher = UltimateSocketMatcher()
        matcher.obj_pts = obj_pts
        matcher.K = K
        matcher.dist = dist
        matcher.eMc = eMc

        final_pts, status, centers = matcher.solve(
            ellipses_, [*(result[0][:2]), *(result[0][2:] - result[0][:2])])
        print(f"  Found {len(final_pts)} points")

        if final_pts is None or centers.shape[0] < 4:
            print(f"  Not enough points, skipping")
            continue

        pts3d = matcher.obj_pts[matcher.r_idx]
        pts2d = centers

        # ============ Two-round PnP ============
        rvec, tvec, valid, inlier_mask, per_point_errors, round1_error, used_threshold = two_round_pnp(
            pts2d, pts3d, K, dist, error_threshold=FIXED_ERROR_THRESHOLD)
        if not valid:
            print(f"  PnP failed, skipping")
            continue

        n_inliers = inlier_mask.sum() if inlier_mask is not None else 0
        pnp_error = per_point_errors.mean()

        # Print per-point reprojection errors
        print(f"  Per-point reprojection errors (px), threshold={used_threshold:.3f}:")
        for i, (err, pt_name) in enumerate(zip(per_point_errors, obj_pt_names[:len(per_point_errors)])):
            marker = '[INLIER]' if (inlier_mask is not None and inlier_mask[i]) else '[OUTLIER]'
            print(f"    Point {i} ({pt_name}): {err:7.3f} px {marker}")

        # ============ Check if frame should be rejected ============
        # Frame is rejected if ALL points have error > FRAME_REJECT_THRESHOLD
        all_points_exceed = np.all(per_point_errors > FRAME_REJECT_THRESHOLD)
        frame_rejected = False

        if all_points_exceed:
            print(f"  [REJECT] All points exceed threshold {FRAME_REJECT_THRESHOLD} px, frame rejected")
            frame_rejected = True
            rejected_frames += 1
        else:
            accepted_frames += 1

        # ============ Build cMo and bMo ============
        cMo = np.eye(4)
        cMo[:3, :3] = Rotation.from_rotvec(rvec).as_matrix()
        cMo[:3, 3] = tvec
        bMo = bMe @ eMc @ cMo

        # Compute BA version as well
        if n_inliers >= 4:
            pts3d_inlier = pts3d[inlier_mask]
            pts2d_inlier = pts2d[inlier_mask]
        else:
            pts3d_inlier = pts3d
            pts2d_inlier = pts2d
        success, rvec_ba, tvec_ba = cv2.solvePnP(pts3d_inlier, pts2d_inlier, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
        cMo_ba = np.eye(4)
        cMo_ba[:3, :3] = Rotation.from_rotvec(rvec_ba.flatten()).as_matrix()
        cMo_ba[:3, 3] = tvec_ba.flatten()
        bMo_ba = bMe @ eMc @ cMo_ba

        # ============ Update sliding window (only if frame not rejected) ============
        if not frame_rejected:
            rvec_bMo = Rotation.from_matrix(bMo[:3, :3]).as_rotvec()
            window_bMo.append(bMo.copy())
            window_rotvecs.append(rvec_bMo.flatten().copy())
            window_errors.append(pnp_error)
            window_img_files.append(img_file)
            window_frame_ids.append(frame_id)

            # Remove oldest frame if window is full
            if len(window_bMo) > SLIDING_WINDOW_SIZE:
                window_bMo.pop(0)
                window_rotvecs.pop(0)
                window_errors.pop(0)
                window_img_files.pop(0)
                window_frame_ids.pop(0)

            print(f"  Window updated: {len(window_bMo)} frames (accepted: {accepted_frames}, rejected: {rejected_frames})")
        else:
            print(f"  Window unchanged: {len(window_bMo)} frames (accepted: {accepted_frames}, rejected: {rejected_frames})")

        # ============ Compute bMo statistics from window ============
        if len(window_bMo) >= 3:
            bMo_median, bMo_weighted, bMo_ransac, ransac_inlier_mask, pos_threshold, rot_threshold = \
                compute_bMo_statistics(window_bMo, window_rotvecs, window_errors)

            print(f"\n  Window bMo statistics ({len(window_bMo)} frames):")
            print(f"    [Method 1] Median: {pose_to_euler_tvec(bMo_median)}")
            print(f"    [Method 2] Weighted: {pose_to_euler_tvec(bMo_weighted)}")
            print(f"    [Method 3] RANSAC ({ransac_inlier_mask.sum()}/{len(window_bMo)} inliers): {pose_to_euler_tvec(bMo_ransac)}")
            print(f"    RANSAC thresholds: position={pos_threshold:.1f} mm, rotation={rot_threshold:.3f} rad")
            print(f"    RANSAC inliers: {ransac_inlier_mask}"    )
            # Use RANSAC result as default
            bMo_ransac_default = bMo_ransac
            cMo_ransac_default = np.linalg.inv(eMc) @ np.linalg.inv(bMo_ransac_default) @ bMo_ransac_default @ eMc @ cMo
            cMo_ransac_default = np.linalg.inv(eMc) @ np.linalg.inv(bMo_ransac_default) @ bMe @ bMo_ransac_default
        else:
            print(f"\n  Window too small ({len(window_bMo)} frames), need at least 3")

        # ============ Visualization ============
        vis_img = img.copy()

        # Draw estimated axes from current frame (solid, RGB - standard)
        cv2.drawFrameAxes(vis_img, K, dist, cMo[:3, :3], cMo[:3, 3:], 40, 2)

        # Draw RANSAC-refined axes (dashed, white) if we have enough frames
        if len(window_bMo) >= 3:
            # Compute cMo from bMo_ransac and bMe
            cMo_vis = np.linalg.inv(eMc) @ np.linalg.inv(bMe) @ bMo_ransac
            draw_axes_dashed(vis_img, K, dist,
                             Rotation.from_matrix(cMo_vis[:3, :3]).as_rotvec(),
                             cMo_vis[:3, 3],
                             length=50, thickness=2,
                             color=(255, 255, 255))

        # Draw detected points with inlier/outlier coloring
        for i, (x, y) in enumerate(pts2d):
            if inlier_mask is not None and i < len(inlier_mask):
                if inlier_mask[i]:
                    color = (0, 255, 0)   # Green = inlier
                else:
                    color = (0, 0, 255)   # Red = outlier
            else:
                color = (255, 255, 0)     # Cyan = unknown/no mask
            cv2.circle(vis_img, (int(x), int(y)), 3, color, -1)

        # Draw text indicating frame status
        status_text = f"Frame {frame_id} | Accepted: {accepted_frames} | Rejected: {rejected_frames}"
        if frame_rejected:
            status_text += " [REJECTED]"
        cv2.putText(vis_img, status_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Draw window info
        window_text = f"Window: {len(window_bMo)}/{SLIDING_WINDOW_SIZE} | Method: RANSAC"
        cv2.putText(vis_img, window_text, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Crop and resize ROI
        vis_roi = vis_img[roi_y_min:roi_y_max, roi_x_min:roi_x_max]
        vis_roi = cv2.resize(vis_roi, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)

        # Save result
        save_path = os.path.join(RESULT_DIR, f"{ts}_refine.png")
        cv2.imwrite(save_path, vis_roi)

        cv2.imshow("Pose Refine", vis_roi)
        cv2.waitKey(1)

        # Store result for summary
        all_results.append({
            'frame': frame_id,
            'img_file': img_file,
            'ts': ts,
            'bMo': bMo.copy(),
            'bMo_ba': bMo_ba.copy(),
            'cMo': cMo.copy(),
            'cMo_ba': cMo_ba.copy(),
            'reproj_error': pnp_error,
            'frame_rejected': frame_rejected,
            'n_inliers': n_inliers,
            'window_size': len(window_bMo),
        })

        frame_id += 1

    # ============ Summary ============
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)

    total_frames = accepted_frames + rejected_frames
    print(f"\nTotal frames processed: {total_frames}")
    print(f"Accepted frames: {accepted_frames} ({100*accepted_frames/total_frames:.1f}%)")
    print(f"Rejected frames: {rejected_frames} ({100*rejected_frames/total_frames:.1f}%)")

    if window_bMo:
        bMo_median, bMo_weighted, bMo_ransac, ransac_inlier_mask, _, _ = \
            compute_bMo_statistics(window_bMo, window_rotvecs, window_errors)

        print(f"\nFinal window bMo ({len(window_bMo)} frames):")
        print(f"  [Method 1] Median: {pose_to_euler_tvec(bMo_median)}")
        print(f"  [Method 2] Weighted: {pose_to_euler_tvec(bMo_weighted)}")
        print(f"  [Method 3] RANSAC ({ransac_inlier_mask.sum()}/{len(window_bMo)} inliers): {pose_to_euler_tvec(bMo_ransac)}")

    print(f"\nPer-frame results:")
    print(f"{'Frame':<6} {'Rej':<4} {'Win':<4} {'Reproj(px)':<12} {'n_inliers':<10}")
    print("-" * 50)
    for r in all_results:
        rej = 'Y' if r['frame_rejected'] else 'N'
        print(f"{r['frame']:<6} {rej:<4} {r['window_size']:<4} {r['reproj_error']:<12.3f} {r['n_inliers']:<10}")

    cv.destroyAllWindows()
