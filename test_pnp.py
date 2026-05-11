"""
PnP Pose Estimation Test - Compare estimated pose with ground truth.
Tests all images in the given directory and compares bMo/cMo with ground truth.
"""

import cv2 as cv
from ultralytics import YOLO
import time
from gemiEd import *
from scipy.spatial.transform import Rotation
import json
import os
import numpy as np

# Load a model
model = YOLO("checkpoint/best.pt")  # load an official model

# ============ Config ============
DATA_DIR = "dataset/save_data3/20260511_120244"
RESULT_DIR = "result/save_data3/20260511_120244/test_pnp"

# =========== Read metadata config ===========
BEGIN_FRAME_ID = 1180  # Skip frames with frame_id < this value

# PnP threshold config
USE_ADAPTIVE_THRESHOLD = True  # True = adaptive, False = fixed
FIXED_ERROR_THRESHOLD = 0.7    # used when USE_ADAPTIVE_THRESHOLD = False
ADAPTIVE_MULTIPLIER = 2.0      # threshold = median_error * multiplier

# Frame rejection threshold - if mean reprojection error exceeds this, frame is rejected
FRAME_REJECT_THRESHOLD = 1.0   # pixels - if ALL points have error > this, reject frame

# Ground truth bMo_gt (base to object transform)
bMo_gt = np.array([
    [      0.133,   -0.087641,    -0.98723,      1025.3],
    [    -0.9908,    0.013491,    -0.13467,     -299.64],
    [   0.025122,     0.99606,    -0.08504,      539.07],
    [          0,           0,           0,           1]
], dtype=np.float64)

# Camera on robot end-effector (eye-to-hand extrinsic)
eMc = np.array([
    [-7.2267956e-01, 6.9102561e-01, -1.4759262e-02, -5.1758522e+01],
    [-6.9116789e-01, -7.2264087e-01, 8.7790741e-03, 6.0040222e+01],
    [-4.5990809e-03, 1.6545586e-02, 9.9985254e-01, 9.7955963e+01],
    [0.0000000e+00, 0.0000000e+00, 0.0000000e+00, 1.0000000e+00]
], dtype=np.float64)

# Camera intrinsics
K = np.array([
    [2674.7629874104787, 0., 1279.5],
    [0., 2674.7629874104787, 719.5],
    [0., 0., 1.]
], dtype=np.float64)

dist = np.array([-0.11744968686298927, 0.27089153364253454, 0.0012180578884344092,
                 0.00067320963008635703, -0.078845410108757258], dtype=np.float64)

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


def two_round_pnp(pts2d, pts3d, K, dist, error_threshold=0.5):
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


def compute_cMo_from_bMo(bMo, bMe, eMc):
    """Compute cMo from bMo, bMe (robot pose), and eMc (eye-to-hand)
    bMo = bMe @ eMc @ cMo
    => cMo = eMc^-1 @ bMe^-1 @ bMo
    """
    return np.linalg.inv(eMc) @ np.linalg.inv(bMe) @ bMo


def draw_axes_dashed(img, K, dist, rvec, tvec, length, thickness, color):
    """Draw coordinate axes with dashed lines for ground truth"""
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

    # Draw dashed lines
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
    import json

    os.makedirs(RESULT_DIR, exist_ok=True)

    # Load metadata from JSON
    meta_path = os.path.join(DATA_DIR, "metadata.json")
    with open(meta_path, 'r') as f:
        metadata = json.load(f)

    records = metadata['records']
    print("=" * 80)
    print("PnP Pose Estimation Test - vs Ground Truth")
    print("=" * 80)
    print(f"\nbMo_gt:\n{bMo_gt}\n")
    print(f"Loaded {len(records)} frames from {meta_path}")

    frame_id = 0
    all_results = []
    last_timestamp_ns = None

    for record in records:
        # Skip frames before BEGIN_FRAME_ID
        frame_id_val = record['frame_id']
        if frame_id_val < BEGIN_FRAME_ID :
            continue
        # Skip frames with large time diff
        time_diff = record['time_diff_ns']/1e9
        if time_diff > 0.05:
            continue
        # Get current frame timestamp
        current_timestamp_ns = record['camera_timestamp_ns']

        # Check time interval with previous frame
        if last_timestamp_ns is not None:
            time_interval_s = (current_timestamp_ns - last_timestamp_ns) / 1e9
            if time_interval_s < 0.1:
                continue
            elif time_interval_s > 1.0:
                print(f"[WARN] Large timestamp gap: {time_interval_s:.2f}s between consecutive frames")

        # Update last timestamp after potential wait
        last_timestamp_ns = record['camera_timestamp_ns']

        img_relative_path = record['image_path']
        img_path = os.path.join(DATA_DIR, img_relative_path)

        # Extract robot pose from JSON
        bMe = np.array(record['pose_matrix_4x4']).reshape(4, 4)

        print("\n" + "-" * 80)
        print(f"Frame {frame_id}: frame_{frame_id_val:06d}.jpg")
        print("-" * 80)

        # Load image
        img = cv2.imread(img_path)

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
        print(f'  Found {len(final_pts)} points')

        if final_pts is None or centers.shape[0] < 4:
            print(f"  Not enough points, skipping")
            continue

        pts3d = matcher.obj_pts[matcher.r_idx]
        pts2d = centers

        # ============ Two-round PnP estimation ============
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
        if all_points_exceed:
            print(f"  [REJECT] All points exceed threshold {FRAME_REJECT_THRESHOLD} px, frame rejected")
            continue

        # ============ Build cMo from two-round PnP result ============
        cMo = np.eye(4)
        cMo[:3, :3] = Rotation.from_rotvec(rvec).as_matrix()
        cMo[:3, 3] = tvec

        # Compute bMo from two-round PnP
        bMo = bMe @ eMc @ cMo

        # Compute bMo by SOLVEPNP_ITERATIVE for comparison (using inlier points)
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

        # Compute cMo_gt from bMo_gt
        cMo_gt = compute_cMo_from_bMo(bMo_gt, bMe, eMc)

        # ============ Compute errors ============
        # Position error
        pos_error = np.linalg.norm(bMo[:3, 3] - bMo_gt[:3, 3])
        pos_error_ba = np.linalg.norm(bMo_ba[:3, 3] - bMo_gt[:3, 3])

        # Rotation error (geodesic distance on SO(3))
        R_est = bMo[:3, :3]
        R_gt = bMo_gt[:3, :3]
        R_error = Rotation.from_matrix(R_est).inv() * Rotation.from_matrix(R_gt)
        rot_error_vec = R_error.as_rotvec()
        rot_error_deg = np.linalg.norm(rot_error_vec) * 180 / np.pi

        R_est_ba = bMo_ba[:3, :3]
        R_error_ba = Rotation.from_matrix(R_est_ba).inv() * Rotation.from_matrix(R_gt)
        rot_error_vec_ba = R_error_ba.as_rotvec()
        rot_error_deg_ba = np.linalg.norm(rot_error_vec_ba) * 180 / np.pi

        # Reprojection error for estimated pose
        pnp_error = per_point_errors.mean()
        pnp_error_ba = compute_reproj_error(pts3d_inlier, rvec_ba, tvec_ba, pts2d_inlier, K, dist)

        print(f"\n  Estimated bMo_pnp: {pose_to_euler_tvec(bMo)}")
        print(f"  Estimated bMo_ba:  {pose_to_euler_tvec(bMo_ba)}")
        print(f"  GT bMo:          {pose_to_euler_tvec(bMo_gt)}")
        print(f"  Position error (pnp): {pos_error:.2f} mm, (ba): {pos_error_ba:.2f} mm")
        print(f"  Rotation error (pnp): {rot_error_deg:.3f} deg, (ba): {rot_error_deg_ba:.3f} deg")

        print(f"\n  Estimated cMo_pnp: {pose_to_euler_tvec(cMo)}")
        print(f"  Estimated cMo_ba:  {pose_to_euler_tvec(cMo_ba)}")
        print(f"  GT cMo:          {pose_to_euler_tvec(cMo_gt)}")

        print(f"\n  PnP reprojection error: {pnp_error:.4f} (round1: {round1_error:.4f})")
        print(f"  PnP reprojection error (BA, {n_inliers} inliers): {pnp_error_ba:.4f}")

        # Store result
        all_results.append({
            'frame': frame_id,
            'frame_id_val': frame_id_val,
            'bMo': bMo.copy(),
            'bMo_ba': bMo_ba.copy(),
            'cMo': cMo.copy(),
            'cMo_ba': cMo_ba.copy(),
            'pos_error': pos_error,
            'pos_error_ba': pos_error_ba,
            'rot_error_deg': rot_error_deg,
            'rot_error_deg_ba': rot_error_deg_ba,
            'reproj_error': pnp_error,
            'reproj_error_ba': pnp_error_ba,
            'n_inliers': n_inliers,
            'pts3d': pts3d,
            'pts2d': pts2d,
            'rvec': rvec,
            'tvec': tvec,
            'inlier_mask': inlier_mask,
            'rvec_gt': Rotation.from_matrix(cMo_gt[:3, :3]).as_rotvec(),
            'tvec_gt': cMo_gt[:3, 3],
        })

        # ============ Visualization ============
        vis_img = img.copy()

        # Draw ground truth axes (dashed, white/gray)
        draw_axes_dashed(vis_img, K, dist,
                         Rotation.from_matrix(cMo_gt[:3, :3]).as_rotvec(),
                         cMo_gt[:3, 3],
                         length=50, thickness=2,
                         color=(200, 200, 200))

        # Draw estimated BA axes (solid, yellow/magenta/cyan - different from standard RGB)
        cv2.drawFrameAxes(vis_img, K, dist, cMo_ba[:3, :3], cMo_ba[:3, 3:], 35, 2)

        # Draw estimated PnP axes (solid, colored RGB)
        cv2.drawFrameAxes(vis_img, K, dist, cMo[:3, :3], cMo[:3, 3:], 40, 2)

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
            cv2.putText(vis_img, str(i), (int(x)+5, int(y)-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # Crop and resize ROI
        vis_roi = vis_img[roi_y_min:roi_y_max, roi_x_min:roi_x_max]
        vis_roi = cv2.resize(vis_roi, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)

        # Save result
        save_path = os.path.join(RESULT_DIR, f"frame_{frame_id_val:06d}_pnp_test.png")
        cv2.imwrite(save_path, vis_roi)

        cv2.imshow("PnP Test", vis_roi)
        cv2.waitKey(1)

        frame_id += 1

    # ============ Summary ============
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    if all_results:
        pos_errors = [r['pos_error'] for r in all_results]
        pos_errors_ba = [r['pos_error_ba'] for r in all_results]
        rot_errors = [r['rot_error_deg'] for r in all_results]
        rot_errors_ba = [r['rot_error_deg_ba'] for r in all_results]
        reproj_errors = [r['reproj_error'] for r in all_results]
        reproj_errors_ba = [r['reproj_error_ba'] for r in all_results]

        print(f"\nFrames processed: {len(all_results)}")
        print(f"PnP  - Position error: mean={np.mean(pos_errors):.2f} mm, max={np.max(pos_errors):.2f} mm, min={np.min(pos_errors):.2f} mm")
        print(f"PnP  - Rotation error: mean={np.mean(rot_errors):.3f} deg, max={np.max(rot_errors):.3f} deg, min={np.min(rot_errors):.3f} deg")
        print(f"PnP  - Reprojection error: mean={np.mean(reproj_errors):.3f} px, max={np.max(reproj_errors):.3f} px, min={np.min(reproj_errors):.3f} px")
        print(f"BA   - Position error: mean={np.mean(pos_errors_ba):.2f} mm, max={np.max(pos_errors_ba):.2f} mm, min={np.min(pos_errors_ba):.2f} mm")
        print(f"BA   - Rotation error: mean={np.mean(rot_errors_ba):.3f} deg, max={np.max(rot_errors_ba):.3f} deg, min={np.min(rot_errors_ba):.3f} deg")
        print(f"BA   - Reprojection error: mean={np.mean(reproj_errors_ba):.3f} px, max={np.max(reproj_errors_ba):.3f} px, min={np.min(reproj_errors_ba):.3f} px")

        print(f"\nPer-frame results:")
        print(f"{'Frame':<6} {'Pos(pnp)':<10} {'Pos(ba)':<10} {'Rot(pnp)':<10} {'Rot(ba)':<10} {'Rep(pnp)':<10} {'Rep(ba)':<10} {'Inliers':<8}")
        print("-" * 80)
        for r in all_results:
            print(f"{r['frame']:<6} {r['pos_error']:<10.2f} {r['pos_error_ba']:<10.2f} "
                  f"{r['rot_error_deg']:<10.3f} {r['rot_error_deg_ba']:<10.3f} "
                  f"{r['reproj_error']:<10.3f} {r['reproj_error_ba']:<10.3f} {r['n_inliers']:<8}")

    cv.destroyAllWindows()
