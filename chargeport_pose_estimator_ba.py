import cv2
from ultralytics import YOLO
import time
from scipy.spatial.transform import Rotation
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from static_pose_optimizer_ba import StaticPoseOptimizer, pose_to_euler_tvec

# Load a model
model = YOLO("checkpoint/best.pt")  # load an official model

# ============ Config ============
DATA_DIR = "dataset/save_data3/20260511_120244"
RESULT_DIR = "result/save_data3/20260511_120244/pose_estimation_ba"
SAVE_DIR = "dataset/save_data3/chb_20260511_120244"
# DATA_DIR = "dataset/save_data3/20260511_120538"
# RESULT_DIR = "result/save_data3/20260511_120538/pose_estimation_ba"
# SAVE_DIR = "dataset/save_data3/chb_20260511_120538"
SLIDING_WINDOW_SIZE = 8
MAX_FRAMES = -1  # Limit frames for quick test, -1 for all frames

# ============ 数据记录列表 ============
frame_records = []  # 每帧的中间数据和结果列表
pnp_records = []   # PnP结果列表
optimize_records = []  # 优化结果列表


# =========== Read metadata config ===========
BEGIN_FRAME_ID = 1180  # Skip frames with frame_id < this value
PROCESSED_INTERVAL = 0.1
# PnP threshold config
USE_ADAPTIVE_THRESHOLD = True  # True = adaptive, False = fixed
FIXED_ERROR_THRESHOLD = 0.4    # used when USE_ADAPTIVE_THRESHOLD = False
ADAPTIVE_MULTIPLIER = 2.0      # threshold = median_error * multiplier

# Frame rejection threshold - if ALL points have error > this, skip
FRAME_REJECT_THRESHOLD = 2.0   # pixels

# Camera on robot end-effector (eye-to-hand extrinsic)
eMc = np.array([
    [-7.2267956e-01,  6.9102561e-01, -1.4759262e-02 ,-5.1758522e+01],
    [-6.9116789e-01, -7.2264087e-01,  8.7790741e-03,  6.0040222e+01],
    [-4.5990809e-03,  1.6545586e-02,  9.9985254e-01,  9.7955963e+01],
    [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  1.0000000e+00]
    ], dtype=np.float64)
# camera intrinsics
K = np.array([
        [2674.7629874104787,0.,1279.5],
        [0.,2674.7629874104787,719.5],
        [0.,0.,1.]
        ], dtype=np.float64)
dist = np.array([-0.11744968686298927,0.27089153364253454,0.0012180578884344092,0.00067320963008635703,-0.078845410108757258], dtype=np.float64)
# 3D object points in object frame (charge port keypoints)
obj_pts = np.array([
            [-8.0, 11.2, 0.0], [8.0, 11.2, 0.0],
            [-16.0, 0.0, 0.0], [0.0, 0.0, 0.0], [16.0, 0.0, 0.0],
            [-8.0, -13.9, 0.0], [8.0, -13.9, 0.0]
        ], dtype=np.float64)

# Optimization weight configuration
PRIOR_SIGMA = np.array([
    np.deg2rad(5.0),
    np.deg2rad(5.0),
    np.deg2rad(1.0),
    0.5,
    0.5,
    10.0
], dtype=np.float64)

POINT_SIGMAS = np.array([
    1.5,  # top-left small hole
    1.5,  # top-right small hole
    1.0,  # mid-left large hole
    1.0,  # center large hole
    1.0,  # mid-right large hole
    1.0,  # bottom-left large hole
    1.0   # bottom-right large hole
], dtype=np.float64)


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
    return np.linalg.norm(proj.reshape(-1,2) - pts2d, axis=1).mean()


def compute_per_point_reproj_errors(pts3d, rvec, tvec, pts2d, K, dist):
    """Compute reprojection error for each point (in pixels)"""
    proj, _ = cv2.projectPoints(pts3d, rvec, tvec, K, dist)
    return np.linalg.norm(proj.reshape(-1,2) - pts2d, axis=1)


def two_round_pnp(pts2d, pts3d, K, dist, error_threshold=0.4):
    """Two-round PnP with adaptive or fixed threshold.

    Returns:
        rvec, tvec, valid, inlier_mask, per_point_errors, round1_errors, used_threshold
    """
    rvec1, tvec1, valid1 = solvePnP_IPPE(pts2d, pts3d, K, dist)
    if not valid1:
        return None, None, False, None, None, None, None

    per_point_errors = compute_per_point_reproj_errors(pts3d, rvec1, tvec1, pts2d, K, dist)
    round1_error = per_point_errors.mean()

    if USE_ADAPTIVE_THRESHOLD:
        median_error = np.median(per_point_errors)
        current_threshold = median_error * ADAPTIVE_MULTIPLIER
        current_threshold = max(current_threshold, FIXED_ERROR_THRESHOLD)
    else:
        current_threshold = FIXED_ERROR_THRESHOLD

    inlier_mask = per_point_errors < current_threshold
    n_inliers = inlier_mask.sum()

    if n_inliers < 4:
        current_threshold = current_threshold * 2
        inlier_mask = per_point_errors < current_threshold
        n_inliers = inlier_mask.sum()

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
        return False, f"Rotation det={det_R:.4f} (reflection/flip)"

    tvec = cMo[:3, 3]
    if tvec[2] <= 0:
        return False, f"Object behind camera (z={tvec[2]:.2f})"

    dist_norm = np.linalg.norm(tvec)
    if dist_norm < 50 or dist_norm > 3000:
        return False, f"Object distance={dist_norm:.1f}mm (unreasonable)"

    return True, "ok"


def getInferResult(model, img):
    results = model(img)
    if len(results) == 0:
        return []
    return results[0].boxes.xyxy.cpu().numpy()


if __name__ == '__main__':
    import json
    from gemiEd import *

    os.makedirs(RESULT_DIR, exist_ok=True)

    optimizer = StaticPoseOptimizer(K, dist, prior_sigma=PRIOR_SIGMA, point_sigmas=POINT_SIGMAS)
    optimizer.set_extrinsics(eMc)
    optimizer.set_object_pts(obj_pts)

    meta_path = os.path.join(DATA_DIR, "metadata.json")
    with open(meta_path, 'r') as f:
        metadata = json.load(f)

    records = metadata['records']
    print(f"Loaded {len(records)} frames from {meta_path}")

    frame_id = 1
    last_bMo = None
    processed_frames = 0
    last_timestamp_ns = None

    for record in records:
        frame_id_val = record['frame_id']
        if frame_id_val < BEGIN_FRAME_ID:
            continue
        time_diff = record['time_diff_ns']/1e9
        if time_diff > 0.05:
            continue
        if MAX_FRAMES > 0 and processed_frames >= MAX_FRAMES:
            print(f"\nReached max frames limit ({MAX_FRAMES}), stopping...")
            break
        processed_frames += 1

        current_timestamp_ns = record['camera_timestamp_ns']
        if last_timestamp_ns is not None:
            time_diff_s = (current_timestamp_ns - last_timestamp_ns) / 1e9
            if time_diff_s < PROCESSED_INTERVAL:
                continue
        last_timestamp_ns = record['camera_timestamp_ns']

        img_relative_path = record['image_path']
        img_path = os.path.join(DATA_DIR, img_relative_path)
        robot_pose = np.array(record['pose_matrix_4x4']).reshape(4, 4)

        print("\n==============================================")
        print(f"Processing frame {frame_id}: frame_{frame_id_val:06d}.jpg")

        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to read image: {img_path}")
            continue

        img_float = img.astype(np.float32)
        img_bright = img_float - 50
        img_bright = np.clip(img_bright, 0, 255).astype(np.uint8)
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

        detector = pyced.CED(np.ascontiguousarray(roi))
        detector.run_CED()
        rotRects = detector.getEllipsesAfterCluster()
        ellipses_ = []

        for e in rotRects:
            ellipses_.append((*e.center, e.size[0]/2, e.size[1]/2, e.angle))

        matcher = UltimateSocketMatcher()
        matcher.obj_pts = obj_pts
        matcher.K = K
        matcher.dist = dist
        matcher.eMc = eMc

        vis_ellipse = draw_ellipse(roi, ellipses_)

        final_pts, status, centers = matcher.solve(ellipses_, [*(result[0][:2]), *(result[0][2:]-result[0][:2])])
        if final_pts is not None:
            print(f'find {len(final_pts)} points')
        else:
            print('find 0 points')

        inlier_mask = None

        if final_pts is not None and centers.shape[0] >= 7:
            pts3d = matcher.obj_pts[matcher.r_idx]
            pts2d = centers

            rvec, tvec, valid, inlier_mask, per_point_errors_pnp, round1_error, used_threshold = two_round_pnp(
                pts2d, pts3d, K, dist, error_threshold=FIXED_ERROR_THRESHOLD)

            cMo = np.eye(4)
            cMo[:3, :3] = Rotation.from_rotvec(rvec).as_matrix()
            cMo[:3, 3] = tvec

            rvec = np.array(rvec).tolist()
            tvec = np.array(tvec).tolist()

            if not optimizer.is_initialized() and not valid:
                print(f"Frame:{frame_id} frame_{frame_id_val:06d}: PnP failed, skipping pose estimation")
                continue

            n_inliers = inlier_mask.sum() if inlier_mask is not None else 0
            pnp_error = per_point_errors_pnp.mean()

            if last_bMo is not None and optimizer.is_initialized():
                cMo_before = np.linalg.inv(eMc) @ np.linalg.inv(robot_pose) @ last_bMo
                rvec_before = Rotation.from_matrix(cMo_before[:3, :3]).as_rotvec()
                tvec_before = cMo_before[:3, 3]
                per_point_errors_before = compute_per_point_reproj_errors(pts3d, rvec_before, tvec_before, pts2d, K, dist)
                print(f"Per-point reprojection errors of last optimized pose (px):")
                obj_pt_names = ['L-top', 'R-top', 'L-mid', 'center', 'R-mid', 'L-bot', 'R-bot']
                for i, (err, pt_name) in enumerate(zip(per_point_errors_before, obj_pt_names[:len(per_point_errors_before)])):
                    print(f"  Point {i} ({pt_name}): {err:7.3f} px")
                pos_diff = np.linalg.norm(tvec - tvec_before)
                rot_mat_diff = cMo[:3, :3] @ cMo_before[:3, :3].T
                rot_vec_diff = Rotation.from_matrix(rot_mat_diff).as_rotvec()
                rot_diff = np.linalg.norm(rot_vec_diff) * 180 / np.pi
                if pos_diff > 20 or rot_diff > 20:
                    print(f"  [WARN] Large pose difference between PnP and last optimized pose: pos_diff={pos_diff:.1f}mm, rot_diff={rot_diff:.1f}deg")
                    continue

            all_points_exceed = np.all(per_point_errors_pnp > FRAME_REJECT_THRESHOLD)
            if all_points_exceed:
                print(f"  [REJECT] All points exceed threshold {FRAME_REJECT_THRESHOLD} px, frame rejected")
                continue

            bMo_init = robot_pose @ eMc @ cMo
            if not optimizer.is_initialized():
                optimizer.set_initial_pose(bMo_init)

            if optimizer.get_frame_count() >= SLIDING_WINDOW_SIZE:
                optimizer.remove_oldest_frame()

            # # save dataset for chb
            # # centers
            # coords = []
            # for cc in pts2d:
            #     if cc is not None:
            #         coords.append(f"{cc[0]:.4f} {cc[1]:.4f}")
            #     else:
            #         coords.append("-1 -1")
            # txt_path = os.path.join(SAVE_DIR, 'data', f"{frame_id_val}.txt")
            # with open(txt_path, 'w') as f:
            #     f.write(' '.join(coords))
            # # robot poses
            # save_pose_path = os.path.join(SAVE_DIR, 'data', f"{frame_id_val}.npy")
            # np.save(save_pose_path, robot_pose)
            # # images
            # save_img_path = os.path.join(SAVE_DIR, 'image', f"{frame_id_val}.png")
            # cv2.imwrite(save_img_path, img)

            # Add frame with per_point_errors_pnp for dynamic weighting in optimizer
            optimizer.add_frame(frame_id, robot_pose, pts2d, pts3d, per_point_errors_pnp=per_point_errors_pnp)
            result_optimized = optimizer.optimize()

            bMo_optimized = optimizer.get_pose()
            last_bMo = bMo_optimized

            cMo_optimized = optimizer.compute_cMo(robot_pose, frame_id)
            rvec_optimized = Rotation.from_matrix(cMo_optimized[:3, :3]).as_rotvec()
            tvec_optimized = cMo_optimized[:3, 3]

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
            bMo_ba = robot_pose @ eMc @ cMo_ba

            print(f"bMo_optimized: {pose_to_euler_tvec(bMo_optimized)}")
            print(f"bMo_pnp: {pose_to_euler_tvec(bMo_init)}")
            print(f"bMo_ba: {pose_to_euler_tvec(bMo_ba)}")
            print(f"cMo_optimized: {pose_to_euler_tvec(cMo_optimized)}")
            print(f"cMo_pnp: {pose_to_euler_tvec(cMo)}")
            print(f"cMo_ba: {pose_to_euler_tvec(cMo_ba)}")

            pnp_error_ba = compute_reproj_error(pts3d_inlier, rvec_ba, tvec_ba, pts2d_inlier, K, dist)
            print(f"PnP reprojection error: {pnp_error:.4f} (round1: {round1_error:.4f})")
            print(f"PnP reprojection error (BA, {n_inliers} inliers): {pnp_error_ba:.4f}")

            now_error, _ = optimizer.get_frame_error(frame_id)
            print(f"Optimized reprojection error: {now_error:.4f}")
            ave_error = optimizer.get_average_error()
            print(f"Average reprojection error: {ave_error:.4f}")

            bMo_euler, bMo_tvec = pose_to_euler_tvec(bMo_optimized)
            cMo_euler, cMo_tvec = pose_to_euler_tvec(cMo_optimized)

            pnp_record = {
                'frame_id': frame_id_val,
                'frame_idx': frame_id,
                'timestamp_ns': int(current_timestamp_ns),
                'robot_pose': robot_pose.tolist(),
                'n_points': len(pts2d),
                'n_inliers': int(n_inliers),
                'used_threshold': float(used_threshold),
                'pnp_error_round1': float(round1_error),
                'pnp_error_final': float(pnp_error),
                'pnp_error_ba': float(pnp_error_ba),
                'per_point_errors': per_point_errors_pnp.tolist(),
                'inlier_mask': inlier_mask.tolist() if inlier_mask is not None else None,
                'rvec': np.array(rvec).tolist(),
                'tvec': np.array(tvec).tolist(),
                'cMo_euler': cMo_euler.tolist(),
                'cMo_tvec': cMo_tvec.tolist(),
                'bMo_pnp_euler': pose_to_euler_tvec(bMo_init)[0].tolist(),
                'bMo_pnp_tvec': pose_to_euler_tvec(bMo_init)[1].tolist(),
            }
            pnp_records.append(pnp_record)

            opt_record = {
                'frame_id': frame_id_val,
                'frame_idx': frame_id,
                'timestamp_ns': current_timestamp_ns,
                'bMo_euler': bMo_euler.tolist(),
                'bMo_tvec': bMo_tvec.tolist(),
                'cMo_euler': cMo_euler.tolist(),
                'cMo_tvec': cMo_tvec.tolist(),
                'n_frames_in_optimizer': optimizer.get_frame_count(),
                'frame_error': float(now_error),
                'avg_error': float(ave_error) if ave_error is not None else None,
            }
            optimize_records.append(opt_record)

            frame_record = {
                'frame_id': frame_id_val,
                'frame_idx': frame_id,
                'timestamp_ns': int(current_timestamp_ns),
                'robot_pose': robot_pose.tolist(),
                'pts2d': pts2d.tolist(),
                'pts3d': pts3d.tolist(),
                'bMo_init': bMo_init.tolist(),
                'bMo_optimized': bMo_optimized.tolist(),
                'cMo': cMo.tolist(),
                'cMo_optimized': cMo_optimized.tolist(),
                'pnp_error': float(pnp_error),
                'optimized_error': float(now_error),
                'avg_error': float(ave_error) if ave_error is not None else None,
                'per_point_errors_pnp': per_point_errors_pnp.tolist(),
                'inlier_mask': inlier_mask.tolist() if inlier_mask is not None else None,
            }
            frame_records.append(frame_record)

            cv2.drawFrameAxes(img, K, dist, cMo_optimized[:3, :3], cMo_optimized[:3, 3:], 10, 3)
            cv2.drawFrameAxes(img, K, dist, cMo[:3, :3], cMo[:3, 3:], 20, 1)

            frame_id += 1

            proj, _ = cv2.projectPoints(pts3d, rvec_optimized, tvec_optimized, K, dist)
            if centers is not None:
                for i, (x, y) in enumerate(centers):
                    if inlier_mask is not None and i < len(inlier_mask):
                        color = (0, 255, 0) if inlier_mask[i] else (0, 0, 255)
                    else:
                        color = (255, 255, 0)
                    # Detected point: hollow circle with edge (green=inlier, red=outlier, yellow=unknown)
                    cv2.circle(img, (int(x), int(y)), 3, color, 1)
                    x_proj, y_proj = proj[i][0]
                    # Projected point: cross marker (blue)
                    cv2.drawMarker(img, (int(x_proj), int(y_proj)), (255, 0, 0), cv2.MARKER_CROSS, 5, 1)
                    # Connection line
                    cv2.line(img, (int(x), int(y)), (int(x_proj), int(y_proj)), (0, 255, 0), 1)
                    # Point index label
                    cv2.putText(img, str(i), (int(x)-8, int(y)-8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            pad = 50
            vis_result = img[roi_y_min-pad:roi_y_max+pad, roi_x_min-pad:roi_x_max+pad]
            vis_result = cv2.resize(vis_result, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
            vis_result_path = os.path.join(RESULT_DIR, f"frame_{frame_id_val:06d}_vis_result.png")
            cv2.imwrite(vis_result_path, vis_result)
            cv2.waitKey(1)

    cv2.destroyAllWindows()

    # ============ 保存 JSON 记录 ============
    frame_records_path = os.path.join(RESULT_DIR, "frame_records.json")
    with open(frame_records_path, 'w') as f:
        json.dump(frame_records, f, indent=2)
    print(f"\nSaved frame_records to {frame_records_path}")

    pnp_records_path = os.path.join(RESULT_DIR, "pnp_records.json")
    with open(pnp_records_path, 'w') as f:
        json.dump(pnp_records, f, indent=2)
    print(f"Saved pnp_records to {pnp_records_path}")

    optimize_records_path = os.path.join(RESULT_DIR, "optimize_records.json")
    with open(optimize_records_path, 'w') as f:
        json.dump(optimize_records, f, indent=2)
    print(f"Saved optimize_records to {optimize_records_path}")

    # ============ 保存 CSV 记录 ============
    if pnp_records:
        pnp_df = pd.DataFrame(pnp_records)
        pnp_csv_path = os.path.join(RESULT_DIR, "pnp_results.csv")
        pnp_df.to_csv(pnp_csv_path, index=False)
        print(f"Saved pnp_results CSV to {pnp_csv_path}")

    if optimize_records:
        opt_df = pd.DataFrame(optimize_records)
        opt_csv_path = os.path.join(RESULT_DIR, "optimize_results.csv")
        opt_df.to_csv(opt_csv_path, index=False)
        print(f"Saved optimize_results CSV to {opt_csv_path}")

    print("\n" + "=" * 80)
    print("PnP Results Summary")
    print("=" * 80)
    print(f"{'FrameID':>8} {'Pts':>4} {'Inliers':>7} {'Thresh':>8} {'Rnd1Err':>8} {'PnPErr':>8} {'BAErr':>8}")
    print("-" * 80)
    for rec in pnp_records:
        print(f"{rec['frame_id']:>8} {rec['n_points']:>4} {rec['n_inliers']:>7} "
              f"{rec['used_threshold']:>8.3f} {rec['pnp_error_round1']:>8.4f} "
              f"{rec['pnp_error_final']:>8.4f} {rec['pnp_error_ba']:>8.4f}")
    print("-" * 80)

    print("\n" + "=" * 120)
    print("Optimization Results Summary")
    print("=" * 120)
    print(f"{'FrameID':>8} {'Frames':>6} {'bMo_t_x':>10} {'bMo_t_y':>10} {'bMo_t_z':>10} "
          f"{'bMo_rx':>8} {'bMo_ry':>8} {'bMo_rz':>8} "
          f"{'cMo_rx':>8} {'cMo_ry':>8} {'cMo_rz':>8} "
          f"{'FrErr':>8} {'AvgErr':>8}")
    print("-" * 120)
    for rec in optimize_records:
        print(f"{rec['frame_id']:>8} {rec['n_frames_in_optimizer']:>6} "
              f"{rec['bMo_tvec'][0]:>10.2f} {rec['bMo_tvec'][1]:>10.2f} {rec['bMo_tvec'][2]:>10.2f} "
              f"{rec['bMo_euler'][0]:>8.2f} {rec['bMo_euler'][1]:>8.2f} {rec['bMo_euler'][2]:>8.2f} "
              f"{rec['cMo_euler'][0]:>8.2f} {rec['cMo_euler'][1]:>8.2f} {rec['cMo_euler'][2]:>8.2f} "
              f"{rec['frame_error']:>8.4f} {rec['avg_error'] if rec['avg_error'] is not None else 0:>8.4f}")
    print("=" * 120)

    # ============ 绘制各分量随Frame的变化 ============
    def unwrap_angles(angles):
        """ unwrap 角度, 避免 -180/180 跳变 """
        angles = np.array(angles)
        for i in range(1, len(angles)):
            diff = angles[i] - angles[i-1]
            if diff > 180:
                angles[i:] -= 360
            elif diff < -180:
                angles[i:] += 360
        return angles

    if len(frame_records) > 0:
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        fig.suptitle('bMo, cMo, bMe Translation Components vs Frame', fontsize=14)

        frames = [r['frame_id'] for r in frame_records]

        # bMo x, y, z (Optimized vs PnP)
        ax = axes[0, 0]
        ax.plot(frames, [r['bMo_optimized'][0][3] for r in frame_records], 'r-', label='Optimized x')
        ax.plot(frames, [r['bMo_init'][0][3] for r in frame_records], 'r--', label='PnP x', alpha=0.7)
        ax.set_ylabel('X (mm)')
        ax.set_title('bMo X')
        ax.grid(True)
        ax.legend()

        ax = axes[0, 1]
        ax.plot(frames, [r['bMo_optimized'][1][3] for r in frame_records], 'g-', label='Optimized y')
        ax.plot(frames, [r['bMo_init'][1][3] for r in frame_records], 'g--', label='PnP y', alpha=0.7)
        ax.set_ylabel('Y (mm)')
        ax.set_title('bMo Y')
        ax.grid(True)
        ax.legend()

        ax = axes[0, 2]
        ax.plot(frames, [r['bMo_optimized'][2][3] for r in frame_records], 'b-', label='Optimized z')
        ax.plot(frames, [r['bMo_init'][2][3] for r in frame_records], 'b--', label='PnP z', alpha=0.7)
        ax.set_ylabel('Z (mm)')
        ax.set_title('bMo Z')
        ax.grid(True)
        ax.legend()

        # cMo x, y, z (Optimized vs PnP)
        ax = axes[1, 0]
        ax.plot(frames, [r['cMo_optimized'][0][3] for r in frame_records], 'r-', label='Optimized x')
        ax.plot(frames, [r['cMo'][0][3] for r in frame_records], 'r--', label='PnP x', alpha=0.7)
        ax.set_ylabel('X (mm)')
        ax.set_title('cMo X')
        ax.grid(True)
        ax.legend()

        ax = axes[1, 1]
        ax.plot(frames, [r['cMo_optimized'][1][3] for r in frame_records], 'g-', label='Optimized y')
        ax.plot(frames, [r['cMo'][1][3] for r in frame_records], 'g--', label='PnP y', alpha=0.7)
        ax.set_ylabel('Y (mm)')
        ax.set_title('cMo Y')
        ax.grid(True)
        ax.legend()

        ax = axes[1, 2]
        ax.plot(frames, [r['cMo_optimized'][2][3] for r in frame_records], 'b-', label='Optimized z')
        ax.plot(frames, [r['cMo'][2][3] for r in frame_records], 'b--', label='PnP z', alpha=0.7)
        ax.set_ylabel('Z (mm)')
        ax.set_title('cMo Z')
        ax.grid(True)
        ax.legend()

        # bMe x, y, z (robot_pose)
        ax = axes[2, 0]
        ax.plot(frames, [r['robot_pose'][0][3] for r in frame_records], 'r-', label='x')
        ax.set_xlabel('Frame ID')
        ax.set_ylabel('X (mm)')
        ax.set_title('bMe X')
        ax.grid(True)
        ax.legend()

        ax = axes[2, 1]
        ax.plot(frames, [r['robot_pose'][1][3] for r in frame_records], 'g-', label='y')
        ax.set_xlabel('Frame ID')
        ax.set_ylabel('Y (mm)')
        ax.set_title('bMe Y')
        ax.grid(True)
        ax.legend()

        ax = axes[2, 2]
        ax.plot(frames, [r['robot_pose'][2][3] for r in frame_records], 'b-', label='z')
        ax.set_xlabel('Frame ID')
        ax.set_ylabel('Z (mm)')
        ax.set_title('bMe Z')
        ax.grid(True)
        ax.legend()

        plt.tight_layout()
        pose_plot_path = os.path.join(RESULT_DIR, "pose_components_vs_frame.png")
        plt.savefig(pose_plot_path, dpi=150)
        print(f"Saved pose plot to {pose_plot_path}")
        plt.close()

    # ============ 绘制欧拉角随Frame的变化 ============
    if len(frame_records) > 0:
        frames = [r['frame_id'] for r in frame_records]

        # bMo 欧拉角 (optimized)
        bMo_euler_x = [pose_to_euler_tvec(np.array(r['bMo_optimized']))[0][0] for r in frame_records]
        bMo_euler_y = [pose_to_euler_tvec(np.array(r['bMo_optimized']))[0][1] for r in frame_records]
        bMo_euler_z = [pose_to_euler_tvec(np.array(r['bMo_optimized']))[0][2] for r in frame_records]

        # bMo 欧拉角 (PnP)
        bMo_pnp_euler_x = [pose_to_euler_tvec(np.array(r['bMo_init']))[0][0] for r in frame_records]
        bMo_pnp_euler_y = [pose_to_euler_tvec(np.array(r['bMo_init']))[0][1] for r in frame_records]
        bMo_pnp_euler_z = [pose_to_euler_tvec(np.array(r['bMo_init']))[0][2] for r in frame_records]

        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        fig.suptitle('bMo Euler Angles (xyz) vs Frame', fontsize=14)

        ax = axes[0]
        ax.plot(frames, unwrap_angles(bMo_euler_x), 'r-', label='Optimized X')
        ax.plot(frames, unwrap_angles(bMo_pnp_euler_x), 'r--', label='PnP X', alpha=0.7)
        ax.set_ylabel('X (deg)')
        ax.set_title('bMo Euler X')
        ax.grid(True)
        ax.legend()

        ax = axes[1]
        ax.plot(frames, unwrap_angles(bMo_euler_y), 'g-', label='Optimized Y')
        ax.plot(frames, unwrap_angles(bMo_pnp_euler_y), 'g--', label='PnP Y', alpha=0.7)
        ax.set_ylabel('Y (deg)')
        ax.set_title('bMo Euler Y')
        ax.grid(True)
        ax.legend()

        ax = axes[2]
        ax.plot(frames, unwrap_angles(bMo_euler_z), 'b-', label='Optimized Z')
        ax.plot(frames, unwrap_angles(bMo_pnp_euler_z), 'b--', label='PnP Z', alpha=0.7)
        ax.set_xlabel('Frame ID')
        ax.set_ylabel('Z (deg)')
        ax.set_title('bMo Euler Z')
        ax.grid(True)
        ax.legend()

        plt.tight_layout()
        bmo_euler_path = os.path.join(RESULT_DIR, "bmo_euler_vs_frame.png")
        plt.savefig(bmo_euler_path, dpi=150)
        print(f"Saved bMo euler plot to {bmo_euler_path}")
        plt.close()

        # cMo 欧拉角 (optimized)
        cMo_euler_x = [pose_to_euler_tvec(np.array(r['cMo_optimized']))[0][0] for r in frame_records]
        cMo_euler_y = [pose_to_euler_tvec(np.array(r['cMo_optimized']))[0][1] for r in frame_records]
        cMo_euler_z = [pose_to_euler_tvec(np.array(r['cMo_optimized']))[0][2] for r in frame_records]

        # cMo 欧拉角 (PnP)
        cMo_pnp_euler_x = [pose_to_euler_tvec(np.array(r['cMo']))[0][0] for r in frame_records]
        cMo_pnp_euler_y = [pose_to_euler_tvec(np.array(r['cMo']))[0][1] for r in frame_records]
        cMo_pnp_euler_z = [pose_to_euler_tvec(np.array(r['cMo']))[0][2] for r in frame_records]

        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        fig.suptitle('cMo Euler Angles (xyz) vs Frame', fontsize=14)

        ax = axes[0]
        ax.plot(frames, unwrap_angles(cMo_euler_x), 'r-', label='Optimized X')
        ax.plot(frames, unwrap_angles(cMo_pnp_euler_x), 'r--', label='PnP X', alpha=0.7)
        ax.set_ylabel('X (deg)')
        ax.set_title('cMo Euler X')
        ax.grid(True)
        ax.legend()

        ax = axes[1]
        ax.plot(frames, unwrap_angles(cMo_euler_y), 'g-', label='Optimized Y')
        ax.plot(frames, unwrap_angles(cMo_pnp_euler_y), 'g--', label='PnP Y', alpha=0.7)
        ax.set_ylabel('Y (deg)')
        ax.set_title('cMo Euler Y')
        ax.grid(True)
        ax.legend()

        ax = axes[2]
        ax.plot(frames, unwrap_angles(cMo_euler_z), 'b-', label='Optimized Z')
        ax.plot(frames, unwrap_angles(cMo_pnp_euler_z), 'b--', label='PnP Z', alpha=0.7)
        ax.set_xlabel('Frame ID')
        ax.set_ylabel('Z (deg)')
        ax.set_title('cMo Euler Z')
        ax.grid(True)
        ax.legend()

        plt.tight_layout()
        cmo_euler_path = os.path.join(RESULT_DIR, "cmo_euler_vs_frame.png")
        plt.savefig(cmo_euler_path, dpi=150)
        print(f"Saved cMo euler plot to {cmo_euler_path}")
        plt.close()

    # ============ 绘制误差随Frame的变化 ============
    if len(pnp_records) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle('Reprojection Error vs Frame', fontsize=14)

        frames = [r['frame_id'] for r in pnp_records]

        ax = axes[0]
        ax.plot(frames, [r['pnp_error_round1'] for r in pnp_records], 'b-', label='Round1', marker='o')
        ax.plot(frames, [r['pnp_error_final'] for r in pnp_records], 'g-', label='Final', marker='s')
        ax.set_xlabel('Frame ID')
        ax.set_ylabel('Error (px)')
        ax.set_title('PnP Error')
        ax.legend()
        ax.grid(True)

        ax = axes[1]
        opt_frames = [r['frame_id'] for r in optimize_records]
        ax.plot(opt_frames, [r['frame_error'] for r in optimize_records], 'b-', label='Frame Error', marker='o')
        ax.plot(opt_frames, [r['avg_error'] for r in optimize_records if r['avg_error'] is not None], 'g-', label='Avg Error', marker='s')
        ax.set_xlabel('Frame ID')
        ax.set_ylabel('Error (px)')
        ax.set_title('Optimization Error')
        ax.legend()
        ax.grid(True)

        plt.tight_layout()
        error_plot_path = os.path.join(RESULT_DIR, "error_vs_frame.png")
        plt.savefig(error_plot_path, dpi=150)
        print(f"Saved error plot to {error_plot_path}")
        plt.close()

    print("\n" + "=" * 80)
    print("All results saved successfully!")
    print("=" * 80)
