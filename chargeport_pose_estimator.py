import cv2 as cv
from ultralytics import YOLO
import time
from gemiEd import *
from scipy.spatial.transform import Rotation
import os
import numpy as np
from static_pose_optimizer import StaticPoseOptimizer, pose_to_euler_tvec

# Load a model
model = YOLO("checkpoint/best.pt")  # load an official model

# ============ Config ============
DATA_DIR = "dataset/save_data3/20260511_120244"
RESULT_DIR = "result/save_data3/20260511_120244/pose_estimation"
SLIDING_WINDOW_SIZE = 8
MAX_FRAMES = 50  # Limit frames for quick test, -1 for all frames


# =========== Read metadata config ===========
BEGIN_FRAME_ID = 1180  # Skip frames with frame_id < this value

# PnP threshold config
USE_ADAPTIVE_THRESHOLD = True  # True = adaptive, False = fixed
FIXED_ERROR_THRESHOLD = 0.4    # used when USE_ADAPTIVE_THRESHOLD = False
ADAPTIVE_MULTIPLIER = 2.0      # threshold = median_error * multiplier

# Frame rejection threshold - if ALL points have error > this, skip
FRAME_REJECT_THRESHOLD = 1.0   # pixels

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


def solvePnP_IPPE(pts2d, pts3d, K, dist):
    """Wrapper for cv2.solvePnP with IPPE method and validity checks"""
    success, rvec, tvec = cv2.solvePnP(
        pts3d, pts2d, K, dist, flags=cv2.SOLVEPNP_IPPE)
    if not success:
        return None, None, False

    rvec = rvec.flatten()
    tvec = tvec.flatten()
    # Check cMo validity: object should be in front of camera (z > 0)
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
    # Check rotation matrix determinant (should be +1, not -1 for reflection)
    R = cMo[:3, :3]
    det_R = np.linalg.det(R)
    if abs(det_R - 1.0) > 1e-6:
        return False, f"Rotation det={det_R:.4f} (reflection/flip)"

    # Check translation: object origin should be in front of camera (z > 0)
    tvec = cMo[:3, 3]
    if tvec[2] <= 0:
        return False, f"Object behind camera (z={tvec[2]:.2f})"

    # Sanity check: object should be within reasonable distance (50-1000mm)
    dist = np.linalg.norm(tvec)
    if dist < 50 or dist > 3000:
        return False, f"Object distance={dist:.1f}mm (unreasonable)"

    return True, "ok"


def getInferResult(model, img):
    results = model(img)
    if len(results) == 0:
        return []
    return results[0].boxes.xyxy.cpu().numpy()


if __name__ == '__main__':
    import json

    # Create result directory
    os.makedirs(RESULT_DIR, exist_ok=True)

    # Define a static pose optimizer instance
    optimizer = StaticPoseOptimizer(K, dist)
    optimizer.set_extrinsics(eMc)
    optimizer.set_object_pts(obj_pts)

    # Load metadata from JSON
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
        # Skip frames before BEGIN_FRAME_ID
        frame_id_val = record['frame_id']
        if frame_id_val < BEGIN_FRAME_ID:
            continue
        if MAX_FRAMES > 0 and processed_frames >= MAX_FRAMES:
            print(f"\nReached max frames limit ({MAX_FRAMES}), stopping...")
            break
        processed_frames += 1

        # Get current frame timestamp
        current_timestamp_ns = record['camera_timestamp_ns']

        # Check time interval with previous frame
        if last_timestamp_ns is not None:
            time_diff_s = (current_timestamp_ns - last_timestamp_ns) / 1e9
            if time_diff_s < 0.1:
                continue
            elif time_diff_s > 1.0:
                print(f"[WARN] Large timestamp gap: {time_diff_s:.2f}s between consecutive frames")

        # Update last timestamp after potential wait
        last_timestamp_ns = record['camera_timestamp_ns']

        img_relative_path = record['image_path']
        img_path = os.path.join(DATA_DIR, img_relative_path)

        # Extract robot pose from JSON
        robot_pose = np.array(record['pose_matrix_4x4']).reshape(4, 4)

        print("\n==============================================")
        print(f"Processing frame {frame_id}: frame_{frame_id_val:06d}.jpg")

        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to read image: {img_path}")
            continue

        t0 = time.perf_counter_ns()

        img_float = img.astype(np.float32)
        img_bright = img_float - 50

        # 限制范围并转回 uint8
        img_bright = np.clip(
            img_bright, 0, 255).astype(np.uint8)
        result = getInferResult(model, img_bright)
        if result.shape[0] == 0 or result.shape[1] == 0:
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
            ellipses_.append((*e.center, e.size[0]/2, e.size[1]/2, e.angle))

        matcher = UltimateSocketMatcher()
        matcher.obj_pts = obj_pts
        matcher.K = K
        matcher.dist = dist
        matcher.eMc = eMc

        vis_ellipse = draw_ellipse(roi, ellipses_)

        final_pts, status, centers = matcher.solve(ellipses_, [*(result[0][:2]), *(result[0][2:]-result[0][:2])])
        print(f'find {len(final_pts)} points')

        inlier_mask = None  # Initialize for visualization section

        if final_pts is not None and centers.shape[0] >= 4:  # need at least 4 points for solvePnP_IPPE

            pts3d = matcher.obj_pts[matcher.r_idx]
            pts2d = centers

            # ============ Two-round PnP ============
            rvec, tvec, valid, inlier_mask, per_point_errors, round1_error, used_threshold = two_round_pnp(
                pts2d, pts3d, K, dist, error_threshold=FIXED_ERROR_THRESHOLD)
            if not valid:
                print(f"Frame:{frame_id} frame_{frame_id_val:06d}: PnP failed, skipping pose estimation")
                continue

            n_inliers = inlier_mask.sum() if inlier_mask is not None else 0
            pnp_error = per_point_errors.mean()

            # Print per-point reprojection errors
            print(f"Per-point reprojection errors (px), threshold={used_threshold:.3f}:")
            obj_pt_names = ['L-top', 'R-top', 'L-mid', 'center', 'R-mid', 'L-bot', 'R-bot']
            for i, (err, pt_name) in enumerate(zip(per_point_errors, obj_pt_names[:len(per_point_errors)])):
                marker = '[INLIER]' if (inlier_mask is not None and inlier_mask[i]) else '[OUTLIER]'
                print(f"  Point {i} ({pt_name}): {err:7.3f} px {marker}")

            # ============ Check if frame should be rejected ============
            # Frame is rejected if ALL points have error > FRAME_REJECT_THRESHOLD
            all_points_exceed = np.all(per_point_errors > FRAME_REJECT_THRESHOLD)
            if all_points_exceed:
                print(f"  [REJECT] All points exceed threshold {FRAME_REJECT_THRESHOLD} px, frame rejected")
                frame_id += 1
                continue

            # ============ Build cMo from two-round PnP result ============
            cMo = np.eye(4)
            cMo[:3, :3] = Rotation.from_rotvec(rvec).as_matrix()
            cMo[:3, 3] = tvec

            # Compute initial bMo
            bMo_init = robot_pose @ eMc @ cMo

            # Initialize optimizer with first valid bMo
            if not optimizer.is_initialized():
                optimizer.set_initial_pose(bMo_init)

            # Check reprojection error with last optimized bMo
            max_error = None
            if last_bMo is not None and optimizer.is_initialized():
                cMo_before = np.linalg.inv(eMc) @ np.linalg.inv(robot_pose) @ last_bMo
                rvec_before = Rotation.from_matrix(cMo_before[:3, :3]).as_rotvec()
                tvec_before = cMo_before[:3, 3]
                print(f"cMo_before: {pose_to_euler_tvec(cMo_before)}")
                proj, _ = cv2.projectPoints(pts3d, rvec_before, tvec_before, K, dist)
                proj_errors = proj.reshape(-1, 2) - pts2d
                max_error = np.linalg.norm(proj_errors, axis=1).max()
                print(f"cMo_before reprojection error per point:\n {proj_errors} \n mean error: {np.linalg.norm(proj_errors, axis=1).mean():.4f}")

                # find the largest 2 reprojection errors
                max_error_idx = np.argsort(np.linalg.norm(proj_errors, axis=1))[-2:]
                print(f"Indices of 2 largest reprojection errors: {max_error_idx}, errors: {np.linalg.norm(proj_errors[max_error_idx], axis=1)}")

                # visualize all the reprojection errors on the image
                vis_error = img.copy()
                for i, (x, y) in enumerate(pts2d):
                    color = (0, 255, 0) if i not in max_error_idx else (0, 0, 255)
                    cv2.circle(vis_error, (int(x), int(y)), 3, color, -1)
                    x_proj, y_proj = proj[i][0]
                    cv2.circle(vis_error, (int(x_proj), int(y_proj)), 3, (255, 0, 0), -1)
                    cv2.line(vis_error, (int(x), int(y)), (int(x_proj), int(y_proj)), (0, 255, 0), 1)
                # clip the image to the roi and visualize the reprojection error there
                vis_error = vis_error[roi_y_min:roi_y_max, roi_x_min:roi_x_max]
                vis_error = cv2.resize(vis_error, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
                cv.imshow("reprojection_error", vis_error)
                vis_error_path = os.path.join(RESULT_DIR, f"frame_{frame_id_val:06d}_vis_error.png")
                cv.imwrite(vis_error_path, vis_error)

            # Add to optimizer if error is acceptable
            if max_error is None or max_error < 5.0:
                if optimizer.get_frame_count() >= SLIDING_WINDOW_SIZE:
                    optimizer.remove_oldest_frame()

                optimizer.add_frame(frame_id, robot_pose, pts2d, pts3d)
                result_optimized = optimizer.optimize()

                bMo_optimized = optimizer.get_pose()
                last_bMo = bMo_optimized

                # Compute cMo from optimized bMo
                cMo_optimized = optimizer.compute_cMo(robot_pose)

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
                bMo_ba = robot_pose @ eMc @ cMo_ba

                # Print poses in Euler angles + translation
                print(f"bMo_optimized: {pose_to_euler_tvec(bMo_optimized)}")
                print(f"bMo_pnp: {pose_to_euler_tvec(bMo_init)}")
                print(f"bMo_ba: {pose_to_euler_tvec(bMo_ba)}")
                print(f"cMo_optimized: {pose_to_euler_tvec(cMo_optimized)}")
                print(f"cMo_pnp: {pose_to_euler_tvec(cMo)}")
                print(f"cMo_ba: {pose_to_euler_tvec(cMo_ba)}")

                pnp_error = per_point_errors.mean()
                pnp_error_ba = compute_reproj_error(pts3d_inlier, rvec_ba, tvec_ba, pts2d_inlier, K, dist)
                print(f"PnP reprojection error: {pnp_error:.4f} (round1: {round1_error:.4f})")
                print(f"PnP reprojection error (BA, {n_inliers} inliers): {pnp_error_ba:.4f}")

                now_error, _ = optimizer.get_frame_error(frame_id)
                print(f"Optimized reprojection error: {now_error:.4f}")
                ave_error = optimizer.get_average_error()
                print(f"Average reprojection error: {ave_error:.4f}")

                # Draw axes
                cv2.drawFrameAxes(img, K, dist, cMo_optimized[:3, :3], cMo_optimized[:3, 3:], 10, 3)
                cv2.drawFrameAxes(img, K, dist, cMo[:3, :3], cMo[:3, 3:], 20, 1)
            else:
                print(f"frame_{frame_id_val:06d}: Max reprojection error {max_error:.2f} exceeds threshold, skipping optimization update")

            frame_id += 1

        # visualize centers with inlier/outlier coloring
        if centers is not None:
            for i, (x, y) in enumerate(centers):
                if inlier_mask is not None and i < len(inlier_mask):
                    if inlier_mask[i]:
                        color = (0, 255, 0)   # Green = inlier
                    else:
                        color = (0, 0, 255)   # Red = outlier
                else:
                    color = (255, 255, 0)     # Cyan = unknown/no mask
                cv2.circle(img, (int(x), int(y)), 2, color, -1)
                cv2.putText(img, str(i), (int(x)+5, int(y)-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # 裁切到roi显示，并放大2倍
        vis_result = img[roi_y_min:roi_y_max, roi_x_min:roi_x_max]
        vis_result = cv2.resize(vis_result, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
        cv.imshow("vis_result", vis_result)
        # 保存图片
        vis_result_path = os.path.join(RESULT_DIR, f"frame_{frame_id_val:06d}_vis_result.png")
        cv.imwrite(vis_result_path, vis_result)
        cv.waitKey(1)

    cv.destroyAllWindows()