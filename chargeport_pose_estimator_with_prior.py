import cv2 as cv
from ultralytics import YOLO
import time
from gemiEd import *
from scipy.spatial.transform import Rotation
import os
import numpy as np
from sliding_window_pose_optimizer_with_prior import SlidingWindowPoseOptimizerWithPrior, pose_to_euler_tvec, SE3

# Load a model
model = YOLO("checkpoint/best.pt")

# ============ Config ============
DATA_DIR = "dataset/save_data3/20260511_120244"
RESULT_DIR = "result/save_data3_with_prior"
SLIDING_WINDOW_SIZE = 8
MAX_FRAMES = 1000

# =========== Read metadata config ===========
BEGIN_FRAME_ID = 1180  # Skip frames with frame_id < this value

# Optimizer sigmas - configurable
REPROJ_STD = 1.0        # pixels - reprojection residual weight
ROBOT_POSE_STD_TRANS = 5.0  # mm - translation correction prior
ROBOT_POSE_STD_ANGLE = 1.0  # degrees - rotation correction prior

# PnP threshold config
USE_ADAPTIVE_THRESHOLD = True  # True = adaptive, False = fixed
FIXED_ERROR_THRESHOLD = 0.5    # used when USE_ADAPTIVE_THRESHOLD = False
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
    cMo = np.eye(4)
    cMo[:3, :3] = Rotation.from_rotvec(rvec).as_matrix()
    cMo[:3, 3] = tvec

    valid, reason = validate_cMo(cMo)
    if not valid:
        print(f"  [WARN] PnP result invalid: {reason}")
        return None, None, False

    return rvec, tvec, True


def compute_per_point_reproj_errors(pts3d, rvec, tvec, pts2d, K, dist):
    """Compute reprojection error for each point (in pixels)"""
    proj, _ = cv2.projectPoints(pts3d, rvec, tvec, K, dist)
    return np.linalg.norm(proj.reshape(-1,2) - pts2d, axis=1)


def two_round_pnp(pts2d, pts3d, K, dist, error_threshold=5.0):
    """Two-round PnP with adaptive threshold: first round to identify inliers, second round with inliers only."""
    rvec1, tvec1, valid1 = solvePnP_IPPE(pts2d, pts3d, K, dist)
    if not valid1:
        return None, None, False, None, None, None, None

    per_point_errors = compute_per_point_reproj_errors(pts3d, rvec1, tvec1, pts2d, K, dist)
    round1_error = per_point_errors.mean()

    # Adaptive threshold selection
    if USE_ADAPTIVE_THRESHOLD:
        median_err = np.median(per_point_errors)
        used_threshold = median_err * ADAPTIVE_MULTIPLIER
    else:
        used_threshold = error_threshold

    inlier_mask = per_point_errors < used_threshold
    n_inliers = inlier_mask.sum()

    if n_inliers < 4:
        used_threshold = used_threshold * 2
        inlier_mask = per_point_errors < used_threshold
        n_inliers = inlier_mask.sum()

    if n_inliers >= 4:
        pts3d_inlier = pts3d[inlier_mask]
        pts2d_inlier = pts2d[inlier_mask]
        rvec2, tvec2, valid2 = solvePnP_IPPE(pts2d_inlier, pts3d_inlier, K, dist)
        if valid2:
            per_point_errors_round2 = compute_per_point_reproj_errors(pts3d, rvec2, tvec2, pts2d, K, dist)
            return rvec2, tvec2, True, inlier_mask, per_point_errors_round2, round1_error, used_threshold
        else:
            return rvec1, tvec1, True, inlier_mask, per_point_errors, round1_error, used_threshold
    else:
        return rvec1, tvec1, True, inlier_mask, per_point_errors, round1_error, used_threshold


def validate_cMo(cMo):
    """Check if cMo is physically valid"""
    R = cMo[:3, :3]
    det_R = np.linalg.det(R)
    if abs(det_R - 1.0) > 1e-6:
        return False, f"Rotation det={det_R:.4f} (reflection/flip)"

    tvec = cMo[:3, 3]
    if tvec[2] <= 0:
        return False, f"Object behind camera (z={tvec[2]:.2f})"

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

    os.makedirs(RESULT_DIR, exist_ok=True)

    optimizer = SlidingWindowPoseOptimizerWithPrior(
        K, dist,
        reproj_std=REPROJ_STD,
        robot_pose_std_translation=ROBOT_POSE_STD_TRANS,
        robot_pose_std_angle_deg=ROBOT_POSE_STD_ANGLE
    )
    optimizer.set_extrinsics(eMc)
    optimizer.set_object_pts(obj_pts)

    meta_path = os.path.join(DATA_DIR, "metadata.json")
    with open(meta_path, 'r') as f:
        metadata = json.load(f)

    records = metadata['records']
    print(f"Loaded {len(records)} frames from {meta_path}")
    print(f"Optimizer config: reproj_std={REPROJ_STD}, robot_trans_std={ROBOT_POSE_STD_TRANS}mm, robot_angle_std={ROBOT_POSE_STD_ANGLE}deg")

    frame_id = 0
    last_bMo = None
    processed_frames = 0
    last_timestamp_ns = None

    for record in records:
        # Skip frames before BEGIN_FRAME_ID
        frame_id_val = record['frame_id']
        if frame_id_val < BEGIN_FRAME_ID:
            continue

        current_timestamp_ns = record['camera_timestamp_ns']
        if last_timestamp_ns is not None:
            time_diff_s = (current_timestamp_ns - last_timestamp_ns) / 1e9
            if time_diff_s < 0.1:
                continue
            elif time_diff_s > 1.0:
                print(f"[WARN] Large timestamp gap: {time_diff_s:.2f}s between consecutive frames")

        last_timestamp_ns = record['camera_timestamp_ns']

        if MAX_FRAMES > 0 and processed_frames >= MAX_FRAMES:
            print(f"\nReached max frames limit ({MAX_FRAMES}), stopping...")
            break
        processed_frames += 1

        img_relative_path = record['image_path']
        img_path = os.path.join(DATA_DIR, img_relative_path)

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

        inlier_mask = None

        if final_pts is not None and centers.shape[0] >= 4:

            pts3d = matcher.obj_pts[matcher.r_idx]
            pts2d = centers

            # Two-round PnP
            rvec, tvec, valid, inlier_mask, per_point_errors, round1_error, used_threshold = two_round_pnp(
                pts2d, pts3d, K, dist, error_threshold=FIXED_ERROR_THRESHOLD)
            if not valid:
                print(f"Frame:{frame_id} frame_{frame_id_val:06d}: PnP failed, skipping pose estimation")
                continue

            n_inliers = inlier_mask.sum() if inlier_mask is not None else 0

            # Frame rejection check - if ALL points have error > FRAME_REJECT_THRESHOLD, skip
            if per_point_errors is not None and np.all(per_point_errors > FRAME_REJECT_THRESHOLD):
                print(f"Frame:{frame_id} frame_{frame_id_val:06d}: All points exceed error threshold {FRAME_REJECT_THRESHOLD}, skipping")
                continue

            print(f"Per-point reprojection errors (px):")
            obj_pt_names = ['L-top', 'R-top', 'L-mid', 'center', 'R-mid', 'L-bot', 'R-bot']
            for i, (err, pt_name) in enumerate(zip(per_point_errors, obj_pt_names[:len(per_point_errors)])):
                marker = '[INLIER]' if (inlier_mask is not None and inlier_mask[i]) else '[OUTLIER]'
                print(f"  Point {i} ({pt_name}): {err:7.3f} px {marker}")

            # Build cMo from PnP result
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
                # Use corrected robot pose from optimizer for consistency
                # Note: frame_id - 1 may not exist if previous frame was skipped due to timing/validation
                try:
                    corrected_robot_pose = optimizer.get_corrected_robot_pose_with_stored_delta(frame_id - 1)
                except ValueError:
                    # Previous frame not in optimizer (skipped or removed), use raw robot pose
                    corrected_robot_pose = robot_pose
                cMo_before = np.linalg.inv(eMc) @ np.linalg.inv(corrected_robot_pose) @ last_bMo
                rvec_before = Rotation.from_matrix(cMo_before[:3, :3]).as_rotvec()
                tvec_before = cMo_before[:3, 3]
                print(f"cMo_before: {pose_to_euler_tvec(cMo_before)}")
                proj, _ = cv2.projectPoints(pts3d, rvec_before, tvec_before, K, dist)
                proj_errors = proj.reshape(-1, 2) - pts2d
                max_error = np.linalg.norm(proj_errors, axis=1).max()
                print(f"cMo_before reprojection error per point:\n {proj_errors} \n mean error: {np.linalg.norm(proj_errors, axis=1).mean():.4f}")

                max_error_idx = np.argsort(np.linalg.norm(proj_errors, axis=1))[-2:]
                print(f"Indices of 2 largest reprojection errors: {max_error_idx}, errors: {np.linalg.norm(proj_errors[max_error_idx], axis=1)}")

                # visualize reprojection errors on the image
                vis_error = img.copy()
                for i, (x, y) in enumerate(pts2d):
                    color = (0, 255, 0) if i not in max_error_idx else (0, 0, 255)
                    cv2.circle(vis_error, (int(x), int(y)), 3, color, -1)
                    cv2.putText(vis_error, str(i), (int(x)+5, int(y)-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                    x_proj, y_proj = proj[i][0]
                    cv2.circle(vis_error, (int(x_proj), int(y_proj)), 3, (255, 0, 0), -1)
                    cv2.line(vis_error, (int(x), int(y)), (int(x_proj), int(y_proj)), (0, 255, 0), 1)
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

                cMo_optimized = optimizer.compute_cMo(robot_pose)

                # Print poses
                print(f"bMo_optimized: {pose_to_euler_tvec(bMo_optimized)}")
                print(f"bMo_pnp: {pose_to_euler_tvec(bMo_init)}")
                print(f"cMo_optimized: {pose_to_euler_tvec(cMo_optimized)}")
                print(f"cMo_pnp: {pose_to_euler_tvec(cMo)}")

                pnp_error = per_point_errors.mean()
                print(f"PnP reprojection error: {pnp_error:.4f} (round1: {round1_error:.4f}, threshold: {used_threshold:.4f})")

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
                        color = (0, 255, 0)
                    else:
                        color = (0, 0, 255)
                else:
                    color = (255, 255, 0)
                cv2.circle(img, (int(x), int(y)), 2, color, -1)
                cv2.putText(img, str(i), (int(x)+5, int(y)-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        vis_result = img[roi_y_min:roi_y_max, roi_x_min:roi_x_max]
        vis_result = cv2.resize(vis_result, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
        cv.imshow("vis_result", vis_result)
        vis_result_path = os.path.join(RESULT_DIR, f"frame_{frame_id_val:06d}_vis_result.png")
        cv.imwrite(vis_result_path, vis_result)
        cv.waitKey(1)

    cv.destroyAllWindow()