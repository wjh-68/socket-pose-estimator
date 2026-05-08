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
DATA_DIR = "dataset/save_data2"
RESULT_DIR = "result/save_data2"
SLIDING_WINDOW_SIZE = 8
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
# Transform from object frame to model frame (if needed)        
oMo = np.eye(4,dtype=np.float32)

def solvePnP_IPPE(pts2d, pts3d, K, dist):
    """Wrapper for cv2.solvePnP with IPPE method and validity checks"""
    success, rvec, tvec = cv2.solvePnP(
        pts3d, pts2d, K, dist,flags=cv2.SOLVEPNP_IPPE)
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


def two_round_pnp(pts2d, pts3d, K, dist, error_threshold=5.0):
    """Two-round PnP: first round to identify inliers, second round with inliers only.

    Returns:
        rvec, tvec, valid, inlier_mask, per_point_errors, round1_errors
    """
    # ============ Round 1: Initial estimate with all points ============
    rvec1, tvec1, valid1 = solvePnP_IPPE(pts2d, pts3d, K, dist)
    if not valid1:
        return None, None, False, None, None, None

    per_point_errors = compute_per_point_reproj_errors(pts3d, rvec1, tvec1, pts2d, K, dist)
    round1_error = per_point_errors.mean()

    # ============ Filter inliers based on error threshold ============
    inlier_mask = per_point_errors < error_threshold
    n_inliers = inlier_mask.sum()

    # If too few inliers, fall back to all points with higher threshold
    if n_inliers < 4:
        error_threshold = error_threshold * 2
        inlier_mask = per_point_errors < error_threshold
        n_inliers = inlier_mask.sum()

    # ============ Round 2: Refine with inliers only ============
    if n_inliers >= 4:
        pts3d_inlier = pts3d[inlier_mask]
        pts2d_inlier = pts2d[inlier_mask]
        rvec2, tvec2, valid2 = solvePnP_IPPE(pts2d_inlier, pts3d_inlier, K, dist)
        if valid2:
            per_point_errors_round2 = compute_per_point_reproj_errors(pts3d, rvec2, tvec2, pts2d, K, dist)
            return rvec2, tvec2, True, inlier_mask, per_point_errors_round2, round1_error
        else:
            return rvec1, tvec1, True, inlier_mask, per_point_errors, round1_error
    else:
        return rvec1, tvec1, True, inlier_mask, per_point_errors, round1_error


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


def getInferResult(model,img):
    results= model(img)
    if(len(results)==0):
        return []
    return results[0].boxes.xyxy.cpu().numpy()

# from pyAAMED import pyAAMED
if __name__=='__main__':

    # Traverse dataset and feed frames to optimizer
    import re

    # 数字排序
    jpg_files = sorted(
        [f for f in os.listdir(DATA_DIR) if f.endswith('.jpg') and f != 'temp'],
        key=lambda x: int(re.search(r'(\d+)', x).group(1))
    )
    # 字符串排序有误
    # jpg_files = sorted([f for f in os.listdir(DATA_DIR)
    #                     if f.endswith('.jpg') and f != 'temp'])
    # print(f"jpg_files: {jpg_files}")
    npy_files = {f.replace('.npy', '') : f for f in os.listdir(DATA_DIR)
                 if f.endswith('.npy')}

    # ============ bMo_gt estimation config ============
    GT_ESTIMATE_FRAMES = 10  # number of static frames to estimate bMo_gt
    bmo_positions = []       # list of (3,) position arrays
    bmo_rotvecs = []         # list of (3,) rotation vector arrays
    bmo_jpg_files = []       # list of corresponding jpg filenames
    reproj_errors = []       # list of reprojection errors for weighting
    # =================================================

    frame_id = 1
    for jpg_file in jpg_files:
        ts = jpg_file.replace('.jpg', '')
        ts = ts.replace('img_','')
        if int(ts) > GT_ESTIMATE_FRAMES:
            break
        npy_name = 'pose_' + ts
        if npy_name not in npy_files:
            continue
        print("\n==============================================")
        print(f"Processing frame {frame_id}: {jpg_file}, {npy_name}")

        img_path = os.path.join(DATA_DIR, jpg_file)
        img = cv2.imread(img_path)

        
        robot_pose_path = os.path.join(DATA_DIR, npy_files[npy_name])
        robot_pose = np.load(robot_pose_path)
        # print(f"Robot pose:\n{robot_pose}")

        t0 = time.perf_counter_ns()

        img_float = img.astype(np.float32)
        img_bright = img_float -50

        # 限制范围并转回 uint8
        img_bright = np.clip(
            img_bright, 0, 255).astype(np.uint8)
        result = getInferResult(model, img_bright)
        if result.shape[0]==0 or result.shape[1]==0:
            continue
        
        roi = img[int(result[0][1]):int(result[0][3]),
                  int(result[0][0]):int(result[0][2])]

        roi_x_min = int(result[0][0])
        roi_y_min = int(result[0][1])
        roi_x_max = int(result[0][2])
        roi_y_max = int(result[0][3])

        gray_roi = cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)

        detector = pyced.CED(np.ascontiguousarray(roi))
        detector.run_CED()
        rotRects = detector.getEllipsesAfterCluster()
        ellipses_ = []

        for e in rotRects:
            ellipses_.append((*e.center,e.size[0]/2,e.size[1]/2,e.angle))

        matcher = UltimateSocketMatcher()
        matcher.obj_pts = obj_pts
        matcher.K = K
        matcher.dist = dist
        matcher.eMc = eMc

        vis_ellipse = draw_ellipse(roi,ellipses_)
        
        final_pts,status,centers = matcher.solve(ellipses_,[*(result[0][:2]),*(result[0][2:]-result[0][:2])])
        print(f'find {len(final_pts)} points')

        inlier_mask = None  # Initialize for visualization section

        if final_pts is not None and centers.shape[0] >= 4: # need at least 4 points for solvePnP_IPPE

            pts3d = matcher.obj_pts[matcher.r_idx]
            pts2d = centers

            # ============ Two-round PnP ============
            rvec, tvec, valid, inlier_mask, per_point_errors, round1_error = two_round_pnp(
                pts2d, pts3d, K, dist, error_threshold=0.4)
            if not valid:
                print(f"Frame:{frame_id} Timestamp:{ts}: PnP failed, skipping pose estimation")
                continue

            n_inliers = inlier_mask.sum() if inlier_mask is not None else 0

            # Print per-point reprojection errors
            print(f"Per-point reprojection errors (px):")
            obj_pt_names = ['L-top', 'R-top', 'L-mid', 'center', 'R-mid', 'L-bot', 'R-bot']
            for i, (err, pt_name) in enumerate(zip(per_point_errors, obj_pt_names[:len(per_point_errors)])):
                marker = '[INLIER]' if (inlier_mask is not None and inlier_mask[i]) else '[OUTLIER]'
                print(f"  Point {i} ({pt_name}): {err:7.3f} px {marker}")

            # ============ Build cMo from two-round PnP result ============
            cMo = np.eye(4)
            cMo[:3, :3] = Rotation.from_rotvec(rvec).as_matrix()
            cMo[:3, 3] = tvec

            # Compute bMo
            bMo = robot_pose @ eMc @ cMo

            # Compute bMo by SOLVEPNP_ITERATIVE for comparison (using inlier points)
            if n_inliers >= 4:
                pts3d_inlier = pts3d[inlier_mask]
                pts2d_inlier = pts2d[inlier_mask]
            else:
                pts3d_inlier = pts3d
                pts2d_inlier = pts2d
            success, rvec_ba, tvec_ba = cv2.solvePnP(pts3d_inlier, pts2d_inlier, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
            cMo_ba = np.eye(4)
            cMo_ba[:3,:3] = Rotation.from_rotvec(rvec_ba.flatten()).as_matrix()
            cMo_ba[:3,3] = tvec_ba.flatten()
            bMo_ba = robot_pose @ eMc @ cMo_ba

            # Print poses in Euler angles + translation
            print(f"bMo_pnp: {pose_to_euler_tvec(bMo)}")
            print(f"bMo_ba: {pose_to_euler_tvec(bMo_ba)}")
            print(f"cMo_pnp: {pose_to_euler_tvec(cMo)}")
            print(f"cMo_ba: {pose_to_euler_tvec(cMo_ba)}")

            pnp_error = per_point_errors.mean()
            pnp_error_ba = compute_reproj_error(pts3d_inlier, rvec_ba, tvec_ba, pts2d_inlier, K, dist)
            print(f"PnP reprojection error: {pnp_error:.4f} (round1: {round1_error:.4f})")
            print(f"PnP reprojection error (BA, {n_inliers} inliers): {pnp_error_ba:.4f}")

            # ============ Collect bMo for gt estimation (first N static frames) ============
            if frame_id <= GT_ESTIMATE_FRAMES:
                bmo_positions.append(bMo[:3, 3].copy())
                rvec_bMo = Rotation.from_matrix(bMo[:3, :3]).as_rotvec()
                bmo_rotvecs.append(rvec_bMo.flatten().copy())
                bmo_jpg_files.append(jpg_file)
                reproj_errors.append(pnp_error)
                print(f"  [GT estimation] collected frame {frame_id}/{GT_ESTIMATE_FRAMES}")
            # ============================================================================

            # Draw Axes using cMo from PnP
            cv2.drawFrameAxes(img,K,dist,cMo[:3,:3],cMo[:3,3:],20,1)
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
        vis_result_path = os.path.join(RESULT_DIR, f"{ts}_vis_result.png")
        cv.imwrite(vis_result_path, vis_result)
        cv.waitKey(1)

    # ============ Estimate bMo_gt from collected static frames ============
    def rotation_mean_spd(rotvecs, weights=None):
        """Compute mean rotation using quaternion averaging with SDP (signed distributed product).

        Properly handles the non-linear nature of rotation representations.
        """
        n = len(rotvecs)
        if weights is None:
            weights = np.ones(n)

        # Convert to quaternions (w, x, y, z)
        quats = np.array([Rotation.from_rotvec(rv).as_quat() for rv in rotvecs])

        # Ensure all quaternions have positive w (q and -q represent same rotation)
        for i in range(n):
            if quats[i, 0] < 0:
                quats[i] = -quats[i]

        # Compute weighted mean quaternion via SVD on Q.Q^T
        Q = np.zeros((4, 4))
        for i in range(n):
            q = quats[i]
            w = weights[i]
            Q += w * np.outer(q, q)

        # SVD
        _, _, Vt = np.linalg.svd(Q)
        q_mean = Vt[0]

        # Normalize and convert back to rotation vector
        q_mean = q_mean / np.linalg.norm(q_mean)
        return Rotation.from_quat(q_mean).as_rotvec()

    if len(bmo_positions) >= 3:
        bmo_positions = np.array(bmo_positions)
        bmo_rotvecs = np.array(bmo_rotvecs)
        reproj_errors = np.array(reproj_errors)

        # Method 1: Median (robust to outliers)
        pos_median = np.median(bmo_positions, axis=0)
        rotvec_median = np.median(bmo_rotvecs, axis=0)

        # Method 2: Weighted average by inverse reprojection error
        weights = 1.0 / (reproj_errors ** 2 + 1e-6)
        weights /= weights.sum()
        pos_weighted = np.average(bmo_positions, weights=weights, axis=0)
        rotvec_weighted = rotation_mean_spd(bmo_rotvecs, weights=weights)

        # Method 3: RANSAC-style outlier filtering
        pos_dist = np.linalg.norm(bmo_positions - pos_median, axis=1)

        # Compute geodesic rotation distance on SO(3), not linear rotvec subtraction
        rot_median = Rotation.from_rotvec(rotvec_median)
        rot_dist = np.array([
            np.linalg.norm((rot_median * Rotation.from_rotvec(rv).inv()).as_rotvec())
            for rv in bmo_rotvecs
        ])

        # Set thresholds based on distribution (e.g. 75th percentile * 2) or fixed values
        pos_threshold = np.percentile(pos_dist, 75) * 2 if len(pos_dist) > 0 else 5.0
        rot_threshold = np.percentile(rot_dist, 75) * 2 if len(rot_dist) > 0 else 0.05
        ransac_inlier_mask = (pos_dist < pos_threshold) & (rot_dist < rot_threshold)
        n_inliers = ransac_inlier_mask.sum()

        print("\n" + "=" * 70)
        print("bMo_gt Estimation Results (from first {} static frames)".format(len(bmo_positions)))
        print("=" * 70)

        # Print per-frame inlier/outlier status for Method 3
        print(f"\n[Method 3] RANSAC-style filtering (pos_thresh={pos_threshold:.2f}mm, rot_thresh={rot_threshold*180/np.pi:.3f}deg):")
        print(f"{'Frame':<8} {'File':<12} {'Pos dist(mm)':>12} {'Rot dist(deg)':>14} {'Status':<10}")
        print("-" * 60)
        print("-" * 50)
        for i, jpg in enumerate(bmo_jpg_files):
            pos_d = pos_dist[i]
            rot_d = rot_dist[i]
            is_inlier = ransac_inlier_mask[i]
            status = 'INLIER' if is_inlier else 'OUTLIER'
            marker = '  ' if is_inlier else '!!'
            print(f"{marker} {i+1:<6} {jpg:<12} {pos_d:>12.3f} {rot_d*180/np.pi:>14.3f} {status:<10}")

        if n_inliers >= 3:
            pos_ransac = np.mean(bmo_positions[ransac_inlier_mask], axis=0)
            rotvec_ransac = rotation_mean_spd(bmo_rotvecs[ransac_inlier_mask])
        else:
            pos_ransac = pos_median
            rotvec_ransac = rotvec_median
            print(f"  [WARN] Only {n_inliers} inliers, falling back to median")

        # Convert rotation vectors to rotation matrices
        R_median = Rotation.from_rotvec(rotvec_median).as_matrix()
        R_weighted = Rotation.from_rotvec(rotvec_weighted).as_matrix()
        R_ransac = Rotation.from_rotvec(rotvec_ransac).as_matrix()

        # Build bMo_gt candidates
        bMo_gt_median = np.eye(4)
        bMo_gt_median[:3, :3] = R_median
        bMo_gt_median[:3, 3] = pos_median

        bMo_gt_weighted = np.eye(4)
        bMo_gt_weighted[:3, :3] = R_weighted
        bMo_gt_weighted[:3, 3] = pos_weighted

        bMo_gt_ransac = np.eye(4)
        bMo_gt_ransac[:3, :3] = R_ransac
        bMo_gt_ransac[:3, 3] = pos_ransac

        print(f"\n[Method 1] Median:         {pose_to_euler_tvec(bMo_gt_median)}")
        print(f"[Method 2] Weighted:       {pose_to_euler_tvec(bMo_gt_weighted)}")
        print(f"[Method 3] RANSAC ({n_inliers}/{len(bmo_positions)} inliers): {pose_to_euler_tvec(bMo_gt_ransac)}")

        print(f"\nPosition std (x,y,z) mm: {np.std(bmo_positions, axis=0)}")
        # Proper angular std: compute geodesic distance from each rotation to median
        rot_median = Rotation.from_rotvec(rotvec_median)
        angular_dists = np.array([
            np.linalg.norm((rot_median * Rotation.from_rotvec(rv).inv()).as_rotvec())
            for rv in bmo_rotvecs
        ])
        print(f"Rotation angular std (deg): {np.std(angular_dists) * 180/np.pi:.3f}")

        # Save the RANSAC result as the recommended bMo_gt
        bMo_gt = bMo_gt_ransac
        print(f"\n[RECOMMENDED] Using RANSAC-filtered bMo_gt:")
        print(f"bMo_gt = \n{bMo_gt}")
        print(f"bMo_gt Euler+Tvec: {pose_to_euler_tvec(bMo_gt)}")
        print("=" * 70)
    else:
        print("\n[WARN] Not enough frames collected for bMo_gt estimation")
    # ====================================================================

    cv.destroyAllWindows()