"""
Check reprojection error using ground truth bMo_gt.
Compares projected 3D points (using bMo_gt) with detected ellipse centers.
"""

import cv2 as cv
from ultralytics import YOLO
from gemiEd import *
import numpy as np
from scipy.spatial.transform import Rotation
import os
import re

# ============ Config ============
DATA_DIR = "dataset/save_data2"
RESULT_DIR = "result/save_data2/reproj_err"
# Ground truth bMo_gt (base to object transform)
bMo_gt = np.array([
    [0.010125, -0.02251, -0.9997, 987.29],
    [-0.99946, 0.030915, -0.010819, -293.39],
    [0.031149, 0.99927, -0.022185, 493.44],
    [0, 0, 0, 1]
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


def getInferResult(model, img):
    results = model(img)
    if len(results) == 0:
        return []
    return results[0].boxes.xyxy.cpu().numpy()


def project_points(obj_pts, cMo, K, dist):
    """Project 3D points to 2D using camera pose cMo"""
    rvec = Rotation.from_matrix(cMo[:3, :3]).as_rotvec()
    tvec = cMo[:3, 3]
    proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, dist)
    return proj.reshape(-1, 2)


def compute_cMo_from_bMo(bMo, bMe, eMc):
    """Compute cMo from bMo, bMe (robot pose), and eMc (eye-to-hand)
    bMo = bMe @ eMc @ cMo
    => cMo = eMc^-1 @ bMe^-1 @ bMo
    """
    return np.linalg.inv(eMc) @ np.linalg.inv(bMe) @ bMo


if __name__ == '__main__':
    # Load YOLO model
    model = YOLO("checkpoint/best.pt")

    # Get sorted image and pose files
    jpg_files = sorted(
        [f for f in os.listdir(DATA_DIR) if f.endswith('.jpg') and f != 'temp'],
        key=lambda x: int(re.search(r'(\d+)', x).group(1))
    )
    npy_files = {f.replace('.npy', ''): f for f in os.listdir(DATA_DIR)
                 if f.endswith('.npy')}

    print("=" * 80)
    print("Reprojection Error Check using bMo_gt")
    print("=" * 80)
    print(f"\nbMo_gt:\n{bMo_gt}\n")

    all_frame_errors = []

    for jpg_file in jpg_files:
        ts = jpg_file.replace('.jpg', '').replace('img_', '')
        npy_name = 'pose_' + ts

        if npy_name not in npy_files:
            continue

        print("\n" + "-" * 80)
        print(f"Frame: {jpg_file}")
        print("-" * 80)

        # Load image
        img_path = os.path.join(DATA_DIR, jpg_file)
        img = cv2.imread(img_path)

        # Load robot pose
        robot_pose_path = os.path.join(DATA_DIR, npy_files[npy_name])
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

        # Detect ellipses
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        detector = pyced.CED(np.ascontiguousarray(roi))
        detector.run_CED()
        rotRects = detector.getEllipsesAfterCluster()
        ellipses_ = []
        for e in rotRects:
            ellipses_.append((*e.center, e.size[0] / 2, e.size[1] / 2, e.angle))

        # Match ellipses to 3D points
        matcher = UltimateSocketMatcher()
        matcher.obj_pts = obj_pts
        matcher.K = K
        matcher.dist = dist
        matcher.eMc = eMc

        final_pts, status, centers = matcher.solve(
            ellipses_, [*(result[0][:2]), *(result[0][2:] - result[0][:2])])

        if centers is None or centers.shape[0] < 4:
            print(f"  Not enough matched points ({centers.shape[0] if centers is not None else 0}), skipping")
            continue

        # Compute cMo from bMo_gt
        cMo = compute_cMo_from_bMo(bMo_gt, bMe, eMc)

        # Project all 3D object points
        proj_pts = project_points(obj_pts, cMo, K, dist)

        # Get the point indices that were used in matching
        r_idx = matcher.r_idx

        # Print per-point reprojection error
        print(f"\n  Per-point reprojection errors (using bMo_gt):")
        print(f"  {'Point':<10} {'Index':<6} {'Detected':<15} {'Projected':<15} {'Error (px)':<12}")
        print(f"  {'-' * 60}")

        per_point_errors = []
        for i, obj_idx in enumerate(r_idx):
            if i < len(centers) and obj_idx < len(proj_pts):
                det_x, det_y = centers[i]
                proj_x, proj_y = proj_pts[obj_idx]
                error = np.sqrt((det_x - proj_x) ** 2 + (det_y - proj_y) ** 2)
                per_point_errors.append(error)
                print(f"  {obj_pt_names[obj_idx]:<10} {obj_idx:<6} "
                      f"({det_x:7.2f}, {det_y:7.2f}) ({proj_x:7.2f}, {proj_y:7.2f}) {error:10.3f}")

        if per_point_errors:
            mean_error = np.mean(per_point_errors)
            max_error = np.max(per_point_errors)
            print(f"\n  Mean error: {mean_error:.3f} px, Max error: {max_error:.3f} px")
            all_frame_errors.append(mean_error)

            # Visualization
            vis_img = img.copy()

            # Draw detected centers (green) and projected points (blue)
            for i, obj_idx in enumerate(r_idx):
                if i < len(centers) and obj_idx < len(proj_pts):
                    det_x, det_y = centers[i]
                    proj_x, proj_y = proj_pts[obj_idx]

                    # Detected point (green)
                    cv2.circle(vis_img, (int(det_x), int(det_y)), 5, (0, 255, 0), -1)
                    # Projected point (blue)
                    cv2.circle(vis_img, (int(proj_x), int(proj_y)), 5, (255, 0, 0), -1)
                    # Line between them (yellow)
                    cv2.line(vis_img, (int(det_x), int(det_y)), (int(proj_x), int(proj_y)), (0, 255, 255), 1)

            # Crop to ROI and resize
            vis_roi = vis_img[roi_y_min:roi_y_max, roi_x_min:roi_x_max]
            vis_roi = cv2.resize(vis_roi, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)

            # Save result
            save_path = os.path.join(RESULT_DIR, f"{ts}_reproj_err.png")
            cv2.imwrite(save_path, vis_roi)
            cv2.imshow("Reproj Error", vis_roi)
            cv2.waitKey(1)

    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    if all_frame_errors:
        print(f"Frames processed: {len(all_frame_errors)}")
        print(f"Mean error: {np.mean(all_frame_errors):.3f} px")
        print(f"Max error: {np.max(all_frame_errors):.3f} px")
        print(f"Min error: {np.min(all_frame_errors):.3f} px")

        # Per-frame breakdown
        print(f"\nPer-frame mean errors:")
        for i, err in enumerate(all_frame_errors):
            print(f"  Frame {i + 1}: {err:.3f} px")
    else:
        print("No valid frames processed")

    cv.destroyAllWindows()
