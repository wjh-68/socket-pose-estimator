#!/usr/bin/env python3
"""
Optimized offline pose evaluation with bundle adjustment.
Optimizes:
  1. cMo: socket target pose in camera frame
  2. bMo: socket target pose in base frame

Error terms:
  1. BA reprojection error: 3D-2D reprojection in image
  2. Static pose constraint: bMo = robot_pose @ eMc @ cMo

Robot arm measurement is weighted higher (more accurate).
"""
import os
import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.optimize import linear_sum_assignment
from itertools import combinations, permutations
from scipy.spatial.transform import Rotation
import time


# ============ Config ============
DATA_DIR = "dataset"

# Camera on robot end-effector (eye-to-hand extrinsic)
eMc = np.array([[-6.9855857e-01,  7.1512282e-01,  2.4804471e-02, -5.1826664e+01],
                [-7.1555281e-01, -6.9815123e-01, -2.3854841e-02,  5.5274796e+01],
                [ 2.5813223e-04, -3.4412913e-02,  9.9940765e-01,  9.5362617e+01],
                [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  1.0000000e+00]],
               dtype=np.float64)

# Weight for static pose constraint vs reprojection error
# Robot arm measurement is more accurate, so weight its constraint higher
REPROJ_WEIGHT = 1.0
POSE_CONSTRAINT_WEIGHT = 10.0  # robot arm pose constraint weight


# ============ Core Classes ============

class SocketPoseEstimator:
    def __init__(self):
        self.obj_pts = np.array([
            [-8.0, 11.2, 0.0], [8.0, 11.2, 0.0],
            [-16.0, 0.0, 0.0], [0.0, 0.0, 0.0], [16.0, 0.0, 0.0],
            [-8.0, -13.9, 0.0], [8.0, -13.9, 0.0]
        ], dtype=np.float64)
        self.tmp_types = [0, 0, 1, 1, 1, 1, 1]
        self.K = np.array([
            [1015.445938660267, 0., 638.51741890470555],
            [0., 1015.445938660267, 386.838616473841],
            [0., 0., 1.]
        ], dtype=np.float64)
        self.dist = np.array([
            0.11753195467413819, -0.19301774104640848,
            0.00016793575097772418, -0.00061144051421409198, 0.072260521199194336
        ], dtype=np.float64)

    def _clean_and_classify(self, ellipses, dist_thresh=10):
        if not ellipses:
            return []
        nodes = []
        for e in ellipses:
            if 0.9 < e[2] / e[3] < 1.1 and e[2] + e[3] < 80:
                nodes.append({'c': np.array([e[0], e[1]]), 'd': (e[2] + e[3])})
        merged = []
        used = [False] * len(nodes)
        for i in range(len(nodes)):
            if used[i]:
                continue
            cluster = [nodes[i]]
            used[i] = True
            for j in range(i + 1, len(nodes)):
                if not used[j] and np.linalg.norm(nodes[i]['c'] - nodes[j]['c']) < dist_thresh:
                    cluster.append(nodes[j])
                    used[j] = True
            avg_c = np.mean([n['c'] for n in cluster], axis=0)
            max_d = max([n['d'] for n in cluster])
            merged.append({'p': avg_c, 'size': max_d, 'is_double': len(cluster) >= 2})
        merged = [m for m in merged if 10 < m['size'] < 100]
        return merged

    def _gap_method_threshold(self, candidates):
        candidates.sort(key=lambda x: x['size'])
        sizes = [c['size'] for c in candidates]
        gaps = [sizes[i + 1] / sizes[i] for i in range(len(sizes) - 1)]
        split_idx = np.argmax(gaps)
        threshold = (sizes[split_idx] + sizes[split_idx + 1]) / 2
        for c in candidates:
            c['t'] = 1 if c['size'] > threshold else 0

    def _get_signed_area(self, pts):
        return (pts[1][0] - pts[0][0]) * (pts[2][1] - pts[0][1]) - \
               (pts[1][1] - pts[0][1]) * (pts[2][0] - pts[0][0])

    def _evaluate_refined(self, H, template_pts, candidates, dist_thresh=15):
        proj = cv2.perspectiveTransform(template_pts.reshape(-1, 1, 2), H).reshape(-1, 2)
        det_pts = np.array([c['p'] for c in candidates])
        diff = proj[:, np.newaxis, :] - det_pts[np.newaxis, :, :]
        dist_matrix = np.linalg.norm(diff, axis=2)
        row_ind, col_ind = linear_sum_assignment(dist_matrix)
        valid_errors = [dist_matrix[r, c] for r, c in zip(row_ind, col_ind) if dist_matrix[r, c] < dist_thresh]
        inlier_count = len(valid_errors)
        if inlier_count == 0:
            return -9999, proj
        rmse = np.sqrt(np.mean(np.square(valid_errors)))
        score = (inlier_count * 1000) - rmse
        return score, proj

    def solve(self, raw_ellipses):
        candidates = self._clean_and_classify(raw_ellipses)
        if len(candidates) < 4:
            return None, 0
        self._gap_method_threshold(candidates)
        best_H, max_score, final_res = None, 0, None
        tmp_combos = [{'idx': indices, 'types': tuple(sorted([self.tmp_types[i] for i in indices]))}
                      for indices in combinations(range(7), 4)]
        det_indices = list(range(len(candidates)))
        for d_idx_tuple in combinations(det_indices, 4):
            d_subset = [candidates[i] for i in d_idx_tuple]
            d_types_signature = tuple(sorted([d['t'] for d in d_subset]))
            for t_combo in tmp_combos:
                src_pts = self.obj_pts[list(t_combo['idx'])][:, :2]
                src_area_sign = np.sign(self._get_signed_area(src_pts))
                for p_d_subset in permutations(d_subset):
                    dst_pts = np.array([d['p'] for d in p_d_subset], dtype=np.float32)
                    if np.sign(self._get_signed_area(dst_pts)) == src_area_sign:
                        continue
                    H, _ = cv2.findHomography(src_pts, dst_pts)
                    if H is None or np.linalg.det(H[:2, :2]) > 0:
                        continue
                    score, proj = self._evaluate_refined(H, self.obj_pts[:, :2], candidates)
                    if score > max_score:
                        max_score, best_H, final_res = score, H, proj
        return (final_res, max_score) if best_H is not None else (None, 0)

    def estimate_single_pnp(self, pts2d):
        """Solve PnP for single frame, return rvec, tvec"""
        pts2d_arr = np.array(pts2d, dtype=np.float64)
        success, rvec, tvec = cv2.solvePnP(
            self.obj_pts, pts2d_arr, self.K, self.dist,
            flags=cv2.SOLVEPNP_IPPE
        )
        if not success:
            return None
        return rvec.flatten(), tvec.flatten()

    def project_points(self, rvec, tvec):
        """Project 3D points to image"""
        proj, _ = cv2.projectPoints(self.obj_pts, rvec, tvec, self.K, self.dist)
        return proj.reshape(-1, 2)

    def compute_reproj_error(self, rvec, tvec, pts2d):
        """Compute mean reprojection error"""
        proj = self.project_points(rvec, tvec)
        return float(np.mean(np.linalg.norm(proj - pts2d, axis=1)))


# ============ BA Optimization ============

def pose_to_params(bMo, cMo):
    """Convert pose matrices to optimization parameters"""
    bMo_rvec = Rotation.from_matrix(bMo[:3, :3]).as_rotvec()
    bMo_tvec = bMo[:3, 3]
    cMo_rvec = Rotation.from_matrix(cMo[:3, :3]).as_rotvec()
    cMo_tvec = cMo[:3, 3]
    return np.concatenate([bMo_rvec, bMo_tvec, cMo_rvec, cMo_tvec])


def params_to_pose(params):
    """Convert optimization parameters to pose matrices"""
    bMo_rvec = params[:3]
    bMo_tvec = params[3:6]
    cMo_rvec = params[6:9]
    cMo_tvec = params[9:12]

    bMo = np.eye(4)
    bMo[:3, :3] = Rotation.from_rotvec(bMo_rvec).as_matrix()
    bMo[:3, 3] = bMo_tvec

    cMo = np.eye(4)
    cMo[:3, :3] = Rotation.from_rotvec(cMo_rvec).as_matrix()
    cMo[:3, 3] = cMo_tvec

    return bMo, cMo


class BundleAdjustmentOptimizer:
    """
    Optimizes bMo and cMo jointly with two error terms:
    1. Reprojection error: projects bMo points to image and compares with observations
    2. Pose constraint: bMo = robot_pose @ eMc @ cMo (robot arm measurement constraint)

    Weights: REPROJ_WEIGHT for reprojection, POSE_CONSTRAINT_WEIGHT for pose constraint
    """
    def __init__(self, obj_pts, K, dist, observations, robot_poses,
                 reproj_weight=REPROJ_WEIGHT, pose_constraint_weight=POSE_CONSTRAINT_WEIGHT):
        self.obj_pts = obj_pts
        self.K = K
        self.dist = dist
        self.observations = observations  # list of (N, 2) arrays per frame
        self.robot_poses = robot_poses     # list of (4, 4) matrices per frame
        self.reproj_weight = reproj_weight
        self.pose_constraint_weight = pose_constraint_weight
        self.fx, self.fy = K[0, 0], K[1, 1]
        self.cx, self.cy = K[0, 2], K[1, 2]
        self.k1, self.k2, self.p1, self.p2, self.k3 = dist

    def _project_with_distortion(self, pts3d, rvec, tvec):
        """Project 3D points with lens distortion"""
        R, _ = cv2.Rodrigues(rvec)
        pts_cam = R @ pts3d.T + tvec.reshape(3, 1)
        x, y = pts_cam[0] / pts_cam[2], pts_cam[1] / pts_cam[2]
        r2 = x * x + y * y
        radial = 1 + self.k1 * r2 + self.k2 * r2 * r2 + self.k3 * r2 * r2 * r2
        x_dist = x * radial + 2 * self.p1 * x * y + self.p2 * (r2 + 2 * x * x)
        y_dist = y * radial + self.p1 * (r2 + 2 * y * y) + 2 * self.p2 * x * y
        u = self.fx * x_dist + self.cx
        v = self.fy * y_dist + self.cy
        return np.stack([u, v], axis=1)

    def __call__(self, params):
        residuals = []
        bMo, cMo = params_to_pose(params)
        num_frames = len(self.observations)

        # Object rotation: 180 deg around X axis (socket orientation)
        oMo = np.eye(4)
        oMo[:3, :3] = Rotation.from_euler('xyz', [np.pi, 0, 0]).as_matrix()

        for k in range(num_frames):
            robot_pose = self.robot_poses[k]
            obs = self.observations[k]

            # Compute cMo_k from optimized bMo and known transforms
            # bMo = robot_pose @ eMc @ cMo_k => cMo_k = eMc^-1 @ robot_pose^-1 @ bMo
            cMo_k = np.linalg.inv(eMc) @ np.linalg.inv(robot_pose) @ bMo

            # Apply object rotation to get correct socket frame
            cMo_k_oriented = cMo_k @ oMo

            # Extract rvec, tvec
            rvec = Rotation.from_matrix(cMo_k_oriented[:3, :3]).as_rotvec()
            tvec = cMo_k_oriented[:3, 3]

            # Reprojection error
            proj = self._project_with_distortion(self.obj_pts, rvec, tvec)
            reproj_residuals = (proj - obs).flatten()
            residuals.extend(self.reproj_weight * reproj_residuals)

            # Pose constraint: bMo should equal robot_pose @ eMc @ cMo
            # But cMo here is the full optimized cMo, not cMo_k
            # The constraint is: bMo = robot_pose @ eMc @ cMo
            expected_bMo = robot_pose @ eMc @ cMo @ oMo
            pose_residuals = (bMo @ np.linalg.inv(expected_bMo))[:3, 3]  # translation difference
            residuals.extend(self.pose_constraint_weight * pose_residuals)

        return np.array(residuals)


def run_bundle_adjustment(obj_pts, K, dist, observations, robot_poses,
                          bMo_init, cMo_init,
                          reproj_weight=REPROJ_WEIGHT,
                          pose_constraint_weight=POSE_CONSTRAINT_WEIGHT):
    """Run BA optimization"""
    optimizer = BundleAdjustmentOptimizer(
        obj_pts, K, dist, observations, robot_poses,
        reproj_weight, pose_constraint_weight
    )

    initial_params = pose_to_params(bMo_init, cMo_init)

    result = least_squares(
        optimizer, initial_params,
        method='lm',
        ftol=1e-8, xtol=1e-8, max_nfev=5000
    )

    bMo_opt, cMo_opt = params_to_pose(result.x)
    return bMo_opt, cMo_opt, result


# ============ Helpers ============

def init_edge_drawing():
    Params = cv2.ximgproc.EdgeDrawing.Params()
    ed = cv2.ximgproc.createEdgeDrawing()
    Params.EdgeDetectionOperator = 1
    Params.MinPathLength = 45
    Params.PFmode = 0
    Params.NFAValidation = True
    Params.GradientThresholdValue = 30
    ed.setParams(Params)
    return ed


def detect_ellipses(ed, roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    ed.detectEdges(gray)
    ellipses = ed.detectEllipses()
    if ellipses is None:
        return []
    result = []
    for e in ellipses:
        if e[0][2] == 0:
            result.append([e[0][0], e[0][1], e[0][3], e[0][4], e[0][5]])
        else:
            result.append((e[0][0], e[0][1], e[0][2], e[0][2], 0))
    return result


def yolo_detect_roi(model, img, conf=0.5):
    results = model(img)
    if len(results) == 0:
        return None
    boxes = results[0].boxes.xyxy.cpu().numpy()
    if len(boxes) == 0:
        return None
    return boxes[0]


# ============ Main ============

def main():
    from ultralytics import YOLO

    # Model
    model = YOLO("checkpoint/best.pt")

    # Estimator
    estimator = SocketPoseEstimator()
    ed = init_edge_drawing()

    # Collect timestamped files
    png_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.png') and f != 'temp'])
    npy_files = {f.replace('.npy', ''): f for f in os.listdir(DATA_DIR) if f.endswith('.npy')}

    # Collect all valid frame data for batch BA
    frame_data = []

    print("Collecting frame data...")
    for png_file in png_files:
        ts = png_file.replace('.png', '')

        # Get image
        img_path = os.path.join(DATA_DIR, png_file)
        img = cv2.imread(img_path)
        img = np.clip(img.astype(np.float32) * 0.7, 0, 255).astype(np.uint8)

        # Get robot pose
        if ts not in npy_files:
            continue
        robot_pose_path = os.path.join(DATA_DIR, npy_files[ts])
        robot_pose = np.load(robot_pose_path)

        # YOLO detection
        roi_box = yolo_detect_roi(model, img)
        if roi_box is None:
            continue

        roi = img[int(roi_box[1]):int(roi_box[3]), int(roi_box[0]):int(roi_box[2])]
        tl = (int(roi_box[0]), int(roi_box[1]))

        # Ellipse detection + template matching
        ellipses = detect_ellipses(ed, roi)
        final_pts, score = estimator.solve(ellipses)
        if final_pts is None:
            continue

        pts_img = final_pts + np.array(tl)

        # PnP initial estimate
        pnp_result = estimator.estimate_single_pnp(pts_img)
        if pnp_result is None:
            continue

        rvec, tvec = pnp_result

        # Build cMo (socket in camera frame)
        cMo = np.eye(4)
        cMo[:3, :3] = Rotation.from_rotvec(rvec).as_matrix()
        cMo[:3, 3] = tvec

        # Object rotation (180 deg around X)
        oMo = np.eye(4)
        oMo[:3, :3] = Rotation.from_euler('xyz', [np.pi, 0, 0]).as_matrix()
        cMo_oriented = cMo @ oMo

        # Compute bMo initial estimate
        bMo_init = robot_pose @ eMc @ cMo_oriented

        frame_data.append({
            'robot_pose': robot_pose,
            'pts_img': pts_img.astype(np.float64),
            'cMo_init': cMo_oriented.copy(),
            'bMo_init': bMo_init.copy(),
            'img_path': img_path,
            'robot_pose_path': robot_pose_path
        })

    print(f"Collected {len(frame_data)} valid frames")

    if len(frame_data) < 2:
        print("Not enough frames for BA optimization")
        return

    # ============ Run BA Optimization ============
    print("\n=== Running Bundle Adjustment ===")

    observations = [f['pts_img'] for f in frame_data]
    robot_poses = [f['robot_pose'] for f in frame_data]

    # Use median of initial bMo estimates as starting point
    bMo_candidates = [f['bMo_init'] for f in frame_data]
    bMo_init = np.mean(bMo_candidates, axis=0)
    # Ensure proper rotation matrix
    U, S, Vt = np.linalg.svd(bMo_init[:3, :3])
    bMo_init[:3, :3] = U @ Vt

    # Use first frame's cMo as initial
    cMo_init = frame_data[0]['cMo_init']

    # Run BA
    bMo_opt, cMo_opt, ba_result = run_bundle_adjustment(
        estimator.obj_pts, estimator.K, estimator.dist,
        observations, robot_poses,
        bMo_init, cMo_init,
        reproj_weight=REPROJ_WEIGHT,
        pose_constraint_weight=POSE_CONSTRAINT_WEIGHT
    )

    print(f"BA optimization: cost={ba_result.cost:.4f}, nfev={ba_result.nfev}")

    # ============ Evaluate Results ============
    oMo = np.eye(4)
    oMo[:3, :3] = Rotation.from_euler('xyz', [np.pi, 0, 0]).as_matrix()

    print(f"\n{'Frame':<6} {'method':<12} {'t_x(mm)':>10} {'t_y(mm)':>10} {'t_z(mm)':>10} "
          f"{'r_x(deg)':>10} {'r_y(deg)':>10} {'r_z(deg)':>10} {'reproj(pix)':>12}")
    print("-" * 120)

    for idx, f in enumerate(frame_data):
        robot_pose = f['robot_pose']
        pts_img = f['pts_img']

        # Compute cMo from optimized bMo
        cMo_from_bMo = np.linalg.inv(eMc) @ np.linalg.inv(robot_pose) @ bMo_opt
        cMo_oriented = cMo_from_bMo @ oMo

        # Extract rvec, tvec
        rvec = Rotation.from_matrix(cMo_oriented[:3, :3]).as_rotvec()
        tvec = cMo_oriented[:3, 3]

        # Reprojection error
        reproj_err = estimator.compute_reproj_error(rvec, tvec, pts_img)

        # bMo from optimized cMo
        bMo_from_cMo = robot_pose @ eMc @ cMo_opt @ oMo

        euler = Rotation.from_matrix(bMo_opt[:3, :3]).as_euler('xyz', True)

        print(f"{idx:<6} {'BA_opt':<12} {bMo_opt[0,3]:>10.2f} {bMo_opt[1,3]:>10.2f} {bMo_opt[2,3]:>10.2f} "
              f"{euler[0]:>10.2f} {euler[1]:>10.2f} {euler[2]:>10.2f} {reproj_err:>12.4f}")

    # ============ Print Optimization Summary ============
    print("\n=== Optimization Summary ===")
    print(f"Optimized bMo (socket in base):")
    print(f"  Translation: {bMo_opt[:3, 3]}")
    euler_opt = Rotation.from_matrix(bMo_opt[:3, :3]).as_euler('xyz', True)
    print(f"  Euler angles (deg): {euler_opt}")

    print(f"\nOptimized cMo (socket in camera, with object rotation applied):")
    print(f"  Translation: {cMo_opt[:3, 3]}")
    euler_cmo = Rotation.from_matrix(cMo_opt[:3, :3]).as_euler('xyz', True)
    print(f"  Euler angles (deg): {euler_cmo}")

    # Pose consistency check
    print("\n=== Pose Consistency Check ===")
    for idx, f in enumerate(frame_data[:5]):  # Check first 5 frames
        robot_pose = f['robot_pose']
        bMo_check = robot_pose @ eMc @ cMo_opt @ oMo
        trans_diff = np.linalg.norm(bMo_opt[:3, 3] - bMo_check[:3, 3])
        print(f"Frame {idx}: ||bMo_opt - robot@eMc@cMo|| = {trans_diff:.4f} mm")

    print("\nDone!")


if __name__ == '__main__':
    main()
