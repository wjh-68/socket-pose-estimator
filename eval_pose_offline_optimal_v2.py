#!/usr/bin/env python3
"""
Optimized pose evaluation with sliding window BA.
- Processes each frame incrementally: detect -> optimize -> output
- Sliding window controls optimization frame count
- Outputs bMo and cMo after each frame
"""
import os
import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.optimize import linear_sum_assignment
from itertools import combinations, permutations
from scipy.spatial.transform import Rotation
import time
import json
from collections import deque


# ============ Config ============
DATA_DIR = "dataset/savedata4"
DEBUG_DIR = "debug"
SLIDING_WINDOW_SIZE = 20  # Number of frames in optimization window


# Camera on robot end-effector (eye-to-hand extrinsic)
eMc = np.array([[-6.9855857e-01,  7.1512282e-01,  2.4804471e-02, -5.1826664e+01],
                [-7.1555281e-01, -6.9815123e-01, -2.3854841e-02,  5.5274796e+01],
                [ 2.5813223e-04, -3.4412913e-02,  9.9940765e-01,  9.5362617e+01],
                [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  1.0000000e+00]],
               dtype=np.float64)


# ============ Covariance Configuration ============
SIGMA_PIX = 6.0       # pixels
COV_PIXEL = np.eye(2) * (SIGMA_PIX ** 2)

SIGMA_TRANS = 0.5     # mm
SIGMA_ROT = 0.05      # degrees
SIGMA_ROT_RAD = np.radians(SIGMA_ROT)

COV_POSE = np.diag([
    SIGMA_TRANS**2, SIGMA_TRANS**2, SIGMA_TRANS**2,
    SIGMA_ROT_RAD**2, SIGMA_ROT_RAD**2, SIGMA_ROT_RAD**2
])


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

    # 去除形状(太扁),尺寸不合理的椭圆,合并同心圆
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
        t3 = time.perf_counter_ns()
        # 
        centers = self._clean_and_classify(raw_ellipses)

    # solve wxj-version
    def solve2(self, raw_ellipses):
        """Template matching: returns matched points, score, and all candidates"""
        t3 = time.perf_counter_ns()
        candidates = self._clean_and_classify(raw_ellipses)

        t4 = time.perf_counter_ns()
        if len(candidates) < 4:
            return None, 0, []
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
        # after finding H                        
        t5 = time.perf_counter_ns()
        print(f"clean and classify: {(t4-t3)/1e6:.1f}ms\n"
              f"find H: {(t5-t4)/1e6:.1f}ms")

        return (final_res, max_score, candidates) if best_H is not None else (None, 0, candidates)

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
        proj, _ = cv2.projectPoints(self.obj_pts, rvec, tvec, self.K, self.dist)
        return proj.reshape(-1, 2)

    def compute_reproj_error(self, rvec, tvec, pts2d):
        proj = self.project_points(rvec, tvec)
        return float(np.mean(np.linalg.norm(proj - pts2d, axis=1)))


# ============ BA Optimization ============

def pose_to_params(bMo, cMo):
    bMo_rvec = Rotation.from_matrix(bMo[:3, :3]).as_rotvec()
    bMo_tvec = bMo[:3, 3]
    cMo_rvec = Rotation.from_matrix(cMo[:3, :3]).as_rotvec()
    cMo_tvec = cMo[:3, 3]
    return np.concatenate([bMo_rvec, bMo_tvec, cMo_rvec, cMo_tvec])


def params_to_pose(params):
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


def compute_pose_error(bMo_est, bMo_true):
    trans_error = np.linalg.norm(bMo_est[:3, 3] - bMo_true[:3, 3])
    R_diff = bMo_est[:3, :3] @ bMo_true[:3, :3].T
    rot_error = np.arccos(np.clip((np.trace(R_diff) - 1) / 2, -1, 1))
    return trans_error, rot_error


class BundleAdjustmentOptimizer:
    """
    Optimizes bMo and cMo with two covariance-weighted error terms:
    1. Reprojection error (2D per point)
    2. Pose constraint error (6D per frame)
    """

    def __init__(self, obj_pts, K, dist, observations, robot_poses):
        self.obj_pts = obj_pts
        self.K = K
        self.dist = dist
        self.observations = observations
        self.robot_poses = robot_poses
        self.fx, self.fy = K[0, 0], K[1, 1]
        self.cx, self.cy = K[0, 2], K[1, 2]
        self.k1, self.k2, self.p1, self.p2, self.k3 = dist

    def _project_with_distortion(self, pts3d, rvec, tvec):
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

    def _compute_pose_residual_6d(self, bMo, expected_bMo):
        trans_residual = (bMo[:3, 3] - expected_bMo[:3, 3])
        R_diff = bMo[:3, :3] @ expected_bMo[:3, :3].T
        rotvec = Rotation.from_matrix(R_diff).as_rotvec()
        if np.linalg.norm(rotvec) > 1e-6:
            angle = np.linalg.norm(rotvec)
            axis = rotvec / angle
            rot_residual = axis * angle
        else:
            rot_residual = np.zeros(3)
        return np.concatenate([trans_residual, rot_residual])

    def __call__(self, params):
        residuals = []
        bMo, cMo = params_to_pose(params)
        num_frames = len(self.observations)

        oMo = np.eye(4)
        oMo[:3, :3] = Rotation.from_euler('xyz', [np.pi, 0, 0]).as_matrix()

        for k in range(num_frames):
            robot_pose = self.robot_poses[k]
            obs = self.observations[k]

            cMo_k = np.linalg.inv(eMc) @ np.linalg.inv(robot_pose) @ bMo
            cMo_k_oriented = cMo_k @ oMo

            rvec = Rotation.from_matrix(cMo_k_oriented[:3, :3]).as_rotvec()
            tvec = cMo_k_oriented[:3, 3]

            # Reprojection error weighted by pixel covariance
            proj = self._project_with_distortion(self.obj_pts, rvec, tvec)
            reproj_residuals = (proj - obs)
            for residual_2d in reproj_residuals:
                weighted_residual = residual_2d / SIGMA_PIX
                residuals.extend(weighted_residual)

            # Pose constraint error weighted by pose covariance
            expected_bMo = robot_pose @ eMc @ cMo @ oMo
            pose_residual_6d = self._compute_pose_residual_6d(bMo, expected_bMo)
            weighted_trans = pose_residual_6d[:3] / SIGMA_TRANS
            weighted_rot = pose_residual_6d[3:] / SIGMA_ROT_RAD
            residuals.extend(np.concatenate([weighted_trans, weighted_rot]))

        return np.array(residuals)


def run_bundle_adjustment(obj_pts, K, dist, observations, robot_poses,
                          bMo_init, cMo_init):
    """Run BA optimization"""
    optimizer = BundleAdjustmentOptimizer(obj_pts, K, dist, observations, robot_poses)
    initial_params = pose_to_params(bMo_init, cMo_init)

    result = least_squares(
        optimizer, initial_params,
        method='lm',
        ftol=1e-8, xtol=1e-8, max_nfev=5000
    )

    bMo_opt, cMo_opt = params_to_pose(result.x)
    return bMo_opt, cMo_opt, result


# ============ Visualization ============

def draw_debug_visualization(img, candidates, pts_img, rvec_ba, tvec_ba,
                              bMo, cMo, frame_id, reproj_err_pnp, reproj_err_ba,
                              window_size, method='BA'):
    """Draw comprehensive debug visualization"""
    vis = img.copy()

    # Draw all detected candidates (gray)
    for i, c in enumerate(candidates):
        cv2.circle(vis, (int(c['p'][0]), int(c['p'][1])), 3, (100, 100, 100), -1)

    # Draw matched circle centers with IDs
    colors = [(0, 255, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0),
              (255, 0, 0), (0, 165, 255), (128, 0, 128)]
    for i, (x, y) in enumerate(pts_img):
        color = colors[i % len(colors)]
        cv2.circle(vis, (int(x), int(y)), 8, color, 2)
        cv2.putText(vis, str(i), (int(x) + 10, int(y) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Draw camera axes (safer version with boundary check)
    if rvec_ba is not None and tvec_ba is not None:
        try:
            # Project axis endpoints manually to check bounds
            axis_length = 20  # shorter axis to reduce chance of going out of frame
            K = np.eye(3)
            dist = np.zeros(5)
            axis_points, _ = cv2.projectPoints(
                np.array([[0,0,0], [axis_length,0,0], [0,axis_length,0], [0,0,axis_length]], dtype=np.float32),
                rvec_ba.astype(np.float32), tvec_ba.astype(np.float32), K, dist
            )
            axis_points = axis_points.reshape(-1, 2)

            # Check if all points are within image bounds
            h, w = vis.shape[:2]
            all_in_bounds = all(0 <= p[0] < w and 0 <= p[1] < h for p in axis_points)

            if all_in_bounds:
                # Draw axes manually (B, G, R for X, Y, Z)
                origin = tuple(axis_points[0].astype(int))
                cv2.line(vis, origin, tuple(axis_points[1].astype(int)), (255, 0, 0), 2)  # X - red
                cv2.line(vis, origin, tuple(axis_points[2].astype(int)), (0, 255, 0), 2)  # Y - green
                cv2.line(vis, origin, tuple(axis_points[3].astype(int)), (0, 0, 255), 2)  # Z - blue
            else:
                # Draw small indicator at origin instead
                cv2.circle(vis, tuple(axis_points[0].astype(int)), 5, (0, 255, 255), -1)
        except Exception:
            pass  # Skip axes drawing if projection fails

    # Draw info text
    euler = Rotation.from_matrix(bMo[:3, :3]).as_euler('xyz', True) if bMo is not None else [0, 0, 0]
    info_lines = [
        f"Frame: {frame_id} | Window: {window_size} | Method: {method}",
        f"bMo: [{bMo[0,3]:.1f}, {bMo[1,3]:.1f}, {bMo[2,3]:.1f}]" if bMo is not None else "bMo: N/A",
        f"Euler: [{euler[0]:.1f}, {euler[1]:.1f}, {euler[2]:.1f}]",
        f"Reproj PnP: {reproj_err_pnp:.2f} pix",
        f"Reproj BA: {reproj_err_ba:.2f} pix",
    ]

    for i, line in enumerate(info_lines):
        cv2.putText(vis, line, (10, 25 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return vis


def draw_detection_debug(roi, ellipses, candidates, final_pts, score):
    """Draw detection intermediate results"""
    vis = roi.copy()

    for e in ellipses:
        if e[2] == 0:
            center = (int(e[0]), int(e[1]))
            radius = int(e[3])
            cv2.circle(vis, center, radius, (80, 80, 80), 1)
        else:
            center = (int(e[0]), int(e[1]))
            axes = (int(e[2]), int(e[3]))
            angle = int(e[4]) if len(e) > 4 else 0
            cv2.ellipse(vis, center, axes, angle, 0, 360, (80, 80, 80), 1)

    colors_cand = [(0, 200, 200), (200, 200, 0), (0, 200, 0), (200, 0, 200),
                   (0, 150, 255), (255, 150, 0), (150, 0, 255)]
    for i, c in enumerate(candidates):
        color = colors_cand[i % len(colors_cand)]
        t = c.get('t', -1)
        label = "L" if t == 0 else "S" if t == 1 else "?"
        cv2.circle(vis, (int(c['p'][0]), int(c['p'][1])), 4, color, -1)
        cv2.putText(vis, f"{i}({label})", (int(c['p'][0]) + 5, int(c['p'][1]) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        cv2.putText(vis, f"sz:{c['size']:.0f}", (int(c['p'][0]) + 5, int(c['p'][1]) + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1)

    if final_pts is not None:
        match_colors = [(0, 255, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0),
                        (255, 0, 0), (0, 165, 255), (128, 0, 128)]
        for i, (x, y) in enumerate(final_pts):
            color = match_colors[i % len(match_colors)]
            cv2.circle(vis, (int(x), int(y)), 6, color, 2)
            cv2.putText(vis, str(i), (int(x) + 8, int(y) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

    cv2.putText(vis, f"Score: {score:.0f}", (5, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return vis


# ============ Debug File Saving ============

def save_debug_files(frame_id, debug_dir, img, roi, ellipses, candidates,
                     final_pts, score, rvec_pnp, tvec_pnp, rvec_ba, tvec_ba,
                     bMo, cMo, robot_pose, reproj_err_pnp, reproj_err_ba,
                     ba_result=None, window_size=0, method='PnP'):
    """Save all debug files for a frame"""
    frame_dir = os.path.join(debug_dir, f"frame_{frame_id:04d}")
    os.makedirs(frame_dir, exist_ok=True)

    # 1. Detection intermediate results
    det_vis = draw_detection_debug(roi, ellipses, candidates, final_pts, score)
    cv2.imwrite(os.path.join(frame_dir, "01_detection_raw.png"), det_vis)

    # 2. Full visualization
    vis = draw_debug_visualization(img, candidates, final_pts if final_pts is not None else [],
                                   rvec_ba, tvec_ba, bMo, cMo, frame_id,
                                   reproj_err_pnp, reproj_err_ba, window_size, method)
    cv2.imwrite(os.path.join(frame_dir, "02_visualization.png"), vis)

    # 3. ROI with matched points
    roi_vis = roi.copy()
    if final_pts is not None:
        match_colors = [(0, 255, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0),
                        (255, 0, 0), (0, 165, 255), (128, 0, 128)]
        for i, (x, y) in enumerate(final_pts):
            color = match_colors[i % len(match_colors)]
            cv2.circle(roi_vis, (int(x), int(y)), 8, color, 2)
            cv2.putText(roi_vis, str(i), (int(x) + 10, int(y) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    cv2.imwrite(os.path.join(frame_dir, "03_roi_matched.png"), roi_vis)

    # 4. Pose data JSON
    pose_data = {
        'frame_id': frame_id,
        'bMo': bMo.tolist() if bMo is not None else None,
        'cMo': cMo.tolist() if cMo is not None else None,
        'robot_pose': robot_pose.tolist() if robot_pose is not None else None,
        'eMc': eMc.tolist(),
        'rvec_pnp': rvec_pnp.tolist() if rvec_pnp is not None else None,
        'tvec_pnp': tvec_pnp.tolist() if tvec_pnp is not None else None,
        'rvec_ba': rvec_ba.tolist() if rvec_ba is not None else None,
        'tvec_ba': tvec_ba.tolist() if tvec_ba is not None else None,
        'reproj_err_pnp': float(reproj_err_pnp) if reproj_err_pnp is not None else None,
        'reproj_err_ba': float(reproj_err_ba) if reproj_err_ba is not None else None,
        'score': float(score),
        'window_size': window_size,
        'method': method,
        'ba_cost': float(ba_result.cost) if ba_result is not None else None,
        'ba_nfev': ba_result.nfev if ba_result is not None else None,
        'covariance_config': {
            'sigma_pixel': SIGMA_PIX,
            'sigma_trans_mm': SIGMA_TRANS,
            'sigma_rot_deg': SIGMA_ROT,
        }
    }
    with open(os.path.join(frame_dir, "04_pose_data.json"), 'w') as f:
        json.dump(pose_data, f, indent=2)

    # 5. Error metrics
    if bMo is not None and robot_pose is not None:
        oMo = np.eye(4)
        oMo[:3, :3] = Rotation.from_euler('xyz', [np.pi, 0, 0]).as_matrix()
        expected_bMo = robot_pose @ eMc @ cMo @ oMo
        trans_diff, rot_diff = compute_pose_error(bMo, expected_bMo)

        error_data = {
            'frame_id': frame_id,
            'pose_consistency_translation_mm': float(trans_diff),
            'pose_consistency_rotation_deg': float(np.degrees(rot_diff)),
            'reproj_err_pnp_pix': float(reproj_err_pnp) if reproj_err_pnp is not None else None,
            'reproj_err_ba_pix': float(reproj_err_ba) if reproj_err_ba is not None else None,
            'bMo_euler_xyz_deg': Rotation.from_matrix(bMo[:3, :3]).as_euler('xyz', True).tolist(),
        }
        with open(os.path.join(frame_dir, "05_error_metrics.json"), 'w') as f:
            json.dump(error_data, f, indent=2)

    # 6. 2D points
    if final_pts is not None:
        pts_data = {
            'frame_id': frame_id,
            'num_points': len(final_pts),
            'points_2d': [{'id': i, 'x': float(p[0]), 'y': float(p[1])} for i, p in enumerate(final_pts)],
        }
        with open(os.path.join(frame_dir, "06_points_2d.json"), 'w') as f:
            json.dump(pts_data, f, indent=2)

    # 7. Original image
    cv2.imwrite(os.path.join(frame_dir, "00_original.png"), img)


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

    os.makedirs(DEBUG_DIR, exist_ok=True)

    print("=== Covariance-Based Sliding Window BA ===")
    print(f"Window size: {SLIDING_WINDOW_SIZE}")
    print(f"Pixel sigma: {SIGMA_PIX} pix")
    print(f"Pose sigma: trans={SIGMA_TRANS} mm, rot={SIGMA_ROT} deg")
    print()

    model = YOLO("checkpoint/best.pt")
    estimator = SocketPoseEstimator()
    ed = init_edge_drawing()

    # Sliding window for frame data
    frame_window = deque(maxlen=SLIDING_WINDOW_SIZE)

    # Collect timestamped files
    png_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.png') and f != 'temp'])
    npy_files = {f.replace('.npy', ''): f for f in os.listdir(DATA_DIR) if f.endswith('.npy')}

    print(f"\n{'Frame':<6} {'Method':<8} {'t_x(mm)':>10} {'t_y(mm)':>10} {'t_z(mm)':>10} "
          f"{'r_x(deg)':>10} {'r_y(deg)':>10} {'r_z(deg)':>10} {'reproj':>8} {'Win':>4}")
    print("-" * 110)

    global_bMo_opt = None
    global_cMo_opt = None

    for png_file in png_files:
        ts = png_file.replace('.png', '')

        img_path = os.path.join(DATA_DIR, png_file)
        img = cv2.imread(img_path)
        # 降低亮度
        # img = np.clip(img.astype(np.float32) * 0.7, 0, 255).astype(np.uint8)

        if ts not in npy_files:
            continue
        robot_pose_path = os.path.join(DATA_DIR, npy_files[ts])
        robot_pose = np.load(robot_pose_path)

        # YOLO detection
        t0 = time.perf_counter_ns()
        roi_box = yolo_detect_roi(model, img)
        if roi_box is None:
            continue

        roi = img[int(roi_box[1]):int(roi_box[3]), int(roi_box[0]):int(roi_box[2])]
        tl = (int(roi_box[0]), int(roi_box[1]))

        # Ellipse detection + template matching
        t1 = time.perf_counter_ns()
        ellipses = detect_ellipses(ed, roi)
        t2 = time.perf_counter_ns()
        
        print(f"YOLO: {(t1-t0)/1e6:.1f}ms\n"
              f"detect ellipses: {(t2-t1)/1e6:.1f}ms")
        


        final_pts, score, candidates = estimator.solve(ellipses)
        if final_pts is None:
            continue

        pts_img = final_pts + np.array(tl)

        # PnP initial estimate
        t6 = time.perf_counter_ns()
        pnp_result = estimator.estimate_single_pnp(pts_img)
        t7 = time.perf_counter_ns()
        print(f"pnp: {(t7-t6)/1e6:.1f}ms")

        if pnp_result is None:
            continue

        rvec, tvec = pnp_result

        # Build cMo (socket in camera frame)
        cMo = np.eye(4)
        cMo[:3, :3] = Rotation.from_rotvec(rvec).as_matrix()
        cMo[:3, 3] = tvec

        # Object rotation (180 deg around X)
        oMo = np.eye(4)
        # oMo[:3, :3] = Rotation.from_euler('xyz', [np.pi, 0, 0]).as_matrix()
        oMo[:3, :3] = Rotation.from_euler('xyz', [0, 0, 0]).as_matrix()
        cMo_oriented = cMo @ oMo

        # Compute bMo initial estimate from PnP
        bMo_init = robot_pose @ eMc @ cMo_oriented

        reproj_err_pnp = estimator.compute_reproj_error(rvec, tvec, pts_img)
        t8 = time.perf_counter_ns()
        print(f"compute_reproj_error: {(t8-t7)/1e6:.1f}ms")

        # ============ Process this frame ============
        frame_id = len(frame_window)

        # Add to sliding window
        frame_window.append({
            'robot_pose': robot_pose,
            'pts_img': pts_img.astype(np.float64),
            'cMo_init': cMo_oriented.copy(),
            'bMo_init': bMo_init.copy(),
            'img': img,
            'roi': roi,
            'ellipses': ellipses,
            'candidates': candidates,
            'final_pts': pts_img,
            'score': score,
            'rvec_pnp': rvec,
            'tvec_pnp': tvec,
        })

        current_window_size = len(frame_window)

        # ============ Optimization based on window size ============
        if current_window_size == 1:
            # Single frame: just use PnP result
            bMo_out = bMo_init.copy()
            cMo_out = cMo_oriented.copy()
            rvec_ba = rvec
            tvec_ba = tvec
            reproj_err_ba = reproj_err_pnp
            ba_result = None
            method = 'PnP'

            print(f"{frame_id:<6} {'PnP':<8} {bMo_out[0,3]:>10.2f} {bMo_out[1,3]:>10.2f} {bMo_out[2,3]:>10.2f} "
                  f"{Rotation.from_matrix(bMo_out[:3,:3]).as_euler('xyz',True)[0]:>10.2f} "
                  f"{Rotation.from_matrix(bMo_out[:3,:3]).as_euler('xyz',True)[1]:>10.2f} "
                  f"{Rotation.from_matrix(bMo_out[:3,:3]).as_euler('xyz',True)[2]:>10.2f} "
                  f"{reproj_err_pnp:>8.2f} {current_window_size:>4}")

        else:
            # Multiple frames: run BA with sliding window
            observations = [f['pts_img'] for f in frame_window]
            robot_poses = [f['robot_pose'] for f in frame_window]

            # Initialize from global solution if available, else mean of window
            if global_bMo_opt is not None:
                bMo_init_ba = global_bMo_opt.copy()
                cMo_init_ba = global_cMo_opt.copy()
            else:
                bMo_candidates = [f['bMo_init'] for f in frame_window]
                bMo_init_ba = np.mean(bMo_candidates, axis=0)
                U, S, Vt = np.linalg.svd(bMo_init_ba[:3, :3])
                bMo_init_ba[:3, :3] = U @ Vt
                cMo_init_ba = frame_window[0]['cMo_init']

            # Run BA
            bMo_opt, cMo_opt, ba_result = run_bundle_adjustment(
                estimator.obj_pts, estimator.K, estimator.dist,
                observations, robot_poses,
                bMo_init_ba, cMo_init_ba
            )
            t9 = time.perf_counter_ns()
            print(f"optimization: {(t9-t8)/1e6:.1f}ms")

            # Update global solution
            global_bMo_opt = bMo_opt.copy()
            global_cMo_opt = cMo_opt.copy()

            # Compute per-frame output using optimized global bMo
            latest_robot_pose = frame_window[-1]['robot_pose']
            cMo_from_bMo = np.linalg.inv(eMc) @ np.linalg.inv(latest_robot_pose) @ bMo_opt
            cMo_oriented_out = cMo_from_bMo @ oMo

            rvec_ba = Rotation.from_matrix(cMo_oriented_out[:3, :3]).as_rotvec()
            tvec_ba = cMo_oriented_out[:3, 3]
            reproj_err_ba = estimator.compute_reproj_error(rvec_ba, tvec_ba, frame_window[-1]['pts_img'])

            bMo_out = bMo_opt
            cMo_out = cMo_opt
            method = 'BA'

            # bMo: after optimation
            euler_out = Rotation.from_matrix(bMo_out[:3, :3]).as_euler('xyz', True)
            print(f"{frame_id:<6} {'bMo_opt':<8} {bMo_out[0,3]:>10.2f} {bMo_out[1,3]:>10.2f} {bMo_out[2,3]:>10.2f} "
                  f"{euler_out[0]:>10.2f} {euler_out[1]:>10.2f} {euler_out[2]:>10.2f} "
                  f"{reproj_err_ba:>8.2f} {current_window_size:>4}")
            # bMo: pnp init
            bMo_pnp = frame_window[-1]['bMo_init']
            euler_out = Rotation.from_matrix(bMo_pnp[:3, :3]).as_euler('xyz', True)
            print(f"{frame_id:<6} {'bMo_pnp':<8} {bMo_pnp[0,3]:>10.2f} {bMo_pnp[1,3]:>10.2f} {bMo_pnp[2,3]:>10.2f} "
                  f"{euler_out[0]:>10.2f} {euler_out[1]:>10.2f} {euler_out[2]:>10.2f} \n")

            # cMo: after optimation
            euler_out = Rotation.from_matrix(cMo_out[:3, :3]).as_euler('xyz', True)
            print(f"{frame_id:<6} {'cMo_opt':<8} {cMo_out[0,3]:>10.2f} {cMo_out[1,3]:>10.2f} {cMo_out[2,3]:>10.2f} "
                  f"{euler_out[0]:>10.2f} {euler_out[1]:>10.2f} {euler_out[2]:>10.2f} ")
            # cMo: pnp init
            cMo_pnp = frame_window[-1]['cMo_init']
            euler_out = Rotation.from_matrix(cMo_pnp[:3, :3]).as_euler('xyz', True)
            print(f"{frame_id:<6} {'cMo_pnp':<8} {cMo_pnp[0,3]:>10.2f} {cMo_pnp[1,3]:>10.2f} {cMo_pnp[2,3]:>10.2f} "
                  f"{euler_out[0]:>10.2f} {euler_out[1]:>10.2f} {euler_out[2]:>10.2f} ")

        # ============ Save debug files ============
        t10 = time.perf_counter_ns()
        save_debug_files(
            frame_id=frame_id,
            debug_dir=DEBUG_DIR,
            img=img,
            roi=roi,
            ellipses=ellipses,
            candidates=candidates,
            final_pts=pts_img,
            score=score,
            rvec_pnp=rvec,
            tvec_pnp=tvec,
            rvec_ba=rvec_ba,
            tvec_ba=tvec_ba,
            bMo=bMo_out,
            cMo=cMo_out,
            robot_pose=robot_pose,
            reproj_err_pnp=reproj_err_pnp,
            reproj_err_ba=reproj_err_ba,
            ba_result=ba_result,
            window_size=current_window_size,
            method=method
        )
        t11 = time.perf_counter_ns()
        print(f"save files: {(t11-t10)/1e6:.1f}ms")

        # ============ Visualize ============
        vis = draw_debug_visualization(
            img, candidates, pts_img,
            rvec_ba, tvec_ba,
            bMo_out, cMo_out, frame_id,
            reproj_err_pnp, reproj_err_ba,
            current_window_size, method
        )
        t12 = time.perf_counter_ns()
        print(f"visualization: {(t12-t11)/1e6:.1f}ms")

        cv2.imshow("result", vis)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()

    print(f"\n=== Final Summary ===")
    if global_bMo_opt is not None:
        print(f"Final bMo (socket in base):")
        print(f"  Translation: {global_bMo_opt[:3, 3]}")
        euler_final = Rotation.from_matrix(global_bMo_opt[:3, :3]).as_euler('xyz', True)
        print(f"  Euler angles (deg): {euler_final}")
        print(f"\nFinal cMo (socket in camera):")
        print(f"  Translation: {global_cMo_opt[:3, 3]}")

    print(f"\nDebug files saved to: {DEBUG_DIR}/")
    print("Done!")


if __name__ == '__main__':
    main()
