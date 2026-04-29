#!/usr/bin/env python3
"""
EKF-based pose estimation combining camera and robot measurements.
- State: object pose in base frame (bMo), 6D [position(3) + euler(3)]
- Prediction: constant velocity model (optional, or static)
- Update: camera-derived bMo from PnP/BA + robot_pose @ eMc
"""
import os
import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.optimize import linear_sum_assignment
from itertools import permutations, combinations
from scipy.spatial.transform import Rotation
from dataclasses import dataclass, field
from typing import Optional
import time

# ============ Config ============
DATA_DIR = "dataset"

# 眼手标定外参: 相机在机械臂末端坐标系上的位姿
eMc = np.array([[-6.9855857e-01,  7.1512282e-01,  2.4804471e-02, -5.1826664e+01],
                [-7.1555281e-01, -6.9815123e-01, -2.3854841e-02,  5.5274796e+01],
                [ 2.5813223e-04, -3.4412913e-02,  9.9940765e-01,  9.5362617e+01],
                [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  1.0000000e+00]],
               dtype=np.float64)


# ============ EKF Implementation ============

@dataclass
class EKFState:
    """EKF state: 6D pose [position(3), euler(3)]"""
    mean: np.ndarray  # 6D: [x, y, z, rx, ry, rz] (euler degrees)
    cov: np.ndarray   # 6x6 covariance

    @staticmethod
    def from_bmo(bmo: np.ndarray):
        pos = bmo[:3, 3]
        euler = Rotation.from_matrix(bmo[:3, :3]).as_euler('xyz', True)
        return EKFState(mean=np.concatenate([pos, euler]), cov=np.eye(6) * 10.0)

    def to_bmo(self) -> np.ndarray:
        bmo = np.eye(4)
        bmo[:3, 3] = self.mean[:3]
        bmo[:3, :3] = Rotation.from_euler('xyz', self.mean[3:6], True).as_matrix()
        return bmo


class EKFFilter:
    """
    Static EKF: T_bo is constant throughout the process.
    State: 6D [position(3), euler(3)]
    Measurement: 6D pose from camera (noisy)
    Purpose: fuse noisy measurements into a stable estimate.
    """

    def __init__(self, meas_noise_pos: float = 20.0, meas_noise_rot: float = 2.0):
        self.R = np.diag([meas_noise_pos] * 3 + [meas_noise_rot] * 3)
        self.state: Optional[EKFState] = None
        self.initialized = False

    def initialize(self, bmo: np.ndarray):
        self.state = EKFState.from_bmo(bmo)
        self.initialized = True

    def predict(self, dt: float = 1.0):
        """Static model: state unchanged, no motion."""
        if not self.initialized:
            return

    def update(self, z_bmo: np.ndarray):
        """Update with 6D pose measurement."""
        if not self.initialized:
            self.initialize(z_bmo)
            return

        z = np.concatenate([z_bmo[:3, 3],
                            Rotation.from_matrix(z_bmo[:3, :3]).as_euler('xyz', True)])

        innov = z - self.state.mean
        innov[3:] = ((innov[3:] + 180) % 360) - 180

        H = np.eye(6)
        S = H @ self.state.cov @ H.T + self.R
        K = self.state.cov @ H.T @ np.linalg.inv(S)

        self.state.mean = self.state.mean + K @ innov
        self.state.cov = (np.eye(6) - K @ H) @ self.state.cov

    def get_state(self) -> Optional[EKFState]:
        return self.state


# ============ Core Classes (from eval_pose_offline.py) ============

class ReprojectionErrorFunctor:
    def __init__(self, pts3d, observations, K, dist):
        self.pts3d = pts3d
        self.observations = observations
        self.K = K
        self.dist = dist
        self.fx, self.fy = K[0, 0], K[1, 1]
        self.cx, self.cy = K[0, 2], K[1, 2]
        self.k1, self.k2, self.p1, self.p2, self.k3 = dist

    def __call__(self, params):
        residuals = []
        num_frames = len(self.observations)
        for k in range(num_frames):
            pose = params[k * 6:(k + 1) * 6]
            rvec, tvec = pose[:3], pose[3:6]
            R, _ = cv2.Rodrigues(rvec)
            for i in range(len(self.pts3d)):
                pt3d = self.pts3d[i]
                pt_cam = R @ pt3d + tvec
                x, y = pt_cam[0] / pt_cam[2], pt_cam[1] / pt_cam[2]
                r2 = x * x + y * y
                radial = 1 + self.k1 * r2 + self.k2 * r2 * r2 + self.k3 * r2 * r2 * r2
                x_dist = x * radial + 2 * self.p1 * x * y + self.p2 * (r2 + 2 * x * x)
                y_dist = y * radial + self.p1 * (r2 + 2 * y * y) + 2 * self.p2 * x * y
                u = self.fx * x_dist + self.cx
                v = self.fy * y_dist + self.cy
                residuals.append(u - self.observations[k][i][0])
                residuals.append(v - self.observations[k][i][1])
        return np.array(residuals)


class SingleFrameEstimator:
    def __init__(self, obj_pts, K, dist):
        self.obj_pts = obj_pts.astype(np.float64)
        self.K = K.astype(np.float64)
        self.dist = dist.astype(np.float64)

    def compute_reproj_error(self, rvec, tvec, pts2d):
        proj, _ = cv2.projectPoints(self.obj_pts, rvec, tvec, self.K, self.dist)
        proj = proj.reshape(-1, 2)
        errors = np.linalg.norm(proj - pts2d, axis=1)
        return float(np.mean(errors))

    def solve(self, pts2d):
        pts2d_arr = np.array(pts2d, dtype=np.float64)
        success, rvec, tvec = cv2.solvePnP(
            self.obj_pts, pts2d_arr, self.K, self.dist,
            flags=cv2.SOLVEPNP_IPPE
        )
        if not success:
            return None

        reproj_err_pnp = self.compute_reproj_error(rvec, tvec, pts2d_arr)

        params = np.concatenate([rvec.flatten(), tvec.flatten()])
        functor = ReprojectionErrorFunctor(self.obj_pts, [pts2d], self.K, self.dist)
        result = least_squares(
            functor, params, method='lm',
            ftol=1e-6, xtol=1e-6, max_nfev=100
        )
        rvec_ba = result.x[:3]
        tvec_ba = result.x[3:6]
        reproj_err_ba = self.compute_reproj_error(rvec_ba, tvec_ba, pts2d_arr)

        return {
            'rvec_pnp': rvec.flatten(), 'tvec_pnp': tvec.flatten(),
            'rvec_ba': rvec_ba, 'tvec_ba': tvec_ba,
            'reproj_err_pnp': reproj_err_pnp, 'reproj_err_ba': reproj_err_ba,
            'cost': result.cost
        }


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
        self.estimator = SingleFrameEstimator(self.obj_pts, self.K, self.dist)

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

    def estimate_single(self, pts2d):
        return self.estimator.solve(pts2d)


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


def bmo_from_camera(cMo: np.ndarray, robot_pose: np.ndarray) -> np.ndarray:
    """
    Compute object pose in base frame from camera measurement.
    bMo = robot_pose @ eMc @ cMo
    where cMo = object in camera frame (with X-180 rotation applied)
    """
    # Object rotation: 180 deg around X
    oMo = np.eye(4, dtype=np.float64)
    oMo[:3, :3] = Rotation.from_euler('xyz', [np.pi, 0, 0]).as_matrix()
    
    cMo_ = cMo @ oMo
    bMo = robot_pose @ eMc @ cMo_
    return bMo


# ============ Main ============

def main():
    from ultralytics import YOLO

    # Model
    model = YOLO("checkpoint/best.pt")

    # Estimator
    estimator = SocketPoseEstimator()
    ed = init_edge_drawing()

    # EKF filter
    ekf = EKFFilter(meas_noise_pos=20.0, meas_noise_rot=2.0)

    # Collect timestamped files
    png_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.png') and f != 'temp'])
    npy_files = {f.replace('.npy', ''): f for f in os.listdir(DATA_DIR) if f.endswith('.npy')}

    print(f"\n{'Frame':<6} {'method':<12} {'t_x(mm)':>10} {'t_y(mm)':>10} {'t_z(mm)':>10} "
          f"{'r_x(deg)':>10} {'r_y(deg)':>10} {'r_z(deg)':>10} {'reproj(pix)':>12}")
    print("-" * 110)

    frame_id = 0
    prev_ts_sec = None

    for png_file in png_files:
        ts = png_file.replace('.png', '')

        # Parse timestamp from filename (format: 2026-04-23_09_25_20_584726976776803)
        parts = ts.split('_')
        date_str = parts[0]  # 2026-04-23
        hour, minute, second = int(parts[1]), int(parts[2]), int(parts[3])
        nanos = int(parts[4])
        curr_ts_sec = hour * 3600 + minute * 60 + second + nanos / 1e9

        # Get image
        img_path = os.path.join(DATA_DIR, png_file)
        img = cv2.imread(img_path)
        if img is None:
            continue
        # 全局降低亮度 (亮度太高)
        img = np.clip(img.astype(np.float32) * 0.7, 0, 255).astype(np.uint8)

        # Get robot pose
        if ts in npy_files:
            robot_pose = np.load(os.path.join(DATA_DIR, npy_files[ts]))
        else:
            robot_pose = np.eye(4, dtype=np.float64)

        t0 = time.perf_counter_ns()

        # YOLO detection
        roi_box = yolo_detect_roi(model, img)
        if roi_box is None:
            cv2.imshow("result", img)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            continue

        roi = img[int(roi_box[1]):int(roi_box[3]), int(roi_box[0]):int(roi_box[2])]
        tl = (int(roi_box[0]), int(roi_box[1]))

        # Ellipse detection + template matching
        ellipses = detect_ellipses(ed, roi)
        final_pts, score = estimator.solve(ellipses)
        if final_pts is None:
            print(f"Frame {frame_id}: matching failed, score={score}")
            cv2.imshow("result", img)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            continue

        pts_img = final_pts + np.array(tl)

        # PnP + BA
        result = estimator.estimate_single(pts_img)
        if result is None:
            continue

        # Build camera pose matrix (cMo)
        cMo = np.eye(4, dtype=np.float64)
        cMo[:3, :3] = Rotation.from_rotvec(result['rvec_ba']).as_matrix()
        cMo[:3, 3] = result['tvec_ba'].flatten()

        # Compute bMo from camera
        bMo_camera = bmo_from_camera(cMo, robot_pose)

        # EKF predict
        if prev_ts_sec is not None:
            dt = curr_ts_sec - prev_ts_sec
            ekf.predict(dt)
        prev_ts_sec = curr_ts_sec

        # EKF update
        ekf.update(bMo_camera)

        # Get filtered state
        state = ekf.get_state()
        if state is None:
            continue

        bMo_ekf = state.to_bmo()

        elapsed = (time.perf_counter_ns() - t0) / 1e6

        # Euler angles
        euler_camera = Rotation.from_matrix(bMo_camera[:3, :3]).as_euler('xyz', True)
        euler_ekf = Rotation.from_matrix(bMo_ekf[:3, :3]).as_euler('xyz', True)

        print(f"\n--- Frame {frame_id} ({elapsed:.1f}ms) ---")
        print(f"{frame_id:<6} {'Raw(camera)':<12} {bMo_camera[0,3]:>10.2f} {bMo_camera[1,3]:>10.2f} {bMo_camera[2,3]:>10.2f} "
              f"{euler_camera[0]:>10.2f} {euler_camera[1]:>10.2f} {euler_camera[2]:>10.2f} {result['reproj_err_ba']:>12.4f}")
        print(f"{frame_id:<6} {'EKF':<12} {bMo_ekf[0,3]:>10.2f} {bMo_ekf[1,3]:>10.2f} {bMo_ekf[2,3]:>10.2f} "
              f"{euler_ekf[0]:>10.2f} {euler_ekf[1]:>10.2f} {euler_ekf[2]:>10.2f} {'---':>12}")

        # Position difference between raw and filtered
        diff = np.linalg.norm(bMo_camera[:3, 3] - bMo_ekf[:3, 3])
        print(f"  [Filter delta position: {diff:.2f}mm]")

        # Visualize
        vis = img.copy()
        for i, (x, y) in enumerate(pts_img):
            cv2.circle(vis, (int(x), int(y)), 5, (0, 255, 0), -1)
            cv2.putText(vis, str(i), (int(x) + 5, int(y)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.drawFrameAxes(vis, estimator.K.astype(np.float32), estimator.dist.astype(np.float32),
                          result['rvec_ba'].astype(np.float32), result['tvec_ba'].astype(np.float32), 40, 2)
        cv2.imshow("result", vis)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        frame_id += 1

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
