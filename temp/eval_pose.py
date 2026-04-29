#!/usr/bin/env python3
"""
Pose estimation evaluation script.
For each image in dataset/images/, runs multiple solves and computes RMSE across results.
"""
import os
import cv2
import numpy as np
from scipy.optimize import least_squares
from itertools import permutations, combinations

# ============ Config ============
IMG_DIR = "dataset/images"
MODEL_PATH = "checkpoint/best.pt"
WINDOW_SIZE = 1  # single frame BA
NUM_TRIALS = 10  # number of solves per image


# ============ Core Classes (copied from socket_pose_estimator.py) ============

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
    """Single frame pose estimator with BA optimization."""

    def __init__(self, obj_pts, K, dist):
        self.obj_pts = obj_pts.astype(np.float64)
        self.K = K.astype(np.float64)
        self.dist = dist.astype(np.float64)

    def compute_reproj_error(self, rvec, tvec, pts2d):
        """Compute average reprojection error."""
        proj, _ = cv2.projectPoints(self.obj_pts, rvec, tvec, self.K, self.dist)
        proj = proj.reshape(-1, 2)
        errors = np.linalg.norm(proj - pts2d, axis=1)
        return float(np.mean(errors))

    def solve(self, pts2d):
        """Solve pose for single frame: PnP + BA, return both with reproj errors."""
        pts2d_arr = np.array(pts2d, dtype=np.float64)
        success, rvec, tvec = cv2.solvePnP(
            self.obj_pts, pts2d_arr, self.K, self.dist,
            flags=cv2.SOLVEPNP_IPPE
        )
        if not success:
            return None

        reproj_err_pnp = self.compute_reproj_error(rvec, tvec, pts2d_arr)

        # BA optimization
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
        from scipy.optimize import linear_sum_assignment
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
        """Estimate pose for single frame with BA."""
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
    return boxes[0] if len(boxes) > 0 else None


def compute_pose_rmse(poses):
    """Compute RMSE across multiple pose estimates."""
    if len(poses) < 2:
        return 0.0
    poses_arr = np.array(poses)
    mean_pose = poses_arr.mean(axis=0)
    mse = np.mean((poses_arr - mean_pose) ** 2)
    return np.sqrt(mse)


def pose_to_vec(rvec, tvec):
    """Flatten pose to 6-vector [rx,ry,rz,tx,ty,tz]"""
    return np.concatenate([rvec.flatten(), tvec.flatten()])


# ============ Main ============

def main():
    from ultralytics import YOLO

    model = YOLO(MODEL_PATH)
    estimator = SocketPoseEstimator()
    ed = init_edge_drawing()

    image_files = sorted([f for f in os.listdir(IMG_DIR) if f.endswith(('.jpg', '.png'))])

    print(f"{'Image':<30} {'Trial':<6} {'rvec_x':>10} {'rvec_y':>10} {'rvec_z':>10} "
          f"{'tvec_x':>10} {'tvec_y':>10} {'tvec_z':>10} {'cost':>12}")
    print("-" * 120)

    for fname in image_files:
        img_path = os.path.join(IMG_DIR, fname)
        img = cv2.imread(img_path)
        if img is None:
            print(f"{fname}: failed to load")
            continue

        roi_box = yolo_detect_roi(model, img)
        if roi_box is None:
            print(f"{fname}: no YOLO detection")
            continue

        roi = img[int(roi_box[1]):int(roi_box[3]), int(roi_box[0]):int(roi_box[2])]
        tl = (int(roi_box[0]), int(roi_box[1]))
        ellipses = detect_ellipses(ed, roi)
        final_pts, score = estimator.solve(ellipses)

        if final_pts is None:
            print(f"{fname}: matching failed, score={score}")
            continue

        pts_img = final_pts + np.array(tl)
        poses_pnp = []
        poses_ba = []

        print(f"\n=== {fname} ===")
        print(f"{'Trial':<6} {'method':<6} {'rvec_x':>10} {'rvec_y':>10} {'rvec_z':>10} "
              f"{'tvec_x':>10} {'tvec_y':>10} {'tvec_z':>10} {'reproj_err':>10}")
        print("-" * 80)

        for trial in range(NUM_TRIALS):
            # Full pipeline: ROI detection -> ellipse detection -> template matching -> PnP + BA
            roi_box = yolo_detect_roi(model, img)
            if roi_box is None:
                continue
            roi = img[int(roi_box[1]):int(roi_box[3]), int(roi_box[0]):int(roi_box[2])]
            tl = (int(roi_box[0]), int(roi_box[1]))
            ellipses = detect_ellipses(ed, roi)
            final_pts, score = estimator.solve(ellipses)
            if final_pts is None:
                continue
            pts_img = final_pts + np.array(tl)

            result = estimator.estimate_single(pts_img)
            if result is None:
                continue

            vec_pnp = pose_to_vec(result['rvec_pnp'], result['tvec_pnp'])
            vec_ba = pose_to_vec(result['rvec_ba'], result['tvec_ba'])
            poses_pnp.append(vec_pnp)
            poses_ba.append(vec_ba)

            print(f"{trial+1:<6} {'PnP':<6} {vec_pnp[0]:>10.4f} {vec_pnp[1]:>10.4f} {vec_pnp[2]:>10.4f} "
                  f"{vec_pnp[3]:>10.2f} {vec_pnp[4]:>10.2f} {vec_pnp[5]:>10.2f} {result['reproj_err_pnp']:>10.4f}")
            print(f"{trial+1:<6} {'BA':<6} {vec_ba[0]:>10.4f} {vec_ba[1]:>10.4f} {vec_ba[2]:>10.4f} "
                  f"{vec_ba[3]:>10.2f} {vec_ba[4]:>10.2f} {vec_ba[5]:>10.2f} {result['reproj_err_ba']:>10.4f}")

        if len(poses_ba) >= 2:
            rmse_pnp = compute_pose_rmse(poses_pnp)
            rmse_ba = compute_pose_rmse(poses_ba)
            print(f"{'RMSE':<6} {'':<6} {'':<10} {'':<10} {'':<10} {'':<10} {'':<10} {'':<10} {rmse_pnp:>10.4f} (PnP)")
            print(f"{'RMSE':<6} {'':<6} {'':<10} {'':<10} {'':<10} {'':<10} {'':<10} {'':<10} {rmse_ba:>10.4f} (BA)")
        print()


if __name__ == '__main__':
    main()