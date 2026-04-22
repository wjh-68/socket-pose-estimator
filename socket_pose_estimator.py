#!/usr/bin/env python3
"""
Sliding window pose estimator with bundle adjustment optimization.
- Uses solvePnP (IPPE) for initial pose estimation per frame
- Maintains a sliding window of N frames
- Jointly optimizes all poses in window to minimize reprojection error
"""
import os
# os.environ['QT_QPA_PLATFORM'] = 'offscreen'
os.environ['OPENCV_VIDEOIO_DEBUG'] = '0'

import cv2
import numpy as np
from scipy.optimize import least_squares
from collections import deque
import time

VISUALIZE = True        # 是否显示可视化窗口
VISUALIZE_DELAY = 1500  # 可视化时每张图片停留时间(ms)，仅在 VISUALIZE=True 时生效


class ReprojectionErrorFunctor:
    """Functor for reprojection error computation with distortion."""

    def __init__(self, pts3d, observations, K, dist):
        self.pts3d = pts3d
        self.observations = observations
        self.K = K
        self.dist = dist
        self.fx, self.fy = K[0, 0], K[1, 1]
        self.cx, self.cy = K[0, 2], K[1, 2]
        self.k1, self.k2, self.p1, self.p2, self.k3 = dist

    def __call__(self, params):
        """
        params: concatenated [pose0(6), pose1(6), ..., poseN(6)]
        returns: residuals (2 * num_pts * num_frames,)
        """
        num_frames = len(self.observations)
        residuals = []

        for k in range(num_frames):
            pose = params[k * 6:(k + 1) * 6]
            rvec = pose[:3]
            tvec = pose[3:6]

            R, _ = cv2.Rodrigues(rvec)
            for i in range(len(self.pts3d)):
                pt3d = self.pts3d[i]
                pt_cam = R @ pt3d + tvec

                x = pt_cam[0] / pt_cam[2]
                y = pt_cam[1] / pt_cam[2]

                r2 = x * x + y * y
                radial = 1 + self.k1 * r2 + self.k2 * r2 * r2 + self.k3 * r2 * r2 * r2
                x_dist = x * radial + 2 * self.p1 * x * y + self.p2 * (r2 + 2 * x * x)
                y_dist = y * radial + self.p1 * (r2 + 2 * y * y) + 2 * self.p2 * x * y

                u = self.fx * x_dist + self.cx
                v = self.fy * y_dist + self.cy

                residuals.append(u - self.observations[k][i][0])
                residuals.append(v - self.observations[k][i][1])

        return np.array(residuals)


class SlidingWindowEstimator:
    """
    Sliding window pose estimator.
    - Uses IPPE (solvePnP with SOLVEPNP_IPPE) for initial pose per frame
    - Jointly optimizes all poses in window using bundle adjustment
    """

    def __init__(self, obj_pts, K, dist, window_size=5):
        self.obj_pts = obj_pts.astype(np.float64)
        self.K = K.astype(np.float64)
        self.dist = dist.astype(np.float64)
        self.window_size = window_size
        self.window = deque(maxlen=window_size)

    def add_frame(self, frame):
        """Add a frame with 2D keypoints to the sliding window."""
        self.window.append(frame)

    def _init_pose_ippe(self, pts2d):
        """Initialize pose using IPPE (solvePnP with IPPE method)."""
        if len(pts2d) != len(self.obj_pts):
            return None
        pts2d_arr = np.array(pts2d, dtype=np.float64)
        success, rvec, tvec = cv2.solvePnP(
            self.obj_pts, pts2d_arr, self.K, self.dist,
            flags=cv2.SOLVEPNP_IPPE
        )
        if not success:
            return None
        return np.concatenate([rvec.flatten(), tvec.flatten()])

    def _optimize_poses(self, observations):
        """
        Jointly optimize all poses to minimize reprojection error.
        Uses Levenberg-Marquardt algorithm via scipy least_squares.
        """
        num_frames = len(observations)
        if num_frames == 0:
            return [], 0.0

        init_poses = []
        for k in range(num_frames):
            pose = self._init_pose_ippe(observations[k])
            if pose is None:
                pose = np.zeros(6)
            init_poses.append(pose)

        params = np.concatenate(init_poses)

        functor = ReprojectionErrorFunctor(self.obj_pts, observations, self.K, self.dist)

        result = least_squares(
            functor,
            params,
            method='lm',
            ftol=1e-6,
            xtol=1e-6,
            max_nfev=100
        )

        optimized_poses = []
        for k in range(num_frames):
            pose = result.x[k * 6:(k + 1) * 6]
            optimized_poses.append(pose)

        return optimized_poses, result.cost

    def estimate(self):
        """
        Estimate poses for all frames in the sliding window.
        Returns (optimized_poses, avg_error, init_poses).
        """
        if len(self.window) == 0:
            return [], 0.0, []

        observations = [frame['keypoints'] for frame in self.window]
        num_frames = len(observations)

        init_poses = []
        for k in range(num_frames):
            pose = self._init_pose_ippe(observations[k])
            if pose is None:
                pose = np.zeros(6)
            init_poses.append(pose)

        params = np.concatenate(init_poses)
        functor = ReprojectionErrorFunctor(self.obj_pts, observations, self.K, self.dist)

        result = least_squares(
            functor,
            params,
            method='lm',
            ftol=1e-6,
            xtol=1e-6,
            max_nfev=100
        )

        optimized_poses = []
        for k in range(num_frames):
            pose = result.x[k * 6:(k + 1) * 6]
            optimized_poses.append(pose)

        num_residuals = len(self.obj_pts) * 2 * num_frames
        avg_error = np.sqrt(result.cost / num_residuals) if num_residuals > 0 else 0.0

        return optimized_poses, avg_error, init_poses

    def compute_reprojection_error(self, poses, observations):
        """Compute average reprojection error for given poses."""
        total_error = 0.0
        count = 0

        for k, pose in enumerate(poses):
            rvec = pose[:3]
            tvec = pose[3:6]
            R, _ = cv2.Rodrigues(rvec)

            for i, pt3d in enumerate(self.obj_pts):
                pt_cam = R @ pt3d + tvec
                x = pt_cam[0] / pt_cam[2]
                y = pt_cam[1] / pt_cam[2]

                r2 = x * x + y * y
                radial = 1 + self.dist[0] * r2 + self.dist[1] * r2 * r2 + self.dist[4] * r2 * r2 * r2
                x_dist = x * radial + 2 * self.dist[2] * x * y + self.dist[3] * (r2 + 2 * x * x)
                y_dist = y * radial + self.dist[2] * (r2 + 2 * y * y) + 2 * self.dist[3] * x * y

                u = self.K[0, 0] * x_dist + self.K[0, 2]
                v = self.K[1, 1] * y_dist + self.K[1, 2]

                du = u - observations[k][i][0]
                dv = v - observations[k][i][1]
                total_error += np.sqrt(du * du + dv * dv)
                count += 1

        return total_error / count if count > 0 else 0.0


class SocketPoseEstimator:
    """
    Full socket pose estimation pipeline combining detection and optimization.
    Inherits detection logic from UltimateSocketMatcher and adds sliding window BA.
    """

    def __init__(self, window_size=5):
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
        self.window_size = window_size
        self.matcher = None
        self.estimator = SlidingWindowEstimator(
            self.obj_pts, self.K, self.dist, window_size
        )
        self.frames = deque(maxlen=window_size)

    def _clean_and_classify(self, ellipses, dist_thresh=10):
        """Merge concentric circles, classify by size."""
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
                if not used[j]:
                    dist = np.linalg.norm(nodes[i]['c'] - nodes[j]['c'])
                    if dist < dist_thresh:
                        cluster.append(nodes[j])
                        used[j] = True
            avg_c = np.mean([n['c'] for n in cluster], axis=0)
            max_d = max([n['d'] for n in cluster])
            merged.append({'p': avg_c, 'size': max_d, 'is_double': len(cluster) >= 2})

        merged = [m for m in merged if 10 < m['size'] < 100]
        return merged

    def _gap_method_threshold(self, candidates):
        """Classify holes as large/small based on gap method."""
        candidates.sort(key=lambda x: x['size'])
        sizes = [c['size'] for c in candidates]
        gaps = [sizes[i + 1] / sizes[i] for i in range(len(sizes) - 1)]
        split_idx = np.argmax(gaps)
        threshold = (sizes[split_idx] + sizes[split_idx + 1]) / 2
        for c in candidates:
            c['t'] = 1 if c['size'] > threshold else 0

    def _get_signed_area(self, pts):
        p0, p1, p2 = pts[0], pts[1], pts[2]
        return (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0])

    def _evaluate_refined(self, H, template_pts, candidates, dist_thresh=15):
        from scipy.optimize import linear_sum_assignment

        proj = cv2.perspectiveTransform(template_pts.reshape(-1, 1, 2), H).reshape(-1, 2)
        det_pts = np.array([c['p'] for c in candidates])

        diff = proj[:, np.newaxis, :] - det_pts[np.newaxis, :, :]
        dist_matrix = np.linalg.norm(diff, axis=2)

        row_ind, col_ind = linear_sum_assignment(dist_matrix)

        valid_errors = []
        for r, c in zip(row_ind, col_ind):
            d = dist_matrix[r, c]
            if d < dist_thresh:
                valid_errors.append(d)

        inlier_count = len(valid_errors)
        if inlier_count == 0:
            return -9999, proj

        rmse = np.sqrt(np.mean(np.square(valid_errors)))
        score = (inlier_count * 1000) - rmse
        return score, proj

    def solve(self, raw_ellipses):
        """Template matching for 7-hole socket detection."""
        from itertools import permutations, combinations

        candidates = self._clean_and_classify(raw_ellipses)
        if len(candidates) < 4:
            return None, 0

        self._gap_method_threshold(candidates)

        best_H, max_score = None, 0
        tmp_combos = []
        for indices in combinations(range(7), 4):
            types = tuple(sorted([self.tmp_types[i] for i in indices]))
            tmp_combos.append({'idx': indices, 'types': types})

        det_indices = list(range(len(candidates)))
        final_res = None

        for d_idx_tuple in combinations(det_indices, 4):
            d_subset = [candidates[i] for i in d_idx_tuple]
            d_types_signature = tuple(sorted([d['t'] for d in d_subset]))

            for t_combo in tmp_combos:
                src_pts = self.obj_pts[list(t_combo['idx'])][:, :2]
                src_types = [self.tmp_types[i] for i in t_combo['idx']]
                src_area_sign = self._get_signed_area(src_pts)

                for p_d_subset in permutations(d_subset):
                    dst_pts = np.array([d['p'] for d in p_d_subset], dtype=np.float32)

                    if np.sign(self._get_signed_area(dst_pts)) == src_area_sign:
                        continue

                    H, _ = cv2.findHomography(src_pts, dst_pts)
                    if H is None:
                        continue

                    det_sign = np.linalg.det(H[:2, :2])
                    if det_sign > 0:
                        continue

                    score, proj = self._evaluate_refined(H, self.obj_pts[:, :2], candidates)
                    if score > max_score:
                        max_score = score
                        best_H = H
                        final_res = proj

        return (final_res, max_score) if best_H is not None else (None, 0)

    def add_frame(self, pts2d, frame_id=0):
        """
        Add a detected frame to the sliding window.
        pts2d: 7x2 array of 2D keypoints in image coordinates.
        """
        frame = {
            'id': frame_id,
            'keypoints': [np.array(pt, dtype=np.float64) for pt in pts2d]
        }
        self.frames.append(frame)
        self.estimator.add_frame(frame)
        return len(self.frames)

    def estimate(self):
        """
        Run bundle adjustment on all frames in sliding window.
        Returns optimized poses and reprojection error.
        """
        return self.estimator.estimate()

    def estimate_single(self, pts2d):
        """
        Estimate pose for a single frame (no BA, just IPPE).
        Returns rvec, tvec, and projected points.
        """
        pts2d_arr = np.array(pts2d, dtype=np.float64)
        success, rvec, tvec = cv2.solvePnP(
            self.obj_pts, pts2d_arr,
            self.K, self.dist,
            flags=cv2.SOLVEPNP_IPPE
        )
        if success:
            proj_back, _ = cv2.projectPoints(
                self.obj_pts, rvec, tvec, self.K, self.dist
            )
            return rvec, tvec, proj_back.reshape(-1, 2)
        return None, None, None


def main():
    """Test with camera or image directory."""
    from ultralytics import YOLO

    model = YOLO("checkpoint/best.pt")
    estimator = SocketPoseEstimator(window_size=5)

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

    ed = init_edge_drawing()
    IMG_DIR = "dataset/images"

    frame_id = 0
    all_poses = []

    for fname in sorted(os.listdir(IMG_DIR)):
        if not fname.endswith((".jpg", ".png")):
            continue
        img_path = os.path.join(IMG_DIR, fname)
        img = cv2.imread(img_path)
        if img is None:
            continue

        t0 = time.perf_counter_ns()

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

        frame_count = estimator.add_frame(pts_img, frame_id)
        frame_id += 1

        print(f"{fname}: detected score={score:.2f}, window_size={frame_count}")

        if frame_count >= 2:
            poses, avg_error, init_poses = estimator.estimate()
            if poses:
                last_init = init_poses[-1]
                last_pose = poses[-1]
                rvec_ippe = last_init[:3]
                tvec_ippe = last_init[3:6]
                rvec_ba = last_pose[:3]
                tvec_ba = last_pose[3:6]
                print(f"  IPPE: rvec={rvec_ippe}, tvec={tvec_ippe}")
                cv2.drawFrameAxes(img, estimator.K.astype(np.float32),
                                 estimator.dist.astype(np.float32),
                                 rvec_ippe.astype(np.float32), tvec_ippe.astype(np.float32), 40, 2)
                print(f"  BA:   rvec={rvec_ba}, tvec={tvec_ba}, error={avg_error:.2f}")

        elapsed = (time.perf_counter_ns() - t0) / 1e6
        print(f"  Total: {elapsed:.1f}ms")

        if VISUALIZE:
            vis = img.copy()
            for i, (x, y) in enumerate(pts_img):
                cv2.circle(vis, (int(x), int(y)), 5, (0, 255, 0), -1)
                cv2.putText(vis, str(i), (int(x) + 5, int(y)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.imshow("result", vis)
            cv2.waitKey(VISUALIZE_DELAY)

    if VISUALIZE:
        cv2.destroyAllWindows()

    if len(all_poses) > 0:
        print("\n=== Final Results ===")
        print(f"Total frames processed: {len(all_poses)}")


if __name__ == '__main__':
    main()
