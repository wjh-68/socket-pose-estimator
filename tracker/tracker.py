import numpy as np
import cv2
import json
from scipy.spatial.transform import Rotation
import time
import os
import pandas as pd
import matplotlib.pyplot as plt
from static_pose_optimizer_ba import StaticPoseOptimizer, pose_to_euler_tvec
from core.error import ConfigError
class Tracker:
    def __init__(self, cfg):
        # check cfg
        validate_cfg(cfg)
        self.cfg = cfg

        # load camera params
        cam_prms = self.cfg.get('camera', {})
        self.K = np.asarray(cam_prms.get('K'),dtype=np.float64)
        self.dist = np.asarray(cam_prms.get('dist'),dtype=np.float64)
        self.eMc = np.asarray(cam_prms.get('eMc'),dtype=np.float64)

        # load pnp threshold
        tracker_prms = self.cfg.get('tracker', {})
        self.use_adptive_threshold = tracker_prms.get(
            'use_adptive_threshold', True)
        self.reproj_threshold = tracker_prms.get(
            'reproj_threshold', 1.0) 
        self.adaptive_multiplier = tracker_prms.get(
            'adaptive_multiplier', 2.0) 
        
        # load object pose rejection threshold
        self.max_translation = tracker_prms.get(
            'max_translation', 20.0) 
        self.max_rotation_deg = tracker_prms.get(
            'max_rotation_deg', 20.0) 

        # load optimizer params
        optimizer_prms = self.cfg.get('optimizer', {})
        self.sliding_window_size = optimizer_prms.get(
            'window_size', 5)
        self.obj_pts = np.asarray(
            optimizer_prms.get('obj_pts'),dtype=np.float64)
        self.prior_sigmas = np.asarray(
            optimizer_prms.get('prior_sigmas'),dtype=np.float64)
        self.point_sigmas = np.asarray(
            optimizer_prms.get('point_sigmas'),dtype=np.float64)

        # load result_dir
        self.result_dir = self.cfg.get('save', {}).get('result_dir', './result')
        # init optimizer
        self.optimizer = StaticPoseOptimizer(
            self.K, self.dist, 
            prior_sigma=self.prior_sigmas, point_sigmas=self.point_sigmas)
        self.optimizer.set_extrinsics(self.eMc)
        self.optimizer.set_object_pts(self.obj_pts)
        self._reset_statistics()

    def _reset_statistics(self):
        self.frame_records = []
        self.pnp_records = []
        self.optimize_records = []

        self.cnt_no_detection = 0
        self.cnt_rejected_frames = 0
        self.cnt_less_7pts = 0
        self.cnt_pnp_failed = 0
        self.last_bMo = None
        
        
    def track(self, packet):
        pts3d = self.obj_pts.copy()
        # extract data from packet
        frame_id = packet.frame_id
        timestamp = packet.timestamp
        pts2d = packet.refined_pts2d
        robot_pose = packet.robot_pose
        img = packet.image
        roi = packet.roi
        roi_img, roi_x_min, roi_y_min, roi_x_max, roi_y_max = extract_roi(img, roi)

        t0 = time.perf_counter_ns()
        pnp_results = self._compute_pnp_results(pts2d, pts3d)
        if not pnp_results['valid']:
            self.cnt_pnp_failed += 1
            print(f"Frame:{frame_id}: PnP failed, skipping pose estimation")
            return False

        # if np.all(pnp_results['per_point_errors'] > self.fixed_error_threshold):
        #     self.cnt_rejected_frames += 1
        #     print(f"  [REJECT] All points exceed threshold {self.fixed_error_threshold} px")
        #     return False

        cMo = np.eye(4)
        cMo[:3, :3] = Rotation.from_rotvec(pnp_results['rvec']).as_matrix()
        cMo[:3, 3] = pnp_results['tvec']

        if self._should_reject_pose_diff(cMo, robot_pose):
            self.cnt_rejected_frames += 1
            return False

        bMo_init = robot_pose @ self.eMc @ cMo
        duration = (time.perf_counter_ns()-t0)*1e-6
        print(f"init and filter frames: {duration:.1f} ms")

        t0 = time.perf_counter_ns()
        bMo_optimized, cMo_optimized = self._prepare_optimizer(
            frame_id,
            robot_pose,
            pts2d,
            pts3d,
            pnp_results['per_point_errors'],
            bMo_init,
        )
        duration = (time.perf_counter_ns()-t0)*1e-6
        print(f"total optimization: {duration:.1f} ms")

        t0 = time.perf_counter_ns()
        rvec_optimized = Rotation.from_matrix(cMo_optimized[:3, :3]).as_rotvec()
        tvec_optimized = cMo_optimized[:3, 3]

        if pnp_results['inlier_mask'] is not None and pnp_results['inlier_mask'].sum() >= 4:
            pts3d_inlier = pts3d[pnp_results['inlier_mask']]
            pts2d_inlier = pts2d[pnp_results['inlier_mask']]
        else:
            pts3d_inlier = pts3d
            pts2d_inlier = pts2d

        success, rvec_ba, tvec_ba = cv2.solvePnP(pts3d_inlier, pts2d_inlier, self.K, self.dist, flags=cv2.SOLVEPNP_ITERATIVE)
        cMo_ba = np.eye(4)
        cMo_ba[:3, :3] = Rotation.from_rotvec(rvec_ba.flatten()).as_matrix()
        cMo_ba[:3, 3] = tvec_ba.flatten()
        bMo_ba = robot_pose @ self.eMc @ cMo_ba

        print(f"bMo_optimized: {pose_to_euler_tvec(bMo_optimized)}")
        print(f"bMo_pnp: {pose_to_euler_tvec(bMo_init)}")
        print(f"bMo_ba: {pose_to_euler_tvec(bMo_ba)}")
        print(f"cMo_optimized: {pose_to_euler_tvec(cMo_optimized)}")
        print(f"cMo_pnp: {pose_to_euler_tvec(cMo)}")
        print(f"cMo_ba: {pose_to_euler_tvec(cMo_ba)}")

        pnp_error_ba = compute_reproj_error(pts3d_inlier, rvec_ba, tvec_ba, pts2d_inlier, self.K, self.dist)
        print(f"PnP reprojection error: {pnp_results['per_point_errors'].mean():.4f} (round1: {pnp_results['round1_error']:.4f})")
        print(f"PnP reprojection error (BA, {int(pnp_results['inlier_mask'].sum()) if pnp_results['inlier_mask'] is not None else 0} inliers): {pnp_error_ba:.4f}")

        now_error, _ = self.optimizer.get_frame_error(frame_id)
        print(f"Optimized reprojection error: {now_error:.4f}")
        ave_error = self.optimizer.get_average_error()
        print(f"Average reprojection error: {ave_error:.4f}")

        bMo_euler, bMo_tvec = pose_to_euler_tvec(bMo_optimized)
        cMo_euler, cMo_tvec = pose_to_euler_tvec(cMo_optimized)

        self._append_records(
            frame_id=frame_id,
            frame_id_val=frame_id,
            timestamp_ns=timestamp,
            robot_pose=robot_pose,
            pts2d=pts2d,
            pts3d=pts3d,
            bMo_init=bMo_init,
            bMo_optimized=bMo_optimized,
            cMo=cMo,
            cMo_optimized=cMo_optimized,
            pnp_results=pnp_results,
            pnp_error_ba=pnp_error_ba,
            rvec_ba=rvec_ba,
            tvec_ba=tvec_ba,
            now_error=now_error,
            ave_error=ave_error,
            bMo_euler=bMo_euler,
            bMo_tvec=bMo_tvec,
            cMo_euler=cMo_euler,
            cMo_tvec=cMo_tvec,
        )
        duration = (time.perf_counter_ns()-t0)*1e-6
        print(f"print and record: {duration:.1f} ms")

        t0 = time.perf_counter_ns()
        self._render_frame(
            img=img,
            roi_bounds=(roi_x_min, roi_y_min, roi_x_max, roi_y_max),
            centers=pts2d,
            pts3d=pts3d,
            rvec_optimized=rvec_optimized,
            tvec_optimized=tvec_optimized,
            cMo=cMo_optimized,
            inlier_mask=pnp_results['inlier_mask'],
            frame_id_val=frame_id,
        )
        duration = (time.perf_counter_ns()-t0)*1e-6
        print(f"visualization: {duration:.1f} ms")
    def _should_reject_pose_diff(self, cMo, robot_pose):
        if self.last_bMo is None or not self.optimizer.is_initialized():
            return False

        cMo_before = np.linalg.inv(self.eMc) @ np.linalg.inv(robot_pose) @ self.last_bMo
        tvec_before = cMo_before[:3, 3]
        pos_diff = float(np.linalg.norm(cMo[:3, 3] - tvec_before))

        rot_mat_diff = cMo[:3, :3] @ cMo_before[:3, :3].T
        rot_vec_diff = Rotation.from_matrix(rot_mat_diff).as_rotvec()
        rot_diff = float(np.degrees(np.linalg.norm(rot_vec_diff)))

        if pos_diff > self.max_translation or rot_diff > self.max_rotation_deg:
            print(f"  [REJECT] Large pose difference: pos_diff={pos_diff:.1f}mm, rot_diff={rot_diff:.1f}deg")
            return True
        return False
    def _compute_pnp_results(self, pts2d, pts3d):
        rvec, tvec, valid, inlier_mask, per_point_errors, round1_error, used_threshold = self._two_round_pnp(
            pts2d, pts3d, self.K, self.dist, error_threshold=self.reproj_threshold)
        return {
            'valid': valid,
            'rvec': rvec,
            'tvec': tvec,
            'inlier_mask': inlier_mask,
            'per_point_errors': per_point_errors,
            'round1_error': round1_error,
            'used_threshold': used_threshold,
        }

    def _prepare_optimizer(self, frame_id, robot_pose, pts2d, pts3d, per_point_errors_pnp, bMo_init):
        if not self.optimizer.is_initialized():
            self.optimizer.set_initial_pose(bMo_init)

        if self.optimizer.get_frame_count() >= self.sliding_window_size:
            self.optimizer.remove_oldest_frame()

        self.optimizer.add_frame(frame_id, robot_pose, pts2d, pts3d, per_point_errors_pnp=per_point_errors_pnp)

        self.optimizer.optimize()

        bMo_optimized = self.optimizer.get_pose()
        self.last_bMo = bMo_optimized
        cMo_optimized = self.optimizer.compute_cMo(robot_pose, frame_id)
        return bMo_optimized, cMo_optimized

    def _append_records(
        self,
        frame_id,
        frame_id_val,
        timestamp_ns,
        robot_pose,
        pts2d,
        pts3d,
        bMo_init,
        bMo_optimized,
        cMo,
        cMo_optimized,
        pnp_results,
        pnp_error_ba,
        rvec_ba,
        tvec_ba,
        now_error,
        ave_error,
        bMo_euler,
        bMo_tvec,
        cMo_euler,
        cMo_tvec,
    ):
        inlier_mask = pnp_results['inlier_mask']
        per_point_errors = pnp_results['per_point_errors']
        round1_error = pnp_results['round1_error']
        used_threshold = pnp_results['used_threshold']
        rvec = pnp_results['rvec']
        tvec = pnp_results['tvec']

        self.pnp_records.append({
            'frame_id': frame_id_val,
            'frame_idx': frame_id,
            'timestamp_ns': int(timestamp_ns),
            'robot_pose': robot_pose.tolist(),
            'n_points': len(pts2d),
            'n_inliers': int(inlier_mask.sum()) if inlier_mask is not None else 0,
            'used_threshold': float(used_threshold),
            'pnp_error_round1': float(round1_error),
            'pnp_error_final': float(per_point_errors.mean()),
            'pnp_error_ba': float(pnp_error_ba),
            'per_point_errors': per_point_errors.tolist(),
            'inlier_mask': inlier_mask.tolist() if inlier_mask is not None else None,
            'rvec': np.array(rvec).tolist(),
            'tvec': np.array(tvec).tolist(),
            'cMo_euler': cMo_euler.tolist(),
            'cMo_tvec': cMo_tvec.tolist(),
            'bMo_pnp_euler': pose_to_euler_tvec(bMo_init)[0].tolist(),
            'bMo_pnp_tvec': pose_to_euler_tvec(bMo_init)[1].tolist(),
        })

        self.optimize_records.append({
            'frame_id': frame_id_val,
            'frame_idx': frame_id,
            'timestamp_ns': int(timestamp_ns),
            'bMo_euler': bMo_euler.tolist(),
            'bMo_tvec': bMo_tvec.tolist(),
            'cMo_euler': cMo_euler.tolist(),
            'cMo_tvec': cMo_tvec.tolist(),
            'n_frames_in_optimizer': self.optimizer.get_frame_count(),
            'frame_error': float(now_error),
            'avg_error': float(ave_error) if ave_error is not None else None,
        })

        self.frame_records.append({
            'frame_id': frame_id_val,
            'frame_idx': frame_id,
            'timestamp_ns': int(timestamp_ns),
            'robot_pose': robot_pose.tolist(),
            'pts2d': pts2d.tolist(),
            'pts3d': pts3d.tolist(),
            'bMo_init': bMo_init.tolist(),
            'bMo_optimized': bMo_optimized.tolist(),
            'cMo': cMo.tolist(),
            'cMo_optimized': cMo_optimized.tolist(),
            'pnp_error': float(per_point_errors.mean()),
            'optimized_error': float(now_error),
            'avg_error': float(ave_error) if ave_error is not None else None,
            'per_point_errors_pnp': per_point_errors.tolist(),
            'inlier_mask': inlier_mask.tolist() if inlier_mask is not None else None,
        })
    
    def _two_round_pnp(self, pts2d, pts3d, K, dist, error_threshold=0.4):
        """Two-round PnP with adaptive or fixed threshold.

        Returns:
            rvec, tvec, valid, inlier_mask, per_point_errors, round1_errors, used_threshold
        """
        rvec1, tvec1, valid1 = solvePnP_IPPE(pts2d, pts3d, K, dist)
        if not valid1:
            return None, None, False, None, None, None, None

        per_point_errors = compute_per_point_reproj_errors(pts3d, rvec1, tvec1, pts2d, K, dist)
        round1_error = per_point_errors.mean()

        if self.use_adptive_threshold:
            median_error = np.median(per_point_errors)
            current_threshold = median_error * self.adaptive_multiplier
            current_threshold = min(current_threshold, self.reproj_threshold)
        else:
            current_threshold = self.reproj_threshold

        inlier_mask = per_point_errors < current_threshold
        n_inliers = inlier_mask.sum()

        if n_inliers >= 4:
            pts3d_inlier = pts3d[inlier_mask]
            pts2d_inlier = pts2d[inlier_mask]
            rvec2, tvec2, valid2 = solvePnP_IPPE(pts2d_inlier, pts3d_inlier, K, dist)
            if valid2:
                per_point_errors_round2 = compute_per_point_reproj_errors(pts3d, rvec2, tvec2, pts2d, K, dist)
                return rvec2, tvec2, True, inlier_mask, per_point_errors_round2, round1_error, current_threshold
            return rvec1, tvec1, True, inlier_mask, per_point_errors, round1_error, current_threshold

        return rvec1, tvec1, True, inlier_mask, per_point_errors, round1_error, current_threshold
    def _render_frame(
            self,
            img,
            roi_bounds,
            centers,
            pts3d,
            rvec_optimized,
            tvec_optimized,
            cMo,
            inlier_mask,
            frame_id_val,
        ):
            roi_x_min, roi_y_min, roi_x_max, roi_y_max = roi_bounds
            cv2.drawFrameAxes(img, self.K, self.dist, cMo[:3, :3], cMo[:3, 3:], 10, 3)
            proj, _ = cv2.projectPoints(pts3d, rvec_optimized, tvec_optimized, self.K, self.dist)

            for i, (x, y) in enumerate(centers):
                if inlier_mask is not None and i < len(inlier_mask):
                    color = (0, 255, 0) if inlier_mask[i] else (0, 0, 255)
                else:
                    color = (255, 255, 0)
                cv2.circle(img, (int(x), int(y)), 3, color, 1)
                x_proj, y_proj = proj[i][0]
                cv2.drawMarker(img, (int(x_proj), int(y_proj)), (255, 0, 0), cv2.MARKER_CROSS, 5, 1)
                cv2.line(img, (int(x), int(y)), (int(x_proj), int(y_proj)), (0, 255, 0), 1)
                cv2.putText(img, str(i), (int(x)-8, int(y)-8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            pad = 50
            roi_y_min_clamped = max(0, roi_y_min - pad)
            roi_y_max_clamped = min(img.shape[0], roi_y_max + pad)
            roi_x_min_clamped = max(0, roi_x_min - pad)
            roi_x_max_clamped = min(img.shape[1], roi_x_max + pad)
            vis_result = img[roi_y_min_clamped:roi_y_max_clamped, roi_x_min_clamped:roi_x_max_clamped]
            vis_result = cv2.resize(vis_result, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
            vis_result_path = os.path.join(self.result_dir, f"frame_{frame_id_val:06d}_vis_result.png")
            cv2.imwrite(vis_result_path, vis_result)
    def save_results(self):

        cv2.destroyAllWindows()
        self._save_json_records()
        self._save_csv_records()
        self._save_plots()
        self._print_summary()

    def _save_json_records(self):
        frame_records_path = os.path.join(self.result_dir, "frame_records.json")
        with open(frame_records_path, 'w') as f:
            json.dump(self.frame_records, f, indent=2)
        print(f"\nSaved frame_records to {frame_records_path}")

        pnp_records_path = os.path.join(self.result_dir, "pnp_records.json")
        with open(pnp_records_path, 'w') as f:
            json.dump(self.pnp_records, f, indent=2)
        print(f"Saved pnp_records to {pnp_records_path}")

        optimize_records_path = os.path.join(self.result_dir, "optimize_records.json")
        with open(optimize_records_path, 'w') as f:
            json.dump(self.optimize_records, f, indent=2)
        print(f"Saved optimize_records to {optimize_records_path}")

    def _save_csv_records(self):
        if self.pnp_records:
            pnp_df = pd.DataFrame(self.pnp_records)
            pnp_csv_path = os.path.join(self.result_dir, "pnp_results.csv")
            pnp_df.to_csv(pnp_csv_path, index=False)
            print(f"Saved pnp_results CSV to {pnp_csv_path}")

        if self.optimize_records:
            opt_df = pd.DataFrame(self.optimize_records)
            opt_csv_path = os.path.join(self.result_dir, "optimize_results.csv")
            opt_df.to_csv(opt_csv_path, index=False)
            print(f"Saved optimize_results CSV to {opt_csv_path}")

    def _save_plots(self):
        if not self.frame_records:
            return

        def unwrap_angles(angles):
            angles = np.array(angles)
            for i in range(1, len(angles)):
                diff = angles[i] - angles[i-1]
                if diff > 180:
                    angles[i:] -= 360
                elif diff < -180:
                    angles[i:] += 360
            return angles

        self._save_translation_plots()
        self._save_euler_plots(unwrap_angles)
        self._save_error_plots()

    def _save_translation_plots(self):
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        fig.suptitle('bMo, cMo, bMe Translation Components vs Frame', fontsize=14)

        frames = [r['frame_id'] for r in self.frame_records]

        ax = axes[0, 0]
        ax.plot(frames, [r['bMo_optimized'][0][3] for r in self.frame_records], 'r-', label='Optimized x')
        ax.plot(frames, [r['bMo_init'][0][3] for r in self.frame_records], 'r--', label='PnP x', alpha=0.7)
        ax.set_ylabel('X (mm)')
        ax.set_title('bMo X')
        ax.grid(True)
        ax.legend()

        ax = axes[0, 1]
        ax.plot(frames, [r['bMo_optimized'][1][3] for r in self.frame_records], 'g-', label='Optimized y')
        ax.plot(frames, [r['bMo_init'][1][3] for r in self.frame_records], 'g--', label='PnP y', alpha=0.7)
        ax.set_ylabel('Y (mm)')
        ax.set_title('bMo Y')
        ax.grid(True)
        ax.legend()

        ax = axes[0, 2]
        ax.plot(frames, [r['bMo_optimized'][2][3] for r in self.frame_records], 'b-', label='Optimized z')
        ax.plot(frames, [r['bMo_init'][2][3] for r in self.frame_records], 'b--', label='PnP z', alpha=0.7)
        ax.set_ylabel('Z (mm)')
        ax.set_title('bMo Z')
        ax.grid(True)
        ax.legend()

        ax = axes[1, 0]
        ax.plot(frames, [r['cMo_optimized'][0][3] for r in self.frame_records], 'r-', label='Optimized x')
        ax.plot(frames, [r['cMo'][0][3] for r in self.frame_records], 'r--', label='PnP x', alpha=0.7)
        ax.set_ylabel('X (mm)')
        ax.set_title('cMo X')
        ax.grid(True)
        ax.legend()

        ax = axes[1, 1]
        ax.plot(frames, [r['cMo_optimized'][1][3] for r in self.frame_records], 'g-', label='Optimized y')
        ax.plot(frames, [r['cMo'][1][3] for r in self.frame_records], 'g--', label='PnP y', alpha=0.7)
        ax.set_ylabel('Y (mm)')
        ax.set_title('cMo Y')
        ax.grid(True)
        ax.legend()

        ax = axes[1, 2]
        ax.plot(frames, [r['cMo_optimized'][2][3] for r in self.frame_records], 'b-', label='Optimized z')
        ax.plot(frames, [r['cMo'][2][3] for r in self.frame_records], 'b--', label='PnP z', alpha=0.7)
        ax.set_ylabel('Z (mm)')
        ax.set_title('cMo Z')
        ax.grid(True)
        ax.legend()

        ax = axes[2, 0]
        ax.plot(frames, [r['robot_pose'][0][3] for r in self.frame_records], 'r-', label='x')
        ax.set_xlabel('Frame ID')
        ax.set_ylabel('X (mm)')
        ax.set_title('bMe X')
        ax.grid(True)
        ax.legend()

        ax = axes[2, 1]
        ax.plot(frames, [r['robot_pose'][1][3] for r in self.frame_records], 'g-', label='y')
        ax.set_xlabel('Frame ID')
        ax.set_ylabel('Y (mm)')
        ax.set_title('bMe Y')
        ax.grid(True)
        ax.legend()

        ax = axes[2, 2]
        ax.plot(frames, [r['robot_pose'][2][3] for r in self.frame_records], 'b-', label='z')
        ax.set_xlabel('Frame ID')
        ax.set_ylabel('Z (mm)')
        ax.set_title('bMe Z')
        ax.grid(True)
        ax.legend()

        plt.tight_layout()
        pose_plot_path = os.path.join(self.result_dir, "pose_components_vs_frame.png")
        plt.savefig(pose_plot_path, dpi=150)
        print(f"Saved pose plot to {pose_plot_path}")
        plt.close()

    def _save_euler_plots(self, unwrap_angles):
        frames = [r['frame_id'] for r in self.frame_records]

        bMo_euler_x = [pose_to_euler_tvec(np.array(r['bMo_optimized']))[0][0] for r in self.frame_records]
        bMo_euler_y = [pose_to_euler_tvec(np.array(r['bMo_optimized']))[0][1] for r in self.frame_records]
        bMo_euler_z = [pose_to_euler_tvec(np.array(r['bMo_optimized']))[0][2] for r in self.frame_records]

        bMo_pnp_euler_x = [pose_to_euler_tvec(np.array(r['bMo_init']))[0][0] for r in self.frame_records]
        bMo_pnp_euler_y = [pose_to_euler_tvec(np.array(r['bMo_init']))[0][1] for r in self.frame_records]
        bMo_pnp_euler_z = [pose_to_euler_tvec(np.array(r['bMo_init']))[0][2] for r in self.frame_records]

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
        bmo_euler_path = os.path.join(self.result_dir, "bmo_euler_vs_frame.png")
        plt.savefig(bmo_euler_path, dpi=150)
        print(f"Saved bMo euler plot to {bmo_euler_path}")
        plt.close()

        cMo_euler_x = [pose_to_euler_tvec(np.array(r['cMo_optimized']))[0][0] for r in self.frame_records]
        cMo_euler_y = [pose_to_euler_tvec(np.array(r['cMo_optimized']))[0][1] for r in self.frame_records]
        cMo_euler_z = [pose_to_euler_tvec(np.array(r['cMo_optimized']))[0][2] for r in self.frame_records]

        cMo_pnp_euler_x = [pose_to_euler_tvec(np.array(r['cMo']))[0][0] for r in self.frame_records]
        cMo_pnp_euler_y = [pose_to_euler_tvec(np.array(r['cMo']))[0][1] for r in self.frame_records]
        cMo_pnp_euler_z = [pose_to_euler_tvec(np.array(r['cMo']))[0][2] for r in self.frame_records]

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
        cmo_euler_path = os.path.join(self.result_dir, "cmo_euler_vs_frame.png")
        plt.savefig(cmo_euler_path, dpi=150)
        print(f"Saved cMo euler plot to {cmo_euler_path}")
        plt.close()

    def _save_error_plots(self):
        if not self.pnp_records or not self.optimize_records:
            return

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle('Reprojection Error vs Frame', fontsize=14)

        frames = [r['frame_id'] for r in self.pnp_records]
        ax = axes[0]
        ax.plot(frames, [r['pnp_error_round1'] for r in self.pnp_records], 'b-', label='Round1', marker='o')
        ax.plot(frames, [r['pnp_error_final'] for r in self.pnp_records], 'g-', label='Final', marker='s')
        ax.set_xlabel('Frame ID')
        ax.set_ylabel('Error (px)')
        ax.set_title('PnP Error')
        ax.legend()
        ax.grid(True)

        opt_frames = [r['frame_id'] for r in self.optimize_records]
        ax = axes[1]
        ax.plot(opt_frames, [r['frame_error'] for r in self.optimize_records], 'b-', label='Frame Error', marker='o')
        ax.plot(opt_frames, [r['avg_error'] for r in self.optimize_records if r['avg_error'] is not None], 'g-', label='Avg Error', marker='s')
        ax.set_xlabel('Frame ID')
        ax.set_ylabel('Error (px)')
        ax.set_title('Optimization Error')
        ax.legend()
        ax.grid(True)

        plt.tight_layout()
        error_plot_path = os.path.join(self.result_dir, "error_vs_frame.png")
        plt.savefig(error_plot_path, dpi=150)
        print(f"Saved error plot to {error_plot_path}")
        plt.close()

    def _print_summary(self):
        print("\n" + "=" * 80)
        print("PnP Results Summary")
        print("=" * 80)
        print(f"{'FrameID':>8} {'Pts':>4} {'Inliers':>7} {'Thresh':>8} {'Rnd1Err':>8} {'PnPErr':>8} {'BAErr':>8}")
        print("-" * 80)
        for rec in self.pnp_records:
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
        for rec in self.optimize_records:
            print(f"{rec['frame_id']:>8} {rec['n_frames_in_optimizer']:>6} "
                  f"{rec['bMo_tvec'][0]:>10.2f} {rec['bMo_tvec'][1]:>10.2f} {rec['bMo_tvec'][2]:>10.2f} "
                  f"{rec['bMo_euler'][0]:>8.2f} {rec['bMo_euler'][1]:>8.2f} {rec['bMo_euler'][2]:>8.2f} "
                  f"{rec['cMo_euler'][0]:>8.2f} {rec['cMo_euler'][1]:>8.2f} {rec['cMo_euler'][2]:>8.2f} "
                  f"{rec['frame_error']:>8.4f} {rec['avg_error'] if rec['avg_error'] is not None else 0:>8.4f}")
        print("=" * 120)
        print(f"No Detection:{self.cnt_no_detection}, Less 7pts:{self.cnt_less_7pts}, PnP Failed:{self.cnt_pnp_failed}, Large_Pose_Diff_Rejected:{self.cnt_rejected_frames}")
        print("All results saved successfully!")
        print("=" * 80)

def extract_roi(img, box):
    x0, y0, x1, y1 = map(int, box[:4])
    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(img.shape[1], x1)
    y1 = min(img.shape[0], y1)
    roi = img[y0:y1, x0:x1]
    return roi, x0, y0, x1, y1

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

def compute_reproj_error(pts3d, rvec, tvec, pts2d, K, dist):
    proj, _ = cv2.projectPoints(pts3d, rvec, tvec, K, dist)
    return np.linalg.norm(proj.reshape(-1,2) - pts2d, axis=1).mean()


def compute_per_point_reproj_errors(pts3d, rvec, tvec, pts2d, K, dist):
    """Compute reprojection error for each point (in pixels)"""
    proj, _ = cv2.projectPoints(pts3d, rvec, tvec, K, dist)
    return np.linalg.norm(proj.reshape(-1,2) - pts2d, axis=1)


def validate_cfg(cfg):
    if cfg is None:
        raise ConfigError("Config is None")

    # check num_keypoints, obj_pts and sigmas dimensions consistency
    num_kps = cfg.get('detector', {}).get('num_keypoints')
    obj_pts = cfg.get('optimizer', {}).get('obj_pts')
    if num_kps is None or obj_pts is None:
        raise ConfigError(
            "num_keypoints and obj_pts must be specified in config")
    if len(obj_pts) != num_kps:
        raise ConfigError(
            f"num_keypoints ({num_kps}) does not match \
            number of obj_pts ({len(obj_pts)})")

    # prior_sigmas = cfg.get('optimizer', {}).get('prior_sigmas')
    point_sigmas = cfg.get('optimizer', {}).get('point_sigmas')
    if point_sigmas is not None and len(point_sigmas) != num_kps:
        raise ConfigError(
            f"num_keypoints ({num_kps}) does not match \
                number of point_sigmas ({len(point_sigmas)})")
    
    # check camera params
    cam_cfg = cfg.get('camera', {})
    cam_prms_n = ['K', 'dist', 'eMc']
    for n in cam_prms_n:
        if not cam_cfg.get(n):
            raise ConfigError(f"Camera param {n} is None")
