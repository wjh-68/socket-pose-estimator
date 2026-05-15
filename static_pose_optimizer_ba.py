#!/usr/bin/env python3
"""
Probabilistic static pose bundle adjustment with per-frame SE(3) perturbation.

This optimizer maintains a shared global object pose in the base frame (bMo)
plus one per-frame SE(3) slack variable to absorb timing and robot/camera
uncertainties.
"""

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


class StaticPoseOptimizer:
    """Shared global pose + per-frame perturbation optimizer."""

    def __init__(self, K, dist, prior_sigma=None, point_sigmas=None):
        self.K = np.array(K, dtype=np.float64)
        self.dist = np.array(dist, dtype=np.float64)
        self.eMc = np.eye(4, dtype=np.float64)
        self.obj_pts = None
        self.point_sigmas = None  # per-point reprojection uncertainty in pixels

        self._frames = []
        self._pose = None
        self._pose_params = None
        self._last_optimized_params = None
        self._initial_pose = None

        self._optimization_history = []
        self._frame_results = []
        self._diagnostics = {}

        self.loss = 'huber'
        self.loss_scale = 0.3
        self.ftol = 1e-3
        self.xtol = 1e-3
        self.max_nfev = 300

        self.fx = self.K[0, 0]
        self.fy = self.K[1, 1]
        self.cx = self.K[0, 2]
        self.cy = self.K[1, 2]
        self.k1, self.k2, self.p1, self.p2, self.k3 = self.dist

        # Default prior sigma for per-frame perturbations 
        # (rx, ry, rz in radians, tx, ty in mm, tz in mm)
        if prior_sigma is None:
            self.prior_sigma = np.array([
                np.deg2rad(5.0),
                np.deg2rad(5.0),
                np.deg2rad(1.0),
                0.5,
                0.5,
                10.0
            ], dtype=np.float64)
        else:
            self.prior_sigma = np.array(prior_sigma, dtype=np.float64)

        # Set point sigmas if provided
        if point_sigmas is not None:
            self.set_point_sigmas(point_sigmas)

    def set_extrinsics(self, eMc):
        self.eMc = np.array(eMc, dtype=np.float64)

    def set_object_pts(self, obj_pts):
        self.obj_pts = np.array(obj_pts, dtype=np.float64)
        # Initialize point_sigmas to default (all 1.0) if not set
        if self.point_sigmas is None:
            self.point_sigmas = np.ones(len(self.obj_pts), dtype=np.float64)

    def set_point_sigmas(self, point_sigmas):
        """Set per-point reprojection uncertainty (in pixels).
        
        Args:
            point_sigmas: array of shape (N,) with sigma for each 3D point.
                         sigma_i > 0: larger sigma = lower weight (noisier point)
                         Default: all 1.0 (isotropic, equal weight)
        """
        point_sigmas = np.array(point_sigmas, dtype=np.float64)
        
        # Validate dimension if obj_pts is already set
        if self.obj_pts is not None:
            if len(point_sigmas) != len(self.obj_pts):
                raise ValueError(
                    f"point_sigmas length {len(point_sigmas)} does not match "
                    f"obj_pts length {len(self.obj_pts)}"
                )
        
        # Validate all sigmas > 0
        if np.any(point_sigmas <= 0):
            raise ValueError("All point_sigmas must be positive")
        
        self.point_sigmas = point_sigmas

    def is_initialized(self):
        return self._pose is not None and len(self._frames) > 0

    def set_initial_pose(self, pose):
        self._pose = np.array(pose, dtype=np.float64)
        self._pose_params = self._pose_to_params(self._pose)
        self._initial_pose = self._pose.copy()
        self._last_optimized_params = None

    def add_frame(self, frame_index, robot_pose, pts2d, pts3d=None):
        robot_pose = np.array(robot_pose, dtype=np.float64)
        pts2d = np.array(pts2d, dtype=np.float64)

        if pts3d is None:
            if self.obj_pts is None:
                raise ValueError("Must set obj_pts before using default pts3d")
            pts3d = self.obj_pts
        else:
            pts3d = np.array(pts3d, dtype=np.float64)

        frame = {
            'robot_pose': robot_pose,
            'pts2d': pts2d,
            'pts3d': pts3d,
            'index': frame_index
        }
        self._frames.append(frame)

    def remove_frame(self, frame_index):
        self._frames = [f for f in self._frames if f['index'] != frame_index]

    def remove_frames(self, frame_indices):
        indices_to_remove = set(frame_indices)
        self._frames = [f for f in self._frames if f['index'] not in indices_to_remove]

    def remove_oldest_frame(self):
        if self._frames:
            self._frames.pop(0)

    def get_frame_by_index(self, frame_index):
        for frame in self._frames:
            if frame['index'] == frame_index:
                return frame
        return None

    def get_pose(self):
        return self._pose.copy() if self._pose is not None else None

    def get_pose_euler(self, unit='deg'):
        if self._pose is None:
            return None
        euler = Rotation.from_matrix(self._pose[:3, :3]).as_euler('xyz')
        return np.degrees(euler) if unit == 'deg' else euler

    def get_pose_rotvec(self, unit='deg'):
        if self._pose is None:
            return None
        rv = Rotation.from_matrix(self._pose[:3, :3]).as_rotvec()
        return np.degrees(rv) if unit == 'deg' else rv

    def get_frame_count(self):
        return len(self._frames)

    def get_frame_error(self, frame_index):
        frame = self.get_frame_by_index(frame_index)
        if frame is None:
            raise ValueError(f"Frame with index {frame_index} not found")
        if self._pose is None:
            raise ValueError("No pose optimized yet. Call optimize() first.")

        cMo = self.compute_cMo(frame['robot_pose'], frame_index)
        proj = self._project(frame['pts3d'], cMo[:3, :3], cMo[:3, 3])
        errors = np.linalg.norm(proj - frame['pts2d'], axis=1)
        return float(np.mean(errors)), {
            'pts3d': frame['pts3d'].copy(),
            'pts2d': frame['pts2d'].copy(),
            'proj': proj.copy(),
            'errors': errors.copy()
        }

    def get_average_error(self):
        if not self._frames or self._pose is None:
            return None
        errors = self.get_all_errors()
        return sum(errors.values()) / len(errors)

    def get_all_errors(self):
        if not self._frames or self._pose is None:
            return None
        errors = {}
        for frame in self._frames:
            err, _ = self.get_frame_error(frame['index'])
            errors[frame['index']] = err
        return errors

    def optimize(self, pose_init=None):
        if not self._frames:
            raise ValueError("No frames added. Add at least one frame before optimizing.")
        if pose_init is not None:
            self.set_initial_pose(pose_init)
        elif self._pose is None:
            raise ValueError("No initial pose set. Call set_initial_pose() or pass pose_init.")

        if self.point_sigmas is None:
            if self.obj_pts is not None:
                self.point_sigmas = np.ones(len(self.obj_pts), dtype=np.float64)
            else:
                self.point_sigmas = np.ones(len(self._frames[0]['pts3d']), dtype=np.float64)

        # Sanity check: ensure point_sigmas length matches each frame's point count
        for frame in self._frames:
            if len(self.point_sigmas) != len(frame['pts3d']):
                raise ValueError(
                    f"point_sigmas length {len(self.point_sigmas)} does not match "
                    f"frame pts3d length {len(frame['pts3d'])}"
                )

        n_frames = len(self._frames)
        initial_params = np.zeros(6 + 6 * n_frames, dtype=np.float64)
        initial_params[:6] = self._pose_params

        optimizer = _StaticPoseOptimizerFunctor(
            self._frames, self.K, self.dist,
            self.eMc, self.prior_sigma, self.point_sigmas,
            self.loss, self.loss_scale
        )

        self._optimization_history = []
        self._frame_results = []
        self._diagnostics = {}

        initial_residuals = optimizer(initial_params)
        frame_errors = self._compute_all_frame_errors(optimizer, initial_params)
        self._optimization_history.append({
            'iteration': 0,
            'cost': float(np.sum(initial_residuals ** 2)),
            'avg_error': float(np.mean(frame_errors)),
            'max_error': float(np.max(frame_errors)),
            'params': initial_params.copy(),
            'pose': self._params_to_pose(initial_params[:6]).copy()
        })

        result = least_squares(
            optimizer,
            initial_params,
            method='dogbox',
            ftol=self.ftol,
            xtol=self.xtol,
            max_nfev=self.max_nfev,
            loss=self.loss,
            f_scale=self.loss_scale
        )

        self._pose = self._params_to_pose(result.x[:6])
        self._pose_params = result.x[:6].copy()
        self._last_optimized_params = result.x.copy()

        final_frame_errors = self._compute_all_frame_errors(optimizer, result.x)
        unweighted_rms, weighted_rms = self._compute_reprojection_statistics(result.x)
        self._optimization_history.append({
            'iteration': result.nfev,
            'cost': float(np.sum(optimizer(result.x) ** 2)),
            'avg_error': float(np.mean(final_frame_errors)),
            'max_error': float(np.max(final_frame_errors)),
            'weighted_reproj_rms': weighted_rms,
            'unweighted_reproj_rms': unweighted_rms,
            'params': result.x.copy(),
            'pose': self._pose.copy(),
            'success': result.success,
            'message': result.message,
            'condition_number': self._compute_condition_number(result.jac)
        })

        self._record_frame_results(optimizer, result.x)
        self._record_diagnostics(result)

        return result

    def _compute_all_frame_errors(self, optimizer, params):
        residuals = optimizer(params)
        errors = []
        idx = 0
        for frame in self._frames:
            n_pts = len(frame['pts2d'])
            frame_residuals = residuals[idx:idx + n_pts * 2]
            frame_errors = np.sqrt(np.sum(frame_residuals.reshape(n_pts, 2) ** 2, axis=1))
            errors.append(np.mean(frame_errors))
            idx += n_pts * 2
        return np.array(errors)

    def _record_frame_results(self, optimizer, params):
        self._frame_results = []
        for frame in self._frames:
            cMo = self.compute_cMo(frame['robot_pose'], frame['index'])
            pts3d = frame['pts3d']
            pts2d = frame['pts2d']
            proj = self._project(pts3d, cMo[:3, :3], cMo[:3, 3])
            errors = np.linalg.norm(proj - pts2d, axis=1)

            self._frame_results.append({
                'frame_index': frame['index'],
                'n_points': len(pts2d),
                'mean_error': float(np.mean(errors)),
                'max_error': float(np.max(errors)),
                'min_error': float(np.min(errors)),
                'std_error': float(np.std(errors)),
                'total_error': float(np.sum(errors)),
                'pts2d': pts2d.copy(),
                'proj': proj.copy(),
                'errors': errors.copy(),
                'cMo': cMo.copy()
            })

    def get_optimization_history(self):
        return self._optimization_history.copy()

    def get_frame_results(self):
        return self._frame_results.copy()

    def get_initial_pose(self):
        return self._initial_pose.copy() if self._initial_pose is not None else None

    def get_diagnostics(self):
        return self._diagnostics.copy()

    def print_summary(self):
        if not self._optimization_history:
            print("No optimization history available.")
            return

        initial = self._optimization_history[0]
        final = self._optimization_history[-1]
        print("\n" + "=" * 60)
        print("Optimization Summary")
        print("=" * 60)
        print(f"Initial avg error: {initial['avg_error']:.4f} px")
        print(f"Final avg error:   {final['avg_error']:.4f} px")
        print(f"Error reduction:  {(1 - final['avg_error'] / initial['avg_error']) * 100:.1f}%")
        print(f"Condition number (JTJ): {final.get('condition_number', np.nan):.3g}")
        if self._diagnostics:
            print(f"Average perturbation magnitude: {self._diagnostics['avg_perturbation_magnitude']:.4f}")
            print("Per-axis perturbation std (rx,ry,rz,tx,ty,tz): " +
                  ", ".join(f"{x:.4f}" for x in self._diagnostics['perturbation_std']))
            if 'weighted_reproj_rms' in self._diagnostics:
                print(f"Weighted reprojection RMS:   {self._diagnostics['weighted_reproj_rms']:.4f} px")
            if 'unweighted_reproj_rms' in self._diagnostics:
                print(f"Unweighted reprojection RMS: {self._diagnostics['unweighted_reproj_rms']:.4f} px")
            
            if 'point_sigmas' in self._diagnostics:
                print(f"\nPoint Sigma Configuration (per-point reprojection uncertainty):")
                print(f"  Mean sigma:   {self._diagnostics['avg_point_sigma']:.4f} px")
                print(f"  Min sigma:    {self._diagnostics['min_point_sigma']:.4f} px")
                print(f"  Max sigma:    {self._diagnostics['max_point_sigma']:.4f} px")
                sigmas = self._diagnostics['point_sigmas']
                for i, sigma in enumerate(sigmas):
                    print(f"  Point {i}: sigma={sigma:.4f} px")

        if self._frame_results:
            print("\n--- Per-Frame Results ---")
            print(f"{'Frame':>6} {'N_pts':>6} {'Mean':>8} {'Max':>8} {'Std':>8}")
            print("-" * 40)
            for fr in self._frame_results:
                print(f"{fr['frame_index']:>6} {fr['n_points']:>6} "
                      f"{fr['mean_error']:>8.3f} {fr['max_error']:>8.3f} {fr['std_error']:>8.3f}")
            total_error = sum(fr['total_error'] for fr in self._frame_results)
            total_pts = sum(fr['n_points'] for fr in self._frame_results)
            print(f"{'Total':>6} {total_pts:>6} {total_error/total_pts:>8.3f}")
        print("\n" + "=" * 60)

    def export_to_dict(self):
        if not self._optimization_history:
            return None
        final = self._optimization_history[-1]
        return {
            'initial_pose': self._initial_pose.tolist() if self._initial_pose is not None else None,
            'optimized_pose': final['pose'].tolist(),
            'initial_avg_error': self._optimization_history[0]['avg_error'],
            'final_avg_error': final['avg_error'],
            'error_reduction_percent': (1 - final['avg_error'] / self._optimization_history[0]['avg_error']) * 100,
            'condition_number': final.get('condition_number'),
            'diagnostics': self._diagnostics.copy(),
            'optimization_history': [
                {'iteration': h['iteration'], 'cost': h['cost'], 'avg_error': h['avg_error'], 'max_error': h['max_error']}
                for h in self._optimization_history
            ],
            'frame_results': [
                {k: fr[k] for k in ('frame_index', 'n_points', 'mean_error', 'max_error', 'min_error', 'std_error', 'total_error')}
                for fr in self._frame_results
            ]
        }

    def set_ba_config(self, loss='huber', loss_scale=1.0, ftol=1e-8, xtol=1e-8, max_nfev=500):
        self.loss = loss
        self.loss_scale = loss_scale
        self.ftol = ftol
        self.xtol = xtol
        self.max_nfev = max_nfev

    def compute_cMo(self, robot_pose, frame_index=None):
        bMe = robot_pose
        eMb = np.linalg.inv(bMe)
        eMc_inv = np.linalg.inv(self.eMc)
        cMo_nominal = eMc_inv @ eMb @ self._pose
        if self._last_optimized_params is None or frame_index is None:
            return cMo_nominal
        frame_pos = self._find_frame_position(frame_index)
        if frame_pos is None:
            return cMo_nominal
        delta = self._last_optimized_params[6 + 6 * frame_pos: 6 + 6 * (frame_pos + 1)]
        return self.apply_perturbation(cMo_nominal, delta)

    @staticmethod
    def apply_perturbation(T, delta):
        dR = Rotation.from_rotvec(np.array(delta[:3], dtype=np.float64)).as_matrix()
        dt = np.array(delta[3:], dtype=np.float64)
        T_delta = np.eye(4, dtype=np.float64)
        T_delta[:3, :3] = dR
        T_delta[:3, 3] = dt
        return T @ T_delta

    def _find_frame_position(self, frame_index):
        for idx, frame in enumerate(self._frames):
            if frame['index'] == frame_index:
                return idx
        return None

    def _project(self, pts3d, R, tvec):
        pts3d = np.array(pts3d, dtype=np.float64)
        pts_cam = R @ pts3d.T + np.array(tvec, dtype=np.float64).reshape(3, 1)
        x = pts_cam[0] / pts_cam[2]
        y = pts_cam[1] / pts_cam[2]
        r2 = x * x + y * y
        radial = 1 + self.k1 * r2 + self.k2 * r2 * r2 + self.k3 * r2 * r2 * r2
        x_dist = x * radial + 2 * self.p1 * x * y + self.p2 * (r2 + 2 * x * x)
        y_dist = y * radial + self.p1 * (r2 + 2 * y * y) + 2 * self.p2 * x * y
        u = self.fx * x_dist + self.cx
        v = self.fy * y_dist + self.cy
        return np.stack([u, v], axis=1)

    @staticmethod
    def _pose_to_params(pose):
        rvec = Rotation.from_matrix(pose[:3, :3]).as_rotvec()
        tvec = pose[:3, 3]
        return np.concatenate([rvec, tvec])

    @staticmethod
    def _params_to_pose(params):
        rvec = params[:3]
        tvec = params[3:6]
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = Rotation.from_rotvec(rvec).as_matrix()
        pose[:3, 3] = tvec
        return pose

    def _compute_condition_number(self, jac):
        if jac is None:
            return np.inf
        try:
            jtj = jac.T @ jac
            return float(np.linalg.cond(jtj))
        except Exception:
            return np.inf

    def _compute_reprojection_statistics(self, params):
        bMo = self._params_to_pose(params[:6])
        all_errors = []
        all_weighted_errors = []

        for frame_index, frame in enumerate(self._frames):
            bMe = frame['robot_pose']
            eMb = np.linalg.inv(bMe)
            eMc_inv = np.linalg.inv(self.eMc)
            cMo_nominal = eMc_inv @ eMb @ bMo
            delta = params[6 + 6 * frame_index: 6 + 6 * (frame_index + 1)]
            cMo = self.apply_perturbation(cMo_nominal, delta)
            proj = self._project(frame['pts3d'], cMo[:3, :3], cMo[:3, 3])
            err = proj - frame['pts2d']
            all_errors.append(err)

            if self.point_sigmas is not None:
                if len(self.point_sigmas) != len(frame['pts3d']):
                    raise ValueError(
                        f"point_sigmas length {len(self.point_sigmas)} does not match "
                        f"frame pts3d length {len(frame['pts3d'])}"
                    )
                weighted_err = err / self.point_sigmas[:, np.newaxis]
            else:
                weighted_err = err
            all_weighted_errors.append(weighted_err)

        if len(all_errors) == 0:
            return 0.0, 0.0

        all_errors = np.vstack(all_errors)
        all_weighted_errors = np.vstack(all_weighted_errors)
        unweighted_rms = float(np.sqrt(np.mean(all_errors ** 2)))
        weighted_rms = float(np.sqrt(np.mean(all_weighted_errors ** 2)))
        return unweighted_rms, weighted_rms

    def _record_diagnostics(self, result):
        if self._last_optimized_params is None:
            return
        deltas = self._last_optimized_params[6:].reshape(-1, 6)
        norms = np.linalg.norm(deltas, axis=1)
        self._diagnostics['avg_perturbation_magnitude'] = float(np.mean(norms)) if norms.size else 0.0
        self._diagnostics['max_perturbation_magnitude'] = float(np.max(norms)) if norms.size else 0.0
        self._diagnostics['perturbation_std'] = np.std(deltas, axis=0).tolist() if norms.size else [0.0] * 6
        self._diagnostics['perturbation_mean'] = np.mean(deltas, axis=0).tolist() if norms.size else [0.0] * 6
        self._diagnostics['condition_number'] = self._compute_condition_number(result.jac)

        if result.x is not None:
            unweighted_rms, weighted_rms = self._compute_reprojection_statistics(result.x)
            self._diagnostics['unweighted_reproj_rms'] = unweighted_rms
            self._diagnostics['weighted_reproj_rms'] = weighted_rms
        
        # Record point sigma configuration
        if self.point_sigmas is not None:
            self._diagnostics['point_sigmas'] = self.point_sigmas.tolist()
            self._diagnostics['avg_point_sigma'] = float(np.mean(self.point_sigmas))
            self._diagnostics['min_point_sigma'] = float(np.min(self.point_sigmas))
            self._diagnostics['max_point_sigma'] = float(np.max(self.point_sigmas))


class _StaticPoseOptimizerFunctor:
    def __init__(self, frames, K, dist, eMc, prior_sigma, point_sigmas, loss, loss_scale):
        self.frames = frames
        self.eMc = np.array(eMc, dtype=np.float64)
        self.prior_sigma = np.array(prior_sigma, dtype=np.float64)
        self.point_sigmas = point_sigmas  # shape (N,), per-point uncertainty
        self.loss = loss
        self.loss_scale = loss_scale

        self.fx = K[0, 0]
        self.fy = K[1, 1]
        self.cx = K[0, 2]
        self.cy = K[1, 2]
        self.k1, self.k2, self.p1, self.p2, self.k3 = dist

    def __call__(self, params):
        bMo = StaticPoseOptimizer._params_to_pose(params[:6])
        residuals = []

        for frame_index, frame in enumerate(self.frames):
            bMe = frame['robot_pose']
            eMb = np.linalg.inv(bMe)
            eMc_inv = np.linalg.inv(self.eMc)
            cMo_nominal = eMc_inv @ eMb @ bMo
            delta = params[6 + 6 * frame_index: 6 + 6 * (frame_index + 1)]
            cMo = StaticPoseOptimizer.apply_perturbation(cMo_nominal, delta)

            proj = self._project(frame['pts3d'], cMo[:3, :3], cMo[:3, 3])
            
            # Compute reprojection error
            err = proj - frame['pts2d']  # shape (N, 2)
            
            # Apply per-point whitening: divide by point sigma
            # point_sigmas shape: (N,), err shape: (N, 2)
            if self.point_sigmas is not None:
                if len(frame['pts3d']) != len(self.point_sigmas):
                    raise ValueError(
                        f"point_sigmas length {len(self.point_sigmas)} does not match "
                        f"frame pts3d length {len(frame['pts3d'])}"
                    )
                whitened_err = err / self.point_sigmas[:, np.newaxis]
            else:
                whitened_err = err
            
            # Flatten and add to residuals
            residuals.extend(whitened_err.flatten())
            
            # Add prior penalty on perturbation delta
            residuals.extend((delta / self.prior_sigma).flatten())

        return np.array(residuals, dtype=np.float64)

    def _project(self, pts3d, R, tvec):
        pts3d = np.array(pts3d, dtype=np.float64)
        pts_cam = R @ pts3d.T + np.array(tvec, dtype=np.float64).reshape(3, 1)
        x = pts_cam[0] / pts_cam[2]
        y = pts_cam[1] / pts_cam[2]
        r2 = x * x + y * y
        radial = 1 + self.k1 * r2 + self.k2 * r2 * r2 + self.k3 * r2 * r2 * r2
        x_dist = x * radial + 2 * self.p1 * x * y + self.p2 * (r2 + 2 * x * x)
        y_dist = y * radial + self.p1 * (r2 + 2 * y * y) + 2 * self.p2 * x * y
        u = self.fx * x_dist + self.cx
        v = self.fy * y_dist + self.cy
        return np.stack([u, v], axis=1)


def rotation_diff_deg(R1, R2):
    R_diff = np.array(R1) @ np.array(R2).T
    rotvec = Rotation.from_matrix(R_diff).as_rotvec()
    return float(np.degrees(np.linalg.norm(rotvec)))


def pose_to_euler_tvec(pose, unit='deg'):
    if pose is None:
        return None
    if not isinstance(pose, np.ndarray) or pose.shape != (4, 4):
        raise ValueError("Pose must be a 4x4 numpy array")
    if not np.allclose(pose[3, :], [0, 0, 0, 1]):
        raise ValueError("Invalid pose: last row must be [0, 0, 0, 1]")
    if not np.allclose(pose[:3, :3] @ pose[:3, :3].T, np.eye(3), atol=1e-6):
        raise ValueError("Invalid pose: rotation part must be orthogonal")
    euler = Rotation.from_matrix(pose[:3, :3]).as_euler('xyz')
    tvec = pose[:3, 3]
    if unit == 'deg':
        euler = np.degrees(euler)
    return euler, tvec


if __name__ == '__main__':
    K = np.array([
        [1015.4, 0, 638.5],
        [0, 1015.4, 386.8],
        [0, 0, 1]
    ], dtype=np.float64)
    dist = np.array([0.1, -0.2, 0, 0, 0.07], dtype=np.float64)

    optimizer = StaticPoseOptimizer(K, dist)
    obj_pts = np.array([
        [-8.0, 11.2, 0.0], [8.0, 11.2, 0.0],
        [-16.0, 0.0, 0.0], [0.0, 0.0, 0.0], [16.0, 0.0, 0.0],
        [-8.0, -13.9, 0.0], [8.0, -13.9, 0.0]
    ], dtype=np.float64)
    optimizer.set_object_pts(obj_pts)
    optimizer.set_extrinsics(np.eye(4))

    bMo_init = np.eye(4)
    bMo_init[:3, 3] = [348, -1084, 498]
    optimizer.set_initial_pose(bMo_init)

    for i in range(5):
        robot_pose = np.eye(4)
        robot_pose[:3, 3] = [i * 10.0, i * 5.0, 100.0 + i * 50.0]
        pts2d = np.random.rand(7, 2) * 200 + 400
        optimizer.add_frame(i, robot_pose, pts2d)

    result = optimizer.optimize()
    print('optimized pose:')
    print(optimizer.get_pose())
    print('avg error:', optimizer.get_average_error())
    print('diagnostics:', optimizer.get_diagnostics())
