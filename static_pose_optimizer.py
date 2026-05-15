#!/usr/bin/env python3
"""
Generic Pose Optimizer for bundle adjustment over multiple frames.
Optimizes a single static pose (e.g., bMo) to minimize reprojection error.
"""

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from scipy.spatial.transform import Rotation as R
import cv2


class StaticPoseOptimizer:
    """
    Generic pose optimizer for bundle adjustment.

    Optimizes a single static pose between robot's base frame and object frame
    to minimize reprojection error across multiple frames.
    Each frame provides 2D-3D correspondences and robot pose.

    Usage:
        optimizer = StaticPoseOptimizer(K, dist)
        optimizer.set_extrinsics(eMc)  # eye-to-hand calibration
        optimizer.set_object_pts(obj_pts)  # 3D model points
        optimizer.add_frame(robot_pose, pts2d, pts3d)  # add observation
        optimizer.set_initial_pose(bMo_init)  # set initial value
        optimizer.optimize()
        result_pose = optimizer.get_pose()
        avg_err = optimizer.get_average_error()
    """

    def __init__(self, K, dist):
        """
        Args:
            K: Camera intrinsic matrix (3x3)
            dist: Distortion coefficients (5,)
        """
        self.K = np.array(K, dtype=np.float64)
        self.dist = np.array(dist, dtype=np.float64)
        self.eMc = np.eye(4, dtype=np.float64)  # default identity
        self.obj_pts = None # 3D object points (Nx3), you can set it or provide per-frame

        self._frames = []  # list of {robot_pose, pts2d, pts3d}
        self._pose = None  # current optimized pose (4x4)
        self._pose_params = None  # 6D params (rvec, tvec)

        # 记录中间数据和结果
        self._optimization_history = []  # 优化过程记录列表
        self._frame_results = []  # 每帧的结果列表
        self._initial_pose = None  # 初始位姿

        # BA configuration
        self.loss = 'huber'
        self.loss_scale = 0.3
        self.ftol = 1e-3
        self.xtol = 1e-3
        self.max_nfev = 300

        # Cache camera params
        self.fx = self.K[0, 0]
        self.fy = self.K[1, 1]
        self.cx = self.K[0, 2]
        self.cy = self.K[1, 2]
        self.k1, self.k2, self.p1, self.p2, self.k3 = self.dist

    def set_extrinsics(self, eMc):
        """Set eye-to-hand calibration matrix (camera to end-effector)."""
        self.eMc = np.array(eMc, dtype=np.float64)

    def set_object_pts(self, obj_pts):
        """Set 3D object model points (Nx3)."""
        self.obj_pts = np.array(obj_pts, dtype=np.float64)

    def is_initialized(self):
        """Check if optimizer has been initialized with at least one frame and initial pose."""
        return self._pose is not None and len(self._frames) > 0

    def set_initial_pose(self, pose):
        """
        Set initial pose for optimization (4x4 matrix).
        Can be called before add_frame or after.
        """
        self._pose = np.array(pose, dtype=np.float64)
        self._pose_params = self._pose_to_params(self._pose)
        self._initial_pose = self._pose.copy()

    def add_frame(self, frame_index, robot_pose, pts2d, pts3d=None):
        """
        Add a frame to the optimization.

        Args:
            robot_pose: End-effector pose in base frame (4x4 matrix, bMe)
            pts2d: Observed 2D points (Nx2)
            pts3d: Corresponding 3D points (Nx3). If None, uses self.obj_pts.
            frame_index: User-defined identifier for this frame. Used for later removal.
        """
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
        """Remove a frame by its user-defined index. No-op if frame not found."""
        self._frames = [f for f in self._frames if f['index'] != frame_index]

    def remove_frames(self, frame_indices):
        """Remove multiple frames by their user-defined indices."""
        indices_to_remove = set(frame_indices)
        self._frames = [f for f in self._frames if f['index'] not in indices_to_remove]

    def remove_oldest_frame(self):
        """Remove the oldest frame (first in list). No-op if empty."""
        if self._frames:
            self._frames.pop(0)

    def get_frame_by_index(self, frame_index):
        """Get frame data by its user-defined index."""
        for frame in self._frames:
            if frame['index'] == frame_index:
                return frame
        return None

    def get_pose(self):
        """Get current optimized pose (4x4 matrix)."""
        return self._pose.copy() if self._pose is not None else None

    def get_pose_euler(self, unit='deg'):
        """
        Get pose as euler angles (xyz convention).

        Args:
            unit: 'deg' or 'rad'
        """
        if self._pose is None:
            return None
        euler = Rotation.from_matrix(self._pose[:3, :3]).as_euler('xyz')
        if unit == 'deg':
            euler = np.degrees(euler)
        return euler
    
    def get_pose_rotvec(self, unit='deg'):
        """
        Get pose as rotation vector.

        Args:
            unit: 'deg' or 'rad'
        """
        if self._pose is None:
            return None
        rv = Rotation.from_matrix(self._pose[:3, :3]).as_rotvec()
        if unit == 'deg':
            rv = np.degrees(rv)
        return rv

    def get_frame_count(self):
        """Get number of frames in optimization."""
        return len(self._frames)

    def get_frame_error(self, frame_index):
        """
        Get reprojection error for a specific frame.

        Returns:
            float: mean reprojection error in pixels
            dict: {pts3d, pts2d, proj} for the frame
        """
        frame = self.get_frame_by_index(frame_index)
        if frame is None:
            raise ValueError(f"Frame with index {frame_index} not found")

        if self._pose is None:
            raise ValueError("No pose optimized yet. Call optimize() first.")

        cMo = self.compute_cMo(frame['robot_pose'])

        proj = self._project(frame['pts3d'], cMo[:3, :3], cMo[:3, 3])
        errors = np.linalg.norm(proj - frame['pts2d'], axis=1)

        return float(np.mean(errors)), {
            'pts3d': frame['pts3d'].copy(),
            'pts2d': frame['pts2d'].copy(),
            'proj': proj.copy(),
            'errors': errors.copy()
        }

    def get_average_error(self):
        """Get average reprojection error across all frames."""
        if not self._frames or self._pose is None:
            return None

        errors = self.get_all_errors()
        return sum(errors.values()) / len(errors)

    def get_all_errors(self):
        """Get reprojection error for all frames as a dict {frame_index: error}."""
        if not self._frames or self._pose is None:
            return None
        errors = {}
        for frame in self._frames:
            err, _ = self.get_frame_error(frame['index'])
            errors[frame['index']] = err
        return errors

    def optimize(self, pose_init=None):
        """
        Run bundle adjustment optimization.

        Args:
            pose_init: Initial pose (4x4). If None, uses previously set initial pose.
        """
        if not self._frames:
            raise ValueError("No frames added. Add at least one frame before optimizing.")

        if pose_init is not None:
            self.set_initial_pose(pose_init)
        elif self._pose is None:
            raise ValueError("No initial pose set. Call set_initial_pose() or pass pose_init.")

        initial_params = self._pose_params.copy()

        # 创建优化器functor
        optimizer = _StaticPoseOptimizerFunctor(
            self._frames, self.K, self.dist,
            self.eMc, self.loss, self.loss_scale
        )

        # 记录初始状态
        self._optimization_history = []
        self._frame_results = []

        # 记录初始误差
        initial_errors = self._compute_all_frame_errors(optimizer, initial_params)
        self._optimization_history.append({
            'iteration': 0,
            'cost': float(np.sum(initial_errors)),
            'avg_error': float(np.mean(initial_errors)),
            'max_error': float(np.max(initial_errors)),
            'params': initial_params.copy(),
            'pose': self._params_to_pose(initial_params).copy()
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

        self._pose = self._params_to_pose(result.x)
        self._pose_params = result.x

        # 记录最终状态
        final_errors = self._compute_all_frame_errors(optimizer, result.x)
        self._optimization_history.append({
            'iteration': result.nfev,
            'cost': float(np.sum(final_errors)),
            'avg_error': float(np.mean(final_errors)),
            'max_error': float(np.max(final_errors)),
            'params': result.x.copy(),
            'pose': self._pose.copy(),
            'success': result.success,
            'message': result.message
        })

        # 记录每帧详细结果
        self._record_frame_results(optimizer, result.x)

        return result

    def _compute_all_frame_errors(self, optimizer, params):
        """计算所有帧的残差平方和。"""
        residuals = optimizer(params)
        n_frames = len(self._frames)
        errors_per_frame = []
        idx = 0
        for frame in self._frames:
            n_pts = len(frame['pts2d'])
            frame_residuals = residuals[idx:idx + n_pts * 2]
            errors = np.sqrt(np.sum(frame_residuals.reshape(n_pts, 2) ** 2, axis=1))
            errors_per_frame.append(np.mean(errors))
            idx += n_pts * 2
        return np.array(errors_per_frame)

    def _record_frame_results(self, optimizer, params):
        """记录每帧的详细结果。"""
        bMo = self._params_to_pose(params)
        for frame in self._frames:
            bMe = frame['robot_pose']
            pts3d = frame['pts3d']
            pts2d = frame['pts2d']

            eMb = np.linalg.inv(bMe)
            eMc_inv = np.linalg.inv(self.eMc)
            cMo = eMc_inv @ eMb @ bMo

            rvec = Rotation.from_matrix(cMo[:3, :3]).as_rotvec()
            tvec = cMo[:3, 3]

            proj = optimizer._project(pts3d, rvec, tvec)
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
        """
        获取优化过程记录列表。

        Returns:
            list: 优化历史列表，每个元素包含迭代信息
        """
        return self._optimization_history.copy()

    def get_frame_results(self):
        """
        获取每帧的详细结果列表。

        Returns:
            list: 每帧结果列表
        """
        return self._frame_results.copy()

    def get_initial_pose(self):
        """获取初始位姿。"""
        return self._initial_pose.copy() if self._initial_pose is not None else None

    def print_summary(self):
        """打印优化结果摘要。"""
        if not self._optimization_history:
            print("No optimization history available.")
            return

        print("\n" + "=" * 60)
        print("Optimization Summary")
        print("=" * 60)

        # 初始和最终位姿
        initial = self._optimization_history[0]
        final = self._optimization_history[-1]

        print("\n--- Pose Comparison ---")
        print(f"Initial: t={initial['pose'][:3, 3]}")
        print(f"Final:   t={final['pose'][:3, 3]}")

        initial_euler = Rotation.from_matrix(initial['pose'][:3, :3]).as_euler('xyz', degrees=True)
        final_euler = Rotation.from_matrix(final['pose'][:3, :3]).as_euler('xyz', degrees=True)
        print(f"Initial euler (xyz deg): {initial_euler}")
        print(f"Final euler (xyz deg): {final_euler}")

        print("\n--- Error Evolution ---")
        print(f"Initial avg error: {initial['avg_error']:.4f} px")
        print(f"Final avg error:   {final['avg_error']:.4f} px")
        print(f"Error reduction:  {(1 - final['avg_error']/initial['avg_error'])*100:.1f}%")

        # 每帧结果
        if self._frame_results:
            print("\n--- Per-Frame Results ---")
            print(f"{'Frame':>6} {'N_pts':>6} {'Mean':>8} {'Max':>8} {'Std':>8}")
            print("-" * 40)
            for fr in self._frame_results:
                print(f"{fr['frame_index']:>6} {fr['n_points']:>6} "
                      f"{fr['mean_error']:>8.3f} {fr['max_error']:>8.3f} {fr['std_error']:>8.3f}")
            print("-" * 40)
            total_error = sum(fr['total_error'] for fr in self._frame_results)
            total_pts = sum(fr['n_points'] for fr in self._frame_results)
            print(f"{'Total':>6} {total_pts:>6} {total_error/total_pts:>8.3f}")

        print("\n" + "=" * 60)

    def export_to_dict(self):
        """
        导出会话数据到字典格式，便于序列化和进一步分析。

        Returns:
            dict: 包含所有中间数据和结果的字典
        """
        if not self._optimization_history:
            return None

        final = self._optimization_history[-1]
        return {
            'initial_pose': self._initial_pose.tolist() if self._initial_pose is not None else None,
            'optimized_pose': final['pose'].tolist(),
            'initial_avg_error': self._optimization_history[0]['avg_error'],
            'final_avg_error': final['avg_error'],
            'error_reduction_percent': (1 - final['avg_error'] / self._optimization_history[0]['avg_error']) * 100,
            'optimization_history': [
                {
                    'iteration': h['iteration'],
                    'cost': h['cost'],
                    'avg_error': h['avg_error'],
                    'max_error': h['max_error']
                } for h in self._optimization_history
            ],
            'frame_results': [
                {
                    'frame_index': fr['frame_index'],
                    'n_points': fr['n_points'],
                    'mean_error': fr['mean_error'],
                    'max_error': fr['max_error'],
                    'min_error': fr['min_error'],
                    'std_error': fr['std_error'],
                    'total_error': fr['total_error']
                } for fr in self._frame_results
            ]
        }

    def set_ba_config(self, loss='huber', loss_scale=1.0, ftol=1e-8, xtol=1e-8, max_nfev=500):
        """Configure bundle adjustment parameters."""
        self.loss = loss
        self.loss_scale = loss_scale
        self.ftol = ftol
        self.xtol = xtol
        self.max_nfev = max_nfev

    def compute_cMo(self, robot_pose):
        """
        Compute camera pose from base pose and calibration.

        cMo = (bMe @ eMc)^{-1} @ bMo = eMc^{-1} @ bMe^{-1} @ bMo
        """
        bMe = robot_pose
        eMb = np.linalg.inv(bMe)
        eMc_inv = np.linalg.inv(self.eMc)
        bMo = self._pose
        cMo = eMc_inv @ eMb @ bMo
        return cMo

    def _project(self, pts3d, R, tvec):
        """Project 3D points to 2D with distortion."""
        pts3d = np.array(pts3d, dtype=np.float64)
        R_mat = np.array(R, dtype=np.float64)

        # Rotate and translate
        pts_cam = R_mat @ pts3d.T + np.array(tvec).reshape(3, 1)

        # Normalize
        x = pts_cam[0] / pts_cam[2]
        y = pts_cam[1] / pts_cam[2]

        # Distortion
        r2 = x * x + y * y
        radial = 1 + self.k1 * r2 + self.k2 * r2 * r2 + self.k3 * r2 * r2 * r2
        x_dist = x * radial + 2 * self.p1 * x * y + self.p2 * (r2 + 2 * x * x)
        y_dist = y * radial + self.p1 * (r2 + 2 * y * y) + 2 * self.p2 * x * y

        # Final projection
        u = self.fx * x_dist + self.cx
        v = self.fy * y_dist + self.cy
        return np.stack([u, v], axis=1)

    @staticmethod
    def _pose_to_params(pose):
        """Convert 4x4 pose to 6D params (rvec + tvec)."""
        rvec = Rotation.from_matrix(pose[:3, :3]).as_rotvec()
        tvec = pose[:3, 3]
        return np.concatenate([rvec, tvec])

    @staticmethod
    def _params_to_pose(params):
        """Convert 6D params to 4x4 pose."""
        rvec = params[:3]
        tvec = params[3:6]
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = Rotation.from_rotvec(rvec).as_matrix()
        pose[:3, 3] = tvec
        return pose

    def compare_poses(self, other_pose):
        """
        Compare current pose with another pose.

        Returns:
            float: Angle difference in degrees
        """
        if self._pose is None:
            raise ValueError("No pose optimized yet.")

        other_pose = np.array(other_pose)
        R1 = self._pose[:3, :3]
        R2 = other_pose[:3, :3]

        R_diff = R1 @ R2.T
        diff_rotvec = Rotation.from_matrix(R_diff).as_rotvec()
        return float(np.degrees(np.linalg.norm(diff_rotvec)))


class _StaticPoseOptimizerFunctor:
    """Internal functor for scipy optimization."""

    def __init__(self, frames, K, dist, eMc, loss, loss_scale):
        self.frames = frames
        self.eMc = np.array(eMc, dtype=np.float64)

        # Unpack camera params
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        k1, k2, p1, p2, k3 = dist
        self.fx, self.fy = fx, fy
        self.cx, self.cy = cx, cy
        self.k1, self.k2, self.p1, self.p2, self.k3 = k1, k2, p1, p2, k3

    def __call__(self, params):
        bMo = StaticPoseOptimizer._params_to_pose(params)
        residuals = []

        for frame in self.frames:
            bMe = frame['robot_pose']
            pts3d = frame['pts3d']
            pts2d = frame['pts2d']

            # Compute cMo
            eMb = np.linalg.inv(bMe)
            eMc_inv = np.linalg.inv(self.eMc)
            cMo = eMc_inv @ eMb @ bMo

            rvec = Rotation.from_matrix(cMo[:3, :3]).as_rotvec()
            tvec = cMo[:3, 3]

            # Project
            proj = self._project(pts3d, rvec, tvec)
            residuals.extend((proj - pts2d).flatten())

        return np.array(residuals)

    def _project(self, pts3d, rvec, tvec):
        """Project 3D points to 2D with distortion."""
        R, _ = cv2.Rodrigues(np.array(rvec, dtype=np.float64))
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


# ============ Utility Functions ============

def rotation_diff_deg(R1, R2):
    """
    Compute angle difference between two rotation matrices in degrees.
    """
    R_diff = np.array(R1) @ np.array(R2).T
    rotvec = Rotation.from_matrix(R_diff).as_rotvec()
    return float(np.degrees(np.linalg.norm(rotvec)))

def pose_to_euler_tvec(pose, unit='deg'):
    """Convert 4x4 pose to euler angles and translation vector."""
    if pose is None:
        return None
    # check pose are 4x4 numpy array
    if not isinstance(pose, np.ndarray) or pose.shape != (4, 4):
        raise ValueError("Pose must be a 4x4 numpy array")
    # check if rotation part is valid    
    if not np.allclose(pose[3, :], [0, 0, 0, 1]):
        raise ValueError("Invalid pose: last row must be [0, 0, 0, 1]")   
    if not np.allclose(pose[:3, :3] @ pose[:3, :3].T, np.eye(3), atol=1e-6):
        raise ValueError("Invalid pose: rotation part must be orthogonal")
    
    euler = Rotation.from_matrix(pose[:3, :3]).as_euler('xyz')
    tvec = pose[:3, 3]
    if unit == 'deg':
        euler = np.degrees(euler)
    return euler, tvec

if __name__ == "__main__":
    # Simple test
    K = np.array([
        [1015.4, 0, 638.5],
        [0, 1015.4, 386.8],
        [0, 0, 1]
    ])
    dist = np.array([0.1, -0.2, 0, 0, 0.07])

    optimizer = StaticPoseOptimizer(K, dist)

    # Set 3D model points
    obj_pts = np.array([
        [-8.0, 11.2, 0.0], [8.0, 11.2, 0.0],
        [-16.0, 0.0, 0.0], [0.0, 0.0, 0.0], [16.0, 0.0, 0.0],
        [-8.0, -13.9, 0.0], [8.0, -13.9, 0.0]
    ], dtype=np.float64)
    optimizer.set_object_pts(obj_pts)

    # Set extrinsics
    eMc = np.eye(4)
    optimizer.set_extrinsics(eMc)

    # Add some dummy frames
    bMo_init = np.eye(4)
    bMo_init[:3, 3] = [348, -1084, 498]

    for i in range(5):
        robot_pose = np.eye(4)
        robot_pose[:3, 3] = [i * 10, i * 5, 100 + i * 50]
        pts2d = np.random.rand(7, 2) * 200 + 400  # dummy 2D points
        optimizer.add_frame(i, robot_pose, pts2d)

    optimizer.set_initial_pose(bMo_init)
    result = optimizer.optimize()

    print("Optimized pose:")
    print(optimizer.get_pose())
    print(f"Average error: {optimizer.get_average_error():.4f}")