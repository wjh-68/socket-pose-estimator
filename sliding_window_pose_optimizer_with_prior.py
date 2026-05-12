#!/usr/bin/env python3
"""
Sliding Window Pose Optimizer with Robot Pose Prior.

Optimizes bMo (object pose in base frame) and per-frame robot pose corrections
using a sliding window approach. The optimizer uses manifold-inspired
parameterization (SE(3) via 6D rvec+tvec) with two residual terms:

1. Reprojection residual: geometric distance between projected and observed 2D points
2. Robot pose prior: constrains robot pose corrections to be small (assumes
   measurement is mostly correct, only small errors exist)

The information matrix is constructed from configured sigmas:
- reproj_std: standard deviation for reprojection residuals (pixels)
- robot_pose_std: standard deviation for robot pose corrections (mm and radians)
"""

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
import cv2
from typing import List, Dict, Tuple, Optional


def pose_to_euler_tvec(pose, unit='deg'):
    """Convert 4x4 pose to euler angles and translation vector."""
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


class SE3:
    """SE(3) operations using 6D rvec+tvec parameterization."""

    @staticmethod
    def from_rvec_tvec(rvec, tvec):
        """Create 4x4 SE(3) matrix from rvec (rotvec) and tvec."""
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = Rotation.from_rotvec(rvec).as_matrix()
        pose[:3, 3] = tvec
        return pose

    @staticmethod
    def to_rvec_tvec(pose):
        """Convert 4x4 SE(3) to rvec, tvec."""
        rvec = Rotation.from_matrix(pose[:3, :3]).as_rotvec()
        tvec = pose[:3, 3]
        return rvec, tvec

    @staticmethod
    def compose(pose1, pose2):
        """Compose two SE(3) poses: pose1 @ pose2."""
        return pose1 @ pose2

    @staticmethod
    def inverse(pose):
        """Compute SE(3) inverse."""
        R = pose[:3, :3]
        t = pose[:3, 3]
        result = np.eye(4, dtype=np.float64)
        result[:3, :3] = R.T
        result[:3, 3] = -R.T @ t
        return result

    @staticmethod
    def exp_se3(delta):
        """Exponential map: 6D delta (omega, v) -> SE(3)"""
        if len(delta) == 6:
            omega = delta[0:3]  # rotation part (rotvec)
            v = delta[3:6]      # translation part
        else:
            raise ValueError("Delta must be 6D")

        theta = np.linalg.norm(omega)
        if theta < 1e-8:
            R = np.eye(3)
            V = np.eye(3)
        else:
            axis = omega / theta
            K = np.array([
                [0, -axis[2], axis[1]],
                [axis[2], 0, -axis[0]],
                [-axis[1], axis[0], 0]
            ])
            R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * K @ K
            V = np.eye(3) + (1 - np.cos(theta)) / theta * K + (theta - np.sin(theta)) / theta * K @ K

        t = V @ v
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = R
        pose[:3, 3] = t
        return pose

    @staticmethod
    def log_se3(pose):
        """Logarithmic map: SE(3) -> 6D delta (omega, v)"""
        R = pose[:3, :3]
        t = pose[:3, 3]

        trace_R = np.trace(R)
        if trace_R > 3 - 1e-10:
            omega = np.zeros(3)
            theta = 0
        elif trace_R < -1 + 1e-10:
            if abs(R[0,0] + 1) < 1e-6:
                axis = np.array([1, 0, 0])
            elif abs(R[1,1] + 1) < 1e-6:
                axis = np.array([0, 1, 0])
            else:
                axis = np.array([0, 0, 1])
            omega = np.pi * axis
            theta = np.pi
        else:
            theta = np.arccos((trace_R - 1) / 2)
            omega_skew = (R - R.T) / (2 * np.sin(theta))
            omega = np.array([
                omega_skew[2, 1],
                omega_skew[0, 2],
                omega_skew[1, 0]
            ])
            omega = omega * theta

        if theta < 1e-8:
            V_inv = np.eye(3)
        else:
            axis = omega / theta
            K = np.array([
                [0, -axis[2], axis[1]],
                [axis[2], 0, -axis[0]],
                [-axis[1], axis[0], 0]
            ])
            V_inv = np.eye(3) - 0.5 * K + (1 - theta / (2 * np.tan(theta / 2))) / theta ** 2 * K @ K

        v = V_inv @ t
        return np.concatenate([omega, v])


class SlidingWindowPoseOptimizerWithPrior:
    """
    Sliding window pose optimizer that optimizes bMo and per-frame robot pose corrections.

    Key features:
    - Sliding window over frames for online operation
    - Per-frame robot pose error correction (6D delta per frame)
    - Two residual terms: reprojection error + robot pose prior
    - Configurable sigmas for information matrix weighting
    - SE(3) manifold methods via 6D rvec+tvec parameterization
    """

    def __init__(self, K, dist,
                 reproj_std=1.0,
                 robot_pose_std_translation=5.0,
                 robot_pose_std_angle_deg=1.0):
        """
        Args:
            K: Camera intrinsic matrix (3x3)
            dist: Distortion coefficients (5,)
            reproj_std: Standard deviation for reprojection residuals (pixels)
            robot_pose_std_translation: Standard deviation for translation corrections (mm)
            robot_pose_std_angle_deg: Standard deviation for angle corrections (degrees)
        """
        self.K = np.array(K, dtype=np.float64)
        self.dist = np.array(dist, dtype=np.float64)
        self.eMc = np.eye(4, dtype=np.float64)

        self.reproj_std = reproj_std
        self.robot_pose_std_translation = robot_pose_std_translation
        self.robot_pose_std_angle_deg = robot_pose_std_angle_deg
        self.robot_pose_std_angle = np.deg2rad(robot_pose_std_angle_deg)

        self.obj_pts = None
        self._frames: List[Dict] = []
        self._bMo: Optional[np.ndarray] = None
        self._last_deltas: List[np.ndarray] = []  # Store optimized deltas per frame

        self.loss = 'huber'
        self.loss_scale = 1.0
        self.ftol = 1e-4
        self.xtol = 1e-4
        self.max_nfev = 500

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

    def set_initial_pose(self, pose):
        """Set initial bMo pose (4x4 matrix)."""
        self._bMo = np.array(pose, dtype=np.float64)

    def is_initialized(self):
        """Check if optimizer has been initialized."""
        return self._bMo is not None and len(self._frames) > 0

    def add_frame(self, frame_index: int, robot_pose: np.ndarray,
                  pts2d: np.ndarray, pts3d: np.ndarray = None):
        """Add a frame to the sliding window."""
        robot_pose = np.array(robot_pose, dtype=np.float64)
        pts2d = np.array(pts2d, dtype=np.float64)

        if pts3d is None:
            if self.obj_pts is None:
                raise ValueError("Must set obj_pts before using default pts3d")
            pts3d = self.obj_pts
        else:
            pts3d = np.array(pts3d, dtype=np.float64)

        frame = {
            'index': frame_index,
            'robot_pose': robot_pose,
            'pts2d': pts2d,
            'pts3d': pts3d
        }
        self._frames.append(frame)

    def remove_frame(self, frame_index: int):
        """Remove a frame by its user-defined index."""
        self._frames = [f for f in self._frames if f['index'] != frame_index]

    def remove_oldest_frame(self):
        """Remove the oldest frame in the sliding window (smallest index)."""
        if not self._frames:
            return
        min_index = min(f['index'] for f in self._frames)
        self._frames = [f for f in self._frames if f['index'] != min_index]

    def get_frame_count(self):
        """Get number of frames in the sliding window."""
        return len(self._frames)

    def get_pose(self):
        """Get current optimized bMo (4x4 matrix)."""
        return self._bMo.copy() if self._bMo is not None else None

    def get_frame_error(self, frame_index):
        """Get reprojection error for a specific frame."""
        frame = self.get_frame_by_index(frame_index)
        if frame is None:
            raise ValueError(f"Frame {frame_index} not found")
        if self._bMo is None:
            raise ValueError("No pose optimized yet")

        cMo = self._compute_cMo_with_correction(frame['robot_pose'], np.zeros(6))
        proj = self._project(frame['pts3d'], cMo)
        errors = np.linalg.norm(proj - frame['pts2d'], axis=1)
        return float(np.mean(errors)), {
            'pts3d': frame['pts3d'].copy(),
            'pts2d': frame['pts2d'].copy(),
            'proj': proj.copy(),
            'errors': errors.copy()
        }

    def get_average_error(self):
        """Get average reprojection error across all frames."""
        if not self._frames or self._bMo is None:
            return None
        errors = []
        for frame in self._frames:
            err, _ = self.get_frame_error(frame['index'])
            errors.append(err)
        return np.mean(errors)

    def get_frame_by_index(self, frame_index):
        """Get frame data by its user-defined index."""
        for frame in self._frames:
            if frame['index'] == frame_index:
                return frame
        return None

    def optimize(self, pose_init: np.ndarray = None):
        """Run the optimization."""
        if not self._frames:
            raise ValueError("No frames added. Add at least one frame before optimizing.")

        if pose_init is not None:
            self._bMo = np.array(pose_init, dtype=np.float64)
        elif self._bMo is None:
            raise ValueError("No initial pose set. Call set_initial_pose() or pass pose_init.")

        initial_params = self._build_params(np.zeros(6))

        optimizer = _SlidingWindowOptimizerFunctor(
            self._frames,
            self._bMo,
            self.K, self.dist,
            self.eMc,
            self.reproj_std,
            self.robot_pose_std_translation,
            self.robot_pose_std_angle
        )

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

        self._bMo = SE3.from_rvec_tvec(result.x[0:3], result.x[3:6])

        # Store optimized deltas per frame
        self._last_deltas = []
        for i, _ in enumerate(self._frames):
            delta = result.x[6 + i * 6: 6 + (i + 1) * 6]
            self._last_deltas.append(delta.copy())

        return result

    def set_ba_config(self, loss='huber', loss_scale=1.0,
                      ftol=1e-4, xtol=1e-4, max_nfev=500):
        """Configure optimization parameters."""
        self.loss = loss
        self.loss_scale = loss_scale
        self.ftol = ftol
        self.xtol = xtol
        self.max_nfev = max_nfev

    def _build_params(self, delta):
        """Build parameter vector from bMo and per-frame deltas."""
        bMo_rvec, bMo_tvec = SE3.to_rvec_tvec(self._bMo)
        params = np.concatenate([bMo_rvec, bMo_tvec])
        for _ in self._frames:
            params = np.concatenate([params, delta])
        return params

    def _compute_cMo_with_correction(self, robot_pose, delta):
        """Compute cMo using corrected robot pose."""
        correction = SE3.exp_se3(delta)
        corrected_robot_pose = SE3.compose(robot_pose, correction)

        eMc_inv = SE3.inverse(self.eMc)
        bMe_inv = SE3.inverse(corrected_robot_pose)
        cMo = eMc_inv @ bMe_inv @ self._bMo
        return cMo

    def _project(self, pts3d, cMo):
        """Project 3D points to 2D using camera model."""
        rvec, tvec = SE3.to_rvec_tvec(cMo)
        proj, _ = cv2.projectPoints(pts3d, rvec, tvec, self.K, self.dist)
        return proj.reshape(-1, 2)

    def get_corrected_robot_pose(self, frame_index, delta):
        """Get corrected robot pose for a given frame."""
        frame = self.get_frame_by_index(frame_index)
        if frame is None:
            raise ValueError(f"Frame {frame_index} not found")
        correction = SE3.exp_se3(delta)
        return SE3.compose(frame['robot_pose'], correction)

    def get_corrected_robot_pose_with_stored_delta(self, frame_index):
        """Get corrected robot pose for a given frame using stored delta from last optimization."""
        frame_idx = self._get_frame_index(frame_index)
        if frame_idx < 0:
            raise ValueError(f"Frame {frame_index} not found")
        if not self._last_deltas or frame_idx >= len(self._last_deltas):
            # No stored delta, return original robot pose
            return self._frames[frame_idx]['robot_pose'].copy()
        delta = self._last_deltas[frame_idx]
        return self.get_corrected_robot_pose(frame_index, delta)

    def get_delta_for_frame(self, frame_index, params):
        """Extract delta for a specific frame from parameter vector."""
        frame_idx = self._get_frame_index(frame_index)
        if frame_idx < 0:
            raise ValueError(f"Frame {frame_index} not found")
        delta_start = 6 + frame_idx * 6
        return params[delta_start:delta_start + 6]

    def _get_frame_index(self, frame_index):
        """Get internal frame list index for a frame_index."""
        for i, f in enumerate(self._frames):
            if f['index'] == frame_index:
                return i
        return -1

    def compute_cMo(self, robot_pose):
        """Compute cMo from robot pose using current bMo (no correction)."""
        eMc_inv = SE3.inverse(self.eMc)
        bMe_inv = SE3.inverse(robot_pose)
        return eMc_inv @ bMe_inv @ self._bMo


class _SlidingWindowOptimizerFunctor:
    """Internal functor for scipy optimization."""

    def __init__(self, frames, bMo, K, dist, eMc,
                 reproj_std, robot_std_trans, robot_std_angle):
        self.frames = frames
        self.bMo = np.array(bMo, dtype=np.float64)
        self.eMc = np.array(eMc, dtype=np.float64)
        self.K = np.array(K, dtype=np.float64)
        self.dist = np.array(dist, dtype=np.float64)
        self.reproj_std = reproj_std
        self.robot_std_trans = robot_std_trans
        self.robot_std_angle = robot_std_angle
        self.eMc_inv = SE3.inverse(self.eMc)

    def __call__(self, params):
        """Compute residuals."""
        bMo = SE3.from_rvec_tvec(params[0:3], params[3:6])
        residuals = []

        for i, frame in enumerate(self.frames):
            delta = params[6 + i * 6: 6 + (i + 1) * 6]

            correction = SE3.exp_se3(delta)
            corrected_robot_pose = SE3.compose(frame['robot_pose'], correction)
            bMe_inv = SE3.inverse(corrected_robot_pose)
            cMo = self.eMc_inv @ bMe_inv @ bMo

            proj = self._project_pts(frame['pts3d'], cMo)

            reproj_residual = (proj - frame['pts2d']).flatten() / self.reproj_std

            delta_trans = delta[3:6]
            delta_rot_ang = np.linalg.norm(delta[0:3])
            prior_residual = np.zeros(6)
            prior_residual[0:3] = delta[0:3] / self.robot_std_angle
            prior_residual[3:6] = delta_trans / self.robot_std_trans

            residuals.extend(reproj_residual)
            residuals.extend(prior_residual)

        return np.array(residuals)

    def _project_pts(self, pts3d, cMo):
        """Project 3D points to 2D."""
        rvec, tvec = SE3.to_rvec_tvec(cMo)
        proj, _ = cv2.projectPoints(pts3d, rvec, tvec, self.K, self.dist)
        return proj.reshape(-1, 2)


if __name__ == "__main__":
    K = np.array([
        [1015.4, 0, 638.5],
        [0, 1015.4, 386.8],
        [0, 0, 1]
    ])
    dist = np.array([0.1, -0.2, 0, 0, 0.07])

    optimizer = SlidingWindowPoseOptimizerWithPrior(
        K, dist,
        reproj_std=1.0,
        robot_pose_std_translation=5.0,
        robot_pose_std_angle_deg=1.0
    )

    obj_pts = np.array([
        [-8.0, 11.2, 0.0], [8.0, 11.2, 0.0],
        [-16.0, 0.0, 0.0], [0.0, 0.0, 0.0], [16.0, 0.0, 0.0],
        [-8.0, -13.9, 0.0], [8.0, -13.9, 0.0]
    ], dtype=np.float64)
    optimizer.set_object_pts(obj_pts)
    optimizer.set_extrinsics(np.eye(4))

    bMo_init = np.eye(4)
    bMo_init[:3, 3] = [348, -1084, 498]

    np.random.seed(42)
    for i in range(5):
        robot_pose = np.eye(4)
        robot_pose[:3, 3] = [i * 10, i * 5, 100 + i * 50]

        noise_rotvec = np.random.randn(3) * 0.01
        noise_trans = np.random.randn(3) * 2
        noise_pose = SE3.from_rvec_tvec(noise_rotvec, noise_trans)
        robot_pose = robot_pose @ noise_pose

        pts2d = np.random.rand(7, 2) * 10 + 400 + np.random.randn(7, 2) * 2
        optimizer.add_frame(i, robot_pose, pts2d)

    optimizer.set_initial_pose(bMo_init)
    result = optimizer.optimize()

    print("Optimized pose:")
    print(optimizer.get_pose())
    print(f"Average error: {optimizer.get_average_error():.4f}")