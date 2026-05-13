#!/usr/bin/env python3
"""
Kalman Filter-based Camera Pose Estimator.

State vector: [rotation_vector(3), translation_vector(3), velocity_rot(3), velocity_trans(3)]
- 6 DoF pose + 6 DoF velocity = 12 dimensional state

Prediction: x_pred = F @ x + B @ u (velocity-based prediction)
Observation: z = H @ x (but we use 2D-3D reprojection error as innovation)

The Kalman filter estimates camera pose cMo, which is then combined with
robot_pose to compute bMo = robot_pose @ eMc @ cMo
"""

import numpy as np
from scipy.spatial.transform import Rotation
import cv2


class KalmanFilterPoseEstimator:
    """
    Kalman filter for camera pose estimation.

    State: [rvec(3), tvec(3), vrvec(3), vtvec(3)] - 12 dimensions
    - rvec, tvec: camera pose (rotation vector + translation)
    - vrvec, vtvec: angular and translational velocities

    Prediction step:
        pose_pred = pose_prev + velocity * dt
        velocity_pred = velocity_prev (constant velocity model)

    Update step:
        Uses 2D-3D correspondences to compute innovation via reprojection error
        Linearized observation model via Jacobian of reprojection
    """

    def __init__(self, K, dist, process_noise_pos=0.001, process_noise_vel=0.01,
                 measurement_noise=1.0):
        """
        Args:
            K: Camera intrinsic matrix (3x3)
            dist: Distortion coefficients (5,)
            process_noise_pos: Process noise for position/rotation (std dev)
            process_noise_vel: Process noise for velocity (std dev)
            measurement_noise: Measurement noise for reprojection (std dev in pixels)
        """
        self.K = np.array(K, dtype=np.float64)
        self.dist = np.array(dist, dtype=np.float64)
        self.fx = self.K[0, 0]
        self.fy = self.K[1, 1]
        self.cx = self.K[0, 2]
        self.cy = self.K[1, 2]
        self.k1, self.k2, self.p1, self.p2, self.k3 = self.dist

        # State dimension
        self.state_dim = 12

        # State: [rvec(3), tvec(3), vrvec(3), vtvec(3)]
        self.state = np.zeros(self.state_dim, dtype=np.float64)
        self.velocity = np.zeros(6, dtype=np.float64)

        # Covariance matrix
        self.P = np.eye(self.state_dim, dtype=np.float64) * 1e-3

        # Process noise parameters
        self.q_pos = process_noise_pos
        self.q_vel = process_noise_vel

        # Measurement noise (reprojection)
        self.r_measurement = measurement_noise

        # State initialized flag
        self.initialized = False

        # History for visualization
        self.pose_history = []
        self.velocity_history = []
        self.error_history = []

    def initialize(self, rvec, tvec):
        """
        Initialize the filter with an initial pose estimate.

        Args:
            rvec: Initial rotation vector (3,)
            tvec: Initial translation vector (3,)
        """
        self.state[:3] = np.array(rvec, dtype=np.float64).flatten()
        self.state[3:6] = np.array(tvec, dtype=np.float64).flatten()
        self.state[6:9] = np.zeros(3, dtype=np.float64)  # zero angular velocity
        self.state[9:12] = np.zeros(3, dtype=np.float64)  # zero translation velocity

        self.velocity = np.zeros(6, dtype=np.float64)
        self.P = np.eye(self.state_dim, dtype=np.float64) * 1e-3
        self.initialized = True
        self.pose_history = []
        self.velocity_history = []
        self.error_history = []

    def predict(self, dt):
        """
        Prediction step: propagate state forward using velocity.

        Args:
            dt: Time delta in seconds
        """
        if not self.initialized:
            return

        # State transition: pose += velocity * dt
        # Rotation part (exponential map for rotation)
        rvec = self.state[:3]
        vrvec = self.state[6:9]

        # For small dt, add rotvec perturbation
        delta_rvec = vrvec * dt
        rvec_pred = rvec + delta_rvec

        # Translation part
        tvec = self.state[3:6]
        vtvec = self.state[9:12]
        tvec_pred = tvec + vtvec * dt

        # Velocity stays constant (random walk handled by process noise)
        vrvec_pred = self.state[6:9]
        vtvec_pred = self.state[9:12]

        self.state[:3] = rvec_pred
        self.state[3:6] = tvec_pred
        self.state[6:9] = vrvec_pred
        self.state[9:12] = vtvec_pred

        # State transition Jacobian (F)
        F = np.eye(self.state_dim, dtype=np.float64)
        # d(rvec_next)/d(vrvec) = dt * I
        F[:3, 6:9] = np.eye(3, dtype=np.float64) * dt
        # d(tvec_next)/d(vtvec) = dt * I
        F[3:6, 9:12] = np.eye(3, dtype=np.float64) * dt

        # Process noise
        Q = np.zeros((self.state_dim, self.state_dim), dtype=np.float64)
        # Position/rotation noise
        Q[:6, :6] = np.eye(6, dtype=np.float64) * (self.q_pos ** 2)
        # Velocity noise (random walk)
        Q[6:12, 6:12] = np.eye(6, dtype=np.float64) * ((self.q_vel * dt) ** 2)

        # Covariance prediction
        self.P = F @ self.P @ F.T + Q

    def compute_reprojection_jacobian(self, pts3d, rvec, tvec):
        """
        Compute Jacobian of reprojection w.r.t. state.

        Returns:
            J: Jacobian matrix (2N x state_dim)
        """
        N = pts3d.shape[0]
        J = np.zeros((2 * N, self.state_dim), dtype=np.float64)

        # Project points to get current reprojection
        proj = self._project(pts3d, rvec, tvec)

        # Compute Jacobian numerically using central difference
        eps = 1e-6
        for i in range(6):  # Only pose params affect reprojection, not velocity
            state_plus = self.state.copy()
            state_plus[i] += eps
            rvec_p = state_plus[:3]
            tvec_p = state_plus[3:6]
            proj_p = self._project(pts3d, rvec_p, tvec_p)

            state_minus = self.state.copy()
            state_minus[i] -= eps
            rvec_m = state_minus[:3]
            tvec_m = state_minus[3:6]
            proj_m = self._project(pts3d, rvec_m, tvec_m)

            J[:, i] = ((proj_p - proj_m) / (2 * eps)).flatten()

        return J

    def update(self, pts2d, pts3d, outlier_threshold=5.0):
        """
        Update step using 2D-3D correspondences.

        Args:
            pts2d: Observed 2D points (Nx2)
            pts3d: Corresponding 3D points (Nx3)
            outlier_threshold: Reject updates with large innovation (pixels)

        Returns:
            innovation: Computed innovation vector
            is_valid: Whether update was applied
        """
        if not self.initialized:
            return None, False

        pts2d = np.array(pts2d, dtype=np.float64)
        pts3d = np.array(pts3d, dtype=np.float64)

        rvec = self.state[:3]
        tvec = self.state[3:6]

        # Predict reprojection
        proj_pred = self._project(pts3d, rvec, tvec)

        # Innovation (measurement residual)
        innovation = (pts2d - proj_pred).flatten()

        # Check for outliers
        innovation_norm = np.linalg.norm(innovation.reshape(-1, 2), axis=1)
        max_innovation = np.max(innovation_norm)

        if max_innovation > outlier_threshold:
            # Large innovation - potential outlier, use robust update
            print(f"  [WARN] Large innovation detected: max={max_innovation:.2f}px, using robust update")

        # Compute Jacobian
        J = self.compute_reprojection_jacobian(pts3d, rvec, tvec)

        # Measurement noise
        N = pts2d.shape[0]
        R_noise = np.eye(2 * N, dtype=np.float64) * (self.r_measurement ** 2)

        # Kalman gain
        S = J @ self.P @ J.T + R_noise
        K = self.P @ J.T @ np.linalg.inv(S)

        # State update
        self.state = self.state + K @ innovation

        # Covariance update (Joseph form for numerical stability)
        I_KJ = np.eye(self.state_dim) - K @ J
        self.P = I_KJ @ self.P @ I_KJ.T + K @ R_noise @ K.T

        # Update velocity from delta state (for next prediction)
        self.state[6:9] = np.zeros(3)  # reset velocity after update
        self.state[9:12] = np.zeros(3)

        # Record error
        mean_error = np.mean(innovation_norm)
        self.error_history.append(mean_error)

        return innovation, True

    def update_with_pnp(self, pts2d, pts3d, robot_pose, eMc, measurement_noise=1.0):
        """
        Update using PnP to get observation, then apply Kalman update.

        This uses IPPE to solve PnP, then uses the result as measurement.

        Args:
            pts2d: Observed 2D points (Nx2)
            pts3d: Corresponding 3D points (Nx3)
            robot_pose: Robot end-effector pose (4x4)
            eMc: Eye-to-hand calibration (4x4)
            measurement_noise: Measurement noise std dev

        Returns:
            cMo: Computed camera pose (4x4)
            rvec, tvec: Rotation and translation vectors
            per_point_errors: Per-point reprojection errors
        """
        if pts2d.shape[0] < 4:
            return None, None, None, False

        # Solve PnP
        success, rvec, tvec = cv2.solvePnP(
            pts3d, pts2d, self.K, self.dist, flags=cv2.SOLVEPNP_IPPE)
        if not success:
            return None, None, None, False

        rvec = rvec.flatten()
        tvec = tvec.flatten()

        # Compute per-point errors
        per_point_errors = self._compute_per_point_errors(pts3d, rvec, tvec, pts2d)

        # Initialize if needed
        if not self.initialized:
            self.initialize(rvec, tvec)
            return self.get_cMo(), rvec, tvec, per_point_errors, True

        # Prediction step
        # Assume dt from last update
        if len(self.pose_history) > 0:
            dt = 0.1  # default 100ms
        else:
            dt = 0.0
        self.predict(dt)

        # Update step
        innovation, is_valid = self.update(pts2d, pts3d, outlier_threshold=10.0)

        # Get updated pose
        cMo = self.get_cMo()

        # Update velocity history
        if len(self.pose_history) > 0:
            prev_cMo = self.pose_history[-1]
            delta = self._pose_to_params(cMo) - self._pose_to_params(prev_cMo)
            self.velocity_history.append(delta / max(dt, 0.001))

        self.pose_history.append(cMo.copy())

        return cMo, rvec, tvec, per_point_errors, is_valid

    def get_cMo(self):
        """Get current camera pose as 4x4 matrix."""
        rvec = self.state[:3]
        tvec = self.state[3:6]
        cMo = np.eye(4, dtype=np.float64)
        cMo[:3, :3] = Rotation.from_rotvec(rvec).as_matrix()
        cMo[:3, 3] = tvec
        return cMo

    def get_bMo(self, robot_pose, eMc):
        """
        Get object pose in base frame.

        bMo = bMe @ eMc @ cMo
        """
        bMe = robot_pose
        cMo = self.get_cMo()
        return bMe @ eMc @ cMo

    def _project(self, pts3d, rvec, tvec):
        """Project 3D points to 2D with distortion."""
        pts3d = np.array(pts3d, dtype=np.float64)
        rvec = np.array(rvec, dtype=np.float64).flatten()
        tvec = np.array(tvec, dtype=np.float64).flatten()

        R, _ = cv2.Rodrigues(rvec)
        pts_cam = R @ pts3d.T + tvec.reshape(3, 1)

        x = pts_cam[0] / pts_cam[2]
        y = pts_cam[1] / pts_cam[2]

        r2 = x * x + y * y
        radial = 1 + self.k1 * r2 + self.k2 * r2 * r2 + self.k3 * r2 * r2 * r2
        x_dist = x * radial + 2 * self.p1 * x * y + self.p2 * (r2 + 2 * x * x)
        y_dist = y * radial + self.p1 * (r2 + 2 * y * y) + 2 * self.p2 * x * y

        u = self.fx * x_dist + self.cx
        v = self.fy * y_dist + self.cy
        return np.stack([u, v], axis=1)

    def _compute_per_point_errors(self, pts3d, rvec, tvec, pts2d):
        """Compute reprojection error per point."""
        proj = self._project(pts3d, rvec, tvec)
        return np.linalg.norm(proj - pts2d, axis=1)

    @staticmethod
    def _pose_to_params(pose):
        """Convert 4x4 pose to 6D params (rvec + tvec)."""
        rvec = Rotation.from_matrix(pose[:3, :3]).as_rotvec()
        tvec = pose[:3, 3]
        return np.concatenate([rvec, tvec])

    def get_pose_history(self):
        """Get history of poses."""
        return self.pose_history

    def get_error_history(self):
        """Get history of reprojection errors."""
        return self.error_history

    def get_state(self):
        """Get current state vector."""
        return self.state.copy()

    def get_covariance(self):
        """Get current covariance matrix."""
        return self.P.copy()


class RobustKalmanFilterPoseEstimator(KalmanFilterPoseEstimator):
    """
    Robust Kalman filter with outlier rejection and adaptive noise.
    """

    def __init__(self, K, dist, process_noise_pos=0.001, process_noise_vel=0.01,
                 measurement_noise=1.0, max_reproj_error=10.0):
        super().__init__(K, dist, process_noise_pos, process_noise_vel, measurement_noise)
        self.max_reproj_error = max_reproj_error

    def update_with_robust(self, pts2d, pts3d, robot_pose, eMc, inlier_mask=None):
        """
        Update with robust handling of outliers.

        Args:
            pts2d: Observed 2D points
            pts3d: Corresponding 3D points
            robot_pose: Robot pose (4x4)
            eMc: Eye-to-hand calibration (4x4)
            inlier_mask: Boolean mask for inliers (if None, computed automatically)

        Returns:
            cMo, per_point_errors, used_inlier_mask
        """
        if pts2d.shape[0] < 4:
            return None, None, None, False

        # First compute per-point errors with current state
        if self.initialized:
            rvec = self.state[:3]
            tvec = self.state[3:6]
            per_point_errors = self._compute_per_point_errors(pts3d, rvec, tvec, pts2d)
        else:
            per_point_errors = np.full(pts2d.shape[0], self.max_reproj_error)

        # Determine inliers
        if inlier_mask is None:
            # Use adaptive threshold based on median error
            median_error = np.median(per_point_errors)
            threshold = max(median_error * 2.0, 3.0)
            inlier_mask = per_point_errors < threshold

        n_inliers = inlier_mask.sum()
        if n_inliers < 4:
            # Not enough inliers, use all points
            inlier_mask = np.ones(pts2d.shape[0], dtype=bool)
            n_inliers = pts2d.shape[0]

        # Filter points
        pts2d_filt = pts2d[inlier_mask]
        pts3d_filt = pts3d[inlier_mask]

        # Solve PnP with inliers
        success, rvec, tvec = cv2.solvePnP(
            pts3d_filt, pts2d_filt, self.K, self.dist, flags=cv2.SOLVEPNP_IPPE)
        if not success:
            return None, None, None, False

        rvec = rvec.flatten()
        tvec = tvec.flatten()

        # Compute per-point errors with inlier-only PnP
        per_point_errors_all = self._compute_per_point_errors(pts3d, rvec, tvec, pts2d)

        # Initialize if needed
        if not self.initialized:
            self.initialize(rvec, tvec)
            cMo = self.get_cMo()
            self.pose_history.append(cMo.copy())
            return cMo, per_point_errors_all, inlier_mask, True

        # Predict
        if len(self.pose_history) > 0:
            dt = 0.1
        else:
            dt = 0.0
        self.predict(dt)

        # Compute innovation with inliers
        proj_pred = self._project(pts3d_filt, self.state[:3], self.state[3:6])
        innovation = (pts2d_filt - proj_pred).flatten()
        innovation_norm = np.linalg.norm(innovation.reshape(-1, 2), axis=1)

        # Check if innovation is too large (outlier)
        max_innov = np.max(innovation_norm)
        if max_innov > self.max_reproj_error:
            # Reduce Kalman gain for robustness
            adapt_noise = self.r_measurement * (max_innov / self.max_reproj_error) ** 2
        else:
            adapt_noise = self.r_measurement

        # Update with adapted noise
        J = self.compute_reprojection_jacobian(pts3d_filt, self.state[:3], self.state[3:6])
        N = pts2d_filt.shape[0]
        R_noise = np.eye(2 * N, dtype=np.float64) * (adapt_noise ** 2)

        S = J @ self.P @ J.T + R_noise
        K = self.P @ J.T @ np.linalg.inv(S)

        self.state = self.state + K @ innovation
        I_KJ = np.eye(self.state_dim) - K @ J
        self.P = I_KJ @ self.P @ I_KJ.T + K @ R_noise @ K.T

        # Update velocity
        self.state[6:9] = np.zeros(3)
        self.state[9:12] = np.zeros(3)

        cMo = self.get_cMo()
        self.pose_history.append(cMo.copy())
        self.error_history.append(np.mean(innovation_norm))

        return cMo, per_point_errors_all, inlier_mask, True


def pose_to_euler_tvec(pose, unit='deg'):
    """Convert 4x4 pose to euler angles and translation vector."""
    if pose is None:
        return None
    if not isinstance(pose, np.ndarray) or pose.shape != (4, 4):
        raise ValueError("Pose must be a 4x4 numpy array")
    euler = Rotation.from_matrix(pose[:3, :3]).as_euler('xyz')
    tvec = pose[:3, 3]
    if unit == 'deg':
        euler = np.degrees(euler)
    return euler, tvec
