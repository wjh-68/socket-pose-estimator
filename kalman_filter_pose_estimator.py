#!/usr/bin/env python3
"""
Kalman Filter-based Camera Pose Estimator.

State vector: [rotation_vector(3), translation_vector(3), velocity_rot(3), velocity_trans(3)]
- 6 DoF pose + 6 DoF velocity = 12 dimensional state

Prediction: x_pred = F @ x + B @ u (velocity-based prediction)
Observation: z = H @ x (using 2D-3D reprojection error as innovation)

The Kalman filter estimates camera pose cMo, which is then combined with
robot_pose to compute bMo = robot_pose @ eMc @ cMo

For planar scenes (object on Z=0 plane, camera observing vertically):
- X, Y translation: well observable (direct mapping to image)
- Z (depth): poorly observable (small error -> large reprojection error)
- rx, ry (tilt): poorly observable (similar to depth)
- rz (yaw): well observable (rotation about optical axis)
"""

import numpy as np
from scipy.spatial.transform import Rotation
import cv2


class KalmanFilterPoseEstimator:
    """
    Kalman filter for camera pose estimation with anisotropic noise.

    State: [rvec(3), tvec(3), vrvec(3), vtvec(3)] - 12 dimensions
    - rvec, tvec: camera pose (rotation vector + translation)
    - vrvec, vtvec: angular and translational velocities

    Noise configuration for planar scenes:
        - process_noise: 6D vector [qx, qy, qz, qrx, qry, qrz]
        - measurement_noise: scalar or 2D per-point noise
    """

    def __init__(self, K, dist,
                 process_noise=(0.1, 0.1, 0.5, 0.01, 0.01, 0.05),
                 process_noise_vel=(0.05, 0.05, 0.2, 0.005, 0.005, 0.02),
                 measurement_noise=2.0):
        """
        Args:
            K: Camera intrinsic matrix (3x3)
            dist: Distortion coefficients (5,)
            process_noise: 6D process noise std dev [qx, qy, qz, qrx, qry, qrz]
                         Higher for poorly observable directions (Z, rx, ry)
            process_noise_vel: 6D velocity random walk noise std dev
            measurement_noise: Measurement noise for reprojection (pixels)
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

        # Covariance matrix
        self.P = np.eye(self.state_dim, dtype=np.float64) * 1e-3

        # Process noise parameters (6D, anisotropic)
        self.q_pos = np.array(process_noise, dtype=np.float64)  # [qx, qy, qz, qrx, qry, qrz]
        self.q_vel = np.array(process_noise_vel, dtype=np.float64)

        # Measurement noise
        self.r_measurement = measurement_noise

        # State initialized flag
        self.initialized = False
        self.last_update_time = None

        # History for visualization
        self.pose_history = []
        self.error_history = []

        # Track update status for diagnostics
        self.last_update_success = False

    def initialize(self, rvec, tvec, timestamp=None):
        """
        Initialize the filter with an initial pose estimate.

        Args:
            rvec: Initial rotation vector (3,)
            tvec: Initial translation vector (3,)
            timestamp: Optional timestamp for dt calculation
        """
        self.state[:3] = np.array(rvec, dtype=np.float64).flatten()
        self.state[3:6] = np.array(tvec, dtype=np.float64).flatten()
        self.state[6:9] = np.zeros(3, dtype=np.float64)
        self.state[9:12] = np.zeros(3, dtype=np.float64)

        self.P = np.eye(self.state_dim, dtype=np.float64) * 1e-3
        self.initialized = True
        self.last_update_time = timestamp
        self.last_update_success = True
        self.pose_history = []
        self.error_history = []

    def predict(self, dt):
        """
        Prediction step: propagate state forward using velocity.

        Args:
            dt: Time delta in seconds
        """
        if not self.initialized or dt <= 0:
            return

        # State transition: pose += velocity * dt
        rvec = self.state[:3]
        vrvec = self.state[6:9]
        delta_rvec = vrvec * dt
        rvec_pred = rvec + delta_rvec

        tvec = self.state[3:6]
        vtvec = self.state[9:12]
        tvec_pred = tvec + vtvec * dt

        self.state[:3] = rvec_pred
        self.state[3:6] = tvec_pred

        # State transition Jacobian (F)
        F = np.eye(self.state_dim, dtype=np.float64)
        F[:3, 6:9] = np.eye(3, dtype=np.float64) * dt
        F[3:6, 9:12] = np.eye(3, dtype=np.float64) * dt

        # Anisotropic process noise
        Q = np.zeros((self.state_dim, self.state_dim), dtype=np.float64)
        # Position/rotation noise (diagonal, anisotropic)
        Q[:3, :3] = np.diag(self.q_pos[3:] ** 2)      # rotation noise
        Q[3:6, 3:6] = np.diag(self.q_pos[:3] ** 2)    # translation noise
        # Velocity noise (random walk, diagonal)
        Q[6:9, 6:9] = np.diag((self.q_vel[3:] ** 2) * dt)
        Q[9:12, 9:12] = np.diag((self.q_vel[:3] ** 2) * dt)

        # Covariance prediction
        self.P = F @ self.P @ F.T + Q

    def compute_reprojection_jacobian(self, pts3d, rvec, tvec):
        """
        Compute Jacobian of reprojection w.r.t. state (only pose params).

        Returns:
            J: Jacobian matrix (2N x 6) - only for pose params
        """
        N = pts3d.shape[0]
        J = np.zeros((2 * N, 6), dtype=np.float64)

        eps = 1e-6
        for i in range(6):
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

    def update(self, pts2d, pts3d, measurement_noise=None):
        """
        Update step using 2D-3D correspondences.

        Args:
            pts2d: Observed 2D points (Nx2)
            pts3d: Corresponding 3D points (Nx3)
            measurement_noise: Override measurement noise (pixels)

        Returns:
            mean_error: Mean reprojection error
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

        # Innovation
        innovation = (pts2d - proj_pred).flatten()
        innovation_norm = np.linalg.norm(innovation.reshape(-1, 2), axis=1)
        mean_error = np.mean(innovation_norm)

        # Compute Jacobian
        J = self.compute_reprojection_jacobian(pts3d, rvec, tvec)

        # Measurement noise
        N = pts2d.shape[0]
        r = measurement_noise if measurement_noise is not None else self.r_measurement
        R_noise = np.eye(2 * N, dtype=np.float64) * (r ** 2)

        # Kalman gain
        S = J @ self.P[:6, :6] @ J.T + R_noise
        K = self.P[:6, :6] @ J.T @ np.linalg.inv(S)

        # State update (only pose part)
        delta = K @ innovation
        self.state[:3] += delta[:3]
        self.state[3:6] += delta[3:6]

        # Covariance update (Joseph form)
        I_KJ = np.eye(6) - K @ J
        self.P[:6, :6] = I_KJ @ self.P[:6, :6] @ I_KJ.T + K @ R_noise @ K.T

        # Reset velocity after update (constant velocity model)
        self.state[6:9] = np.zeros(3)
        self.state[9:12] = np.zeros(3)

        self.last_update_success = True
        self.error_history.append(mean_error)

        return mean_error, True

    def update_with_observation(self, pts2d, pts3d, rvec_pnp=None, tvec_pnp=None,
                                inlier_mask=None, timestamp=None):
        """
        Update with PnP observation.

        This method:
        1. Always predicts first (if initialized)
        2. Attempts update with all points (or inliers)
        3. Handles update failures gracefully

        Args:
            pts2d: Observed 2D points (Nx2)
            pts3d: Corresponding 3D points (Nx3)
            rvec_pnp: PnP rotation vector (optional, for display only)
            tvec_pnp: PnP translation vector (optional, for display only)
            inlier_mask: Boolean mask for inliers
            timestamp: Timestamp for dt calculation

        Returns:
            cMo, rvec_pnp, tvec_pnp, per_point_errors, is_updated
        """
        if pts2d.shape[0] < 4:
            return None, rvec_pnp, tvec_pnp, None, False

        # Compute per-point errors with current state
        if self.initialized:
            rvec = self.state[:3]
            tvec = self.state[3:6]
            per_point_errors = self._compute_per_point_errors(pts3d, rvec, tvec, pts2d)
        else:
            per_point_errors = None

        # Determine inliers if not provided
        if inlier_mask is None and per_point_errors is not None:
            median_error = np.median(per_point_errors)
            threshold = max(median_error * 2.0, 2.0)
            inlier_mask = per_point_errors < threshold

        # Always predict first (if initialized)
        if self.initialized and self.last_update_time is not None and timestamp is not None:
            dt = (timestamp - self.last_update_time) / 1e9
            dt = max(min(dt, 0.5), 0.001)  # clamp to reasonable range
        elif self.initialized:
            dt = 0.1  # default 100ms
        else:
            dt = 0.0

        if self.initialized:
            self.predict(dt)

        # Solve PnP for observation
        use_pts2d = pts2d
        use_pts3d = pts3d
        use_mask = inlier_mask

        if inlier_mask is not None:
            n_inliers = inlier_mask.sum()
            if n_inliers >= 4:
                use_pts2d = pts2d[inlier_mask]
                use_pts3d = pts3d[inlier_mask]
            else:
                use_mask = None

        success, rvec, tvec = cv2.solvePnP(
            use_pts3d, use_pts2d, self.K, self.dist, flags=cv2.SOLVEPNP_IPPE)

        if not success:
            # PnP failed - increase process noise to account for prediction uncertainty
            if self.initialized:
                self._increase_process_noise(2.0)
                self.last_update_success = False
            return None, rvec_pnp, tvec_pnp, per_point_errors, False

        rvec = rvec.flatten()
        tvec = tvec.flatten()

        # Compute per-point errors with PnP result
        per_point_errors = self._compute_per_point_errors(pts3d, rvec, tvec, pts2d)

        # Initialize if needed
        if not self.initialized:
            self.initialize(rvec, tvec, timestamp)
            cMo = self.get_cMo()
            self.pose_history.append(cMo.copy())
            return cMo, rvec, tvec, per_point_errors, True

        # Compute innovation based on current state prediction
        proj_pred = self._project(pts3d, self.state[:3], self.state[3:6])
        innovation = (pts2d - proj_pred).flatten()
        innovation_norm = np.linalg.norm(innovation.reshape(-1, 2), axis=1)
        max_innov = np.max(innovation_norm)

        # Adaptive measurement noise based on innovation
        if max_innov > self.r_measurement * 3:
            adapt_r = self.r_measurement * (max_innov / (self.r_measurement * 3)) ** 2
        else:
            adapt_r = self.r_measurement

        # Attempt update
        mean_error, updated = self.update(pts2d, pts3d, measurement_noise=adapt_r)

        if not updated:
            # Update failed - increase process noise
            self._increase_process_noise(1.5)
            self.last_update_success = False
            cMo = self.get_cMo()
            return cMo, rvec, tvec, per_point_errors, False

        # Check if innovation is too large (state diverged from observation)
        if max_innov > 20.0:
            print(f"  [WARN] Large innovation: {max_innov:.2f}px, resetting velocity")
            self.state[6:9] = np.zeros(3)
            self.state[9:12] = np.zeros(3)

        self.last_update_time = timestamp
        self.last_update_success = True

        cMo = self.get_cMo()
        self.pose_history.append(cMo.copy())

        return cMo, rvec, tvec, per_point_errors, True

    def _increase_process_noise(self, factor):
        """Increase process noise when update fails (covariance inflation)."""
        self.P[:6, :6] *= factor ** 2
        self.P[6:12, 6:12] *= (factor * 0.1) ** 2

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

    def is_initialized(self):
        """Check if filter is initialized."""
        return self.initialized

    def was_last_update_successful(self):
        """Check if last update was successful."""
        return self.last_update_success


class RobustKalmanFilterPoseEstimator(KalmanFilterPoseEstimator):
    """
    Robust Kalman filter with outlier rejection and adaptive noise.

    Optimized for planar scenes where:
    - Object is on Z=0 plane
    - Camera observes roughly vertically (along -Z)
    - Camera motion is roughly in XY plane
    """

    def __init__(self, K, dist,
                 process_noise=(0.1, 0.1, 0.5, 0.01, 0.01, 0.05),
                 process_noise_vel=(0.05, 0.05, 0.2, 0.005, 0.005, 0.02),
                 measurement_noise=2.0,
                 max_innovation=15.0):
        """
        Args:
            process_noise: 6D [qx, qy, qz, qrx, qry, qrz]
                          - qx, qy: small (well observable in planar scene)
                          - qz: larger (depth not well observable)
                          - qrx, qry: larger (tilt not well observable)
                          - qrz: small (yaw about optical axis well observable)
            measurement_noise: Base measurement noise in pixels
            max_innovation: Maximum allowed innovation before adaptation
        """
        super().__init__(K, dist, process_noise, process_noise_vel, measurement_noise)
        self.max_innovation = max_innovation


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
