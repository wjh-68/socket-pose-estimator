#!/usr/bin/env python3
"""
Kalman Filter-based Camera Pose Estimator - Constant Velocity EKF.

State vector: [rotation_vector(3), translation_vector(3), velocity_rot(3), velocity_trans(3)]
- 6 DoF pose + 6 DoF velocity = 12 dimensional state

Key features:
1. SO(3) rotation prediction using exponential map
2. Constant velocity model with proper discretization
3. Anisotropic process noise for planar scenes
4. Velocity estimation from pose deltas
5. Mahalanobis gating for outlier rejection
6. Velocity damping for stability

For planar scenes (object on Z=0 plane, camera observing vertically):
- X, Y translation: well observable
- Z (depth): poorly observable
- rx, ry (tilt): poorly observable
- rz (yaw): well observable about optical axis
"""

import numpy as np
from scipy.spatial.transform import Rotation
import cv2


def lie_algebra_exp(v):
    """
    Exponential map for Lie algebra (rotation vector to rotation matrix).
    Equivalent to cv2.Rodrigues but using scipy for consistency.
    """
    return Rotation.from_rotvec(v)


def lie_algebra_log(R):
    """
    Logarithmic map for Lie algebra (rotation matrix to rotation vector).
    """
    return Rotation.from_matrix(R.as_matrix()).as_rotvec()


class KalmanFilterPoseEstimator:
    """
    Constant Velocity EKF for camera pose estimation.

    State: [rvec(3), tvec(3), vrvec(3), vtvec(3)] - 12 dimensions
    - rvec, tvec: camera pose (rotation vector + translation)
    - vrvec, vtvec: angular and translational velocities

    Process model (constant velocity):
        R_pred = R @ Exp(omega * dt)          # SO(3) rotation
        t_pred = t + v * dt                   # Euclidean translation
        v_pred = v                            # constant velocity
        omega_pred = omega                    # constant angular velocity

    Measurement model:
        innovation = observed_pixels - projected_pixels
        reprojection-based EKF update
    """

    def __init__(self, K, dist,
                 process_noise=(0.1, 0.1, 0.5, 0.01, 0.01, 0.05),  # [qx, qy, qz, qrx, qry, qrz]
                 velocity_noise=(0.05, 0.05, 0.2, 0.005, 0.005, 0.02),
                 measurement_noise=2.0,
                 velocity_damping=0.98,
                 velocity_alpha=0.9,
                 mahalanobis_threshold=9.21):  # chi2(0.01, 2 DOF)
        """
        Args:
            K: Camera intrinsic matrix (3x3)
            dist: Distortion coefficients (5,)
            process_noise: 6D position/rotation process noise std dev
            velocity_noise: 6D velocity random walk noise std dev
            measurement_noise: Base measurement noise (pixels)
            velocity_damping: Damping factor for velocity (0.95-0.99)
            velocity_alpha: Low-pass filter coefficient for velocity (0.8-0.95)
            mahalanobis_threshold: Chi-squared threshold for innovation gating
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

        # Covariance matrix (12x12)
        self.P = np.eye(self.state_dim, dtype=np.float64) * 1e-4

        # Process noise parameters (anisotropic)
        self.q_pos = np.array(process_noise, dtype=np.float64)
        self.q_vel = np.array(velocity_noise, dtype=np.float64)

        # Measurement noise
        self.r_measurement = measurement_noise

        # Velocity damping factor (prevents velocity drift)
        self.velocity_damping = velocity_damping

        # Velocity low-pass filter coefficient
        self.velocity_alpha = velocity_alpha

        # Mahalanobis gating threshold (chi-squared for 2 DOF, p=0.01)
        self.mahalanobis_threshold = mahalanobis_threshold

        # State initialized flag
        self.initialized = False
        self.last_update_time = None

        # History for diagnostics
        self.pose_history = []
        self.velocity_history = []
        self.error_history = []
        self.mahal_history = []
        self.cond_history = []

        # Track update status
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
        self.state[6:9] = np.zeros(3, dtype=np.float64)  # initial angular velocity
        self.state[9:12] = np.zeros(3, dtype=np.float64)  # initial translation velocity

        # Initial covariance
        self.P = np.eye(self.state_dim, dtype=np.float64) * 1e-4
        # Position uncertainty can be larger initially
        self.P[3:6, 3:6] = np.diag([0.01, 0.01, 0.1])  # translation uncertainty
        self.P[:3, :3] = np.diag([0.001, 0.001, 0.001])  # rotation uncertainty

        self.initialized = True
        self.last_update_time = timestamp
        self.last_update_success = True
        self.pose_history = []
        self.velocity_history = []
        self.error_history = []
        self.mahal_history = []
        self.cond_history = []

    def predict(self, dt):
        """
        Prediction step using constant velocity model.

        Args:
            dt: Time delta in seconds
        """
        if not self.initialized or dt <= 0:
            return

        # Current state
        rvec = self.state[:3]
        tvec = self.state[3:6]
        omega = self.state[6:9]   # angular velocity (rotation vector)
        v = self.state[9:12]      # translation velocity

        # ========== SO(3) rotation prediction ==========
        # R_pred = R @ Exp(omega * dt)
        R_current = Rotation.from_rotvec(rvec)
        R_delta = Rotation.from_rotvec(omega * dt)
        R_pred = R_current * R_delta
        rvec_pred = R_pred.as_rotvec()

        # ========== Translation prediction ==========
        # t_pred = t + v * dt
        tvec_pred = tvec + v * dt

        # ========== Velocity prediction (constant) ==========
        # Velocity stays constant (random walk handled by process noise)

        # Update state
        self.state[:3] = rvec_pred
        self.state[3:6] = tvec_pred
        # state[6:9] and state[9:12] unchanged

        # ========== Velocity damping ==========
        self.state[6:9] *= self.velocity_damping
        self.state[9:12] *= self.velocity_damping

        # ========== Compute state transition Jacobian (F) ==========
        # For SO(3), the Jacobian is more complex, but we approximate
        # F = [I  0  dt*I  0
        #      0  I   0   dt*I
        #      0  0   I   0
        #      0  0   0   I]
        F = np.eye(self.state_dim, dtype=np.float64)
        F[:3, 6:9] = np.eye(3, dtype=np.float64) * dt
        F[3:6, 9:12] = np.eye(3, dtype=np.float64) * dt

        # ========== Compute process noise covariance (Q) ==========
        # Proper discretization for constant velocity model:
        # Q = sigma_a^2 * [dt^4/4, dt^3/2; dt^3/2, dt^2] for each axis
        # For simplicity, we separate position and velocity noise

        Q = np.zeros((self.state_dim, self.state_dim), dtype=np.float64)

        # Rotation noise (for rotation vector)
        # Using simplified: Q_rot = (dt^2) * sigma_rot^2
        dt2 = dt * dt
        dt4 = dt2 * dt2

        # Position/rotation uncertainty from velocity
        Q[:3, :3] = np.diag(self.q_pos[3:] ** 2) * dt2      # rotation from omega noise
        Q[3:6, 3:6] = np.diag(self.q_pos[:3] ** 2) * dt2    # translation from v noise

        # Velocity random walk
        Q[6:9, 6:9] = np.diag((self.q_vel[3:] ** 2) * dt)   # angular velocity random walk
        Q[9:12, 9:12] = np.diag((self.q_vel[:3] ** 2) * dt)  # translation velocity random walk

        # Cross-correlation (position affected by velocity noise)
        # Simplified: small cross terms
        Q[:3, 6:9] = np.diag(self.q_pos[3:] * self.q_vel[3:] * dt2 * 0.5)
        Q[3:6, 9:12] = np.diag(self.q_pos[:3] * self.q_vel[:3] * dt2 * 0.5)

        # Covariance prediction
        self.P = F @ self.P @ F.T + Q

    def compute_reprojection_jacobian(self, pts3d, rvec, tvec):
        """
        Compute Jacobian of reprojection w.r.t. state (only pose params).

        Returns:
            J: Jacobian matrix (2N x 6) - only for pose params [rvec, tvec]
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
            mahal_distance: Mahalanobis distance
            is_valid: Whether update was applied
        """
        if not self.initialized:
            return None, None, False

        pts2d = np.array(pts2d, dtype=np.float64)
        pts3d = np.array(pts3d, dtype=np.float64)

        rvec = self.state[:3]
        tvec = self.state[3:6]

        # Predict reprojection
        proj_pred = self._project(pts3d, rvec, tvec)

        # Innovation (measurement residual)
        innovation = (pts2d - proj_pred).flatten()
        innovation_norm = np.linalg.norm(innovation.reshape(-1, 2), axis=1)
        mean_error = np.mean(innovation_norm)

        # Compute Jacobian
        J = self.compute_reprojection_jacobian(pts3d, rvec, tvec)

        # Measurement noise
        N = pts2d.shape[0]
        r = measurement_noise if measurement_noise is not None else self.r_measurement
        R_noise = np.eye(2 * N, dtype=np.float64) * (r ** 2)

        # ========== Mahalanobis gating ==========
        # S = H @ P @ H^T + R
        P_pose = self.P[:6, :6]
        S = J @ P_pose @ J.T + R_noise

        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            print("  [WARN] S matrix singular, using pseudo-inverse")
            S_inv = np.linalg.pinv(S)

        mahal_sq = innovation @ S_inv @ innovation
        mahal_distance = np.sqrt(mahal_sq)

        # Store for diagnostics
        self.mahal_history.append(mahal_distance)

        # Check condition number of JTJ for degeneracy detection
        try:
            JTJ = J.T @ np.linalg.inv(R_noise) @ J
            cond = np.linalg.cond(JTJ)
            self.cond_history.append(cond)
        except:
            cond = 1e10
            self.cond_history.append(cond)

        # Innovation gating
        if mahal_distance > self.mahalanobis_threshold:
            print(f"  [WARN] Mahalanobis distance {mahal_distance:.2f} exceeds threshold {self.mahalanobis_threshold:.2f}")
            # Inflate measurement noise instead of rejecting
            adapt_r = r * (mahal_distance / self.mahalanobis_threshold)
            R_noise = np.eye(2 * N, dtype=np.float64) * (adapt_r ** 2)

        # Kalman gain
        S = J @ P_pose @ J.T + R_noise
        K = P_pose @ J.T @ np.linalg.inv(S)

        # State update (only pose part, NOT velocity)
        delta = K @ innovation
        self.state[:3] += delta[:3]
        self.state[3:6] += delta[3:6]

        # Covariance update using Joseph form for numerical stability
        I_KH = np.eye(self.state_dim) - self._expand_K(K, J)
        self.P = I_KH @ self.P @ I_KH.T + self._expand_K(K, R_noise, K.T)

        self.last_update_success = True
        self.error_history.append(mean_error)

        return mean_error, mahal_distance, True

    def _expand_K(self, K, M, KT=None):
        """Expand Kalman gain for full state dimension."""
        # K is (12 x 2N), M is (2N x 6) or (2N x 2N)
        # Result should be (12 x 12)
        result = np.zeros((self.state_dim, self.state_dim), dtype=np.float64)
        result[:6, :6] = K @ M if KT is None else K @ M @ KT
        return result

    def update_velocity_from_deltas(self, prev_rvec, prev_tvec, curr_rvec, curr_tvec, dt):
        """
        Estimate velocity from pose deltas and apply low-pass filtering.

        Args:
            prev_rvec, prev_tvec: Previous pose
            curr_rvec, curr_tvec: Current pose
            dt: Time delta
        """
        if dt <= 0:
            return

        # Translation velocity
        v_measured = (curr_tvec - prev_tvec) / dt

        # Rotation velocity using Lie algebra
        R_prev = Rotation.from_rotvec(prev_rvec)
        R_curr = Rotation.from_rotvec(curr_rvec)
        R_delta = R_prev.inv() * R_curr
        omega_measured = R_delta.as_rotvec() / dt

        # Low-pass filtering
        alpha = self.velocity_alpha
        self.state[6:9] = alpha * self.state[6:9] + (1 - alpha) * omega_measured
        self.state[9:12] = alpha * self.state[9:12] + (1 - alpha) * v_measured

    def update_with_observation(self, pts2d, pts3d, rvec_pnp=None, tvec_pnp=None,
                                inlier_mask=None, timestamp=None):
        """
        Update with PnP observation.

        This method implements the full EKF cycle:
        1. Predict (if initialized)
        2. Solve PnP for observation
        3. Compute innovation
        4. Mahalanobis gating
        5. Update state and covariance
        6. Estimate velocity from pose delta

        Args:
            pts2d: Observed 2D points (Nx2)
            pts3d: Corresponding 3D points (Nx3)
            rvec_pnp: PnP rotation vector (optional, for diagnostics)
            tvec_pnp: PnP translation vector (optional, for diagnostics)
            inlier_mask: Boolean mask for inliers
            timestamp: Timestamp for dt calculation

        Returns:
            cMo, rvec_pnp, tvec_pnp, per_point_errors, is_updated
        """
        if pts2d.shape[0] < 4:
            return None, rvec_pnp, tvec_pnp, None, False

        # Store previous state for velocity estimation
        prev_rvec = self.state[:3].copy() if self.initialized else None
        prev_tvec = self.state[3:6].copy() if self.initialized else None

        # Compute dt
        if self.initialized and self.last_update_time is not None and timestamp is not None:
            dt = (timestamp - self.last_update_time) / 1e9
            dt = max(min(dt, 0.5), 0.001)
        elif self.initialized:
            dt = 0.1
        else:
            dt = 0.0

        # ========== Always predict first ==========
        if self.initialized:
            self.predict(dt)

        # Determine inliers if not provided
        inlier_mask_use = inlier_mask
        if inlier_mask is None:
            if self.initialized:
                per_point_errors = self._compute_per_point_errors(pts3d, self.state[:3], self.state[3:6], pts2d)
                median_error = np.median(per_point_errors)
                threshold = max(median_error * 2.0, 2.0)
                inlier_mask_use = per_point_errors < threshold
            else:
                inlier_mask_use = np.ones(pts2d.shape[0], dtype=bool)

        n_inliers = inlier_mask_use.sum() if inlier_mask_use is not None else pts2d.shape[0]
        if n_inliers < 4:
            inlier_mask_use = np.ones(pts2d.shape[0], dtype=bool)
            n_inliers = pts2d.shape[0]

        # ========== Solve PnP ==========
        use_pts2d = pts2d[inlier_mask_use] if inlier_mask_use is not None else pts2d
        use_pts3d = pts3d[inlier_mask_use] if inlier_mask_use is not None else pts3d

        success, rvec, tvec = cv2.solvePnP(
            use_pts3d, use_pts2d, self.K, self.dist, flags=cv2.SOLVEPNP_IPPE)

        if not success:
            if self.initialized:
                self._inflate_covariance(2.0)
                self.last_update_success = False
            return None, rvec_pnp, tvec_pnp, None, False

        rvec = rvec.flatten()
        tvec = tvec.flatten()

        # Compute per-point errors with PnP result
        per_point_errors = self._compute_per_point_errors(pts3d, rvec, tvec, pts2d)

        # ========== Initialize if needed ==========
        if not self.initialized:
            self.initialize(rvec, tvec, timestamp)
            cMo = self.get_cMo()
            self.pose_history.append(cMo.copy())
            return cMo, rvec, tvec, per_point_errors, True

        # ========== Update ==========
        mean_error, mahal_dist, updated = self.update(pts2d, pts3d)

        if not updated:
            self._inflate_covariance(1.5)
            self.last_update_success = False
            cMo = self.get_cMo()
            return cMo, rvec, tvec, per_point_errors, False

        # ========== Estimate velocity from pose delta ==========
        self.update_velocity_from_deltas(prev_rvec, prev_tvec, self.state[:3], self.state[3:6], dt)

        self.last_update_time = timestamp
        self.last_update_success = True

        # Store velocity for diagnostics
        self.velocity_history.append(self.state[6:12].copy())

        cMo = self.get_cMo()
        self.pose_history.append(cMo.copy())

        return cMo, rvec, tvec, per_point_errors, True

    def _inflate_covariance(self, factor):
        """Inflate covariance when update fails."""
        self.P *= factor ** 2

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

    def get_velocity(self):
        """Get current velocity state."""
        return self.state[6:12].copy()

    def get_pose_history(self):
        """Get history of poses."""
        return self.pose_history

    def get_velocity_history(self):
        """Get history of velocities."""
        return self.velocity_history

    def get_error_history(self):
        """Get history of reprojection errors."""
        return self.error_history

    def get_mahal_history(self):
        """Get history of Mahalanobis distances."""
        return self.mahal_history

    def get_cond_history(self):
        """Get history of condition numbers."""
        return self.cond_history

    def get_state(self):
        """Get current state vector."""
        return self.state.copy()

    def get_covariance(self):
        """Get current covariance matrix."""
        return self.P.copy()

    def get_covariance_diagonal(self):
        """Get diagonal elements of covariance for diagnostics."""
        return np.diag(self.P).copy()

    def is_initialized(self):
        """Check if filter is initialized."""
        return self.initialized

    def was_last_update_successful(self):
        """Check if last update was successful."""
        return self.last_update_success


class RobustKalmanFilterPoseEstimator(KalmanFilterPoseEstimator):
    """
    Robust Kalman filter with outlier rejection and adaptive noise.

    Optimized for planar scenes.
    """

    def __init__(self, K, dist,
                 process_noise=(0.1, 0.1, 0.5, 0.01, 0.01, 0.05),
                 velocity_noise=(0.05, 0.05, 0.2, 0.005, 0.005, 0.02),
                 measurement_noise=2.0,
                 velocity_damping=0.98,
                 velocity_alpha=0.9,
                 mahalanobis_threshold=9.21):
        """
        Args:
            process_noise: 6D [qx, qy, qz, qrx, qry, qrz]
            velocity_noise: 6D velocity random walk noise
            measurement_noise: Base measurement noise (pixels)
            velocity_damping: Damping factor (0.95-0.99)
            velocity_alpha: Low-pass filter coefficient (0.8-0.95)
            mahalanobis_threshold: Chi-squared threshold for gating
        """
        super().__init__(K, dist, process_noise, velocity_noise, measurement_noise,
                         velocity_damping, velocity_alpha, mahalanobis_threshold)


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