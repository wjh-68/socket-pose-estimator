import cv2
import numpy as np
from scipy.spatial.transform import Rotation

# -----------------------------
# PnP/IPPE Wrapper
# -----------------------------
def solvePnP_IPPE(pts2d, pts3d, K, dist, min_depth=50, max_depth=3000):
    """Solve PnP using IPPE for planar targets, with basic physical checks.

    Args:
        pts2d (ndarray): 2D points, shape (N,2)
        pts3d (ndarray): Corresponding 3D points, shape (N,3)
        K (ndarray): Camera intrinsics, shape (3,3)
        dist (ndarray or None): Distortion coefficients
        min_depth (float): Minimum allowed distance from camera (mm)
        max_depth (float): Maximum allowed distance from camera (mm)

    Returns:
        rvec (ndarray): Rotation vector (3,)
        tvec (ndarray): Translation vector (3,)
        valid (bool): Whether the solution passed basic checks
        reason (str): If invalid, reason why
    """
    success, rvec, tvec = cv2.solvePnP(
        pts3d, pts2d, K, dist, flags=cv2.SOLVEPNP_IPPE
    )

    if not success:
        return None, None, False, "PnP solver failed"

    rvec = rvec.flatten()
    tvec = tvec.flatten()

    # 基本物理约束
    z = tvec[2]
    dist_norm = np.linalg.norm(tvec)
    if z <= 0:
        return None, None, False, f"Object behind camera (z={z:.1f}mm)"
    if dist_norm < min_depth or dist_norm > max_depth:
        return None, None, False, f"Unreasonable object distance={dist_norm:.1f}mm"

    return rvec, tvec, True, "ok"


# -----------------------------
# Reprojection Errors
# -----------------------------
def compute_per_point_reproj_errors(pts3d, rvec, tvec, pts2d, K, dist):
    """Compute per-point reprojection error in pixels"""
    proj, _ = cv2.projectPoints(pts3d, rvec, tvec, K, dist)
    return np.linalg.norm(proj.reshape(-1, 2) - pts2d, axis=1)


def compute_reproj_error(pts3d, rvec, tvec, pts2d, K, dist):
    """Compute mean reprojection error over all points"""
    return compute_per_point_reproj_errors(pts3d, rvec, tvec, pts2d, K, dist).mean()

from dataclasses import dataclass
from typing import Optional

@dataclass(slots=True)
class Pose:
    rvec: np.ndarray
    tvec: np.ndarray

@dataclass(slots=True)
class PnPDiagnostics:
    inlier_mask: np.ndarray
    per_point_errors: np.ndarray
    round1_error: float
    used_threshold: float

@dataclass(slots=True)
class PnPResult:
    valid: bool
    reason: str = ""
    pose: Optional[Pose] = None
    diagnostics: Optional[PnPDiagnostics] = None
    
# -----------------------------
# Two-round PnP for initial pose
# -----------------------------
def two_round_pnp(
    pts2d,
    pts3d,
    K,
    dist,
    fixed_threshold=0.4,
    use_adaptive_threshold=True,
    adaptive_multiplier=2.0,
    min_inliers=4
):
    """Two-round PnP for robust pose initialization.

    1. Solve PnP on all points
    2. Compute per-point reprojection errors
    3. Optionally adaptively threshold outliers
    4. Re-solve PnP on inliers if enough points remain

    Args:
        pts2d (ndarray): 2D points, shape (N,2)
        pts3d (ndarray): 3D points, shape (N,3)
        K (ndarray): Camera intrinsics
        dist (ndarray or None): Distortion coefficients
        fixed_threshold (float): Max reproj error in pixels for inliers
        use_adaptive_threshold (bool): If True, threshold = median * multiplier
        adaptive_multiplier (float): Multiplier for adaptive threshold
        min_inliers (int): Minimum number of inliers required to refine pose

    Returns:
        dict: Dictionary containing PnP results
        - valid (bool): True if pose is usable
        - reason (str): If invalid, reason why
        - rvec (ndarray): Rotation vector (3,)
        - tvec (ndarray): Translation vector (3,)
        - inlier_mask (ndarray): Boolean array indicating inlier points
        - per_point_errors (ndarray): Reprojection error per point (after round2 if refined)
        - round1_error (float): Mean reproj error of round1
        - used_threshold (float): Threshold used to classify inliers
    """
    rvec1, tvec1, valid1, reason = solvePnP_IPPE(pts2d, pts3d, K, dist)
    if not valid1:
        return PnPResult(valid=False, reason=reason, pose=None, diagnostics=None)

    # Round 1: compute reprojection errors
    per_point_errors = compute_per_point_reproj_errors(pts3d, rvec1, tvec1, pts2d, K, dist)
    round1_error = per_point_errors.mean()

    # Compute threshold for inlier selection
    if use_adaptive_threshold:
        median_error = np.median(per_point_errors)
        current_threshold = min(median_error * adaptive_multiplier, fixed_threshold)
    else:
        current_threshold = fixed_threshold

    inlier_mask = per_point_errors < current_threshold
    n_inliers = inlier_mask.sum()

    # Round 2: refine pose if enough inliers
    if n_inliers >= min_inliers:
        pts3d_inlier = pts3d[inlier_mask]
        pts2d_inlier = pts2d[inlier_mask]
        rvec2, tvec2, valid2, reason2 = solvePnP_IPPE(pts2d_inlier, pts3d_inlier, K, dist)
        if valid2:
            per_point_errors_round2 = compute_per_point_reproj_errors(pts3d, rvec2, tvec2, pts2d, K, dist)
            return PnPResult(
                valid=True, reason="", pose=Pose(rvec2, tvec2), \
                diagnostics=PnPDiagnostics(
                    inlier_mask, per_point_errors_round2, round1_error, current_threshold))
        else:
            return PnPResult(valid=False, reason=reason2, pose=None, diagnostics=None)
    # Fallback: return round1 results
    return PnPResult(
        valid=True, reason="", pose=Pose(rvec1, tvec1), \
        diagnostics=PnPDiagnostics(
            inlier_mask, per_point_errors, round1_error, current_threshold))

def rvec_tvec_to_transform(rvec, tvec):
    """Convert rotation vector and translation vector to 4x4 transformation matrix.

    Args:
        rvec: Rotation vector (3x1 or 1x3 array)
        tvec: Translation vector (3x1 or 1x3 array)

    Returns:
        transform: 4x4 homogeneous transformation matrix
    """
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_rotvec(rvec).as_matrix()
    transform[:3, 3] = tvec.flatten()
    return transform

# convert transform mat to rvec and tvec
def transform_to_rvec_tvec(transform):
    rvec = np.zero(3)
    tvec = np.zero(3)
    rvec = Rotation.from_matrix(transform[:3, :3]).as_rotvec()
    tvec = transform[:3, 3]
    return rvec, tvec