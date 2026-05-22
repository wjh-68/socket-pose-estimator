import cv2
from ultralytics import YOLO
import json
from scipy.spatial.transform import Rotation
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import threading
import time
from static_pose_optimizer_ba import StaticPoseOptimizer, pose_to_euler_tvec
from gemiEd import UltimateSocketMatcher, getInferResult
import pycylinderedsf as pyced

# ============ Config ============
# Data source mode: 'offline' for file-based, 'online' for real-time hardware
DATA_SOURCE_MODE = "offline"  # 'offline' or 'online'

# Offline mode config
DATA_DIR = "dataset/save_data3/20260511_120244"
# RESULT_DIR = "result/save_data3/20260511_120244/pose_estimation_ba"
RESULT_DIR = "result/0515"
SAVE_DIR = "dataset/save_data3/chb_20260511_120244"
# DATA_DIR = "dataset/save_data3/20260511_120538"
# RESULT_DIR = "result/save_data3/20260511_120538/pose_estimation_ba"
# SAVE_DIR = "dataset/save_data3/chb_20260511_120538"
SLIDING_WINDOW_SIZE = 8
MAX_FRAMES = -1  # Limit frames for quick test, -1 for all frames

# Online mode config
ROBOT_IP = "192.168.1.20"
ROBOT_PORT = 30004
CAMERA_ID = 0
CAMERA_WIDTH = 2560
CAMERA_HEIGHT = 1440
CAMERA_BRIGHTNESS = 128
FRAME_POSE_SYNC_TOLERANCE_NS = 20_000_000  # 20ms in nanoseconds

# =========== Read metadata config ===========
BEGIN_FRAME_ID = 1180  # Skip frames with frame_id < this value
PROCESSED_INTERVAL = 0.03
# =========== PnP threshold config ===========
USE_ADAPTIVE_THRESHOLD = True  # True = adaptive (but <= fixed), False = fixed
FIXED_ERROR_THRESHOLD = 1.0    
ADAPTIVE_MULTIPLIER = 2.0      # threshold = median_error * multiplier

# Frame rejection threshold - if pose diff (PnP - Optimized) exceed threshold, skip
MAX_TRANSLATION = 20    # mm
MAX_ROTATION_DEG = 20

# Camera on robot end-effector (eye-to-hand extrinsic)
eMc=np.array(
[[-7.2849429e-01,  6.8505180e-01, -3.0797155e-04, -6.3927837e+01],
 [-6.8492085e-01, -7.2834611e-01,  1.9882789e-02,  6.0054520e+01],
 [ 1.3396430e-02 , 1.4695434e-02,  9.9980229e-01, -1.7391582e+02],
 [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  1.0000000e+00]
 ],dtype=np.float64)

# camera intrinsics
K = np.array([
    [1359.1199645944478,0.,640.54132556823811],
    [0.,1359.1199645944478,362.00605351844041],
    [0.,0.,1.]
    ],dtype=np.float64)
dist = np.array([-0.11507587466685387,0.28954640142800997,0.002523795531233719,-0.0003497505689869382,-0.093769042874787809],dtype=np.float64)
# 3D object points in object frame (charge port keypoints)
obj_pts = np.array([
            [-8.0, 11.2, 0.0], [8.0, 11.2, 0.0],
            [-16.0, 0.0, 0.0], [0.0, 0.0, 0.0], [16.0, 0.0, 0.0],
            [-8.0, -13.9, 0.0], [8.0, -13.9, 0.0]
        ], dtype=np.float64)

# Optimization weight configuration
PRIOR_SIGMA = np.array([
    np.deg2rad(5.0),
    np.deg2rad(5.0),
    np.deg2rad(1.0),
    0.5,
    0.5,
    10.0
], dtype=np.float64)

POINT_SIGMAS = np.array([
    1.5,  # top-left small hole
    1.5,  # top-right small hole
    1.0,  # mid-left large hole
    1.0,  # center large hole
    1.0,  # mid-right large hole
    1.0,  # bottom-left large hole
    1.0   # bottom-right large hole
], dtype=np.float64)


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


def compute_reproj_error(pts3d, rvec, tvec, pts2d, K, dist):
    proj, _ = cv2.projectPoints(pts3d, rvec, tvec, K, dist)
    return np.linalg.norm(proj.reshape(-1,2) - pts2d, axis=1).mean()


def compute_per_point_reproj_errors(pts3d, rvec, tvec, pts2d, K, dist):
    """Compute reprojection error for each point (in pixels)"""
    proj, _ = cv2.projectPoints(pts3d, rvec, tvec, K, dist)
    return np.linalg.norm(proj.reshape(-1,2) - pts2d, axis=1)


def two_round_pnp(pts2d, pts3d, K, dist, error_threshold=0.4):
    """Two-round PnP with adaptive or fixed threshold.

    Returns:
        rvec, tvec, valid, inlier_mask, per_point_errors, round1_errors, used_threshold
    """
    rvec1, tvec1, valid1 = solvePnP_IPPE(pts2d, pts3d, K, dist)
    if not valid1:
        return None, None, False, None, None, None, None

    per_point_errors = compute_per_point_reproj_errors(pts3d, rvec1, tvec1, pts2d, K, dist)
    round1_error = per_point_errors.mean()

    if USE_ADAPTIVE_THRESHOLD:
        median_error = np.median(per_point_errors)
        current_threshold = median_error * ADAPTIVE_MULTIPLIER
        current_threshold = min(current_threshold, FIXED_ERROR_THRESHOLD)
    else:
        current_threshold = FIXED_ERROR_THRESHOLD

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


def draw_ellipse(img, ellipses):
    vis = img.copy()
    for ellipse in ellipses:
        center = (int(ellipse[0]), int(ellipse[1]))
        axes = (int(ellipse[2]), int(ellipse[3]))
        angle = ellipse[4]
        cv2.ellipse(vis, center, axes, angle, 0, 360, (0, 0, 255), 1, cv2.LINE_AA)
    return vis


def get_robot_pose_from_rpc(robot_rpc_client, robot_name):
    """Get robot TCP pose from rpc client, convert to 4x4 matrix."""
    tcp_pose = robot_rpc_client.getRobotInterface(robot_name).getRobotState().getTcpPose()
    r = Rotation.from_euler('xyz', tcp_pose[3:])
    t = np.array(tcp_pose[:3]).reshape((3, 1))
    robot_pose = np.eye(4)
    robot_pose[:3, :3] = r.as_matrix()
    robot_pose[:3, 3] = t.flatten() * 1000  # mm
    return robot_pose, tcp_pose


class SensorDataManager:
    """Unified interface for both offline (file-based) and online (hardware) data acquisition."""
    
    def __init__(self, mode='offline', **kwargs):
        """
        Args:
            mode: 'offline' for file-based, 'online' for real-time hardware
            **kwargs: mode-specific parameters
        """
        self.mode = mode
        self.is_running = False
        
        if mode == 'offline':
            self._init_offline(**kwargs)
        elif mode == 'online':
            self._init_online(**kwargs)
        else:
            raise ValueError(f"Unknown data source mode: {mode}")
    
    def _init_offline(self, data_dir=None, **kwargs):
        """Initialize offline mode (file-based)."""
        self.data_dir = data_dir
        self.current_frame_idx = 0
        self.img_files = []
        self.npy_files = {}
        
        if data_dir and os.path.exists(data_dir):
            self.img_files = sorted([f for f in os.listdir(data_dir) 
                                    if f.endswith('.jpg') and f != 'temp'])
            self.npy_files = {f.replace('.npy', ''): f for f in os.listdir(data_dir) 
                             if f.endswith('.npy')}
    
    def _init_online(self, robot_ip=None, robot_port=None, camera_id=0, 
                     camera_width=2560, camera_height=1440, brightness=128,
                     sync_tolerance_ns=20_000_000, **kwargs):
        """Initialize online mode (real-time hardware)."""
        self.robot_ip = robot_ip
        self.robot_port = robot_port
        self.camera_id = camera_id
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.brightness = brightness
        self.sync_tolerance_ns = sync_tolerance_ns
        
        # Thread-safe data storage
        self.latest_frame = None
        self.latest_frame_ts = 0
        self.latest_pose = None
        self.latest_pose_ts = 0
        self.frame_lock = threading.Lock()
        self.pose_lock = threading.Lock()
        
        # Robot connection
        self.robot_rpc_client = None
        self.robot_name = None
        self.cap = None
        
        self.camera_thread = None
        self.robot_thread = None
    
    def start_online(self):
        """Start online data acquisition (camera and robot threads)."""
        if self.mode != 'online':
            raise ValueError("Cannot start online mode when initialized in offline mode")
        
        try:
            import pyaubo_sdk
        except ImportError:
            raise ImportError("pyaubo_sdk is required for online mode. Install it or use offline mode.")
        
        # Initialize camera
        self.cap = cv2.VideoCapture(self.camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)
        self.cap.set(cv2.CAP_PROP_BRIGHTNESS, self.brightness)
        
        # Initialize robot connection
        self.robot_rpc_client = pyaubo_sdk.RpcClient()
        self.robot_rpc_client.connect(self.robot_ip, self.robot_port)
        
        if not self.robot_rpc_client.hasConnected():
            raise RuntimeError("Failed to connect to robot")
        
        self.robot_rpc_client.login("aubo", "123456")
        if not self.robot_rpc_client.hasLogined():
            raise RuntimeError("Failed to login to robot")
        
        self.robot_name = self.robot_rpc_client.getRobotNames()[0]
        print(f"Connected to robot: {self.robot_name}")
        
        # Start acquisition threads
        self.is_running = True
        self.camera_thread = threading.Thread(target=self._read_camera_loop, daemon=True)
        self.robot_thread = threading.Thread(target=self._read_robot_loop, daemon=True)
        self.camera_thread.start()
        self.robot_thread.start()
        
        print("Online data acquisition started")
    
    def _read_camera_loop(self):
        """Camera thread function."""
        while self.is_running:
            ret, frame = self.cap.read()
            if ret:
                with self.frame_lock:
                    self.latest_frame = frame.copy()
                    self.latest_frame_ts = time.perf_counter_ns()
            time.sleep(0.05)  # 20Hz sampling
    
    def _read_robot_loop(self):
        """Robot thread function."""
        while self.is_running:
            if self.robot_rpc_client is None or self.robot_name is None:
                time.sleep(0.5)
                continue
            
            try:
                robot_pose, _ = get_robot_pose_from_rpc(self.robot_rpc_client, self.robot_name)
                with self.pose_lock:
                    self.latest_pose = robot_pose
                    self.latest_pose_ts = time.perf_counter_ns()
            except Exception as e:
                print(f"Error reading robot pose: {e}")
                time.sleep(0.01)
                continue
            time.sleep(0.02)  # 50Hz sampling
    
    def get_next_frame_pose_online(self):
        """
        Get synchronized frame and pose in online mode.
        
        Returns:
            tuple: (frame, pose, time_diff_ns, success)
                - frame: numpy array or None
                - pose: 4x4 numpy array or None
                - time_diff_ns: time difference between frame and pose timestamps
                - success: whether data is valid and synchronized
        """
        if self.mode != 'online':
            raise ValueError("This method only works in online mode")
        
        with self.frame_lock:
            frame = self.latest_frame
            frame_ts = self.latest_frame_ts
        
        with self.pose_lock:
            pose = self.latest_pose
            pose_ts = self.latest_pose_ts
        
        if frame is None or pose is None:
            return None, None, -1, False
        
        diff_ns = abs(frame_ts - pose_ts)
        
        if diff_ns > self.sync_tolerance_ns:
            return frame, pose, diff_ns, False
        
        return frame, pose, diff_ns, True
    
    def get_next_frame_pose_offline(self):
        """
        Get next frame and pose in offline mode.
        
        Returns:
            tuple: (frame, pose, timestamp_ns, success)
        """
        if self.mode != 'offline':
            raise ValueError("This method only works in offline mode")
        
        if self.current_frame_idx >= len(self.img_files):
            return None, None, 0, False
        
        img_file = self.img_files[self.current_frame_idx]
        img_path = os.path.join(self.data_dir, img_file)
        
        # Load image
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"Failed to load image: {img_path}")
            return None, None, 0, False
        
        # Load pose (optional in offline mode)
        pose = None
        base_name = os.path.splitext(img_file)[0]
        if base_name in self.npy_files:
            npy_path = os.path.join(self.data_dir, self.npy_files[base_name])
            try:
                pose_data = np.load(npy_path)
                if pose_data.shape == (4, 4):
                    pose = pose_data
            except Exception as e:
                print(f"Failed to load pose: {e}")
        
        timestamp_ns = int(time.perf_counter_ns())
        self.current_frame_idx += 1
        
        return frame, pose, timestamp_ns, True
    
    def get_next_frame_pose(self):
        """
        Get next frame and pose (automatically handles mode).
        
        Returns:
            tuple: (frame, pose, timestamp_ns, success)
                For online: (frame, pose, time_diff_ns, synchronized)
                For offline: (frame, pose, timestamp_ns, loaded)
        """
        if self.mode == 'online':
            return self.get_next_frame_pose_online()
        else:
            return self.get_next_frame_pose_offline()
    
    def stop_online(self):
        """Stop online data acquisition."""
        if self.mode == 'online':
            self.is_running = False
            if self.cap:
                self.cap.release()
            if self.camera_thread:
                self.camera_thread.join(timeout=2.0)
            if self.robot_thread:
                self.robot_thread.join(timeout=2.0)
            print("Online data acquisition stopped")
    
    def reset_offline(self):
        """Reset offline frame counter."""
        if self.mode == 'offline':
            self.current_frame_idx = 0


class ChargeportPoseEstimator:
    def __init__(
        self,
        data_source_mode=DATA_SOURCE_MODE,
        data_dir=DATA_DIR,
        result_dir=RESULT_DIR,
        save_dir=SAVE_DIR,
        model_path='checkpoint/best.pt',
        sliding_window_size=SLIDING_WINDOW_SIZE,
        max_frames=MAX_FRAMES,
        begin_frame_id=BEGIN_FRAME_ID,
        processed_interval=PROCESSED_INTERVAL,
        use_adaptive_threshold=USE_ADAPTIVE_THRESHOLD,
        fixed_error_threshold=FIXED_ERROR_THRESHOLD,
        adaptive_multiplier=ADAPTIVE_MULTIPLIER,
        max_translation=MAX_TRANSLATION,
        max_rotation_deg=MAX_ROTATION_DEG,
        # Online mode parameters
        robot_ip=ROBOT_IP,
        robot_port=ROBOT_PORT,
        camera_id=CAMERA_ID,
        camera_width=CAMERA_WIDTH,
        camera_height=CAMERA_HEIGHT,
        camera_brightness=CAMERA_BRIGHTNESS,
        sync_tolerance_ns=FRAME_POSE_SYNC_TOLERANCE_NS,
    ):
        self.data_source_mode = data_source_mode
        self.data_dir = data_dir
        self.result_dir = result_dir
        self.save_dir = save_dir
        self.sliding_window_size = sliding_window_size
        self.max_frames = max_frames
        self.begin_frame_id = begin_frame_id
        self.processed_interval = processed_interval
        self.use_adaptive_threshold = use_adaptive_threshold
        self.fixed_error_threshold = fixed_error_threshold
        self.adaptive_multiplier = adaptive_multiplier
        self.max_translation = max_translation
        self.max_rotation_deg = max_rotation_deg

        self.model = YOLO(model_path)

        self.optimizer = StaticPoseOptimizer(K, dist, prior_sigma=PRIOR_SIGMA, point_sigmas=POINT_SIGMAS)
        self.optimizer.set_extrinsics(eMc)
        self.optimizer.set_object_pts(obj_pts)

        # Initialize data source manager
        if data_source_mode == 'offline':
            self.data_source = SensorDataManager(
                mode='offline',
                data_dir=data_dir
            )
        elif data_source_mode == 'online':
            self.data_source = SensorDataManager(
                mode='online',
                robot_ip=robot_ip,
                robot_port=robot_port,
                camera_id=camera_id,
                camera_width=camera_width,
                camera_height=camera_height,
                brightness=camera_brightness,
                sync_tolerance_ns=sync_tolerance_ns,
            )
        else:
            raise ValueError(f"Unknown data source mode: {data_source_mode}")

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

    def _load_metadata(self):
        metadata_path = os.path.join(self.data_dir, 'metadata.json')
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        return metadata['records']

    def _parse_robot_pose(self, record):
        return np.array(record['pose_matrix_4x4']).reshape(4, 4)

    def _is_record_eligible(self, record):
        if record['frame_id'] < self.begin_frame_id:
            return False
        if record['time_diff_ns'] / 1e9 > 0.05:
            return False
        return True

    def _should_skip_by_interval(self, current_timestamp_ns, last_timestamp_ns):
        if last_timestamp_ns is None:
            return False
        return (current_timestamp_ns - last_timestamp_ns) / 1e9 < self.processed_interval

    def _load_image(self, record):
        image_path = os.path.join(self.data_dir, record['image_path'])
        img = cv2.imread(image_path)
        if img is None:
            print(f"Failed to read image: {image_path}")
        return img

    def _detect_socket(self, img):
        img_bright = np.clip(img.astype(np.float32) - 50, 0, 255).astype(np.uint8)
        infer_result = getInferResult(self.model, img_bright)
        if not infer_result or len(infer_result) != 2:
            return None, None

        boxes, keypoints = infer_result
        if boxes is None or len(boxes) == 0:
            return None, None

        return boxes, keypoints

    def _extract_roi(self, img, box):
        x0, y0, x1, y1 = map(int, box[:4])
        x0 = max(0, x0)
        y0 = max(0, y0)
        x1 = min(img.shape[1], x1)
        y1 = min(img.shape[0], y1)
        roi = img[y0:y1, x0:x1]
        return roi, x0, y0, x1, y1

    def _normalize_keypoints(self, keypoints, top_left):
        if keypoints is None:
            return None
        keypoints = np.array(keypoints, dtype=np.float64)
        if keypoints.ndim == 1:
            keypoints = keypoints.reshape(-1, 2)
        return keypoints - np.array(top_left, dtype=np.float64)

    def _create_matcher(self):
        matcher = UltimateSocketMatcher(True)
        matcher.obj_pts = obj_pts
        matcher.K = K
        matcher.dist = dist
        matcher.eMc = eMc
        return matcher

    def _detect_ellipses(self, roi):
        detector = pyced.CED(np.ascontiguousarray(roi))
        detector.run_CED()
        rot_rects = detector.getEllipsesAfterCluster()
        return [(*e.center, e.size[0] / 2, e.size[1] / 2, e.angle) for e in rot_rects]

    def _match_socket(self, box, keypoints, roi_top_left, roi):
        normalized_keypoints = self._normalize_keypoints(keypoints, roi_top_left)
        ellipses = self._detect_ellipses(roi)
        if len(ellipses) == 0:
            print("  No ellipses detected in ROI, skipping")
            return None

        matcher = self._create_matcher()
        box_xywh = [*(box[:2]), *(box[2:] - box[:2])]
        final_pts, status, centers = matcher.solve(ellipses, box_xywh, keypoints=normalized_keypoints)
        if final_pts is None or len(final_pts) == 0:
            print("  Matcher failed to find valid correspondences, skipping")
            return None
        return final_pts, status, centers, matcher

    def _compute_pnp_results(self, pts2d, pts3d):
        rvec, tvec, valid, inlier_mask, per_point_errors, round1_error, used_threshold = two_round_pnp(
            pts2d, pts3d, K, dist, error_threshold=self.fixed_error_threshold)
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
        cv2.drawFrameAxes(img, K, dist, cMo[:3, :3], cMo[:3, 3:], 10, 3)
        proj, _ = cv2.projectPoints(pts3d, rvec_optimized, tvec_optimized, K, dist)

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

    def _process_one_frame(self, img, robot_pose, frame_id, timestamp_ns):
        if img is None:
            print("  Invalid image, skipping")
            return False

        boxes, keypoints = self._detect_socket(img)
        if boxes is None:
            self.cnt_no_detection += 1
            print("  No detection, skipping")
            return False

        roi, roi_x_min, roi_y_min, roi_x_max, roi_y_max = self._extract_roi(img, boxes[0])
        match_result = self._match_socket(boxes[0], keypoints, (roi_x_min, roi_y_min), roi)
        if match_result is None:
            print('find 0 points')
            return False

        final_pts, status, centers, matcher = match_result
        if centers.shape[0] < 7:
            self.cnt_less_7pts += 1
            print(f'find {len(final_pts)} points, less than 7')
            return False

        pts3d = matcher.obj_pts
        pts2d = centers

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

        bMo_init = robot_pose @ eMc @ cMo
        bMo_optimized, cMo_optimized = self._prepare_optimizer(
            frame_id,
            robot_pose,
            pts2d,
            pts3d,
            pnp_results['per_point_errors'],
            bMo_init,
        )

        rvec_optimized = Rotation.from_matrix(cMo_optimized[:3, :3]).as_rotvec()
        tvec_optimized = cMo_optimized[:3, 3]

        if pnp_results['inlier_mask'] is not None and pnp_results['inlier_mask'].sum() >= 4:
            pts3d_inlier = pts3d[pnp_results['inlier_mask']]
            pts2d_inlier = pts2d[pnp_results['inlier_mask']]
        else:
            pts3d_inlier = pts3d
            pts2d_inlier = pts2d

        success, rvec_ba, tvec_ba = cv2.solvePnP(pts3d_inlier, pts2d_inlier, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
        cMo_ba = np.eye(4)
        cMo_ba[:3, :3] = Rotation.from_rotvec(rvec_ba.flatten()).as_matrix()
        cMo_ba[:3, 3] = tvec_ba.flatten()
        bMo_ba = robot_pose @ eMc @ cMo_ba

        print(f"bMo_optimized: {pose_to_euler_tvec(bMo_optimized)}")
        print(f"bMo_pnp: {pose_to_euler_tvec(bMo_init)}")
        print(f"bMo_ba: {pose_to_euler_tvec(bMo_ba)}")
        print(f"cMo_optimized: {pose_to_euler_tvec(cMo_optimized)}")
        print(f"cMo_pnp: {pose_to_euler_tvec(cMo)}")
        print(f"cMo_ba: {pose_to_euler_tvec(cMo_ba)}")

        pnp_error_ba = compute_reproj_error(pts3d_inlier, rvec_ba, tvec_ba, pts2d_inlier, K, dist)
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
            timestamp_ns=timestamp_ns,
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

        self._render_frame(
            img=img,
            roi_bounds=(roi_x_min, roi_y_min, roi_x_max, roi_y_max),
            centers=centers,
            pts3d=pts3d,
            rvec_optimized=rvec_optimized,
            tvec_optimized=tvec_optimized,
            cMo=cMo_optimized,
            inlier_mask=pnp_results['inlier_mask'],
            frame_id_val=frame_id,
        )

        return True

    def _process_frame(self, record, frame_id):
        img = self._load_image(record)
        if img is None:
            return False

        boxes, keypoints = self._detect_socket(img)
        if boxes is None:
            self.cnt_no_detection += 1
            print("  No detection, skipping")
            return False

        roi, roi_x_min, roi_y_min, roi_x_max, roi_y_max = self._extract_roi(img, boxes[0])
        match_result = self._match_socket(boxes[0], keypoints, (roi_x_min, roi_y_min), roi)
        if match_result is None:
            print('find 0 points')
            return False

        final_pts, status, centers, matcher = match_result
        if centers.shape[0] < 7:
            self.cnt_less_7pts += 1
            print(f'find {len(final_pts)} points, less than 7')
            return False

        pts3d = matcher.obj_pts[matcher.r_idx]
        pts2d = centers

        pnp_results = self._compute_pnp_results(pts2d, pts3d)
        if not pnp_results['valid']:
            self.cnt_pnp_failed += 1
            print(f"Frame:{frame_id} frame_{record['frame_id']:06d}: PnP failed, skipping pose estimation")
            return False

        # if np.all(pnp_results['per_point_errors'] > self.fixed_error_threshold):
        #     self.cnt_rejected_frames += 1
        #     print(f"  [REJECT] All points exceed threshold {self.fixed_error_threshold} px")
        #     return False

        cMo = np.eye(4)
        cMo[:3, :3] = Rotation.from_rotvec(pnp_results['rvec']).as_matrix()
        cMo[:3, 3] = pnp_results['tvec']

        robot_pose = self._parse_robot_pose(record)
        if self._should_reject_pose_diff(cMo, robot_pose):
            self.cnt_rejected_frames += 1
            return False

        bMo_init = robot_pose @ eMc @ cMo
        bMo_optimized, cMo_optimized = self._prepare_optimizer(
            frame_id,
            robot_pose,
            pts2d,
            pts3d,
            pnp_results['per_point_errors'],
            bMo_init,
        )

        rvec_optimized = Rotation.from_matrix(cMo_optimized[:3, :3]).as_rotvec()
        tvec_optimized = cMo_optimized[:3, 3]

        if pnp_results['inlier_mask'] is not None and pnp_results['inlier_mask'].sum() >= 4:
            pts3d_inlier = pts3d[pnp_results['inlier_mask']]
            pts2d_inlier = pts2d[pnp_results['inlier_mask']]
        else:
            pts3d_inlier = pts3d
            pts2d_inlier = pts2d

        success, rvec_ba, tvec_ba = cv2.solvePnP(pts3d_inlier, pts2d_inlier, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
        cMo_ba = np.eye(4)
        cMo_ba[:3, :3] = Rotation.from_rotvec(rvec_ba.flatten()).as_matrix()
        cMo_ba[:3, 3] = tvec_ba.flatten()
        bMo_ba = robot_pose @ eMc @ cMo_ba

        print(f"bMo_optimized: {pose_to_euler_tvec(bMo_optimized)}")
        print(f"bMo_pnp: {pose_to_euler_tvec(bMo_init)}")
        print(f"bMo_ba: {pose_to_euler_tvec(bMo_ba)}")
        print(f"cMo_optimized: {pose_to_euler_tvec(cMo_optimized)}")
        print(f"cMo_pnp: {pose_to_euler_tvec(cMo)}")
        print(f"cMo_ba: {pose_to_euler_tvec(cMo_ba)}")

        pnp_error_ba = compute_reproj_error(pts3d_inlier, rvec_ba, tvec_ba, pts2d_inlier, K, dist)
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
            frame_id_val=record['frame_id'],
            timestamp_ns=record['camera_timestamp_ns'],
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

        self._render_frame(
            img=img,
            roi_bounds=(roi_x_min, roi_y_min, roi_x_max, roi_y_max),
            centers=centers,
            pts3d=pts3d,
            rvec_optimized=rvec_optimized,
            tvec_optimized=tvec_optimized,
            cMo=cMo_optimized,
            inlier_mask=pnp_results['inlier_mask'],
            frame_id_val=record['frame_id'],
        )

        return True

    def run(self):
        """Run pose estimation in online or offline mode."""
        os.makedirs(self.result_dir, exist_ok=True)
        
        if self.data_source_mode == 'online':
            self._run_online()
        else:
            self._run_offline()
        
        print(f"Completed. no_detection={self.cnt_no_detection}, less_7pts={self.cnt_less_7pts}, "
              f"pnp_failed={self.cnt_pnp_failed}, rejected={self.cnt_rejected_frames}")
        self._save_results()
    
    def _run_online(self):
        """Run in online mode (real-time hardware acquisition)."""
        print("Starting online pose estimation...")
        
        try:
            # Start data acquisition threads
            self.data_source.start_online()
            
            frame_id = 1
            frame_skip_count = 0
            
            while True:
                # Get synchronized frame and pose
                img, robot_pose, sync_info, success = self.data_source.get_next_frame_pose()
                
                if not success:
                    if img is None or robot_pose is None:
                        time.sleep(0.001)  # Wait for data
                        continue
                    else:
                        # Data not synchronized
                        frame_skip_count += 1
                        if frame_skip_count % 100 == 0:
                            print(f"Skipping unsynchronized frames (diff={sync_info/1e6:.2f}ms)")
                        continue
                
                # Reset skip counter on successful frame
                frame_skip_count = 0
                
                # Process frame
                timestamp_ns = int(time.perf_counter_ns())
                if self._process_one_frame(img, robot_pose, frame_id, timestamp_ns):
                    frame_id += 1
                
                # Check max frames limit
                if self.max_frames > 0 and frame_id > self.max_frames:
                    print(f"Reached max frames limit ({self.max_frames}), stopping...")
                    break
                
                # Display current frame
                cv2.imshow('pose_estimation', img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("User interrupted")
                    break
                    
        except KeyboardInterrupt:
            print("Interrupted by user")
        finally:
            self.data_source.stop_online()
    
    def _run_offline(self):
        """Run in offline mode (file-based)."""
        print("Starting offline pose estimation...")
        
        # Use hardcoded offline path for now (can be made configurable)
        data_dir = "dataset/0515"
        if not os.path.exists(data_dir):
            data_dir = self.data_dir
        
        img_files = sorted([f for f in os.listdir(data_dir) 
                           if f.endswith('.jpg') and f != 'temp'])
        npy_files = {f.replace('.npy', ''): f for f in os.listdir(data_dir) 
                    if f.endswith('.npy')}
        
        frame_id = 1
        processed_frames = 0
        
        for img_file in img_files:
            ts = img_file.replace('_720.jpg', '')
            
            img_path = os.path.join(data_dir, img_file)
            img = cv2.imread(img_path)
            if img is None:
                continue
            
            if f"{ts}" not in npy_files:
                continue
            
            robot_pose_path = os.path.join(data_dir, npy_files[ts])
            robot_pose = np.load(robot_pose_path)
            
            if self._process_one_frame(img, robot_pose, frame_id, timestamp_ns=0):
                frame_id += 1
                processed_frames += 1
            
            # Check max frames limit
            if self.max_frames > 0 and processed_frames >= self.max_frames:
                print(f"Reached max frames limit ({self.max_frames}), stopping...")
                break
            
            # Display current frame
            cv2.imshow('pose_estimation', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("User interrupted")
                break

    def _save_results(self):

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

    def _should_reject_pose_diff(self, cMo, robot_pose):
        if self.last_bMo is None or not self.optimizer.is_initialized():
            return False

        cMo_before = np.linalg.inv(eMc) @ np.linalg.inv(robot_pose) @ self.last_bMo
        tvec_before = cMo_before[:3, 3]
        pos_diff = float(np.linalg.norm(cMo[:3, 3] - tvec_before))

        rot_mat_diff = cMo[:3, :3] @ cMo_before[:3, :3].T
        rot_vec_diff = Rotation.from_matrix(rot_mat_diff).as_rotvec()
        rot_diff = float(np.degrees(np.linalg.norm(rot_vec_diff)))

        if pos_diff > self.max_translation or rot_diff > self.max_rotation_deg:
            print(f"  [REJECT] Large pose difference: pos_diff={pos_diff:.1f}mm, rot_diff={rot_diff:.1f}deg")
            return True
        return False


if __name__ == '__main__':
    estimator = ChargeportPoseEstimator()
    estimator.run()
