import cv2
import os
import numpy as np
import threading
import time
from scipy.spatial.transform import Rotation


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

def get_robot_pose_from_rpc(robot_rpc_client, robot_name):
    """Get robot TCP pose from rpc client, convert to 4x4 matrix."""
    tcp_pose = robot_rpc_client.getRobotInterface(robot_name).getRobotState().getTcpPose()
    r = Rotation.from_euler('xyz', tcp_pose[3:])
    t = np.array(tcp_pose[:3]).reshape((3, 1))
    robot_pose = np.eye(4)
    robot_pose[:3, :3] = r.as_matrix()
    robot_pose[:3, 3] = t.flatten() * 1000  # mm
    return robot_pose, tcp_pose
