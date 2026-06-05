import threading
import cv2
import time
import numpy as np
from collections import deque
from scipy.spatial.transform import Rotation
from core.packet import FramePacket
from .base_data_source import BaseDataSource
from core.logger import get_logger
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional
from config.data_source_config import OnlineDataSourceConfig

class DataSourceState(Enum):

    CREATED = auto()

    INITIALIZED = auto()

    RUNNING = auto()

    STOPPED = auto()

    DESTROYED = auto()


@dataclass
class TimestampedData:
    timestamp_ns: int = 0
    data: object = None


class OnlineDataSource(BaseDataSource):
    def __init__(
            self,
            cfg: OnlineDataSourceConfig,
            stop_event,
    ):
        super().__init__(cfg)
        self.stop_event = stop_event
        self.logger = get_logger(__name__)
        self.state = DataSourceState.CREATED

    def initialize(self):

        if self.state != DataSourceState.CREATED:
            raise RuntimeError(
                f"initialize() invalid state {self.state}"
            )
        if self.stop_event.is_set():
            return False
        
        try:
            import pyaubo_sdk
        except ImportError:
            raise ImportError("Failed to import pyaubo_sdk")
        
        # Initialize camera
        self.cap = cv2.VideoCapture(self.cfg.camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,
                      self.cfg.camera_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT,
                      self.cfg.camera_height)
        self.cap.set(cv2.CAP_PROP_BRIGHTNESS,
                      self.cfg.camera_brightness)
        if not self.cap.isOpened():
            raise RuntimeError("Failed to open camera")
        
        # Initialize robot connection
        self.robot_rpc_client = pyaubo_sdk.RpcClient()
        self.robot_rpc_client.connect(self.cfg.robot_ip,
                                      self.cfg.robot_port)
        self.robot_name = self.robot_rpc_client.getRobotName()[0]
        if not self.robot_rpc_client.hasConnected():
            raise RuntimeError("Failed to connect to robot")
        self.robot_rpc_client.login(self.cfg.robot_login_name,
                                    self.cfg.robot_password)            
        if not self.robot_rpc_client.hasConnected():
            raise RuntimeError("Failed to login to robot")
        if not self.robot_name:
            raise RuntimeError("Failed to get robot name")
        self.frame_id = 0
        self.state = DataSourceState.INITIALIZED
        if self.stop_event.is_set():
            raise InterruptedError()
        return True
    
    def start(self):

        if self.state != DataSourceState.INITIALIZED:
            raise RuntimeError(
                f"start() invalid state {self.state}"
            )

        self.camera_data = TimestampedData()
        self.robot_buffer = deque(maxlen=100)
        self.camera_lock = threading.Lock()
        self.robot_lock = threading.Lock()
        
        # Start threads
        self.camera_thread = threading.Thread(
            target=self._read_camera_loop, daemon=True)
        self.robot_thread = threading.Thread(
            target=self._read_robot_loop, daemon=True)
        self.camera_thread.start()
        self.robot_thread.start()
        self.logger.info("OnlineDataSource started")

        self.state = DataSourceState.RUNNING

    def stop(self):
        if self.state != DataSourceState.RUNNING:
            return
        self.camera_thread.join()
        self.robot_thread.join()
        self.state = DataSourceState.STOPPED

    def cleanup(self):
        if self.state not in(
            DataSourceState.STOPPED,
            DataSourceState.INITIALIZED,
        ):
            return
        if self.cap is not None:
            self.cap.release()
        
        # Close robot client if api provided

        self.state = DataSourceState.DESTROYED

    def get_packet(self)->Optional[FramePacket]:
        with self.camera_lock:
            camera_data = self.camera_data
        if camera_data.data is None:
            return None
        robot_data, time_diff_ns = \
            self._find_closest_robot_data(
                camera_data.timestamp_ns)
        
        if robot_data is None or robot_data.data is None:
            return None
        time_diff_ms = time_diff_ns / 1e6
        if time_diff_ms > self.cfg.sync_tolerance_ms:
            self.logger.warning(f"Time difference \
                {time_diff_ms:.2f} ms exceeds tolerance")
            return None
        
        packet = FramePacket(
            frame_id=self.frame_id,
            timestamp = camera_data.timestamp_ns,
            image =camera_data.data,
            robot_pose=robot_data.data,
            sync_error_ms=time_diff_ms,
        )
        self.frame_id += 1

        return packet
    
    def _read_camera_loop(self):
        self.logger.info("Camera thread started")
        while not self.stop_event.is_set():
            try:
                ret, frame = self.cap.read()
                if not ret:
                    time.sleep(0.01)
                    continue
                ts = time.perf_counter_ns()

                with self.camera_lock:
                    self.camera_data.data = frame
                    self.camera_data.timestamp_ns = ts
            except Exception as e:
                self.logger.error(f"Error reading camera: {e}")
                time.sleep(0.1)
                continue
            time.sleep(0.05)  # 20Hz sampling
        self.logger.info("Camera thread exited")

    def _read_robot_loop(self):
        self.logger.info("Robot thread started")
        while not self.stop_event.is_set():
            try:
                robot_pose, _ = get_robot_pose_from_rpc(
                    self.robot_rpc_client, self.robot_name)
                ts = time.perf_counter_ns()
                with self.robot_lock:
                    self.robot_buffer.append(
                        TimestampedData(ts, robot_pose))
            except Exception as e:
                self.logger.error(
                    f"Error reading robot pose: {e}")
                time.sleep(0.05)
                continue
            time.sleep(0.02)  # 50Hz sampling
        self.logger.info("Robot thread exited")

    def _find_closest_robot_data(self, camera_ts_ns):
        """
        Find robot data closest to the given camera timestamp.

        Returns:
            closest_robot_data: TimestampedData or None
            diff_ns: time difference in nanoseconds
        """
        with self.robot_lock:
            if len(self.robot_buffer) == 0:
                return None, None
            closest_robot_data = min(
                self.robot_buffer,
                key=lambda x: abs(
                    x.timestamp_ns - camera_ts_ns)
            )
            time_diff_ns = abs(
                closest_robot_data.timestamp_ns - camera_ts_ns)
            return closest_robot_data, time_diff_ns

def get_robot_pose_from_rpc(robot_rpc_client, robot_name):
    """Get robot TCP pose from rpc client, convert to 4x4 matrix."""
    tcp_pose = robot_rpc_client.getRobotInterface(robot_name).getRobotState().getTcpPose()
    r = Rotation.from_euler('xyz', tcp_pose[3:])
    t = np.array(tcp_pose[:3]).reshape((3, 1))
    robot_pose = np.eye(4)
    robot_pose[:3, :3] = r.as_matrix()
    robot_pose[:3, 3] = t.flatten() * 1000  # m -> mm
    return robot_pose, tcp_pose