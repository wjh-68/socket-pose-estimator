import threading
import cv2
import time
from core.packet import FramePacket
from .base_data_source import BaseDataSource
from core.logger import setup_logger
from dataclasses import dataclass
from enum import Enum, auto

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

@dataclass
class OnlineDataSourceConfig:
    camera_id: str
    camera_width: int
    camera_height: int
    camera_brightness: int
    robot_ip: str
    robot_port: int
    robot_login_name: str = "aubo"
    robot_password: str = "123456"
    sync_tolerance_ms: float
    read_image_nums: int = 100

class OnlineDataSource(BaseDataSource):
    def __init__(
            self,
            cfg: OnlineDataSourceConfig,
            stop_event,
    ):
        self.cfg = cfg
        self.stop_event = stop_event
        self.logger = setup_logger(__name__)
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
        self.state = DataSourceState.INITIALIZED
        if self.stop_event.is_set():
            raise InterruptedError()
        return True
    
    def start(self):

        if self.state != DataSourceState.INITIALIZED:
            raise RuntimeError(
                f"start() invalid state {self.state}"
            )

         # Start threads
        self.camera_thread = threading.Thread(
            target=self._read_camera_loop, daemon=True)
        self.robot_thread = threading.Thread(
            target=self._read_robot_loop, daemon=True)
        self.camera_thread.start()
        self.robot_thread.start()

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

        
    def _read_camera_loop(self):
        while not self.stop_event.is_set():
            ret, frame = self.cap.read()
            if ret:
                with self.frame_lock:
                    self.latest_frame = frame
                    self.latest_frame_ts = time.perf_counter_ns()
            time.sleep(1.0)

    def _read_robot_loop(self):
        pass

    
