import time
from threading import Thread
from typing import Any
import cv2
from core.logger import setup_logger
from data_reader.offline_loader import OfflineDatasetLoader
from data_reader.sensor_data_manager import get_robot_pose_from_rpc, SensorDataManager


class DataReaderThread(Thread):
    """DataReaderThread is responsible for reading data from the specified source (offline or online) 
    and feeding it into the raw_queue for processing by the pipeline.
    """

    def __init__(self, raw_queue: Any, stop_event, cfg: dict):
        super().__init__(daemon=True)
        self.raw_queue = raw_queue
        self.stop_event = stop_event
        validate_cfg(cfg)
        self.cfg = cfg
        self.logger = setup_logger('DataReader')

    def run(self):
        mode = self.cfg.get('mode', 'offline')
        if mode == 'offline':
            data_path = self.cfg.get('dataset', {}).get('path', 'dataset/0515')
            try:
                loader = OfflineDatasetLoader(data_path, self.cfg)
            except Exception:
                self.logger.exception('init offlinedatasetloader failed')
                self.stop_event.set()
                return
            for pkt in loader.load():
                if self.stop_event.is_set():
                    break
                try:
                    self.raw_queue.put(pkt, block=True)
                except Exception:
                    self.logger.exception('Failed to put packet into raw_queue')
                    self.stop_event.set()
                    break

            self.logger.info('Offline dataset exhausted[EOF]')
            # using None as sentinel value to indicate end of data
            self.raw_queue.put(None)  
            return
        elif mode == 'online':
            sensor_cfg = self.cfg.get('sensor', {})
            robot_ip = sensor_cfg.get('robot_ip')
            robot_port = sensor_cfg.get('robot_port')
            camera_id = sensor_cfg.get('camera_id')
            camera_width = sensor_cfg.get('camera_width')
            camera_height = sensor_cfg.get('camera_height')
            camera_brightness = sensor_cfg.get('camera_brightness')
            sync_tolerance_ns = sensor_cfg.get('sync_tolerance_ns')
            
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
                    # cv2.imshow('pose_estimation', img)
                    # if cv2.waitKey(1) & 0xFF == ord('q'):
                    #     print("User interrupted")
                    #     break
                        
            except KeyboardInterrupt:
                print("Interrupted by user")
            finally:
                self.data_source.stop_online()
        else:
            self.logger.error(f"Unsupported mode: {mode}")
            self.stop_event.set()
            return

def validate_cfg(cfg: dict):
    """Validate config."""
    mode = cfg.get('mode', 'offline')
    if mode not in ['offline', 'online']:
        raise ValueError(f"Unsupported mode: {mode}")
    if mode == 'offline':
        data_path = cfg.get('dataset', {}).get('path')
        if not data_path:
            raise ValueError("Offline mode requires dataset.path in config")
    if mode == 'online':
        sensor_cfg = cfg.get('sensor', {})
        if not sensor_cfg:
            raise ValueError("Online mode requires sensor config in config")
        robot_ip = sensor_cfg.get('robot_ip')
        if not robot_ip:
            raise ValueError("Online mode requires robot_ip in sensor config")
        robot_port = sensor_cfg.get('robot_port')
        if not robot_port:
            raise ValueError("Online mode requires robot_port in sensor config")
        camera_id = sensor_cfg.get('camera_id')
        if not camera_id:
            raise ValueError("Online mode requires camera_id in sensor config")
        camera_width = sensor_cfg.get('camera_width')
        if not camera_width:
            raise ValueError("Online mode requires camera_width in sensor config")
        camera_height = sensor_cfg.get('camera_height')
        if not camera_height:
            raise ValueError("Online mode requires camera_height in sensor config")
        camera_brightness = sensor_cfg.get('camera_brightness')
        if not camera_brightness:
            raise ValueError("Online mode requires camera_brightness in sensor config")
        frame_pose_sync_tolerance_ns = sensor_cfg.get('frame_pose_sync_tolerance_ns')
        if not frame_pose_sync_tolerance_ns:
            raise ValueError("Online mode requires frame_pose_sync_tolerance_ns in sensor config")