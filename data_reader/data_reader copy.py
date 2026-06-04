import time
from threading import Thread
from typing import Any
import cv2
from core.logger import setup_logger
from data_reader.offline_loader import OfflineDatasetLoader
from data_reader.sensor_data_manager import get_robot_pose_from_rpc, SensorDataManager
from core.packet import FramePacket
import queue

class DataReaderThread(Thread):
    """DataReaderThread is responsible for reading data from the specified source (offline or online) 
    and feeding it into the out_q for processing by the pipeline.
    """

    def __init__(self, out_q: Any, stop_event, cfg: dict):
        super().__init__(daemon=True)
        self.out_q = out_q
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
                    self.out_q.put(pkt, block=True)
                except Exception:
                    self.logger.exception('Failed to put packet into out_q')
                    self.stop_event.set()
                    break

            self.logger.info('Offline dataset exhausted[EOF]')
            # using None as sentinel value to indicate end of data
            self.out_q.put(None)  
            return
        elif mode == 'online':
            sensor_cfg = self.cfg.get('sensor', {})
            robot_ip = sensor_cfg.get('robot_ip')
            robot_port = sensor_cfg.get('robot_port')
            camera_id = sensor_cfg.get('camera_id', 0)
            camera_width = sensor_cfg.get('camera_width')
            camera_height = sensor_cfg.get('camera_height')
            camera_brightness = sensor_cfg.get('camera_brightness')
            sync_tolerance_ns = sensor_cfg.get('sync_tolerance_ns')
            read_nums = sensor_cfg.get('read_nums', None)

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
                
                while not self.stop_event.is_set():
                    
                    t0 = time.perf_counter_ns()
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
                                pass
                                # print(f"Skipping unsynchronized frames (diff={sync_info/1e6:.2f}ms)")
                            continue
                    
                    # Reset skip counter on successful frame
                    frame_skip_count = 0
                    
                    # packet a frame
                    timestamp_ns = int(time.perf_counter_ns())
                    packet = FramePacket(
                        frame_id=frame_id, timestamp=timestamp_ns, image=img, robot_pose=robot_pose)
                        
                    try:
                        self.out_q.put_nowait(packet)
                    except queue.Full:
                        try:
                            dropped = self.out_q.get_nowait()
                            self.out_q.task_done()
                            self.logger.warning(
                                "Output queue full, dropping olddest packet")
                        except queue.Empty:
                            pass

                        try:
                            self.out_q.put_nowait(packet)
                        except queue.Full:
                            self.logger.warning(
                                "Output queue still full")
                    frame_id += 1
                    
                    # Check read frames limit
                    if read_nums is not None and frame_id > read_nums:
                        print(f"Reached read_nums limit ({read_nums}), stopping...")
                        break
                    
                    duration = (time.perf_counter_ns() - t0) / 1e6
                    print(f"Frame {frame_id} processed in {duration:.2f}ms")
                    # Display current frame
                    # cv2.imshow('pose_estimation', img)
                    # if cv2.waitKey(1) & 0xFF == ord('q'):
                    #     print("User interrupted")
                    #     break

                try:
                    self.out_q.put_nowait(None)
                except queue.Full:
                    try:
                        dropped = self.out_q.get_nowait()
                        self.out_q.task_done()
                        self.logger.warning(
                            "Output queue full, dropping olddest packet")
                    except queue.Empty:
                        pass

                    try:
                        self.out_q.put_nowait(None)
                    except queue.Full:
                        self.logger.warning(
                            "Output queue still full")      
            
            except Exception:
                self.logger.exception("Online mode: read error")
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
        if sensor_cfg is None:
            raise ValueError("Online mode requires sensor config in config")
        robot_ip = sensor_cfg.get('robot_ip')
        if robot_ip is None:
            raise ValueError("Online mode requires robot_ip in sensor config")
        robot_port = sensor_cfg.get('robot_port')
        if robot_port is None:
            raise ValueError("Online mode requires robot_port in sensor config")
        camera_id = sensor_cfg.get('camera_id')
        if camera_id is None:
            raise ValueError("Online mode requires camera_id in sensor config")
        camera_width = sensor_cfg.get('camera_width')
        if camera_width is None:
            raise ValueError("Online mode requires camera_width in sensor config")
        camera_height = sensor_cfg.get('camera_height')
        if camera_height is None:
            raise ValueError("Online mode requires camera_height in sensor config")
        camera_brightness = sensor_cfg.get('camera_brightness')
        if camera_brightness is None:
            raise ValueError("Online mode requires camera_brightness in sensor config")
        sync_tolerance_ns = sensor_cfg.get('sync_tolerance_ns')
        if sync_tolerance_ns is None:
            raise ValueError("Online mode requires sync_tolerance_ns in sensor config")
