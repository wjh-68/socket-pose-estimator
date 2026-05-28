import threading
import time
import numpy as np
import queue
from concurrent.futures import ThreadPoolExecutor
from core.logger import setup_logger
from static_pose_optimizer_ba import StaticPoseOptimizer, load_camera_parameters
from core.error import ConfigError, QueueError
from tracker.tracker import Tracker

# validate config -> two round pnp -> init/update (BA)optimizer
# -> add result and record data to result queue ->
class TrackThread(threading.Thread):
    def __init__(self, in_q, out_q, stop_event, cfg):
        super().__init__(name="TrackThread", daemon=True)
        self.in_q = in_q
        self.out_q = out_q
        self.stop_event = stop_event
        self.logger = setup_logger("TrackThread")

        # check in/out queue
        if self.in_q is None or self.out_q is None:
            raise QueueError(
                "Input or Output Queue is None")
        # check cfg
        validate_cfg(cfg)
        self.cfg = cfg

        # init tracker
        self.tracker = Tracker(self.cfg)
        

    def process(self, packet, EOF = False):
        if EOF:
            self.tracker.save_results()
        else:
            if not self.validate_packet(packet):
                self.logger.warning("Invalid packet for tracking")
                return
            self.tracker.track(packet)
    
    def validate_packet(self, packet):
        if packet.image is None:
            self.logger.warning(
                "Packet has no image for tracking")
            return False
        if packet.refined_pts2d is None:
            self.logger.warning(
                "Packet has no refined points for tracking")
            return False
        if packet.robot_pose is None:
            self.logger.warning(
                "Packet has no robot pose for tracking")
            return False
        if packet.roi is None:
            self.logger.warning(
                "Packet has no roi for tracking")
            return False
        if packet.timestamp is None:
            self.logger.warning(
                "Packet has no timestamp for tracking")
            return False
        if packet.frame_id is None:
            self.logger.warning(
                "Packet has no frame id for tracking")
            return False
        return True

    def run(self):
        mode = self.cfg.get('mode', 'offline')
        while not self.stop_event.is_set():
            # Get packet from input queue
            try:
                packet = self.in_q.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                # Handle EOF
                EOF = False
                if packet is None:
                    self.logger.info(
                        "Received EOF")
                    EOF = True
                    # Temp: Do nothing
                    # but send the None packet to tracker to 
                    # handle EOF and save result
                    # self.out_q.put(None)
                    # break
                self.process(packet, EOF)
                # TODO: put result packet to out_q
            except Exception:
                self.logger.exception(
                    "RefineThread fatal error, exiting")
                self.stop_event.set()
                return  # finally block will still be executed to mark task done
            finally:
                self.in_q.task_done()



def validate_cfg(cfg):
    if cfg is None:
        raise ConfigError("Config is None")

    # check num_keypoints, obj_pts and sigmas dimensions consistency
    num_kps = cfg.get('detector', {}).get('num_keypoints')
    obj_pts = cfg.get('optimizer', {}).get('obj_pts')
    if num_kps is None or obj_pts is None:
        raise ConfigError(
            "num_keypoints and obj_pts must be specified in config")
    if len(obj_pts) != num_kps:
        raise ConfigError(
            f"num_keypoints ({num_kps}) does not match \
            number of obj_pts ({len(obj_pts)})")

    # prior_sigmas = cfg.get('optimizer', {}).get('prior_sigmas')
    point_sigmas = cfg.get('optimizer', {}).get('point_sigmas')
    if point_sigmas is not None and len(point_sigmas) != num_kps:
        raise ConfigError(
            f"num_keypoints ({num_kps}) does not match \
                number of point_sigmas ({len(point_sigmas)})")
    
    # check mode
    mode = cfg.get('mode', 'offline')
    if mode not in ['offline', 'online']:
        raise ConfigError(f"Unsupported mode: {mode}")
    
    # check camera params
    cam_cfg = cfg.get('camera', {})
    cam_prms_n = ['K', 'dist', 'eMc']
    for n in cam_prms_n:
        if not cam_cfg.get(n):
            raise ConfigError(f"Camera param {n} is None")

