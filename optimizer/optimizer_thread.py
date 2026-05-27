import threading
import time
from collections import deque
import numpy as np
from core.logger import setup_logger

try:
    from static_pose_optimizer_ba import StaticPoseOptimizer, load_camera_parameters
except Exception:
    StaticPoseOptimizer = None
    load_camera_parameters = None


class OptimizerThread(threading.Thread):
    def __init__(self, in_q, out_q, stop_event, cfg):
        super().__init__(name="OptimizerThread", daemon=True)
        self.in_q = in_q
        self.out_q = out_q
        self.stop_event = stop_event
        self.cfg = cfg
        self.logger = setup_logger("Optimizer")
        self.window_size = cfg.get('optimizer', {}).get('window_size', 5)
        self.sliding_window = deque(maxlen=self.window_size)

        self.use_static = cfg.get('optimizer', {}).get('use_static_ba', False)
        self.static_opt = None
        if self.use_static and StaticPoseOptimizer is not None:
            try:
                # Try to load camera params from dataset if available
                dataset_path = cfg.get('dataset', {}).get('path')
                if load_camera_parameters is not None and dataset_path:
                    K, dist, eMc = load_camera_parameters(dataset_path)
                else:
                    K = np.eye(3)
                    dist = np.zeros(5)
                    eMc = np.eye(4)

                self.static_opt = StaticPoseOptimizer(K, dist)
                self.static_opt.set_extrinsics(eMc)
                # set object points if provided in config
                obj_pts = cfg.get('optimizer', {}).get('obj_pts')
                if obj_pts is not None:
                    self.static_opt.set_object_pts(np.array(obj_pts, dtype=np.float64))
                self.logger.info("Initialized StaticPoseOptimizer")
            except Exception:
                self.logger.exception("Failed to initialize StaticPoseOptimizer; falling back to sliding-window")
                self.static_opt = None

    def _sliding_average(self):
        poses = [p.robot_pose for p in self.sliding_window if getattr(p, 'robot_pose', None) is not None]
        if not poses:
            return None
        avg = np.mean(np.stack([p.flatten() for p in poses], axis=0), axis=0)
        return avg.reshape(poses[0].shape)

    def process(self, packet):
        t0 = time.time()
        self.sliding_window.append(packet)

        if self.static_opt is not None and packet.refined_pts2d is not None and packet.valid_mask is not None:
            try:
                # prepare pts2d and pts3d
                pts2d = packet.refined_pts2d
                mask = packet.valid_mask
                if mask is not None:
                    pts2d = np.array(pts2d)[mask]
                pts3d = self.static_opt.obj_pts if self.static_opt.obj_pts is not None else None
                if pts3d is None:
                    # cannot optimize without object points
                    raise RuntimeError("No object_points set for static optimizer")

                if not self.static_opt.is_initialized():
                    # set initial pose to identity if missing
                    self.static_opt.set_initial_pose(np.eye(4, dtype=np.float64))

                self.static_opt.add_frame(packet.frame_id, packet.robot_pose if packet.robot_pose is not None else np.eye(4), pts2d, pts3d)
                result = self.static_opt.optimize()
                packet.optimized_pose = self.static_opt.get_pose()
            except Exception:
                self.logger.exception("Static optimizer error; falling back to sliding average")
                packet.optimized_pose = self._sliding_average() if self._sliding_average() is not None else packet.robot_pose
        else:
            packet.optimized_pose = self._sliding_average() if self._sliding_average() is not None else packet.robot_pose

        packet.timing['optimizer'] = (time.time() - t0) * 1000.0
        return packet

    def run(self):
        while not self.stop_event.is_set():
            try:
                packet = self.in_q.get(timeout=0.1)
            except Exception:
                continue
            try:
                packet = self.process(packet)
                self.out_q.put(packet)
            except Exception:
                self.logger.exception("Optimizer error")
                self.stop_event.set()
