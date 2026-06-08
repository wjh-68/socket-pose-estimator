import threading
import time
from collections import deque
import numpy as np
import queue
from dataclasses import dataclass, field
from core.logger import get_logger
from core.queues import put_latest
from core.packet import FramePacket
from core.errors import PacketValidationError, PacketProcessingError
from config.pose_estimator_config \
    import PoseEstimatorConfig, PoseEstimatorThreadConfig
from .pose_estimator import PoseEstimator


class PoseEstimatorThread(threading.Thread):
    def __init__(self, in_q:queue.Queue,
                 out_q:queue.Queue,
                 stop_event:threading.Event,
                 cfg:PoseEstimatorThreadConfig):
        super().__init__(name="PoseEstimatorThread", daemon=True)
        self.in_q = in_q
        self.out_q = out_q
        self.stop_event = stop_event
        self.cfg = cfg
        self.queue_cfg = self.cfg.queue_cfg
        self.pose_estimator = PoseEstimator(
            self.cfg.pose_estimator_cfg)
        self.logger = get_logger("pose_estimator_thread")

    def _validate_packet(self, packet:FramePacket):
        if packet.refined_pts2d is None:
            raise PacketValidationError(
                "Refined 2D points is None")
        obj_pts_len = self.cfg.pose_estimator_cfg.\
            object_model.obj_pts.shape[0]
        if packet.refined_pts2d.shape != (obj_pts_len,2):
            raise PacketValidationError(
                f"Expect {obj_pts_len} 2D points, but got \
                refined_pts2d shape {packet.refined_pts2d.shape}")
        if packet.robot_pose is None:
            raise PacketValidationError("Robot pose is None")

        # Temp
        if packet.frame_id is None:
            raise PacketValidationError("Frame ID is None")
        if packet.image is None:
            raise PacketValidationError("Image is None")
        if packet.roi is None:
            raise PacketValidationError("ROI is None")
    
    def _process_packet(self, packet:FramePacket):
        
        self._validate_packet(packet)

        t0 = time.perf_counter_ns()
        self.logger.debug(f"refined_pts2d.shape: {packet.refined_pts2d.shape}")
        pose_estimator_result = \
            self.pose_estimator.track(
                packet.frame_id, 
                packet.refined_pts2d,
                packet.robot_pose,
                )
        pose_estimation_cost_ms = \
            (time.perf_counter_ns() - t0) / 1e6

        packet.timing["pose_estimation_time"] = \
            pose_estimation_cost_ms
        self.logger.debug(
            f"Pose estimation cost: \
                {pose_estimation_cost_ms:.6f} ms")

        packet.pose_est_result = pose_estimator_result
        return packet
    
    def run(self):
        while not self.stop_event.is_set():
            try:
                packet = self.in_q.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                # Handle abnormal upstream packet
                if packet is None:
                    self.logger.warning(
                        "received None packet, skipping")
                    continue
                
                # Offline Mode: handle EOF packet from upstream
                if getattr(packet, "eof", False):
                    # Put EOF packet into queue for downstream to handle
                    self._put_packet(packet)    
                    self.logger.info("received EOF packet")
                    break   # finally block will be executed before breaking
                
                # Process packet from upstream
                packet = self._process_packet(packet)

                # Handle failed packet processing
                if packet is None:
                    self.logger.error(
                        "Pose estimation failed, skipping")
                    continue

                # Put packet into queue for downstream to handle
                self._put_packet(packet)
                
            except PacketValidationError as ve:
                self.logger.error(
                    f"Packet validation failed: {ve}"
                )
                continue
            except Exception:
                self.logger.exception("Unexpected exception in PoseEstimatorThread")
                
            finally:
                self.in_q.task_done()

        self.logger.info("PoseEstimatorThread exited cleanly")

    def _put_packet(self, packet: FramePacket):
        """Put packet in output queue."""
        if self.queue_cfg.drop_oldest:
            put_latest(self.out_q, packet, self.logger)
        else:
            self.out_q.put(packet, block=True, 
                            timeout=self.queue_cfg.put_timeout)
