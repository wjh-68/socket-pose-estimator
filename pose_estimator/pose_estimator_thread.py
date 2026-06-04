import threading
import time
from collections import deque
import numpy as np
import queue
from dataclasses import dataclass, field
from core.logger import setup_logger
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
        self.pose_estimator = PoseEstimator(
            self.cfg.pose_estimator_cfg)
        self.logger = setup_logger("PoseEstimatorThread")

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
    
    def _process_packet(self, packet):
        self._validate_packet(packet)
        t0 = time.perf_counter_ns()
        bMo_optimized, cMo_optimized = \
            self.pose_estimator.track(packet)
        packet.timing["pose_estimation_time"] = \
            (time.perf_counter_ns() - t0) / 1e9
        packet.bMo_optimized = bMo_optimized
        packet.cMo_optimized = cMo_optimized
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
                
                # EOF packet from upstream
                if getattr(packet, "eof", False):
                    self.logger.info("received EOF packet")
                    self.out_q.put(packet)
                    break

                # Process packet from upstream
                packet = self._process_packet(packet)

                # Handle failed packet processing
                if packet is None:
                    self.logger.error(
                        "Packet processing failed, skipping")
                    continue

                if self.cfg.queue_cfg.drop_oldest:
                    put_latest(self.out_q, packet)
                else:
                    self.out_q.put(packet)
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
