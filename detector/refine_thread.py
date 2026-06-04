import threading
import time
import numpy as np
import queue
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from core.queues import put_latest
from core.logger import setup_logger
from core.packet import FramePacket
from core.errors import PacketValidationError
from config.queue_config import QueueConfig
from detector.refine_ellipses import *

@dataclass
class RefineThreadConfig:
    max_workers: int = 4
    num_keypoints: int =7
    queue_config: QueueConfig = field(
        default_factory=QueueConfig)

class RefineThread(threading.Thread):
    def __init__(self, in_q:queue.Queue, out_q:queue.Queue,
                 stop_event:threading.Event, cfg:RefineThreadConfig):
        super().__init__(name="RefineThread", daemon=True)
        self.in_q = in_q
        self.out_q = out_q
        self.stop_event = stop_event
        self.cfg = cfg
        self.logger = setup_logger("RefineThread")
        self.executor = ThreadPoolExecutor(
            max_workers=self.cfg.max_workers
        )

    def _extract_sub_roi(self, image, keypoints)-> tuple:
        """Extract sub-ROI images ans top-left corners 
        around each keypoint based on the large and small hole widths.
        """
        # get large and small hole widths
        rect_s = np.linalg.norm(keypoints[1]-keypoints[0])
        rect_l = np.linalg.norm(keypoints[6]-keypoints[5])

        sub_roi_imgs = []
        tls = [] # top-left corners
        h, w = image.shape[:2]
        # extract sub-ROIs around each keypoint
        for i, kp in enumerate(keypoints):
            width = rect_l if i in (0, 1) else rect_s
            # top-left and bottom-right corners of the sub-ROI
            x1 = int(kp[0] - width / 2)
            y1 = int(kp[1] - width / 2)

            x2 = int(kp[0] + width / 2)
            y2 = int(kp[1] + width / 2)

            # avoid out of boundary
            x1_clip = max(0, x1)
            y1_clip = max(0, y1)

            x2_clip = min(w, x2)
            y2_clip = min(h, y2)

            if x2_clip <= x1_clip or y2_clip <= y1_clip:
                raise RuntimeError(
                    f"Invalid ROI for kp[{i}] "
                    f"kp={kp}, "
                    f"roi=({x1},{y1})-({x2},{y2}), "
                    f"image={image.shape}"
                )

            roi = image[
                y1_clip:y2_clip,
                x1_clip:x2_clip
            ]

            sub_roi_imgs.append(
                np.ascontiguousarray(roi)
            )

            tls.append(
                (x1_clip, y1_clip)
            )

        return sub_roi_imgs, np.asarray(tls)
    
    def _refine_point(self, image, keypoints):
        """Refine keypoints by fitting ellipses to the sub-ROIs and 
        finding the best fit points. This is a placeholder implementation.
        """
        try:
            sub_roi_imgs, tls = self._extract_sub_roi(image, keypoints)
        except Exception:
            self.logger.error(
                "Failed to extract sub-ROIs for refinement")
            raise
        t0 = time.perf_counter_ns()
        results = list(self.executor.map(
            detect_and_refine_ellipses, sub_roi_imgs))
        self.logger.info(
            f"threadpool exec detect_and_refine_ellipses time: \
                {(time.perf_counter_ns() - t0) / 1e6:.2f}ms")    
        refined_pts = []
        for i, res in enumerate(results):
            if res is not None:
                # convert to original image coordinates
                refined_pts.append(res['p']+tls[i])
        
        return np.asarray(refined_pts)

    def process(self, packet:FramePacket):
        t0 = time.time()
        self._validate_packet(packet)
        
        try:
            refined_pts = self._refine_point(
                packet.image, packet.keypoints)
        except Exception:
            self.logger.warning("Refinement failed")
            raise

        # check dims
        if refined_pts.shape != (self.cfg.num_keypoints,2):
            self.logger.warning(
                f"Expect refined keypoints shape is \
                {self.cfg.num_keypoints}x2, \
                but got {refined_pts.shape}")
            return None
            # raise ValueError(
            #     f"Expect get a result of refined keypoints \
            #         with shape {self.cfg.num_keypoints}x2, \
            #         but got {refined_pts.shape}")
        
        packet.refined_pts2d = refined_pts
        packet.timing['refine'] = (time.time() - t0) * 1000.0
        self.logger.info(
            f"refine time: {packet.timing['refine']:.2f}ms")
        return packet
    
    def _validate_packet(self, packet:FramePacket):
        kp = packet.keypoints
        if packet.image is None:
            raise PacketValidationError(
                "packet.image is None")
        if not isinstance(kp, np.ndarray):
            raise PacketValidationError(
                "packet.keypoints is not numpy array")
        if kp.shape != (self.cfg.num_keypoints,2):
            raise PacketValidationError(
                f"expect keypoints shape is \
                {self.cfg.num_keypoints}x2,\
                but got {kp.shape}")

    def run(self):
        try:
            while not self.stop_event.is_set():
                # Get packet from input queue
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
                        self.out_q.put(packet)  # pass EOF packet downstream
                        break

                    # Process and add refined points to packet
                    packet = self.process(packet)
                    
                    # Handle failed refinement
                    if packet is None:
                        self.logger.warning(
                            "Refinement failed for packet, skipping")
                        continue

                    if self.cfg.queue_config.drop_oldest:
                        put_latest(self.out_q, packet)
                    else:
                        self.out_q.put(packet, block=True)                    

                except Exception:
                    raise
                finally:
                    self.in_q.task_done()
        except Exception:
            self.logger.exception(
                "RefineThread error")
        finally:
            self.executor.shutdown()
            self.logger.info("RefineThread exited cleanly")
