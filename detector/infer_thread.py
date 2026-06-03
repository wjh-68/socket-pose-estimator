from core.logger import setup_logger
from core.queues import put_latest
from config.queue_config import QueueConfig
import threading
import time
import numpy as np
import queue
import os
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional


@dataclass
class InferThreadConfig:
    engine_path: str
    class_names: list = ["object"]
    num_keypoints: int = 7
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    queue_config: QueueConfig = field(
        default_factory=QueueConfig)

class InferenceState(Enum):
    CREATED = auto()
    INITIALIZED = auto()
    RUNNING = auto()
    ERROR = auto()
    DESTROYED = auto()

class InferThread(threading.Thread):
    """Thread for running inference on frames using a TensorRT model.
    This thread reads frames from an input queue, runs inference using a TensorRT model 
    to detect ROIs and keypoints, and puts the results in an output queue. 
    It also handles CUDA context management to avoid issues in multi-threaded environments.
    """
    def __init__(self, in_q: queue.Queue, out_q: queue.Queue,
                 stop_event: threading.Event, cfg: InferThreadConfig):
        super().__init__(daemon=False)
        self.in_q = in_q
        self.out_q = out_q
        self.stop_event = stop_event
        self.cfg = cfg
        self.logger = setup_logger("InferThread")
        self.state = InferenceState.CREATED

        # Dont load model here(main thread), defer to run()  
        # to avoid GPU context issues in multi-threaded environments

        # GPU and model objects
        self.cuda_ctx = None
        self.model = None

    def _initialize(self):
        """Initialize CUDA context and load TRT engine inside thread."""
        if self.state != InferenceState.CREATED:
            raise RuntimeError(
                f"_initialize() invalid state {self.state}"
            )
    
        try:
            import pycuda.driver as cuda
            from detector.trt_pose_inf import YOLOTRTposeInference
            cuda.init()
            self.cuda_ctx = cuda.Device(0).make_context()

            # Check if engine file exists
            if not os.path.isfile(self.cfg.engine_path):
                raise FileNotFoundError(
                    f"Engine file not found: {self.cfg.engine_path}")
            # Load TRT model
            self.model = YOLOTRTposeInference(
                engine_path = self.cfg.engine_path,
                class_names = self.cfg.class_names,
                num_kps = self.cfg.num_keypoints,
                conf_th = self.cfg.conf_threshold,
                iou_th = self.cfg.iou_threshold,
                )
            self.state = InferenceState.INITIALIZED
            self.logger.info(
                f"Loaded TRT model from {self.cfg.engine_path}")
        except Exception:
            self.logger.exception(
                f"Failed to initialize InferThread")
            self._cleanup()
            self.state = InferenceState.ERROR
            raise
    

    def run(self):
        from detector.trt_pose_inf import getInfer
        try:
            self._initialize()
        except Exception:
            self.logger.error(
                "Initialization failed, stopping thread")
            self.stop_event.set()
            return
        
        self.state = InferenceState.RUNNING

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
                    self.out_q.put(packet)  # pass EOF packet downstream
                    break   # finally block will be executed before breaking

                if packet.image is None:
                    self.logger.warning(
                        "Skipping while no image in received packet")
                    continue
                # Run inference with TRT model
                roi, keypoints = getInfer(self.model, packet.image)
                if roi is None or keypoints is None:
                    self.logger.warning("Inference returned no detections")
                    continue
                if keypoints.shape[0] != self.cfg.num_keypoints:
                    self.logger.warning(
                        f"Inference returned unexpected number of keypoints:\
                        {keypoints.shape[0]}, expected {self.cfg.num_keypoints}")
                    continue
                packet.roi = roi
                packet.keypoints = keypoints

                # Put results in output queue
                if self.cfg.queue_config.drop_oldest:
                    put_latest(self.out_q, packet, self.logger)
                else:
                    self.out_q.put(packet, block=True)

            except Exception as e:
                self.logger.exception(
                    f"Inference processing error: {e}")
            finally:
                self.in_q.task_done()

        self._cleanup()

    def _cleanup(self):
        """Release GPU resources and model safely."""
        try:
            if self.model is not None:
                if hasattr(self.model, "destroy"):
                    self.model.destroy()
                del self.model
                self.model = None
        except Exception:
            self.logger.exception("Failed to cleanup TRT model")
        
        try:
            if self.cuda_ctx is not None:
                self.cuda_ctx.pop()
                self.cuda_ctx.detach()
                self.cuda_ctx = None
        except Exception:
            self.logger.exception("Failed to release CUDA context")
        
        import gc
        gc.collect()
        self.state = InferenceState.DESTROYED
        self.logger.info("InferThread cleanup completed")