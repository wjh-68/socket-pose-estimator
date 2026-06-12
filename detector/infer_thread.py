from core.logger import get_logger
from core.queues import put_latest
from core.packet import FramePacket
from config.queue_config import QueueConfig
import threading
import time
import numpy as np
import queue
import os
from config.infer_config import InferThreadConfig
from enum import Enum, auto


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
        super().__init__(name="InferThread", daemon=False)
        self.in_q = in_q
        self.out_q = out_q
        self.stop_event = stop_event
        self.cfg = cfg
        self.queue_cfg = cfg.queue_config
        self.logger = get_logger("infer_thread")
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
                num_keypoints = self.cfg.num_keypoints,
                conf_threshold = self.cfg.conf_threshold,
                iou_threshold = self.cfg.iou_threshold,
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
            t_received_ns = time.perf_counter_ns()

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
                
                # Handle abnormal packet with no image
                if packet.image is None:
                    self.logger.warning(
                        "Skipping while no image in received packet")
                    continue

                # Run inference with TRT model
                t_proc0 = time.perf_counter_ns()
                roi, keypoints = getInfer(self.model, packet.image)
                infer_cost_ms = (time.perf_counter_ns() - t_proc0) / 1e6
                total_proc_ms = (time.perf_counter_ns() - t_received_ns) / 1e6
                self.logger.debug("Inference cost: %.4f ms (total since recv: %.4f ms)", infer_cost_ms, total_proc_ms)
                packet.timing['infer'] = infer_cost_ms
                
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
                t_put0 = time.perf_counter_ns()
                self._put_packet(packet)
                put_cost_ms = (time.perf_counter_ns() - t_put0) / 1e6
                self.logger.debug("Infer put_cost_ms: %.4f ms", put_cost_ms)

            except Exception as e:
                self.logger.exception(
                    f"Inference processing error: {e}")
            finally:
                self.in_q.task_done()

        self._cleanup()
    
    def _put_packet(self, packet: FramePacket):
        """Put packet in output queue."""
        if self.queue_cfg.drop_oldest:
            put_latest(self.out_q, packet, self.logger)
        else:
            self.out_q.put(packet, block=True, 
                            timeout=self.queue_cfg.put_timeout)

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