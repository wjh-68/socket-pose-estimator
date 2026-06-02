from core.logger import setup_logger
from core.queues import put_latest
import threading
import time
import numpy as np
import queue
import os
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional


@dataclass
class InferThreadConfig:
    engine_path: str
    class_names: list = ["object"]
    num_keypoints: int = 7
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45


class InferenceState(Enum):

    CREATED = auto()

    INITIALIZED = auto()

    RUNNING = auto()

    DESTROYED = auto()

class InferThread(threading.Thread):
    """Thread for running inference on frames using a TensorRT model.
    This thread reads frames from an input queue, runs inference using a TensorRT model 
    to detect ROIs and keypoints, and puts the results in an output queue. 
    It also handles CUDA context management to avoid issues in multi-threaded environments.
    """
    def __init__(self, 
                 in_q: queue.Queue,
                 out_q: queue.Queue,
                 stop_event: threading.Event,
                 cfg: InferThreadConfig):
        super().__init__(daemon=False)
        self.in_q = in_q
        self.out_q = out_q
        self.stop_event = stop_event
        self.cfg = cfg
        self.logger = setup_logger("InferThread")
        self.state = InferenceState.CREATED

        # Dont load model here(main thread), defer to run()  
        # to avoid GPU context issues in multi-threaded environments
        # self.model = None

    def _initialize(self):
        if self.state != InferenceState.CREATED:
            raise RuntimeError(
                f"_initialize() invalid state {self.state}"
            )
    
        import pycuda.driver as cuda
        from detector.trt_pose_inf import YOLOTRTposeInference
        cuda.init()
        self.cuda_ctx = cuda.Device(0).make_context()

        try:
            engine_path = self.cfg.engine_path
            class_names = self.cfg.class_names
            num_kps = self.cfg.num_keypoints
            conf_th = self.cfg.conf_threshold
            iou_th = self.cfg.iou_threshold

            # Check if engine file exists
            if not os.path.isfile(engine_path):
                raise FileNotFoundError(
                    f"Engine file not found: {engine_path}")
            # Load TRT model
            self.model = YOLOTRTposeInference(
                engine_path, class_names, num_kps, conf_th, iou_th)
            self.logger.info(f"Loaded TRT model from {engine_path}")
        except Exception as e:
            self.logger.exception(
                f"Failed to initialize InferThread: {e}")
            self.model = None
            self.stop_event.set()
        finally:
            self.state = InferenceState.INITIALIZED

    def _cleanup(self):
        if self.state != InferenceState.INITIALIZED:
            raise RuntimeError(
                f"_cleanup() invalid state {self.state}"
            )
        if self.model is not None:
            # if the model has any explicit cleanup method, call it here
            self.model.destroy()
            del self.model
            self.model = None

        import gc
        gc.collect()
        self.cuda_ctx.pop()
        self.cuda_ctx.detach()

    def run(self):
        from detector.trt_pose_inf import getInfer
        self._initialize()

        while not self.stop_event.is_set():
            try:
                packet = self.in_q.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                # error handling for None packets (should not happen
                # if upstream threads are well-behaved, but just in case)
                if packet is None:
                    self.logger.warning(
                        "InferThread received None packet, skipping")
                    continue
                
                # handle EOF sentinel
                if packet.eof:
                    self.logger.info("InferThread received EOF")
                    self.out_q.put(packet)  # pass EOF packet downstream
                    break   # finally block will be executed before breaking

                # Process packet with TRT model
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

            except Exception:
                self.logger.exception("Inference error")
            finally:
                self.in_q.task_done()

        self._cleanup()