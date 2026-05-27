from core.logger import setup_logger
import threading
import time
import numpy as np
import os

class InferThread(threading.Thread):
    """Thread for running inference on frames using a TensorRT model.
    This thread reads frames from an input queue, runs inference using a TensorRT model 
    to detect ROIs and keypoints, and puts the results in an output queue. 
    It also handles CUDA context management to avoid issues in multi-threaded environments.
    """
    def __init__(self, in_q, out_q, stop_event: threading.Event, cfg: dict):
        super().__init__(daemon=False)
        self.in_q = in_q
        self.out_q = out_q
        self.stop_event = stop_event
        self.cfg = cfg
        self.mode = cfg.get('mode', 'offline')
        self.logger = setup_logger("InferThread")

        # Dont load model here(main thread), defer to run()  
        # to avoid GPU context issues in multi-threaded environments
        self.model = None


    def run(self):
        import pycuda.driver as cuda
        from detector.trt_pose_inf import YOLOTRTposeInference, getInfer
        # Manually initialize and manage CUDA context in this thread 
        # to avoid issues in multi-threaded environments.
        cuda.init()
        self.cuda_ctx = cuda.Device(0).make_context()

        try:
            # read config parameters and init TRT model
            engine_path = self.cfg.get('detector', {}).get('engine_path')
            class_names = self.cfg.get('detector', {}).get('class_names', ['obj'])
            num_kps = self.cfg.get('detector', {}).get('num_keypoints', 7)
            conf_th = self.cfg.get('detector', {}).get('conf_threshold', 0.5)
            iou_th = self.cfg.get('detector', {}).get('iou_threshold', 0.45)

            if engine_path is not None:
                try:
                    self.model = YOLOTRTposeInference(
                        engine_path, class_names, num_kps, conf_th, iou_th)
                    self.logger.info(f"Loaded TRT model from {engine_path}")
                except Exception:
                    self.logger.exception(
                        "Failed to load TRT model, InferThread will exit")
                    self.model = None
            else:
                self.logger.warning(
                    "No engine_path specified in config; InferThread will exit")
                self.model = None

            if self.model is None:
                self.logger.error("InferThread has no model, exiting")
                self.stop_event.set()

            while True:
                # Get packet from input queue
                try:
                    packet = self.in_q.get(timeout=0.1)
                except Exception:
                    continue
                

                t0 = time.time()
                try:
                    # Check for sentinel value indicating end of data
                    if packet is None:
                        self.logger.info("InferThread received EOF")
                        self.out_q.put(None)
                        break       # finally block will be executed before breaking

                    # Process packet with TRT model
                    roi, keypoints = getInfer(self.model, packet.image)
                    if roi is None or keypoints is None:
                        self.logger.warning("Inference returned no detections")
                        continue
                    packet.roi = roi
                    packet.keypoints = keypoints
                    packet.timing['infer'] = (time.time() - t0) * 1000.0
                    if self.mode == 'offline':
                        # block until space is available to preserve order
                        self.out_q.put(packet, block=True)
                    else:
                        # in online mode, drop old frames if queue is full 
                        # to keep up with real-time
                        if self.out_q.full():
                            try:
                                self.out_q.get_nowait()
                            except Exception:
                                pass
                        try:
                            self.out_q.put(packet, block=False)
                        except Exception:
                            self.logger.warning(
                                "Output inference queue is full, dropping frame")
                except Exception:
                    self.logger.exception("Inference error")
                finally:
                    # ensure we mark the task as done 
                    # even if there was an error to prevent deadlocks
                    self.in_q.task_done()
        finally:
            # Clean up CUDA context when thread exits
            if self.model is not None:
                # if the model has any explicit cleanup method, call it here
                self.model.destroy()
                del self.model
                self.model = None

            import gc
            gc.collect()
            self.cuda_ctx.pop()
            self.cuda_ctx.detach()