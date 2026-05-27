import threading
import time
import numpy as np
from core.logger import setup_logger

try:
    from trt_pose_inf import YOLOTRTposeInference, getInferResults
except Exception:
    YOLOTRTposeInference = None
    getInferResults = None


class DetectorThread(threading.Thread):
    def __init__(self, in_q, out_q, stop_event, cfg):
        super().__init__(name="DetectorThread", daemon=True)
        self.in_q = in_q
        self.out_q = out_q
        self.stop_event = stop_event
        self.cfg = cfg
        self.logger = setup_logger("Detector")

        self.model = None
        det_cfg = cfg.get('detector', {}) if cfg else {}
        engine_path = det_cfg.get('engine_path')
        class_names = det_cfg.get('class_names')
        num_kps = det_cfg.get('num_keypoints', 7)
        conf = det_cfg.get('conf_threshold', 0.25)
        iou = det_cfg.get('iou_threshold', 0.45)

        if engine_path and YOLOTRTposeInference is not None:
            try:
                self.model = YOLOTRTposeInference(engine_path, class_names, num_keypoints=num_kps,
                                                  conf_threshold=conf, iou_threshold=iou)
                self.logger.info(f"Loaded TRT model from {engine_path}")
            except Exception:
                self.logger.exception("Failed to load TRT model, falling back to dummy detector")
                self.model = None
        else:
            if engine_path:
                self.logger.warning("YOLOTRTposeInference not available in environment; using dummy detector")

    def _dummy_detect(self, packet):
        h = getattr(packet.image, 'shape', (480, 640))[0]
        packet.keypoints = np.zeros((6, 2), dtype=float)
        packet.roi = (0, 0, 0, 0)
        return packet

    def process(self, packet):
        t0 = time.time()
        try:
            if self.model is not None and getInferResults is not None:
                img = packet.image
                # ensure 3-channel image for TRT model
                if img is None:
                    raise ValueError("packet.image is None")
                if hasattr(img, 'ndim') and img.ndim == 2:
                    img = np.stack([img, img, img], axis=2)
                elif hasattr(img, 'ndim') and img.ndim == 3 and img.shape[2] == 1:
                    img = np.concatenate([img, img, img], axis=2)
                # Diagnostic: if model exposes preprocess and input/output shapes, log them
                try:
                    input_shape = getattr(self.model, 'input_shape', None)
                    output_shape = getattr(self.model, 'output_shape', None)
                    self.logger.info(f"Model input_shape={input_shape}, output_shape={output_shape}")
                    if hasattr(self.model, 'preprocess'):
                        processed, scale, pad_w, pad_h, ow, oh = self.model.preprocess(img)
                        self.logger.info(
                            f"Processed dtype={processed.dtype}, shape={processed.shape}, contiguous={processed.flags['C_CONTIGUOUS']}, nbytes={processed.nbytes}"
                        )
                except Exception:
                    self.logger.exception("Diagnostic preprocessing failed")

                boxes, kps = getInferResults(self.model, img)
                # kps may be empty or object-dtype; handle safely
                if isinstance(kps, np.ndarray) and kps.size != 0:
                    # if multiple detections, take first
                    if kps.ndim == 3:
                        packet.keypoints = kps[0]
                    elif kps.ndim == 2:
                        # single detection
                        packet.keypoints = kps
                    else:
                        packet.keypoints = np.array(kps[0])
                else:
                    packet.keypoints = None

                if isinstance(boxes, np.ndarray) and boxes.size != 0:
                    packet.roi = boxes[0].tolist() if boxes.ndim == 2 else boxes.tolist()
                else:
                    packet.roi = None
            else:
                packet = self._dummy_detect(packet)
        except Exception:
            self.logger.exception("Inference error, falling back to dummy detect")
            packet = self._dummy_detect(packet)

        # ensure timing dict exists
        if not hasattr(packet, 'timing') or packet.timing is None:
            packet.timing = {}
        packet.timing['detector'] = (time.time() - t0) * 1000.0
        return packet

    def run(self):
        mode = self.cfg.get('mode', 'offline')
        while not self.stop_event.is_set():
            try:
                packet = self.in_q.get(timeout=0.1)
            except Exception:
                continue
            try:
                packet = self.process(packet)
                # push to out_q according to mode
                if mode == 'offline':
                    # block until space available
                    self.out_q.put(packet)
                else:
                    # online: prefer newest, drop oldest if full
                    if self.out_q.full():
                        try:
                            self.out_q.get_nowait()
                        except Exception:
                            pass
                    try:
                        self.out_q.put(packet, block=False)
                    except Exception:
                        # if still can't put, skip this packet
                        self.logger.warning('Failed to enqueue packet in online mode')
            except Exception:
                self.logger.exception("Detector error")
                self.stop_event.set()
