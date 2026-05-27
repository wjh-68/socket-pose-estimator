import threading
import os
import time
import numpy as np
from core.logger import setup_logger


class VisualizerThread(threading.Thread):
    def __init__(self, in_q, stop_event, cfg):
        super().__init__(name="VisualizerThread", daemon=True)
        self.in_q = in_q
        self.stop_event = stop_event
        self.cfg = cfg
        self.logger = setup_logger("Visualizer")
        self.out_dir = cfg.get('output', {}).get('path', 'output')
        os.makedirs(self.out_dir, exist_ok=True)

    def process(self, packet):
        # minimal visualization: record timings and save debug image if requested
        t0 = time.time()
        if self.cfg.get('save', {}).get('save_debug_image', False) and getattr(packet, 'debug_image', None) is not None:
            fn = os.path.join(self.out_dir, f"debug_{packet.frame_id}.npy")
            try:
                np.save(fn, packet.debug_image)
            except Exception:
                pass
        packet.timing['visualizer'] = (time.time() - t0) * 1000.0
        return packet

    def run(self):
        while not self.stop_event.is_set():
            try:
                packet = self.in_q.get(timeout=0.1)
            except Exception:
                continue
            try:
                packet = self.process(packet)
                # could also write CSV row, etc.
            except Exception:
                self.logger.exception("Visualizer error")
                self.stop_event.set()
