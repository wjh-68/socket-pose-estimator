import threading
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from core.logger import setup_logger


class MatcherThread(threading.Thread):
    def __init__(self, in_q, out_q, stop_event, cfg):
        super().__init__(name="MatcherThread", daemon=True)
        self.in_q = in_q
        self.out_q = out_q
        self.stop_event = stop_event
        self.cfg = cfg
        self.logger = setup_logger("Matcher")
        self.executor = ThreadPoolExecutor(max_workers=self.cfg.get('matcher', {}).get('max_workers', 4))

    def refine_point(self, pkt, kp_idx):
        # placeholder refinement
        time.sleep(0)
        return pkt.keypoints[kp_idx]

    def process(self, packet):
        t0 = time.time()
        if packet.keypoints is None:
            packet.refined_pts2d = None
            packet.valid_mask = None
        else:
            n = packet.keypoints.shape[0]
            futures = [self.executor.submit(self.refine_point, packet, i) for i in range(n)]
            pts = [f.result() for f in futures]
            packet.refined_pts2d = np.array(pts)
            packet.valid_mask = np.ones((n,), dtype=bool)
        packet.timing['matcher'] = (time.time() - t0) * 1000.0
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
                self.logger.exception("Matcher error")
                self.stop_event.set()
