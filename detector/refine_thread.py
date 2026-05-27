import threading
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from core.logger import setup_logger
from detector.refine_ellipses import *

class RefineThread(threading.Thread):
    def __init__(self, in_q, out_q, stop_event, cfg):
        super().__init__(name="RefineThread", daemon=True)
        self.in_q = in_q
        self.out_q = out_q
        self.stop_event = stop_event
        self.cfg = cfg
        self.logger = setup_logger("RefineThread")

    def extract_sub_roi(self, image, keypoints)-> tuple:
        """Extract sub-ROI images ans top-left corners 
        around each keypoint based on the large and small hole widths.
        """
        if image is None:
            self.logger.warning("No image for sub-ROI extraction")
            return None, None
        if keypoints is None or len(keypoints) != 7:
            self.logger.warning("Invalid keypoints for sub-ROI extraction")
            return None, None

        # get large and small hole widths
        rect_s = np.linalg.norm(keypoints[1]-keypoints[0])
        rect_l = np.linalg.norm(keypoints[6]-keypoints[5])

        # extract sub-ROIs around each keypoint
        sub_roi_imgs = []
        tls = [] # top-left corners
        for i, keypoint in enumerate(keypoints):
            wh = rect_l if i in [0,1] else rect_s
            # top-left and bottom-right corners of the sub-ROI
            tl = (int(keypoint[0]-wh/2), int(keypoint[1]-wh/2))
            br = (int(keypoint[0]+wh/2), int(keypoint[1]+wh/2))
            sub_roi_img = image[tl[1]:br[1], tl[0]:br[0]]
            sub_roi_imgs.append(sub_roi_img)
            tls.append(tl)
        return sub_roi_imgs, tls

    
    def refine_point(self, image, keypoints):
        """Refine keypoints by fitting ellipses to the sub-ROIs and 
        finding the best fit points. This is a placeholder implementation.
        """
        if image is None or keypoints is None:
            self.logger.warning("No image or keypoints for refinement")
            return None

        sub_roi_imgs, tls = self.extract_sub_roi(image, keypoints)
        if sub_roi_imgs is None:
            self.logger.warning("Failed to extract sub-ROIs for refinement")
            return None
        max_workers = self.cfg.get('refiner', {}).get('max_workers', 8)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            refined_ellipse = executor.map(detect_and_refine_ellipses, sub_roi_imgs)
        # convert to original image coordinates
        refined_pts = refined_ellipse + tls  ##dims ??
        return refined_pts

    def process(self, packet):
        t0 = time.time()
        if not self.validate_packet(packet):
            self.logger.warning("Invalid packet for refinement")
            return None
        refined_pts = self.refine_point(packet.image, packet.keypoints)
        if refined_pts is None:
            self.logger.warning("Refinement failed, returning original keypoints")
            refined_pts = packet.keypoints
        packet.refined_pts2d = refined_pts
        packet.timing['refine'] = (time.time() - t0) * 1000.0
        return packet
    
    def validate_packet(self, packet):
        if packet.image is None:
            self.logger.warning("Packet has no image for refinement")
            return False
        if packet.keypoints is None:
            self.logger.warning("Packet has no keypoints for refinement")
            return False
        return True

    def run(self):
        while True:
            # Get packet from input queue
            try:
                packet = self.in_q.get(timeout=0.1)
            except Exception:
                continue
            try:
                if packet is None:
                    self.logger.info("RefineThread received EOF")
                    self.out_q.put(None)
                    break
                packet = self.process(packet)
                if packet is not None:
                    self.out_q.put(packet)
                else:
                    self.logger.warning("Refinement failed for packet, skipping")
                    continue

            except Exception:
                self.logger.exception("RefineThread error")
                self.stop_event.set()
            finally:
                self.in_q.task_done()