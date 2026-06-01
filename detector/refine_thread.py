import threading
import time
import numpy as np
import queue
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
        if self.in_q is None or self.out_q is None:
            self.logger.error(
                "Input or output queue is None, Thread will exit")
            self.stop_event.set()

    def extract_sub_roi(self, image, keypoints)-> tuple:
        """Extract sub-ROI images ans top-left corners 
        around each keypoint based on the large and small hole widths.
        """
        if image is None:
            self.logger.warning(
                "No image for sub-ROI extraction")
            return None, None
        if keypoints is None or len(keypoints) != 7:
            self.logger.warning(
                "Invalid keypoints for sub-ROI extraction")
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
            sub_roi_imgs.append(np.ascontiguousarray(sub_roi_img))
            tls.append(tl)
        tls = np.asanyarray(tls)  # shape (7,2)
        return sub_roi_imgs, tls

    
    def refine_point(self, image, keypoints):
        """Refine keypoints by fitting ellipses to the sub-ROIs and 
        finding the best fit points. This is a placeholder implementation.
        """
        if image is None or keypoints is None:
            self.logger.warning(
                "No image or keypoints for refinement")
            return None

        sub_roi_imgs, tls = self.extract_sub_roi(image, keypoints)
        if sub_roi_imgs is None:
            self.logger.warning(
                "Failed to extract sub-ROIs for refinement")
            return None
        max_workers = self.cfg.get('refiner', {}).get('max_workers', 8)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            result = executor.map(detect_and_refine_ellipses, sub_roi_imgs)

        # convert to original image coordinates
        # filter out None results from failed detections
        valid_results = [res for res in result if res is not None]
        
        if len(valid_results) == 0:
            self.logger.warning(
                "All ellipse detections failed, no valid results")
            return None
            
        if len(valid_results) != len(sub_roi_imgs):
            self.logger.warning(
                f"Partial detection failure: {len(valid_results)}/{len(sub_roi_imgs)} points detected")
        
        refined_ellipses = np.array([res['p'] for res in valid_results])  # shape (N,2)

        # check dimension consistency
        if refined_ellipses.shape[0] != tls.shape[0]:
            self.logger.warning(
                "Refined ellipses and top-left corners have inconsistent dimensions" \
                "Refined ellipses shape: {}, top-left corners shape: {}".format(
                    refined_ellipses.shape, tls.shape))
            return None
        
        refined_pts = refined_ellipses + tls  # dims should match
        return refined_pts

    def process(self, packet):
        t0 = time.time()
        if not self.validate_packet(packet):
            self.logger.warning("Invalid packet for refinement")
            return None
        refined_pts = self.refine_point(
            packet.image, packet.keypoints)
        if refined_pts is None:
            self.logger.warning(
                "Refinement failed, returning original keypoints")
            refined_pts = packet.keypoints
        packet.refined_pts2d = refined_pts
        packet.timing['refine'] = (time.time() - t0) * 1000.0
        print(f"refine time: {packet.timing['refine']:.2f}ms")
        return packet
    
    def validate_packet(self, packet):
        if packet.image is None:
            self.logger.warning(
                "Packet has no image for refinement")
            return False
        if packet.keypoints is None:
            self.logger.warning(
                "Packet has no keypoints for refinement")
            return False
        return True

    def run(self):
        mode = self.cfg.get('mode', 'offline')
        while not self.stop_event.is_set():
            # Get packet from input queue
            try:
                packet = self.in_q.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                # Handle EOF
                if packet is None:
                    self.logger.info(
                        "RefineThread received EOF")
                    self.out_q.put(None)
                    break
                # Process and add refined points to packet
                packet = self.process(packet)

                # Handle failed refinement
                if packet is None:
                    self.logger.warning(
                        "Refinement failed for packet, skipping")
                    continue
                # offline
                if mode == 'offline':
                    self.out_q.put(packet, block=True)
                # online
                else:
                    # if out_q is full, drop olddest packet
                    try:
                        self.out_q.put_nowait(packet)
                    except queue.Full:
                        try:
                            dropped = self.out_q.get_nowait()
                            self.out_q.task_done()
                            self.logger.warning(
                                "Output queue full, dropping olddest packet")
                        except queue.Empty:
                            pass

                        try:
                            self.out_q.put_nowait(packet)
                        except queue.Full:
                            self.logger.warning(
                                "Output queue still full")
                

            except Exception:
                self.logger.exception(
                    "RefineThread error")
            finally:
                self.in_q.task_done()

# TODO: check keypoints dims for all threads