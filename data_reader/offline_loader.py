import os
import time
from typing import Iterator
import numpy as np
import cv2
from core.packet import FramePacket
from .base_data_source import BaseDatasetLoader
from core.logger import setup_logger



class OfflineDatasetLoader(BaseDatasetLoader):
    """Offline dataset loader that matches PNG/JPG images with corresponding .npy robot poses.

    Behavior follows the project's offline loop: scans `data_dir` for `*.jpg/.png` files,
    finds matching `.npy` entries by filename key, and yields `FramePacket` for each
    successfully loaded pair. If no valid pairs found during validation, raises.
    """

    def __init__(self, root_path: str, cfg: dict):
        self.root = root_path
        self.cfg = cfg
        self._frames = []  # list of (img_file, robot_pose_file, timestamp) tuples
        self.logger = setup_logger('OfflineDatasetLoader')
        self._scan_files()

    def _scan_files(self):
        data_dir = self.root
        if not os.path.isdir(data_dir):
            self.logger.warning(f"Data directory not found: {data_dir}")
            return
        # Scan for image files and corresponding .npy pose files
        files = sorted(os.listdir(data_dir))
        img_exts = (".jpg", ".jpeg", ".png")
        read_nums = self.cfg.get('dataset', {}).get('read_nums', None)
        if read_nums is not None:
            self.logger.info(f"Reading only first {read_nums} image/pose pairs")
            img_files = [f for f in files 
                               if any(f.endswith(ext) for ext in img_exts)][:read_nums]
        else:
            img_files = [f for f in files 
                               if any(f.endswith(ext) for ext in img_exts)]
            
        # parse timestamp
        ts = [(os.path.splitext(f)[0]).replace('_720', '') for f in img_files]
        ts_set = set(ts)
        # find corresponding .npy files for robot poses
        npy_files = [f for f in files if f.endswith('.npy') and os.path.splitext(f)[0] in ts_set]
        # compose list of (img_file, robot_pose_file, timestamp) tuples
        self._frames = list(zip(img_files, npy_files, ts))
        # self._frames = np.stack((img_files, npy_files, ts), axis=1)  # return ndarray, not list of tuples
        self.logger.info(f"Found {len(self._frames)} valid image/pose pairs in {data_dir}")

    def load(self) -> Iterator[FramePacket]:
        data_dir = self.root
        frame_id = 1
        for img_file, pose_file, ts in self._frames:
            img_path = os.path.join(data_dir, img_file)
            pose_path = os.path.join(data_dir, pose_file)
            try:
                img = cv2.imread(img_path)
                if img is None:
                    raise RuntimeError(f"Failed to read image {img_path}")
                robot_pose = np.load(pose_path)
                # temp timestamp
                timestamp_ns = time.time_ns()
                packet = FramePacket(
                    frame_id=frame_id, timestamp=timestamp_ns, image=img, robot_pose=robot_pose)
                yield packet
                frame_id += 1
            except Exception:
                self.logger.exception(f"Failed to load frame {ts} with image {img_file} and pose {pose_file}")
                continue
   

