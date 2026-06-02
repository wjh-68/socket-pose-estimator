import os
import time
from typing import Iterator, Optional
import numpy as np
import cv2
from core.packet import FramePacket
from .base_data_source import BaseDataSource
from core.logger import setup_logger
from dataclasses import dataclass

@dataclass
class OfflineDataSourceConfig:
    dataset_path: str
    read_image_nums: int = 100
    parse_timestamp: bool = False

class OfflineDataSource(BaseDataSource):
    def __init__(
            self,
            cfg: OfflineDataSourceConfig,
    ):
        self.cfg = cfg
        self.logger = setup_logger(__name__)
        self.data_dir = cfg.dataset_path
        self.img_files = []
        self.npy_files = {}
        self.current_frame_idx = 0
        self._scan_files()

    def _scan_files(self):
        if not os.path.isdir(self.data_dir):
            self.logger.warning(f"Data directory not found: {self.data_dir}")
            return
        
        files = sorted(os.listdir(self.data_dir))
        img_exts = (".jpg", ".jpeg", ".png")
        read_nums = self.cfg.read_image_nums
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
        
        npy_files = [f for f in files if f.endswith('.npy') and os.path.splitext(f)[0] in ts_set]
        
        self.img_files = img_files
        self.npy_files = {os.path.splitext(f)[0]: f for f in npy_files}
        
        self.logger.info(f"Found {len(self.img_files)} images and {len(self.npy_files)} poses in {self.data_dir}")

    def get_next_frame_pose_sync(self):
         """
         Get next frame and pose with timestamp synchronization.
         
         Returns:
             tuple: (frame, pose, time_diff_ms, success)
         """
         if self.mode != 'offline':
             raise ValueError("This method only works in offline mode")
         
         if self.current_frame_idx >= len(self.img_files):
             return None, None, 0, False
         
         img_file = self.img_files[self.current_frame_idx]
         img_path = os.path.join(self.data_dir, img_file)
         
         # Load image
         frame = cv2.imread(img_path)
         if frame is None:
             print(f"Failed to load image: {img_path}")
             return None,

