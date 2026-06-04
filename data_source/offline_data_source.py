import os
import time
from typing import Iterator, Optional
import numpy as np
import cv2
from skimage import data_dir
from core.packet import FramePacket
from .base_data_source import BaseDataSource
from core.logger import setup_logger
from dataclasses import dataclass
from config.data_source_config import OfflineDataSourceConfig

class OfflineDataSource(BaseDataSource):
    def __init__(
            self,
            cfg: OfflineDataSourceConfig,
    ):
        super().__init__(cfg)
        self.logger = setup_logger(__name__)

    def initialize(self):
        self._scan_files()
        self.frame_id = 0
        self._idx = 0
        self.logger.info(f"Offline dataset initialized: \
                         {len(self._frames)} samples")

    def start(self):
        self._idx = 0

    def stop(self):
        self.logger.info("Offline dataset stopped")

    def cleanup(self):
        self._frames = []
        self.logger.info("Offline dataset cleaned up")
        
    def get_packet(self):
        while self._idx < len(self._frames):
            img_file, pose_file, ts = self._frames[self._idx]
            self._idx += 1
            try:
                img_path = os.path.join(self.cfg.dataset_path, img_file)
                pose_path = os.path.join(self.cfg.dataset_path, pose_file)

                img = cv2.imread(img_path)
                if img is None:
                    raise RuntimeError(
                        f"Failed to read image: {img_path}")
                robot_pose = np.load(pose_path)
                if robot_pose is None:
                    raise RuntimeError(
                        f"Failed to read robot pose: {pose_path}")
                # TODO: add timestamp parsing
                # if self.cfg.parse_timestamp:

                ts_ns = time.time_ns()
                    
                packet = FramePacket(
                    frame_id=self.frame_id,
                    timestamp=ts_ns,
                    image=img,
                    robot_pose=robot_pose
                )
                self.frame_id+=1
                return packet
            except Exception as e:
                self.logger.warning(f"Skip corrupted frame \
                            {img_file} or {pose_file}: {e}")
                continue

        # return an EOF packet when all data is read
        packet = FramePacket(
            frame_id=-1,
            timestamp=-1,
            image=None,
            robot_pose=None,
            eof=True,
        )
        return packet
        

    def _scan_files(self):
        data_dir = self.cfg.dataset_path
        if not os.path.isdir(data_dir):
            raise FileNotFoundError(
                f"Data directory not found: {data_dir}")
            img_files = []
            npy_files = []
            return
        # Scan for image files and corresponding .npy pose files
        files = sorted(os.listdir(data_dir))
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
        ts = [(os.path.splitext(f)[0]).replace('_720', '') \
              for f in img_files]
        ts_set = set(ts)
        # find corresponding .npy files for robot poses
        npy_files = [f for f in files \
                            if f.endswith('.npy') and \
                            os.path.splitext(f)[0] in ts_set]
        # compose list of (img_file, robot_pose_file, timestamp) tuples
        self._frames = list(zip(img_files, npy_files, ts))
     
