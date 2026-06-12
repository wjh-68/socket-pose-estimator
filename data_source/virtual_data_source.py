import threading
import time
import json
import os
from collections import deque
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional

from core.packet import FramePacket
from .base_data_source import BaseDataSource
from core.logger import get_logger
from config.data_source_config import VirtualDataSourceConfig


class VirtualDataSource(BaseDataSource):
    """A virtual data source that replays locally recorded, time-aligned
    camera images and robot poses at configurable publish rates.

    The dataset folder must contain a `metadata.json` (as in recorded
    sessions) and the image files referenced by it.
    """

    def __init__(self, cfg: VirtualDataSourceConfig, stop_event):
        super().__init__(cfg)
        self.cfg = cfg
        self.stop_event = stop_event
        self.logger = get_logger(__name__)
        self.frame_id = 0

        self.state = "CREATED"

    def initialize(self):
        if self.state != "CREATED":
            raise RuntimeError("initialize() invalid state")
        if self.stop_event.is_set():
            return False

        # support both dataclass and plain dict configs
        cfg = self.cfg
        if isinstance(cfg, dict):
            dataset_path = cfg.get("dataset_path")
            self._camera_rate_hz = float(cfg.get("camera_rate_hz", 20.0))
            self._robot_rate_hz = float(cfg.get("robot_rate_hz", 50.0))
            self._loop = bool(cfg.get("loop", True))
            self._sync_tolerance_ms = float(cfg.get("sync_tolerance_ms", 20.0))
        else:
            dataset_path = getattr(cfg, "dataset_path")
            self._camera_rate_hz = float(getattr(cfg, "camera_rate_hz", 20.0))
            self._robot_rate_hz = float(getattr(cfg, "robot_rate_hz", 50.0))
            self._loop = bool(getattr(cfg, "loop", True))
            self._sync_tolerance_ms = float(getattr(cfg, "sync_tolerance_ms", 20.0))

        # Prefer metadata.json if present (folder with records list)
        meta_path = os.path.join(dataset_path, "metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                meta = json.load(f)
            self.records = meta.get("records", [])
            if not self.records:
                raise RuntimeError("No records found in metadata.json")
        else:
            # Support simple npy+jpg pairs in dataset folder (dataset/0515)
            files = sorted(os.listdir(dataset_path))
            npy_files = [f for f in files if f.lower().endswith('.npy')]
            if not npy_files:
                raise RuntimeError(f"No metadata.json or .npy files found in {dataset_path}")
            records = []
            for i, npy in enumerate(npy_files):
                npy_path = os.path.join(dataset_path, npy)
                try:
                    arr = np.load(npy_path, allow_pickle=True)
                except Exception:
                    arr = None
                # try to find corresponding image: either *_720.jpg or .jpg with same prefix
                base = os.path.splitext(npy)[0]
                img_candidates = [f for f in files if f.startswith(base) and f.lower().endswith(('.jpg', '.png'))]
                img_path = img_candidates[0] if img_candidates else None
                rec = {
                    'frame_id': i,
                    'image_path': img_path,
                    'npy_path': npy,
                    'pose_matrix_4x4': None,
                }
                # normalize pose content if possible
                if arr is not None:
                    a = np.array(arr)
                    if a.size == 16:
                        rec['pose_matrix_4x4'] = a.reshape((4, 4)).tolist()
                    elif a.shape == (4, 4):
                        rec['pose_matrix_4x4'] = a.tolist()
                    else:
                        # fallback: store raw array
                        rec['pose_matrix_4x4'] = a.tolist() if hasattr(a, 'tolist') else None
                records.append(rec)
            self.records = records

        # base path for images (metadata's image_path are relative)
        self.base_path = os.path.join(dataset_path)

        # buffers and locks
        self.camera_data = None
        self.robot_buffer = deque(maxlen=100)
        self.camera_lock = threading.Lock()
        self.robot_lock = threading.Lock()

        self._cam_index = 0
        self._robot_index = 0

        self.state = "INITIALIZED"
        return True

    def start(self):
        if self.state != "INITIALIZED":
            raise RuntimeError("start() invalid state")

        self.camera_thread = threading.Thread(target=self._publish_camera_loop, daemon=True)
        self.robot_thread = threading.Thread(target=self._publish_robot_loop, daemon=True)
        self.camera_thread.start()
        self.robot_thread.start()
        self.state = "RUNNING"
        self.logger.info("VirtualDataSource started")

    def stop(self):
        if self.state != "RUNNING":
            return
        # threads will exit when stop_event is set
        self.camera_thread.join()
        self.robot_thread.join()
        self.state = "STOPPED"

    def cleanup(self):
        self.state = "DESTROYED"

    def get_packet(self) -> Optional[FramePacket]:
        with self.camera_lock:
            camera_data = self.camera_data
        if camera_data is None:
            return None

        robot_data, time_diff_ns = self._find_closest_robot_data(camera_data[0])
        if robot_data is None or time_diff_ns is None:
            return None

        time_diff_ms = time_diff_ns / 1e6
        if time_diff_ms > self._sync_tolerance_ms:
            self.logger.warning(f"Time difference {time_diff_ms:.2f} ms exceeds tolerance")
            return None

        image = camera_data[1]
        packet = FramePacket(
            frame_id=self.frame_id,
            timestamp=camera_data[0],
            image=image,
            robot_pose=robot_data[1],
            sync_error_ms=time_diff_ms,
        )
        self.frame_id += 1
        return packet

    def _publish_camera_loop(self):
        self.logger.info("Virtual camera thread started")
        period = 1.0 / float(self._camera_rate_hz)
        records = self.records
        n = len(records)
        while not self.stop_event.is_set():
            rec = records[self._cam_index]
            img_rel = rec.get("image_path")
            img = None
            if img_rel:
                img_path = os.path.join(self.base_path, img_rel)
                img = cv2.imread(img_path)
            else:
                # no image for this record
                img = None
            if img is None:
                self.logger.debug(f"Image missing or failed to read for record {self._cam_index}")
            ts = int(time.perf_counter_ns())
            with self.camera_lock:
                self.camera_data = (ts, img)
            self._cam_index += 1
            if self._cam_index >= n:
                if self._loop:
                    self._cam_index = 0
                else:
                    break
            time.sleep(period)
        self.logger.info("Virtual camera thread exited")

    def _publish_robot_loop(self):
        self.logger.info("Virtual robot thread started")
        period = 1.0 / float(self._robot_rate_hz)
        records = self.records
        n = len(records)
        while not self.stop_event.is_set():
            rec = records[self._robot_index]
            mat = None
            if rec.get('pose_matrix_4x4') is not None:
                try:
                    mat = np.array(rec.get('pose_matrix_4x4'), dtype=float).reshape((4, 4))
                except Exception:
                    mat = None
            else:
                # try loading npy on demand
                npy_rel = rec.get('npy_path')
                if npy_rel:
                    try:
                        arr = np.load(os.path.join(self.base_path, npy_rel), allow_pickle=True)
                        a = np.array(arr)
                        if a.size == 16:
                            mat = a.reshape((4, 4))
                        elif a.shape == (4, 4):
                            mat = a
                    except Exception:
                        mat = None

            ts = int(time.perf_counter_ns())
            if mat is not None:
                with self.robot_lock:
                    self.robot_buffer.append((ts, mat))
            self._robot_index += 1
            if self._robot_index >= n:
                if self._loop:
                    self._robot_index = 0
                else:
                    break
            time.sleep(period)
        self.logger.info("Virtual robot thread exited")

    def _find_closest_robot_data(self, camera_ts_ns):
        with self.robot_lock:
            if len(self.robot_buffer) == 0:
                return None, None
            closest = min(self.robot_buffer, key=lambda x: abs(x[0] - camera_ts_ns))
            diff = abs(closest[0] - camera_ts_ns)
            return closest, diff
