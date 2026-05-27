import os
import time
from typing import Iterator
import numpy as np
from PIL import Image
from core.packet import FramePacket
from .base_loader import BaseDatasetLoader


class OfflineDatasetLoader(BaseDatasetLoader):
    def __init__(self, root_path: str, cfg: dict = None):
        self.root = root_path
        self.cfg = cfg or {}
        self._files = self._scan_files()
        # By default pre-validate dataset to fail fast if files are corrupted
        if self.cfg.get('prevalidate', True):
            self.validate_or_raise()

    def _scan_files(self):
        exts = (".png", ".jpg", ".jpeg", ".npy")
        files = []
        if not os.path.isdir(self.root):
            return files
        for fn in sorted(os.listdir(self.root)):
            if fn.lower().endswith(exts):
                files.append(os.path.join(self.root, fn))
        return files

    def load(self) -> Iterator[FramePacket]:
        frame_id = 0
        for p in self._files:
            ts = time.time()
            # We assume dataset has been pre-validated; any load error should raise
            if p.lower().endswith('.npy'):
                arr = np.load(p)
                # Distinguish between image-like npy (HxW or HxWxC) and small pose/metadata matrices (e.g. 4x4)
                if arr.ndim >= 2 and max(arr.shape[0], arr.shape[1]) > 16:
                    if arr.ndim == 2:
                        img = np.stack([arr, arr, arr], axis=-1)
                    elif arr.ndim == 3:
                        if arr.shape[2] == 1:
                            img = np.concatenate([arr, arr, arr], axis=2)
                        else:
                            img = arr
                    else:
                        raise RuntimeError(f"Unsupported array shape {arr.shape} for file {p}")
                else:
                    # Likely a small pose/metadata npy (e.g. 4x4); skip as it's not an image
                    continue
            else:
                img = np.array(Image.open(p).convert('RGB'))

            # Normalize image to HxWx3 uint8
            img = np.asarray(img)
            if img.ndim == 2:
                img = np.stack([img, img, img], axis=-1)
            elif img.ndim == 3:
                if img.shape[2] == 1:
                    img = np.concatenate([img, img, img], axis=2)
                elif img.shape[2] > 3:
                    img = img[:, :, :3]

            if img.dtype.kind == 'f':
                mx = float(np.max(img)) if img.size else 0.0
                if mx <= 1.0:
                    img = (img * 255.0).round().astype(np.uint8)
                else:
                    img = np.clip(img, 0, 255).astype(np.uint8)
            elif img.dtype != np.uint8:
                img = np.clip(img, 0, 255).astype(np.uint8)

            packet = FramePacket(frame_id=frame_id, timestamp=ts, image=img)
            frame_id += 1
            yield packet

    def validate_or_raise(self, min_height: int = 32, min_width: int = 32) -> None:
        """Validate all dataset files and raise RuntimeError on first invalid file.

        Checks that images can be opened/loaded and produce at least `min_height` x `min_width` and 3 channels.
        """
        if not self._files:
            raise RuntimeError(f"No data files found in dataset path: {self.root}")

        for p in self._files:
            try:
                if p.lower().endswith('.npy'):
                    arr = np.load(p)
                    # If npy is a small matrix (e.g. 4x4) assume it's a pose/metadata file and skip
                    if arr.ndim >= 2 and max(arr.shape[0], arr.shape[1]) <= 16:
                        # skip pose/metadata npy
                        continue
                    if arr.ndim == 2:
                        h, w = arr.shape
                    elif arr.ndim == 3:
                        h, w = arr.shape[0], arr.shape[1]
                    else:
                        raise RuntimeError(f"Unsupported npy shape {arr.shape} in {p}")
                else:
                    from PIL import Image as PILImage

                    with PILImage.open(p) as im:
                        im = im.convert('RGB')
                        w, h = im.size

                if h < min_height or w < min_width:
                    raise RuntimeError(f"Image too small {w}x{h} in {p}")
            except Exception as e:
                raise RuntimeError(f"Failed to validate dataset file {p}: {e}")
