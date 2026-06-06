from dataclasses import dataclass, field
from typing import Optional, List
from config.queue_config import QueueConfig
import numpy as np


@dataclass
class VizCameraConfig:
    K: np.ndarray
    dist: np.ndarray
    eMc: Optional[np.ndarray] = None


@dataclass
class VizObjectModelConfig:
    obj_pts: List[List[float]]


@dataclass
class VisualizationThreadConfig:
    # Directory to save results (plots and csv)
    result_dir: str = "result/visualization"
    # Optional minimal camera info (intrinsics/distortion/extrinsics)
    camera: Optional[VizCameraConfig] = None
    # Optional object model (3d points)
    object_model: Optional[VizObjectModelConfig] = None
    queue_config: QueueConfig = field(default_factory=QueueConfig)
