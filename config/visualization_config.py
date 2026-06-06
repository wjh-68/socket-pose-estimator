from dataclasses import dataclass, field
from typing import Optional
from config.queue_config import QueueConfig
import numpy as np


@dataclass
class VizCameraConfig:
    K: np.ndarray
    dist: np.ndarray
    eMc: Optional[np.ndarray] = None


@dataclass
class VizObjectModelConfig:
    obj_pts: np.ndarray


@dataclass
class VisualizationThreadConfig:
    # Directory to save results (plots, CSV, and images)
    result_dir: str = "result/visualization"
    # Optional minimal camera info for axis drawing and PnP-based pose computation
    camera: Optional[VizCameraConfig] = None
    # Optional object model 3D points for projection comparison with refined 2D detections
    object_model: Optional[VizObjectModelConfig] = None
    # Control output generation
    save_images: bool = True
    save_csv: bool = True
    save_plots: bool = True
    image_format: str = "png"
    queue_config: QueueConfig = field(default_factory=QueueConfig)
