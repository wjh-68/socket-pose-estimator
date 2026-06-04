from dataclasses import dataclass, field
from queue_config import QueueConfig
import numpy as np

@dataclass
class CameraConfig:
    K: np.ndarray
    dist: np.ndarray
    eMc: np.ndarray

@dataclass
class TrackerConfig:
    # For PnP ransac
    use_adaptive_threshold: bool = True
    reproj_error_threshold: float = 1.0
    adaptive_multiplier: float = 2.0
    min_inliers: int = 4

    max_translation: float = 20.0
    max_rotation_deg: float = 20.0

@dataclass
class ObjectModelConfig:
    # Object model: 3d points
    obj_pts: np.ndarray

@dataclass
class OptimizerConfig:
    prior_sigmas: np.ndarray
    point_sigmas: np.ndarray

@dataclass
class PoseEstimatorConfig:
    window_size: int = 5
    camera: CameraConfig
    object_model: ObjectModelConfig
    tracker: TrackerConfig
    optimizer: OptimizerConfig
    # temp
    result_dir: str = ""

@dataclass
class PoseEstimatorThreadConfig:
    pose_estimator_cfg: PoseEstimatorConfig
    queue_cfg: QueueConfig = field(default_factory=lambda: QueueConfig())
