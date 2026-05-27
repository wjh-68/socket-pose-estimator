from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import numpy as np


@dataclass
class FramePacket:
    frame_id: int
    timestamp: float
    image: np.ndarray
    robot_pose: np.ndarray = field(default_factory=lambda: np.zeros((4, 4)))

    roi: Optional[Any] = None
    keypoints: Optional[np.ndarray] = None

    refined_pts2d: Optional[np.ndarray] = None
    reproj_errs: Optional[np.ndarray] = None
    valid_mask: Optional[np.ndarray] = None

    optimized_pose: Optional[np.ndarray] = None

    debug_image: Optional[np.ndarray] = None

    timing: Dict[str, float] = field(default_factory=dict)
    valid: bool = True
