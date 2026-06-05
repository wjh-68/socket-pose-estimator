from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import numpy as np
from pose_estimator.pose_estimator import PoseEstimatorResult

@dataclass
class FramePacket:
    # raw data
    frame_id: int
    timestamp: int # ns
    image: np.ndarray
    sync_error_ms: Optional[float] = None
    robot_pose: np.ndarray = field(\
        default_factory=lambda: np.identity(4))
    eof: bool = False  # end of data flag

    # inference result
    roi: Optional[Any] = None
    keypoints: Optional[np.ndarray] = None

    # refinement result
    refined_pts2d: Optional[np.ndarray] = None

    # pose estimatation result
    pose_est_result: Optional[PoseEstimatorResult] = None
    
    # profiling info
    timing: Dict[str, float] = field(default_factory=dict)
    valid: bool = True
