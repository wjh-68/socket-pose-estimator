from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import numpy as np


@dataclass
class FramePacket:
    # raw data
    frame_id: int
    timestamp: int # ns
    image: np.ndarray
    sync_error_ms: Optional[float] = None
    robot_pose: np.ndarray = field(\
        default_factory=lambda: np.zeros((4, 4)))
    eof: bool = False  # end of data flag

    # inference results
    roi: Optional[Any] = None
    keypoints: Optional[np.ndarray] = None

    # refinement results
    refined_pts2d: Optional[np.ndarray] = None

    # tracking results
    
    reproj_errs: Optional[np.ndarray] = None
    valid_mask: Optional[np.ndarray] = None

    bMo_optimized: Optional[np.ndarray] = None
    cMo_optimized: Optional[np.ndarray] = None

    debug_image: Optional[np.ndarray] = None

    # profiling info
    timing: Dict[str, float] = field(default_factory=dict)
    valid: bool = True
