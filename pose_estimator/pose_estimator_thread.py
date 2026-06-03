import threading
import time
from collections import deque
import numpy as np
import queue
from core.logger import setup_logger
from core.queues import put_latest
from config.pose_estimator_config import PoseEstimatorConfig
from dataclasses import dataclass, field

from static_pose_optimizer_ba import StaticPoseOptimizer, load_camera_parameters

class PoseEstimatorThread(threading.Thread):
    def __init__(self, in_q:queue.Queue,
                 out_q:queue.Queue,
                 stop_event:threading.Event,
                 cfg:PoseEstimatorConfig):
        super().__init__(name="PoseEstimatorThread", daemon=True)
        self.in_q = in_q
        self.out_q = out_q
        self.stop_event = stop_event
        self.cfg = cfg
        self.logger = setup_logger("PoseEstimatorThread")

    def 