import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from static_pose_optimizer_ba import StaticPoseOptimizer
from config.pose_estimator_config import PoseEstimatorConfig
from utils.pnp_utils import *

class PoseEstimator:
    def __init__(self, cfg:PoseEstimatorConfig):
        self.cfg = cfg
        self.K = self.cfg.camera.K
        self.dist = self.cfg.camera.dist
        self.eMc = self.cfg.camera.eMc
        self.obj_pts = self.cfg.object_model.obj_pts
        # TODO: 优化 StaticPoseOptimizer 接口
        self.optimizer = StaticPoseOptimizer(
            K=self.K, dist=self.dist,
            prior_sigma=self.cfg.optimizer.prior_sigmas,
            point_sigma=self.cfg.optimizer.point_sigmas,
        )
        self.optimizer.set_extrinsics(self.eMc)
        self.optimizer.set_object_pts(self.obj_pts)

    def track(self, image: np.ndarray, keypoints: np.ndarray):
        
        # PnP estimate camera pose
        pnp_results = two_round_pnp(
            keypoints, self.obj_pts, self.K, self.dist,
            self.cfg.tracker.reproj_error_threshold,
            self.cfg.tracker.use_adaptive_threshold,
            self.cfg.tracker.adaptive_multiplier,
            self.cfg.tracker.min_inliers,)
        
            






