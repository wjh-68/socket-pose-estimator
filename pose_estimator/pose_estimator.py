import cv2
import numpy as np
import os
from scipy.spatial.transform import Rotation
from static_pose_optimizer_ba import StaticPoseOptimizer
from config.pose_estimator_config import PoseEstimatorConfig
from utils.pnp_utils import *
from core.logger import setup_logger
from core.packet import FramePacket

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
            prior_sigmas=self.cfg.optimizer.prior_sigmas,
            point_sigmas=self.cfg.optimizer.point_sigmas,
        )
        self.optimizer.set_extrinsics(self.eMc)
        self.optimizer.set_object_pts(self.obj_pts)
        self.logger = setup_logger("PoseEstimator")
        self._reset_statistics()
        # Temp
        self.result_dir = self.cfg.result_dir

    def _reset_statistics(self):
        self.cnt_rejected_frames = 0
        self.cnt_pnp_failed = 0
        self.last_bMo = None

    # TODO: change track() params from packet to below
    # after move record and visualization to new thread
    
    # def track(self, frame_id: int, image: np.ndarray, 
    #           keypoints: np.ndarray, robot_pose: np.adarray):

    def track(self, packet: FramePacket):

        # Temp
        frame_id = packet.frame_id
        refined_pts2d = packet.refined_pts2d
        image = packet.image
        robot_pose = packet.robot_pose
        
        # PnP estimate camera pose
        # two round PnP for better initial pose in optimization
        pnp_result = two_round_pnp(
            refined_pts2d, self.obj_pts, self.K, self.dist,
            self.cfg.tracker.reproj_error_threshold,
            self.cfg.tracker.use_adaptive_threshold,
            self.cfg.tracker.adaptive_multiplier,
            self.cfg.tracker.min_inliers,)
        
        if not pnp_result.valid:
            self.cnt_pnp_failed += 1
            self.logger.warning(
                f"PnP failed: {pnp_result.reason}"
            )
            return None, None
        
        # T_co
        cMo_pnp = rvec_tvec_to_transform(
            pnp_result.pose.rvec, pnp_result.pose.tvec
        )
        bMe = robot_pose
        bMo_pnp = bMe @ self.eMc @ cMo_pnp

        # Initialize pose
        if not self.optimizer.is_initialized():
            self.optimizer.set_initial_pose(bMo_pnp)
            # return bMo_pnp

        # Filter frame
        if self._should_reject_pose_diff(cMo_pnp, robot_pose):
            self.cnt_rejected_frames += 1
            return None, None
        
        # Manage sliding window
        if self.optimizer.get_frame_count() >= self.cfg.window_size:
            self.optimizer.remove_oldest_frame()

        # Add frame
        self.optimizer.add_frame(
            frame_id, robot_pose, refined_pts2d,self.obj_pts)
        
        # Optimize
        self.optimizer.optimize()

        # Get optimized pose
        bMo_optimized = self.optimizer.get_pose()
        cMo_optimized = self.optimizer.compute_cMo(robot_pose, frame_id)
        self.last_bMo = bMo_optimized

        # Temp: Record data for analysis and diagnostics

        # Temp: Visualization
        roi = packet.roi
        inlier_mask = pnp_result.diagnostics.inlier_mask

        # self._render_frame(image,roi,refined_pts2d,self.obj_pts,
        #                    cMo_optimized,inlier_mask,frame_id)
        # TODO: Move visualization and data record to new threads

        return bMo_optimized, cMo_optimized


    def _render_frame(
        self,
        img,
        roi_bounds,
        centers,
        pts3d,
        cMo_optimized,
        inlier_mask,
        frame_id,
    ):
        vis_img = img.copy()
        roi_x_min, roi_y_min, roi_x_max, roi_y_max = roi_bounds
        cMo_rvec, cMo_tvec = transform_to_rvec_tvec(cMo_optimized)
        cv2.drawFrameAxes(vis_img, self.K, self.dist, 
                          cMo_optimized[:3, :3], 
                          cMo_optimized[:3, 3:], 10, 3)
        proj, _ = cv2.projectPoints(
            pts3d, cMo_rvec, cMo_tvec, self.K, self.dist)

        for i, (x, y) in enumerate(centers):
            if inlier_mask is not None and i < len(inlier_mask):
                color = (0, 255, 0) if inlier_mask[i] else (0, 0, 255)
            else:
                color = (255, 255, 0)
            cv2.circle(vis_img, (int(x), int(y)), 3, color, 1)
            x_proj, y_proj = proj[i][0]
            cv2.drawMarker(vis_img, (int(x_proj), int(y_proj)), 
                           (255, 0, 0), cv2.MARKER_CROSS, 5, 1)
            cv2.line(vis_img, (int(x), int(y)), 
                     (int(x_proj), int(y_proj)), (0, 255, 0), 1)
            cv2.putText(vis_img, str(i), (int(x)-8, int(y)-8), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        pad = 50
        roi_y_min_clamped = max(0, roi_y_min - pad)
        roi_y_max_clamped = min(vis_img.shape[0], roi_y_max + pad)
        roi_x_min_clamped = max(0, roi_x_min - pad)
        roi_x_max_clamped = min(vis_img.shape[1], roi_x_max + pad)
        vis_result = vis_img[roi_y_min_clamped:roi_y_max_clamped,
                              roi_x_min_clamped:roi_x_max_clamped]
        vis_result = cv2.resize(vis_result, None, fx=2, fy=2,
                                 interpolation=cv2.INTER_NEAREST)
        vis_result_path = os.path.join(
            self.result_dir, f"frame_{frame_id:06d}_vis_result.png")
        cv2.imwrite(vis_result_path, vis_result)


    def _should_reject_pose_diff(self, cMo, robot_pose):
        if self.last_bMo is None:
            return False

        cMo_before = np.linalg.inv(self.eMc) @ np.linalg.inv(robot_pose) @ self.last_bMo
        tvec_before = cMo_before[:3, 3]
        pos_diff = float(np.linalg.norm(cMo[:3, 3] - tvec_before))

        rot_mat_diff = cMo[:3, :3] @ cMo_before[:3, :3].T
        rot_vec_diff = Rotation.from_matrix(rot_mat_diff).as_rotvec()
        rot_diff = float(np.degrees(np.linalg.norm(rot_vec_diff)))

        if pos_diff > self.cfg.tracker.max_translation or\
            rot_diff > self.cfg.tracker.max_rotation_deg:
            self.logger.warning(f"Rejected large pose diff: \
                pos_diff={pos_diff:.1f}mm, rot_diff={rot_diff:.1f}deg"
            )
            return True
        return False

        
            






