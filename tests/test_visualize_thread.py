import os
import threading
import numpy as np
from pathlib import Path
from queue import Queue

from core.packet import FramePacket
from visualization.visualize_thread import VisualizeThread
from config.visualization_config import VisualizationThreadConfig, VizCameraConfig
from pose_estimator.pose_estimator import PoseEstimatorResult, OptimizerResult
from utils.pnp_utils import PnPResult


def make_dummy_packet(frame_id=1, robot_pose=None, pnp_rvec=None, pnp_tvec=None):
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    packet = FramePacket(
        frame_id=frame_id,
        timestamp=frame_id,
        image=img,
    )

    packet.refined_pts2d = np.array([
        [100 + frame_id * 2, 100 + frame_id * 3],
        [200 - frame_id * 2, 100 + frame_id],
        [150, 200 - frame_id * 5],
        [120 + frame_id, 220 - frame_id],
        [180 - frame_id, 220 + frame_id * 2],
        [90 + frame_id * 4, 300 - frame_id * 2],
        [210 - frame_id, 300 + frame_id],
    ], dtype=np.float32)
    packet.robot_pose = np.eye(4) if robot_pose is None else robot_pose

    pnp = PnPResult(
        valid=True,
        rvec=np.zeros(3) if pnp_rvec is None else pnp_rvec,
        tvec=np.array([0, 0, 500]) if pnp_tvec is None else pnp_tvec,
        inlier_mask=np.ones(7, dtype=bool),
        reproj_errs=np.zeros(7),
        avg_reproj_err=0.0,
        round1_reproj_errs=np.zeros(7),
        used_threshold=1.0,
    )

    opt_rotation = np.eye(4)
    opt_translation = np.eye(4)
    opt_translation[0, 3] = frame_id * 5
    opt_translation[1, 3] = frame_id * -3
    opt_translation[2, 3] = 500 + frame_id * 10
    opt = OptimizerResult(
        bMo=opt_translation,
        cMo=opt_translation,
        reproj_errs=np.zeros(7),
        avg_reproj_err=0.0,
    )
    pose_res = PoseEstimatorResult(valid=True, reason="", optimized=opt, pnp=pnp)
    packet.pose_est_result = pose_res
    return packet


def test_visualize_thread_saves_csv(tmp_path: Path):
    q = Queue()
    stop_event = threading.Event()
    result_dir = tmp_path / "visualization"
    result_dir.mkdir()

    camera_cfg = VizCameraConfig(
        K=np.eye(3),
        dist=np.zeros(5),
        eMc=np.eye(4),
    )
    cfg = VisualizationThreadConfig(result_dir=str(result_dir), camera=camera_cfg)
    vt = VisualizeThread(q, stop_event, cfg)
    vt.start()

    for frame_id in range(1, 5):
        pose = np.eye(4)
        pose[0, 3] = frame_id * 10
        pose[1, 3] = frame_id * -5
        pose[2, 3] = frame_id * 2
        pnp_rvec = np.array([0.0, 0.0, 0.01 * frame_id])
        pnp_tvec = np.array([0.0, 0.0, 500.0 + frame_id * 20])
        q.put(make_dummy_packet(frame_id, robot_pose=pose, pnp_rvec=pnp_rvec, pnp_tvec=pnp_tvec))

    eof = make_dummy_packet(99)
    eof.eof = True
    q.put(eof)

    vt.join(timeout=5)

    expected_files = [
        result_dir / 'pnp_results.csv',
        result_dir / 'optimize_results.csv',
        result_dir / 'robot_pose_translation_vs_frame.png',
        result_dir / 'robot_pose_euler_vs_frame.png',
        result_dir / 'cMo_comparison_tvec_vs_frame.png',
        result_dir / 'cMo_comparison_euler_vs_frame.png',
        result_dir / 'bMo_comparison_tvec_vs_frame.png',
        result_dir / 'bMo_comparison_euler_vs_frame.png',
    ]

    for path in expected_files:
        assert path.exists(), f"Missing expected output: {path}"
