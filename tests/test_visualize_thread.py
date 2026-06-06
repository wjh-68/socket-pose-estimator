import sys
import os
import tempfile
import time
import numpy as np
from queue import Queue

# ensure project root is importable for tests
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.packet import FramePacket
from visualization.visualize_thread import VisualizeThread
from config.visualization_config import VisualizationThreadConfig
from pose_estimator.pose_estimator import PoseEstimatorResult, OptimizerResult
from utils.pnp_utils import PnPResult


def make_dummy_packet(frame_id=1):
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    packet = FramePacket(
        frame_id=frame_id,
        timestamp=0,
        image=img,
    )

    # fake refined points
    packet.refined_pts2d = np.array([[100, 100], [200, 100], [150, 200], [120, 220], [180, 220], [90, 300], [210, 300]])
    # fake robot pose
    packet.robot_pose = np.eye(4)

    pnp = PnPResult(valid=True, rvec=np.zeros(3), tvec=np.array([0,0,500]), inlier_mask=np.ones(7, dtype=bool), reproj_errs=np.zeros(7), avg_reproj_err=0.0, round1_reproj_errs=np.zeros(7), used_threshold=1.0)

    opt = OptimizerResult(bMo=np.eye(4), cMo=np.eye(4), reproj_errs=np.zeros(7), avg_reproj_err=0.0)
    pose_res = PoseEstimatorResult(valid=True, reason="", optimized=opt, pnp=pnp)
    packet.pose_est_result = pose_res
    return packet


def test_visualize_thread_saves_csv():
    q = Queue()
    stop_event = type('E', (), {'is_set': lambda self=False: False})()
    tmpdir = tempfile.mkdtemp()
    cfg = VisualizationThreadConfig(result_dir=tmpdir)
    vt = VisualizeThread(q, stop_event, cfg)
    vt.start()

    pkt = make_dummy_packet(1)
    q.put(pkt)
    eof = make_dummy_packet(2)
    eof.eof = True
    q.put(eof)

    # wait for thread to process
    vt.join(timeout=5)

    pnp_csv = os.path.join(tmpdir, 'pnp_results.csv')
    opt_csv = os.path.join(tmpdir, 'optimize_results.csv')
    assert os.path.exists(pnp_csv)
    assert os.path.exists(opt_csv)
