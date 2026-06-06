import threading
import time
import queue
import os
from collections import deque
from core.logger import get_logger
from core.packet import FramePacket
from config.visualization_config import VisualizationThreadConfig
from core.queues import put_latest
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2

from utils.pnp_utils import rvec_tvec_to_transform, transform_to_rvec_tvec
from static_pose_optimizer_ba import pose_to_euler_tvec


class VisualizeThread(threading.Thread):
    def __init__(self, in_q: queue.Queue, stop_event: threading.Event, cfg: VisualizationThreadConfig):
        super().__init__(name="VisualizeThread", daemon=True)
        self.in_q = in_q
        self.stop_event = stop_event
        self.cfg = cfg
        self.queue_cfg = cfg.queue_config
        self.result_dir = cfg.result_dir
        os.makedirs(self.result_dir, exist_ok=True)
        self.logger = get_logger("visualize_thread")

        # Records collected for CSV / plotting
        self.pnp_records = []
        self.optimize_records = []
        self.frame_records = []

    def _put_packet(self, packet: FramePacket):
        if self.queue_cfg.drop_oldest:
            put_latest(self.in_q, packet, self.logger)
        else:
            # visualizer is a sink so normally we don't put packets back
            pass

    def _validate_packet(self, packet: FramePacket):
        if packet is None:
            raise ValueError("packet is None")
        if packet.image is None:
            raise ValueError("packet.image is None")
        if packet.pose_est_result is None:
            raise ValueError("pose_est_result is None")

    def _build_pnp_record(self, packet: FramePacket):
        res = packet.pose_est_result
        pnp = res.pnp
        rec = {'frame_id': int(packet.frame_id)}
        rec['n_points'] = int(packet.refined_pts2d.shape[0]) if getattr(packet, 'refined_pts2d', None) is not None else None
        if pnp is None:
            rec.update({'n_inliers': None, 'used_threshold': None, 'pnp_error_round1': None, 'pnp_error_final': None})
            return rec

        rec['n_inliers'] = int(np.sum(pnp.inlier_mask)) if getattr(pnp, 'inlier_mask', None) is not None else None
        rec['used_threshold'] = float(pnp.used_threshold) if getattr(pnp, 'used_threshold', None) is not None else None
        rec['pnp_error_round1'] = float(np.mean(pnp.round1_reproj_errs)) if getattr(pnp, 'round1_reproj_errs', None) is not None else None
        rec['pnp_error_final'] = float(getattr(pnp, 'avg_reproj_err', None) or (np.mean(getattr(pnp, 'reproj_errs', [])) if getattr(pnp, 'reproj_errs', None) is not None else None))
        return rec

    def _build_opt_record(self, packet: FramePacket):
        res = packet.pose_est_result
        opt = res.optimized
        rec = {'frame_id': int(packet.frame_id)}
        if opt is None:
            rec.update({'bMo_optimized': None, 'cMo_optimized': None, 'frame_error': None, 'avg_error': None, 'bMo_init': None})
            return rec

        rec['bMo_optimized'] = opt.bMo.tolist() if getattr(opt, 'bMo', None) is not None else None
        rec['cMo_optimized'] = opt.cMo.tolist() if getattr(opt, 'cMo', None) is not None else None
        rec['frame_error'] = float(np.mean(opt.reproj_errs)) if getattr(opt, 'reproj_errs', None) is not None else None
        rec['avg_error'] = float(getattr(opt, 'avg_reproj_err', None) or rec['frame_error'])

        # try compute bMo_init if pnp and robot_pose available and camera eMc provided
        try:
            pnp = res.pnp
            if pnp is not None and getattr(pnp, 'rvec', None) is not None and getattr(packet, 'robot_pose', None) is not None and getattr(self.cfg, 'camera', None) is not None:
                eMc = self.cfg.camera.eMc
                cMo_pnp = rvec_tvec_to_transform(pnp.rvec, pnp.tvec)
                bMo_init = packet.robot_pose @ eMc @ cMo_pnp
                rec['bMo_init'] = bMo_init.tolist()
        except Exception:
            rec['bMo_init'] = None

        return rec

    def _build_frame_record(self, packet: FramePacket, opt_rec: dict):
        frame_rec = {'frame_id': int(packet.frame_id)}
        frame_rec['robot_pose'] = packet.robot_pose.tolist() if getattr(packet, 'robot_pose', None) is not None else None
        frame_rec['bMo_optimized'] = opt_rec.get('bMo_optimized')
        frame_rec['bMo_init'] = opt_rec.get('bMo_init')
        frame_rec['cMo'] = opt_rec.get('cMo_optimized')
        return frame_rec

    def _render_and_save_image(self, packet: FramePacket, opt_rec: dict):
        vis_img = packet.image.copy()
        # draw refined points
        pts = getattr(packet, 'refined_pts2d', None)
        if pts is not None:
            for p in pts:
                cv2.circle(vis_img, (int(p[0]), int(p[1])), 3, (0, 255, 0), -1)

        # draw axes if camera and cMo available
        try:
            if getattr(self.cfg, 'camera', None) is not None and opt_rec.get('cMo_optimized') is not None:
                cMo = np.array(opt_rec['cMo_optimized'])
                K = self.cfg.camera.K
                dist = self.cfg.camera.dist
                cMo_rvec, cMo_tvec = transform_to_rvec_tvec(cMo)
                cv2.drawFrameAxes(vis_img, K, dist, cMo[:3, :3], cMo[:3, 3:], 10, 3)
        except Exception:
            self.logger.debug('skip drawing axes')

        vis_path = os.path.join(self.result_dir, f"frame_{int(packet.frame_id):06d}_vis.png")
        cv2.imwrite(vis_path, vis_img)

    def _process_packet(self, packet: FramePacket):
        self._validate_packet(packet)
        pnp_rec = self._build_pnp_record(packet)
        opt_rec = self._build_opt_record(packet)
        frame_rec = self._build_frame_record(packet, opt_rec)

        self.pnp_records.append(pnp_rec)
        self.optimize_records.append(opt_rec)
        self.frame_records.append(frame_rec)

        try:
            self._render_and_save_image(packet, opt_rec)
        except Exception:
            self.logger.exception("Failed to render/save visualization image")

        return packet

    def _save_csv_records(self):
        # Save pnp
        if self.pnp_records:
            pnp_df = pd.DataFrame(self.pnp_records)
            pnp_csv_path = os.path.join(self.result_dir, "pnp_results.csv")
            pnp_df.to_csv(pnp_csv_path, index=False)
            self.logger.info(f"Saved pnp CSV to {pnp_csv_path}")

        if self.optimize_records:
            opt_df = pd.DataFrame(self.optimize_records)
            opt_csv_path = os.path.join(self.result_dir, "optimize_results.csv")
            opt_df.to_csv(opt_csv_path, index=False)
            self.logger.info(f"Saved optimize CSV to {opt_csv_path}")

    def _save_plots(self):
        # Minimal plotting similar to chargeport implementation
        if not self.frame_records:
            return

        try:
            frames = [r['frame_id'] for r in self.frame_records]
            # translations
            fig, axes = plt.subplots(3, 1, figsize=(8, 8))
            ax = axes[0]
            ax.plot(frames, [r['robot_pose'][0][3] for r in self.frame_records], 'r-')
            ax.set_title('Robot X')
            ax = axes[1]
            ax.plot(frames, [r['robot_pose'][1][3] for r in self.frame_records], 'g-')
            ax.set_title('Robot Y')
            ax = axes[2]
            ax.plot(frames, [r['robot_pose'][2][3] for r in self.frame_records], 'b-')
            ax.set_title('Robot Z')
            plt.tight_layout()
            plt.savefig(os.path.join(self.result_dir, 'robot_pose_translation_vs_frame.png'))
            plt.close()
        except Exception:
            self.logger.exception("Failed to save plots")

    def run(self):
        while not self.stop_event.is_set():
            try:
                packet = self.in_q.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                if packet is None:
                    self.logger.warning("received None packet, skipping")
                    continue

                if getattr(packet, 'eof', False):
                    # save results and exit
                    self._save_csv_records()
                    self._save_plots()
                    self.logger.info("received EOF packet, visualizer exiting")
                    break

                packet = self._process_packet(packet)

            except Exception:
                self.logger.exception("Unexpected exception in VisualizeThread")
            finally:
                try:
                    self.in_q.task_done()
                except Exception:
                    pass

        self.logger.info("VisualizeThread exited cleanly")
