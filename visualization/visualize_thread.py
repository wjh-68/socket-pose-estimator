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
            rec.update({
                'bMo_optimized': None,
                'cMo_optimized': None,
                'cMo_pnp': None,
                'bMo_pnp': None,
                'frame_error': None,
                'avg_error': None,
            })
            return rec

        rec['bMo_optimized'] = opt.bMo.tolist() if getattr(opt, 'bMo', None) is not None else None
        rec['cMo_optimized'] = opt.cMo.tolist() if getattr(opt, 'cMo', None) is not None else None
        rec['frame_error'] = float(np.mean(opt.reproj_errs)) if getattr(opt, 'reproj_errs', None) is not None else None
        rec['avg_error'] = float(getattr(opt, 'avg_reproj_err', None) or rec['frame_error'])

        pnp = res.pnp
        rec['cMo_pnp'] = None
        rec['bMo_pnp'] = None
        if pnp is not None and getattr(pnp, 'rvec', None) is not None and getattr(pnp, 'tvec', None) is not None:
            try:
                cMo_pnp = rvec_tvec_to_transform(pnp.rvec, pnp.tvec)
                rec['cMo_pnp'] = cMo_pnp.tolist()
                if getattr(packet, 'robot_pose', None) is not None and getattr(self.cfg, 'camera', None) is not None:
                    eMc = self.cfg.camera.eMc
                    bMo_pnp = packet.robot_pose @ eMc @ cMo_pnp
                    rec['bMo_pnp'] = bMo_pnp.tolist()
            except Exception:
                rec['cMo_pnp'] = None
                rec['bMo_pnp'] = None

        return rec

    def _pose_components_dict(self, prefix: str, pose):
        if pose is None:
            return {
                f'{prefix}_euler': None,
                f'{prefix}_tvec': None,
            }

        euler, tvec = self._pose_to_components(pose)
        return {
            f'{prefix}_euler': euler.tolist() if euler is not None else None,
            f'{prefix}_tvec': tvec.tolist() if tvec is not None else None,
        }

    def _pose_to_components(self, pose):
        if pose is None:
            return None, None
        try:
            pose_arr = np.asarray(pose, dtype=float)
            euler, tvec = pose_to_euler_tvec(pose_arr)
            return euler, tvec
        except Exception:
            return None, None

    def _build_frame_record(self, packet: FramePacket, opt_rec: dict):
        frame_rec = {'frame_id': int(packet.frame_id)}
        frame_rec['robot_pose'] = packet.robot_pose.tolist() if getattr(packet, 'robot_pose', None) is not None else None
        frame_rec.update(self._pose_components_dict('robot_pose', packet.robot_pose))
        frame_rec.update(self._pose_components_dict('cMo_optimized', opt_rec.get('cMo_optimized')))
        frame_rec.update(self._pose_components_dict('cMo_pnp', opt_rec.get('cMo_pnp')))
        frame_rec.update(self._pose_components_dict('bMo_optimized', opt_rec.get('bMo_optimized')))
        frame_rec.update(self._pose_components_dict('bMo_pnp', opt_rec.get('bMo_pnp')))
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
        if not self.frame_records:
            return

        try:
            frames = [r['frame_id'] for r in self.frame_records]
            self._plot_transform_history(frames, self.frame_records, 'robot_pose', 'Robot base pose', 'robot_pose')
            self._plot_comparison_history(
                frames,
                self.frame_records,
                'cMo',
                ['cMo_optimized', 'cMo_pnp'],
                'Camera pose',
                'cMo_comparison',
            )
            self._plot_comparison_history(
                frames,
                self.frame_records,
                'bMo',
                ['bMo_optimized', 'bMo_pnp'],
                'Body pose',
                'bMo_comparison',
            )
        except Exception:
            self.logger.exception("Failed to save plots")

    def _plot_transform_history(self, frames, records, prefix, title_prefix, filename_prefix):
        translation_series = self._extract_vector_series(records, f'{prefix}_tvec')
        euler_series = self._extract_vector_series(records, f'{prefix}_euler')

        if any(v is not None for v in translation_series.values()):
            self._save_three_series_plot(
                frames,
                translation_series,
                f'{title_prefix} translation over frames',
                os.path.join(self.result_dir, f'{filename_prefix}_translation_vs_frame.png'),
                'mm',
            )

        if any(v is not None for v in euler_series.values()):
            self._save_three_series_plot(
                frames,
                euler_series,
                f'{title_prefix} Euler angles over frames',
                os.path.join(self.result_dir, f'{filename_prefix}_euler_vs_frame.png'),
                'deg',
            )

    def _plot_comparison_history(self, frames, records, base_prefix, compare_prefixes, title_prefix, filename_prefix):
        for suffix, label in [('tvec', 'translation'), ('euler', 'Euler angles')]:
            series_map = {}
            for compare_prefix in compare_prefixes:
                key = f'{compare_prefix}_{suffix}'
                series_map[compare_prefix] = self._extract_vector_series(records, key)

            output_path = os.path.join(self.result_dir, f'{filename_prefix}_{suffix}_vs_frame.png')
            title = f'{title_prefix} {label} over frames'
            self._save_comparison_plot(frames, series_map, title, output_path, 'mm' if suffix == 'tvec' else 'deg')

    def _save_comparison_plot(self, frames, series_map, title, output_path, y_label):
        fig, axes = plt.subplots(3, 1, figsize=(8, 8))
        styles = {
            'cMo_optimized': {'color': 'r', 'label': 'optimized'},
            'cMo_pnp': {'color': 'g', 'label': 'pnp'},
            'bMo_optimized': {'color': 'r', 'label': 'optimized'},
            'bMo_pnp': {'color': 'g', 'label': 'pnp'},
        }

        for axis_idx, axis_name in enumerate(['x', 'y', 'z']):
            ax = axes[axis_idx]
            for series_name, series_values in series_map.items():
                ax.plot(
                    frames,
                    series_values[axis_name],
                    marker='o',
                    linestyle='-',
                    markerfacecolor='none',
                    color=styles[series_name]['color'],
                    label=styles[series_name]['label'],
                )
            ax.set_title(f'{title} ({axis_name.upper()})')
            ax.set_xlabel('frame_id')
            ax.set_ylabel(y_label)
            ax.grid(True)
            if axis_idx == 0:
                ax.legend()

        plt.tight_layout()
        plt.savefig(output_path)
        plt.close(fig)

    def _extract_vector_series(self, records, key):
        series = {'x': [], 'y': [], 'z': []}
        for rec in records:
            values = rec.get(key)
            if values is None:
                values = [np.nan, np.nan, np.nan]
            series['x'].append(float(values[0]) if values[0] is not None else np.nan)
            series['y'].append(float(values[1]) if values[1] is not None else np.nan)
            series['z'].append(float(values[2]) if values[2] is not None else np.nan)
        return series

    def _save_three_series_plot(self, frames, series, title, output_path, y_label):
        fig, axes = plt.subplots(3, 1, figsize=(8, 8))
        colors = {'x': 'r-', 'y': 'g-', 'z': 'b-'}
        labels = {'x': 'X', 'y': 'Y', 'z': 'Z'}

        for ax, axis in zip(axes, ['x', 'y', 'z']):
            ax.plot(frames, series[axis], colors[axis])
            ax.set_title(f'{title} ({labels[axis]})')
            ax.set_xlabel('frame_id')
            ax.set_ylabel(y_label)
            ax.grid(True)

        plt.tight_layout()
        plt.savefig(output_path)
        plt.close(fig)

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
