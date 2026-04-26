#!/usr/bin/env python3
"""
Detection Pipeline Visualization Script
Visualizes each step of the detection pipeline for testing algorithm effectiveness.
- YOLO ROI detection
- Initial ellipse detection
- Used vs discarded ellipses in clean_and_classify
- clean_and_classify filtered ellipses
- gap_method_threshold classified ellipses
"""

import os
import cv2
import numpy as np
from ultralytics import YOLO
from scipy.spatial.transform import Rotation

# ============ Config ============
DATA_DIR = "dataset/savedata4"
DEBUG_DIR = "debug_pipeline"

# Camera parameters (from eval_pose_offline_optimal_v2.py)
K = np.array([
    [1015.445938660267, 0., 638.51741890470555],
    [0., 1015.445938660267, 386.838616473841],
    [0., 0., 1.]
], dtype=np.float64)

dist = np.array([
    0.11753195467413819, -0.19301774104640848,
    0.00016793575097772418, -0.00061144051421409198, 0.072260521199194336
], dtype=np.float64)

# ============ Helper Functions (copied from eval_pose_offline_optimal_v2.py) ============

def init_edge_drawing():
    Params = cv2.ximgproc.EdgeDrawing.Params()
    ed = cv2.ximgproc.createEdgeDrawing()
    Params.EdgeDetectionOperator = 1
    Params.MinPathLength = 45
    Params.PFmode = 0
    Params.NFAValidation = True
    Params.GradientThresholdValue = 30
    ed.setParams(Params)
    return ed

def detect_ellipses(ed, roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    ed.detectEdges(gray)
    ellipses = ed.detectEllipses()
    if ellipses is None:
        return []
    result = []
    for e in ellipses:
        if e[0][2] == 0:
            result.append([e[0][0], e[0][1], e[0][3], e[0][4], e[0][5]])
        else:
            result.append((e[0][0], e[0][1], e[0][2], e[0][2], 0))
    return result

def yolo_detect_roi(model, img, conf=0.5):
    results = model(img)
    if len(results) == 0:
        return None
    boxes = results[0].boxes.xyxy.cpu().numpy()
    if len(boxes) == 0:
        return None
    return boxes[0]

def compute_ellipse_metrics(e):
    a = float(e[2])
    b = float(e[3])
    ratio = a / b if b != 0 else 999.0
    size = a + b
    if a > 0 and b > 0:
        area = np.pi * a * b
        perimeter = np.pi * (3 * (a + b) - np.sqrt((3 * a + b) * (a + 3 * b)))
        circularity = 4 * np.pi * area / (perimeter * perimeter)
    else:
        circularity = 0.0
    return {'ratio': ratio, 'size': size, 'circularity': circularity}

class SocketPoseEstimator:
    def __init__(self):
        self.obj_pts = np.array([
            [-8.0, 11.2, 0.0], [8.0, 11.2, 0.0],
            [-16.0, 0.0, 0.0], [0.0, 0.0, 0.0], [16.0, 0.0, 0.0],
            [-8.0, -13.9, 0.0], [8.0, -13.9, 0.0]
        ], dtype=np.float64)
        self.tmp_types = [0, 0, 1, 1, 1, 1, 1]

    def _clean_and_classify(self, ellipses, dist_thresh=10, cluster_thresh=80):
        if not ellipses:
            return [], [], []

        shape_passed = []
        shape_rejected = []
        for i, e in enumerate(ellipses):
            metrics = compute_ellipse_metrics(e)
            entry = {
                'idx': i,
                'ellipse': e,
                'center': np.array([e[0], e[1]]),
                'ratio': metrics['ratio'],
                'size': metrics['size'],
                'circularity': metrics['circularity'],
            }
            if 0.9 < entry['ratio'] < 1.1 and entry['size'] < 80:
                shape_passed.append(entry)
            else:
                shape_rejected.append(entry)

        nodes = [
            {'idxs': [entry['idx']], 'c': entry['center'], 'd': entry['size']}
            for entry in shape_passed
        ]

        merged = []
        used = [False] * len(nodes)
        for i in range(len(nodes)):
            if used[i]:
                continue
            cluster = [nodes[i]]
            used[i] = True
            for j in range(i + 1, len(nodes)):
                if not used[j] and np.linalg.norm(nodes[i]['c'] - nodes[j]['c']) < dist_thresh:
                    cluster.append(nodes[j])
                    used[j] = True
            avg_c = np.mean([n['c'] for n in cluster], axis=0)
            max_d = max([n['d'] for n in cluster])
            all_idxs = [idx for n in cluster for idx in n['idxs']]
            merged.append({'idxs': all_idxs, 'p': avg_c, 'size': max_d, 'is_double': len(cluster) >= 2})

        merged = [m for m in merged if 10 < m['size'] < 100]
        if not merged:
            return [], [], shape_rejected + shape_passed

        # 聚类 小插孔和大插孔，舍弃
        # groups = []
        # clustered = [False] * len(merged)
        # for i in range(len(merged)):
        #     if clustered[i]:
        #         continue
        #     group = [i]
        #     clustered[i] = True
        #     queue = [i]
        #     while queue:
        #         idx = queue.pop()
        #         for j in range(len(merged)):
        #             if not clustered[j] and np.linalg.norm(merged[idx]['p'] - merged[j]['p']) < cluster_thresh:
        #                 clustered[j] = True
        #                 queue.append(j)
        #                 group.append(j)
        #     groups.append(group)

        # best_group = max(groups, key=lambda g: (len(g), sum(merged[i]['size'] for i in g)))
        kept = []
        kept_indices = []

        for m in merged:
            idxs = m['idxs']

            # 取出对应 ellipse entry
            entries = [e for e in shape_passed if e['idx'] in idxs]

            if len(entries) <= 2:
                # 1个或2个，直接保留
                kept.append(m)
                kept_indices.extend([e['idx'] for e in entries])
                continue

            # >=3 个，做筛选（关键逻辑）
            centers = np.array([e['center'] for e in entries])

            # 方法1（推荐）：选两两距离最小的一对
            min_dist = float('inf')
            best_pair = None

            for i in range(len(centers)):
                for j in range(i + 1, len(centers)):
                    d = np.linalg.norm(centers[i] - centers[j])
                    if d < min_dist:
                        min_dist = d
                        best_pair = (entries[i], entries[j])

            # 保留这两个
            selected_entries = list(best_pair)

            # 重新构造 merged node（更新 center/size）
            avg_c = np.mean([e['center'] for e in selected_entries], axis=0)
            max_d = max([e['size'] for e in selected_entries])

            kept.append({
                'idxs': [e['idx'] for e in selected_entries],
                'p': avg_c,
                'size': max_d,
                'is_double': True
            })

            kept_indices.extend([e['idx'] for e in selected_entries])

        used_ellipses = [entry for entry in shape_passed if entry['idx'] in kept_indices]
        discarded_ellipses = shape_rejected + [entry for entry in shape_passed if entry['idx'] not in kept_indices]
        return kept, used_ellipses, discarded_ellipses

    def _gap_method_threshold(self, candidates):
        candidates.sort(key=lambda x: x['size'])
        sizes = [c['size'] for c in candidates]
        gaps = [sizes[i + 1] / sizes[i] for i in range(len(sizes) - 1)]
        split_idx = np.argmax(gaps)
        threshold = (sizes[split_idx] + sizes[split_idx + 1]) / 2
        for c in candidates:
            c['t'] = 1 if c['size'] > threshold else 0

# ============ Visualization Functions ============

def draw_original_image(img, title="Original Image"):
    """Draw original image with title"""
    vis = img.copy()
    cv2.putText(vis, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return vis

def draw_yolo_roi(img, roi_box, title="YOLO ROI Detection"):
    """Draw YOLO detected ROI"""
    vis = img.copy()
    if roi_box is not None:
        x1, y1, x2, y2 = roi_box
        cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
        cv2.putText(vis, "ROI", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(vis, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return vis

def draw_roi_image(roi, title="ROI Region"):
    """Draw the cropped ROI region"""
    vis = roi.copy()
    cv2.putText(vis, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return vis

def draw_initial_ellipses(roi, ellipses, title="Initial Ellipse Detection"):
    """Draw initial detected ellipses"""
    vis = roi.copy()
    for i, e in enumerate(ellipses):
        if e[2] == 0:  # Circle
            center = (int(e[0]), int(e[1]))
            radius = int(e[3])
            cv2.circle(vis, center, radius, (255, 0, 0), 2)
            cv2.putText(vis, f"{i}", center, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        else:  # Ellipse
            center = (int(e[0]), int(e[1]))
            axes = (int(e[2]), int(e[3]))
            angle = int(e[4]) if len(e) > 4 else 0
            cv2.ellipse(vis, center, axes, angle, 0, 360, (255, 0, 0), 2)
            cv2.putText(vis, f"{i}", center, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.putText(vis, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return vis

def draw_used_ellipses(roi, ellipses, used_ellipses, discarded_ellipses, title="Used vs Discarded Ellipses"):
    """Draw ellipses showing which ones are used (green) and discarded (red)"""
    vis = roi.copy()
    
    # Draw all ellipses first with gray
    for i, e in enumerate(ellipses):
        if e[2] == 0:  # Circle
            center = (int(e[0]), int(e[1]))
            radius = int(e[3])
            cv2.circle(vis, center, radius, (128, 128, 128), 1)
        else:  # Ellipse
            center = (int(e[0]), int(e[1]))
            axes = (int(e[2]), int(e[3]))
            angle = int(e[4]) if len(e) > 4 else 0
            cv2.ellipse(vis, center, axes, angle, 0, 360, (128, 128, 128), 1)
    
    # Draw used ellipses in green
    for entry in used_ellipses:
        idx = entry['idx']
        e = entry['ellipse']
        if e[2] == 0:  # Circle
            center = (int(e[0]), int(e[1]))
            radius = int(e[3])
            cv2.circle(vis, center, radius, (0, 255, 0), 2)
            cv2.putText(vis, f"U{idx}", center, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        else:  # Ellipse
            center = (int(e[0]), int(e[1]))
            axes = (int(e[2]), int(e[3]))
            angle = int(e[4]) if len(e) > 4 else 0
            cv2.ellipse(vis, center, axes, angle, 0, 360, (0, 255, 0), 2)
            cv2.putText(vis, f"U{idx}", center, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    # Draw discarded ellipses in red
    for entry in discarded_ellipses:
        idx = entry['idx']
        e = entry['ellipse']
        if e[2] == 0:  # Circle
            center = (int(e[0]), int(e[1]))
            radius = int(e[3])
            cv2.circle(vis, center, radius, (0, 0, 255), 2)
            cv2.putText(vis, f"D{idx}", center, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        else:  # Ellipse
            center = (int(e[0]), int(e[1]))
            axes = (int(e[2]), int(e[3]))
            angle = int(e[4]) if len(e) > 4 else 0
            cv2.ellipse(vis, center, axes, angle, 0, 360, (0, 0, 255), 2)
            cv2.putText(vis, f"D{idx}", center, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    cv2.putText(vis, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(vis, f"Used: {len(used_ellipses)} | Discarded: {len(discarded_ellipses)}", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    return vis

def format_ellipse_stats(entry):
    e = entry['ellipse']
    center = entry['center']
    return (f"idx={entry['idx']:02d} center=({center[0]:.1f},{center[1]:.1f}) "
            f"ratio={entry['ratio']:.3f} size={entry['size']:.1f} "
            f"circ={entry['circularity']:.3f}")


def print_and_save_ellipse_stats(frame_dir, used_ellipses, discarded_ellipses):
    lines = ["Used ellipses:"]
    for entry in used_ellipses:
        lines.append(format_ellipse_stats(entry))
    lines.append("")
    lines.append("Discarded ellipses:")
    for entry in discarded_ellipses:
        lines.append(format_ellipse_stats(entry))
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(frame_dir, "05_used_discarded_stats.txt"), "w", encoding="utf-8") as f:
        f.write(text)
    return text


def draw_cleaned_candidates(roi, candidates, title="After clean_and_classify"):
    """Draw candidates after clean_and_classify"""
    vis = roi.copy()
    colors = [(0, 255, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0),
              (255, 0, 0), (0, 165, 255), (128, 0, 128)]
    for i, c in enumerate(candidates):
        color = colors[i % len(colors)]
        center = (int(c['p'][0]), int(c['p'][1]))
        cv2.circle(vis, center, 8, color, 2)
        cv2.putText(vis, f"{i}", (center[0] + 10, center[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
        cv2.putText(vis, f"sz:{c['size']:.0f}", (center[0] + 10, center[1] + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    cv2.putText(vis, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return vis

def draw_classified_candidates(roi, candidates, title="After gap_method_threshold"):
    """Draw candidates after gap_method_threshold classification"""
    vis = roi.copy()
    for i, c in enumerate(candidates):
        t = c.get('t', -1)
        if t == 0:  # Large
            color = (0, 255, 0)  # Green
            label = "L"
        elif t == 1:  # Small
            color = (0, 0, 255)  # Red
            label = "S"
        else:
            color = (128, 128, 128)  # Gray
            label = "?"
        center = (int(c['p'][0]), int(c['p'][1]))
        cv2.circle(vis, center, 8, color, 2)
        cv2.putText(vis, f"{i}({label})", (center[0] + 10, center[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
        cv2.putText(vis, f"sz:{c['size']:.0f}", (center[0] + 10, center[1] + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    cv2.putText(vis, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return vis

# ============ Main Function ============

def main():
    # Create debug directory
    os.makedirs(DEBUG_DIR, exist_ok=True)

    # Initialize components
    print("Initializing YOLO model...")
    model = YOLO("checkpoint/best.pt")

    print("Initializing EdgeDrawing...")
    ed = init_edge_drawing()

    print("Initializing SocketPoseEstimator...")
    estimator = SocketPoseEstimator()

    # Get all PNG files
    png_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.png')])
    print(f"Found {len(png_files)} PNG files to process")

    for frame_id, png_file in enumerate(png_files):
        print(f"\nProcessing frame {frame_id}: {png_file}")

        # Load image
        img_path = os.path.join(DATA_DIR, png_file)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to load image: {img_path}")
            continue

        # Create frame directory
        frame_dir = os.path.join(DEBUG_DIR, f"frame_{frame_id:04d}")
        os.makedirs(frame_dir, exist_ok=True)

        # Step 1: Original image
        vis1 = draw_original_image(img)
        cv2.imwrite(os.path.join(frame_dir, "01_original.png"), vis1)

        # Step 2: YOLO ROI detection
        roi_box = yolo_detect_roi(model, img)
        vis2 = draw_yolo_roi(img, roi_box)
        cv2.imwrite(os.path.join(frame_dir, "02_yolo_roi.png"), vis2)

        if roi_box is None:
            print("No ROI detected, skipping ellipse detection")
            continue

        # Extract ROI
        roi = img[int(roi_box[1]):int(roi_box[3]), int(roi_box[0]):int(roi_box[2])]

        # Step 3: ROI region
        vis3 = draw_roi_image(roi)
        cv2.imwrite(os.path.join(frame_dir, "03_roi_region.png"), vis3)

        # Step 4: Initial ellipse detection
        ellipses = detect_ellipses(ed, roi)
        vis4 = draw_initial_ellipses(roi, ellipses)
        cv2.imwrite(os.path.join(frame_dir, "04_initial_ellipses.png"), vis4)

        # Step 5: Used vs Discarded ellipses in clean_and_classify
        candidates, used_ellipses, discarded_ellipses = estimator._clean_and_classify(ellipses)
        vis5 = draw_used_ellipses(roi, ellipses, used_ellipses, discarded_ellipses)
        cv2.imwrite(os.path.join(frame_dir, "05_used_discarded_ellipses.png"), vis5)
        print_and_save_ellipse_stats(frame_dir, used_ellipses, discarded_ellipses)

        # Step 6: After clean_and_classify
        vis6 = draw_cleaned_candidates(roi, candidates)
        cv2.imwrite(os.path.join(frame_dir, "06_cleaned_candidates.png"), vis6)

        # Step 7: After gap_method_threshold
        estimator._gap_method_threshold(candidates)
        vis7 = draw_classified_candidates(roi, candidates)
        cv2.imwrite(os.path.join(frame_dir, "07_classified_candidates.png"), vis7)

        print(f"Saved visualization for frame {frame_id}")

    print(f"\nProcessing complete! Results saved to: {DEBUG_DIR}")

if __name__ == '__main__':
    main()