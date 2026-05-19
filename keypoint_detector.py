import os
import json
import numpy as np
import cv2
import time
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO

# try:
#     os.add_dll_directory(r'D:\SDK\opencv4.7.0_Library\x64\vc17\bin')
#     import pycylinderedsf as pyced
#     HAS_PYCED = True
# except (ImportError, OSError):
#     HAS_PYCED = False
HAS_PYCED = False

# ==============================================================================
# 3D 物体坐标模板 (单位: mm)
# ==============================================================================

Super_KEYPOINT_OBJ_PTS = np.array([
    [-12.0, -21.0, 17.5],   # kp0: S-
    [0.0, -21.0, 12.5],     # kp1: CC2
    [12.0, -21.0, 17.5],    # kp2: S+
    [0.0, -9.0, 17.5],      # kp3: CC1
    [-17.0, 1.2, 7.5],      # kp4: DC-
    [17.0, 1.2, 7.5],       # kp5: DC+
    [-14.25, 21.0, 17.5],   # kp6: A-
    [0.0, 21.0, 7.5],       # kp7: D
    [14.25, 21.0, 17.5],    # kp8: A+
], dtype=np.float32)

Slow_KEYPOINT_OBJ_PTS = np.array([
    [-8.0, -11.2, 7.5],     # tp0: CP
    [8.0, -11.2, 7.5],      # tp1: CC
    [-25.5, 0.0, 7.5],      # tp2: N
    [0.0, 0.0, 7.5],        # tp3: D
    [25.5, 0.0, 7.5],       # tp4: L1
    [-8.0, 13.9, 7.5],      # tp5: L3
    [8.0, 13.9, 7.5],       # tp6: L2
], dtype=np.float32)

# ==============================================================================
# 配置字典
# ==============================================================================

_CAMERA_CONFIGS = {
    'small': {
        'K': np.array([
            [1015.445938660267, 0., 638.51741890470555],
            [0., 1015.445938660267, 386.838616473841],
            [0., 0., 1.]
        ], dtype=np.float64),
        'dist': np.array([
            0.11753195467413819, -0.19301774104640848,
            0.00016793575097772418, -0.00061144051421409198,
            0.072260521199194336
        ], dtype=np.float64),
        'M_cam2end': np.array([
            [-6.9855857e-01,  7.1512282e-01,  2.4804471e-02, -5.1826664e+01],
            [-7.1555281e-01, -6.9815123e-01, -2.3854841e-02,  5.5274796e+01],
            [ 2.5813223e-04, -3.4412913e-02,  9.9940765e-01,  9.5362617e+01],
            [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  1.0000000e+00]
        ], dtype=np.float64),
    },
    'large': {
        'K': np.array([
            [2674.7629874104787, 0., 1279.5],
            [0., 2674.7629874104787, 719.5],
            [0., 0., 1.]
        ], dtype=np.float64),
        'dist': np.array([
            -0.11744968686298927, 0.27089153364253454,
            0.0012180578884344092, 0.00067320963008635703,
            -0.078845410108757258
        ], dtype=np.float64),
        'M_cam2end': np.array([
            [-7.2267956e-01,  6.9102561e-01, -1.4759262e-02, -5.1758522e+01],
            [-6.9116789e-01, -7.2264087e-01,  8.7790741e-03,  6.0040222e+01],
            [-4.5990809e-03,  1.6545586e-02,  9.9985254e-01,  9.7955963e+01],
            [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  1.0000000e+00]
        ], dtype=np.float64),
    },
}

_PORT_CONFIGS = {
    'super': {
        'model_path': './superport_yolo_Pose.pt',
        'keypoint_obj_pts': Super_KEYPOINT_OBJ_PTS,
    },
    'slow': {
        'model_path': './slowport_yolo_Pose.pt',
        'keypoint_obj_pts': Slow_KEYPOINT_OBJ_PTS,
    },
}


class KeypointDetector:
    """充电口关键点检测 + 椭圆拟合 + PnP 位姿估计"""

    def __init__(self, port_type='super', camera='small', detect_method='ed',
                 model_path=None, K=None, dist=None, M_cam2end=None,
                 verbose=False, visual=True):
        """
        初始化关键点检测器，自动加载模型、相机参数和3D模板。

        根据标志位组合自动选择：
        - port_type → YOLO模型 + 3D物体坐标模板
        - camera → 相机内参K、畸变系数dist、手眼标定M_cam2end
        - detect_method → 椭圆检测算法
        以上均可通过可选参数手动覆盖。

        参数:
            port_type: str, 端口类型
                'super': 快充9孔，加载 superport_yolo_Pose.pt + Super_KEYPOINT_OBJ_PTS(9点)
                'slow':  慢充7孔，加载 slowport_yolo_Pose.pt + Slow_KEYPOINT_OBJ_PTS(7点)
            camera: str, 相机类型
                'small': 小相机 1280x720，内参 fx=fy=1015.44
                'large': 大相机 2560x1440，内参 fx=fy=2674.76
            detect_method: str, 椭圆检测方法
                'ed':    使用 OpenCV EdgeDrawing（默认，无需额外安装）
                'pyced': 使用 pycylinderedsf CED（需编译安装 EDSF，未安装时自动回退 'ed'）
            model_path: str, 可选，覆盖自动选择的 YOLO 模型路径
            K: np.ndarray (3,3), 可选，覆盖自动选择的相机内参矩阵
            dist: np.ndarray (5,), 可选，覆盖自动选择的畸变系数
            M_cam2end: np.ndarray (4,4), 可选，覆盖自动选择的手眼标定矩阵（相机→末端）
            verbose: bool, 是否打印详细调试信息，默认 False
            visual: bool, 是否启用可视化绘制，默认 True
        """
        if port_type not in _PORT_CONFIGS:
            raise ValueError(f"port_type 需为 'super' 或 'slow'，当前: '{port_type}'")
        if camera not in _CAMERA_CONFIGS:
            raise ValueError(f"camera 需为 'small' 或 'large'，当前: '{camera}'")

        self.port_type = port_type
        self.camera = camera
        self.verbose = verbose
        self.visual = visual

        # 模型
        _model_path = model_path or _PORT_CONFIGS[port_type]['model_path']
        self.model = YOLO(_model_path)

        # 3D 物体点模板
        self.keypoint_obj_pts = _PORT_CONFIGS[port_type]['keypoint_obj_pts']

        # 相机参数
        cam_cfg = _CAMERA_CONFIGS[camera]
        self.K = K if K is not None else cam_cfg['K'].copy()
        self.dist = dist if dist is not None else cam_cfg['dist'].copy()
        self.M_cam2end = M_cam2end if M_cam2end is not None else cam_cfg['M_cam2end'].copy()

        # 检测方法
        if detect_method == 'pyced' and not HAS_PYCED:
            print("警告: pycylinderedsf 未安装，回退为 'ed'")
            detect_method = 'ed'
        self.detect_method = detect_method

        # EdgeDrawing 检测器（复用，避免重复创建）
        self._ed_detector = cv2.ximgproc.createEdgeDrawing()
        self._ed_params = cv2.ximgproc.EdgeDrawing.Params()
        self._ed_params.EdgeDetectionOperator = 0
        self._ed_params.MinPathLength = 10
        self._ed_params.PFmode = True
        self._ed_params.NFAValidation = True
        self._ed_params.GradientThresholdValue = 20
        self._ed_detector.setParams(self._ed_params)
        self._ed_gradient_threshold = 20

        # 中间结果（每次 process_frame 重置）
        self._reset_state()

    def _reset_state(self):
        """重置所有中间结果属性，在每次 process_frame 开始时调用。"""
        self._frame = None
        self._gray = None
        self._yolo_results = None
        self._current_points = None
        self._best_idx = None
        self._all_centers = None
        self._all_ellipses = None
        self._neighbors = None
        self._min_axis_ratio = 0.9
        self._concentric_results = None
        self._pnp_result = None

    # ==========================================================================
    # 核心流程
    # ==========================================================================

    def process_frame(self, frame, detect_method=None, conf_threshold=0.6, coplanar=False):
        """
        对单帧图像执行完整的关键点检测 + 椭圆拟合 + PnP 流程。

        流程:
            1. YOLO推理 → 提取关键点和检测框
            2. 选择关键点最多的目标
            3. 在目标框内进行椭圆检测
            4. 过滤椭圆（直径过大、长短轴比过小的剔除）
            5. 邻近匹配：关键点 ↔ 椭圆中心（自适应阈值 + 离群点过滤 + 去面积最大）
            6. 同心圆拟合：每个关键点取最小面积椭圆中心作为圆心
            7. solvePnP：RANSAC → ITERATIVE → 离群点剔除 → LM精化

        注意: 此方法只计算不绘制，中间结果存储在 self._* 属性中，
              可通过 draw_* 方法或 get_annotated_frame() 获取可视化。

        参数:
            frame: np.ndarray, BGR图像
            detect_method: str, 可选覆盖椭圆检测方法，None则使用初始化时的设置
            conf_threshold: float, YOLO关键点置信度阈值，默认0.6
            coplanar: bool, 关键点3D坐标是否近似共面，默认False
                True:  使用 RANSAC(ITERATIVE) + IPPE 求解
                False: 使用 RANSAC(ITERATIVE) + ITERATIVE 求解

        返回:
            pnp_result: dict 或 None
                成功时包含: rvec, tvec, R, reproj_error, per_point_errors, valid_indices, num_inliers
                失败时为 None
        """
        self._reset_state()
        self._frame = frame
        self._gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        method = detect_method or self.detect_method
        if method == 'pyced' and not HAS_PYCED:
            method = 'ed'

        # 1. YOLO 关键点检测
        results = self.model(frame, verbose=False)
        self._yolo_results = results
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            if self.verbose: print("  未检测到目标物")
            return None

        # 2. 选择关键点最多的目标
        detected_points = self.get_success_keypoints(results, conf_threshold)
        best_idx, best_num = 0, 0
        for i, pts in enumerate(detected_points):
            if len(pts) > best_num:
                best_num = len(pts)
                best_idx = i
        if best_num < 4:
            if self.verbose: print(f"  关键点不足: {best_num}")
            return None

        self._best_idx = best_idx
        self._current_points = detected_points[best_idx]

        # 3. 在目标框内进行椭圆检测
        boxes_xyxy = boxes.xyxy.cpu().numpy()
        target_box = boxes_xyxy[best_idx:best_idx+1]
        box_w = target_box[0][2] - target_box[0][0]
        box_h = target_box[0][3] - target_box[0][1]
        max_diameter = max(box_w, box_h) / 2

        all_centers, all_ellipses = self.detect_ellipse_centers_in_roi(
            frame, target_box, method=method,
            ed_detector=self._ed_detector, ed_params=self._ed_params,
            cached_gradient_threshold=self._ed_gradient_threshold)
        self._ed_gradient_threshold = self._ed_params.GradientThresholdValue

        # 4. 过滤椭圆：直径过大 或 长短轴比过小
        min_axis_ratio = self._min_axis_ratio
        for i in range(len(all_ellipses)):
            filtered_ellipses = []
            filtered_centers = []
            for j, (cx, cy, a, b, angle) in enumerate(all_ellipses[i]):
                diameter = max(a, b) * 2
                if diameter >= max_diameter:
                    continue
                if min(a, b) / max(a, b) < min_axis_ratio:
                    continue
                filtered_ellipses.append((cx, cy, a, b, angle))
                filtered_centers.append([cx, cy])
            all_ellipses[i] = filtered_ellipses
            all_centers[i] = np.array(filtered_centers) if filtered_centers else np.empty((0, 2))

        self._all_centers = all_centers
        self._all_ellipses = all_ellipses

        # 5. 邻近匹配 + 同心圆拟合
        concentric_results = {}
        for i, (centers, ellipses) in enumerate(zip(all_centers, all_ellipses)):
            if len(centers) == 0:
                continue

            neighbors = self.get_keypoint_neighbors_adaptive(
                [self._current_points], centers, ellipses, scale_factor=0.8)
            neighbors[0] = self.filter_outliers_by_clustering(neighbors[0], std_multiplier=2.0)

            # 邻近点超过3个时去除面积最大的
            for kp_idx in list(neighbors[0].keys()):
                nearby = neighbors[0][kp_idx]
                if len(nearby) >= 3:
                    areas = []
                    for ecx, ecy in nearby:
                        area = 0
                        for (ex, ey, ea, eb, _) in ellipses:
                            if abs(ex - ecx) < 1.5 and abs(ey - ecy) < 1.5:
                                area = ea * eb
                                break
                        areas.append(area)
                    max_area_idx = areas.index(max(areas))
                    neighbors[0][kp_idx].pop(max_area_idx)

            self._neighbors = neighbors

            if self.verbose:
                print("  关键点-椭圆中心匹配结果:")
                for kp_idx, kp_x, kp_y in self._current_points:
                    nearby = neighbors[0].get(kp_idx, [])
                    print(f"    关键点 {kp_idx} ({kp_x:.4f},{kp_y:.4f}): {len(nearby)} 个邻近点")
                    for n_idx, (ncx, ncy) in enumerate(nearby):
                        kp_dist = np.sqrt((ncx - kp_x) ** 2 + (ncy - kp_y) ** 2)
                        print(f"      邻近点 {n_idx}: ({ncx:.4f},{ncy:.4f}), 距离={kp_dist:.4f}")

            for kp_idx, kp_x, kp_y in self._current_points:
                nearby = neighbors[0].get(kp_idx, [])
                if len(nearby) >= 2:
                    best_area = float('inf')
                    concentric_center = None
                    for ecx, ecy in nearby:
                        for (ex, ey, ea, eb, _) in ellipses:
                            if abs(ex - ecx) < 1.5 and abs(ey - ecy) < 1.5:
                                if min(ea, eb) / max(ea, eb) < min_axis_ratio:
                                    continue
                                area = ea * eb
                                if area < best_area:
                                    best_area = area
                                    concentric_center = np.array([ecx, ecy])
                                break
                    if concentric_center is None:
                        concentric_center = np.array(nearby[0])
                elif len(nearby) == 1:
                    concentric_center = np.array(nearby[0])
                else:
                    concentric_center = None

                concentric_results[kp_idx] = {
                    'concentric_center': concentric_center.tolist() if concentric_center is not None else None,
                    'raw_nearby_count': len(nearby),
                    'raw_nearby': nearby
                }

        self._concentric_results = concentric_results

        # 6. solvePnP
        pnp_result = None
        if concentric_results:
            obj_pts_list = []
            img_pts_list = []
            valid_indices = []

            for kp_idx in sorted(concentric_results.keys()):
                cc = concentric_results[kp_idx]['concentric_center']
                if cc is not None and kp_idx < len(self.keypoint_obj_pts):
                    obj_pts_list.append(self.keypoint_obj_pts[kp_idx])
                    img_pts_list.append(cc)
                    valid_indices.append(kp_idx)

            if len(obj_pts_list) >= 4:
                obj_pts = np.array(obj_pts_list, dtype=np.float32)
                img_pts = np.array(img_pts_list, dtype=np.float32)

                if self.verbose:
                    print(f"  DEBUG: K type={type(self.K)}, dtype={self.K.dtype}, shape={self.K.shape}")
                    print(f"  DEBUG: obj_pts dtype={obj_pts.dtype}, shape={obj_pts.shape}")
                    print(f"  DEBUG: img_pts dtype={img_pts.dtype}, shape={img_pts.shape}")

                success = False
                inliers = None
                if coplanar:
                    print("solvePnPRansac (ITERATIVE for filtering)....")
                    success, rvec, tvec, inliers = cv2.solvePnPRansac(
                        obj_pts, img_pts, self.K, self.dist,
                        useExtrinsicGuess=False, reprojectionError=10.0, confidence=0.99,
                        flags=cv2.SOLVEPNP_ITERATIVE)
                    print("inliers:", len(inliers) if inliers is not None else 0)
                    if success and inliers is not None and len(inliers) >= 4:
                        print("solvePnP (IPPE refine)....")
                        inlier_mask = inliers.flatten()
                        success, rvec, tvec = cv2.solvePnP(
                            obj_pts[inlier_mask], img_pts[inlier_mask], self.K, self.dist,
                            flags=cv2.SOLVEPNP_IPPE, rvec=rvec, tvec=tvec)
                    elif not success or inliers is None or len(inliers) < 4:
                        print("RANSAC failed, fallback to IPPE with all points....")
                        success, rvec, tvec = cv2.solvePnP(
                            obj_pts, img_pts, self.K, self.dist,
                            flags=cv2.SOLVEPNP_IPPE)
                else:
                    print("solvePnPRansac....")
                    success, rvec, tvec, inliers = cv2.solvePnPRansac(
                        obj_pts, img_pts, self.K, self.dist,
                        useExtrinsicGuess=False, reprojectionError=3.0, confidence=0.99,
                        flags=cv2.SOLVEPNP_ITERATIVE)
                    print("inliers:", len(inliers) if inliers is not None else 0)
                    if success and inliers is not None and len(inliers) >= 6:
                        print("solvePnPRansac success, solvePnP....")
                        inlier_mask = inliers.flatten()
                        success, rvec, tvec = cv2.solvePnP(
                            obj_pts[inlier_mask], img_pts[inlier_mask], self.K, self.dist,
                            flags=cv2.SOLVEPNP_ITERATIVE, rvec=rvec, tvec=tvec)

                if success:
                    # 识别并剔除误匹配点
                    proj_pts, _ = cv2.projectPoints(obj_pts, rvec, tvec, self.K, self.dist)
                    proj_pts = proj_pts.reshape(-1, 2)
                    per_point_errors = np.linalg.norm(img_pts - proj_pts, axis=1)
                    median_err = np.median(per_point_errors)
                    mad = np.median(np.abs(per_point_errors - median_err))
                    outlier_threshold = median_err + 3.0 * max(mad, 1.0)
                    good_mask = per_point_errors <= outlier_threshold

                    if self.verbose:
                        print(f"  初始误差: median={median_err:.2f}, MAD={mad:.2f}, 阈值={outlier_threshold:.2f}")
                        for j, kp_idx in enumerate(valid_indices):
                            tag = "OUTLIER" if not good_mask[j] else ""
                            print(f"    kp{kp_idx}: {per_point_errors[j]:.2f}px {tag}")

                    # 用好点重新求解 + LM 精化
                    if np.sum(good_mask) >= 4 and np.sum(~good_mask) > 0:
                        good_obj = obj_pts[good_mask]
                        good_img = img_pts[good_mask]
                        print(f"  剔除{np.sum(~good_mask)}个离群点后重解+LM精化....")
                        success2, rvec2, tvec2 = cv2.solvePnP(
                            good_obj, good_img, self.K, self.dist,
                            flags=cv2.SOLVEPNP_ITERATIVE)
                        if success2:
                            rvec2, tvec2 = cv2.solvePnPRefineLM(good_obj, good_img, self.K, self.dist, rvec2, tvec2)
                            proj2, _ = cv2.projectPoints(obj_pts, rvec2, tvec2, self.K, self.dist)
                            proj2 = proj2.reshape(-1, 2)
                            err2 = np.linalg.norm(img_pts - proj2, axis=1)
                            if err2[good_mask].mean() < per_point_errors[good_mask].mean():
                                rvec, tvec = rvec2, tvec2
                                if self.verbose:
                                    print(f"  剔除后好点误差: {per_point_errors[good_mask].mean():.2f} -> {err2[good_mask].mean():.2f}px")
                        else:
                            print("剔除后,cv2.solvePnP失败...")
                    else:
                        print("solvePnPRefineLM....")
                        rvec, tvec = cv2.solvePnPRefineLM(obj_pts, img_pts, self.K, self.dist, rvec, tvec)

                    R, _ = cv2.Rodrigues(rvec)
                    proj_pts, _ = cv2.projectPoints(obj_pts, rvec, tvec, self.K, self.dist)
                    proj_pts = proj_pts.reshape(-1, 2)
                    per_point_errors = np.linalg.norm(img_pts - proj_pts, axis=1)
                    reproj_error = per_point_errors.mean()

                    if self.verbose:
                        print("  逐点重投影误差:")
                        for j, kp_idx in enumerate(valid_indices):
                            print(f"    kp{kp_idx}: {per_point_errors[j]:.2f}px")

                    pnp_result = {
                        'rvec': rvec, 'tvec': tvec, 'R': R,
                        'reproj_error': reproj_error,
                        'per_point_errors': {str(kp_idx): round(float(per_point_errors[j]), 4)
                                             for j, kp_idx in enumerate(valid_indices)},
                        'valid_indices': valid_indices,
                        'num_inliers': len(inliers) if inliers is not None else 0
                    }

                    if self.verbose:
                        print(f"  PnP: rvec={rvec.flatten()}, tvec={tvec.flatten()}, reproj={reproj_error:.2f}px")

        self._pnp_result = pnp_result
        return pnp_result


    # ==========================================================================
    # 可视化方法（读取存储结果，不重新计算）
    # ==========================================================================

    def draw_yolo_keypoints(self, image=None):
        """
        在图像上绘制 YOLO 检测框和关键点。

        绘制内容:
            - 绿色矩形: YOLO检测框
            - 蓝色实心圆: 关键点位置
            - 蓝色文字: 关键点索引

        读取: self._yolo_results（不重新计算）

        参数:
            image: np.ndarray, 可选，在指定图像上绘制；None则使用 self._frame

        返回:
            np.ndarray: 绘制后的图像
        """
        if image is None:
            image = self._frame.copy()
        if self._yolo_results is None:
            return image

        for result in self._yolo_results:
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy().astype(int)
                for box in boxes:
                    x1, y1, x2, y2 = box
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            if result.keypoints is not None:
                kpts_xy = result.keypoints.xy.cpu().numpy()
                kpts_conf = result.keypoints.conf
                if kpts_conf is not None:
                    kpts_conf = kpts_conf.cpu().numpy()
                for i in range(len(kpts_xy)):
                    person_kps = kpts_xy[i]
                    has_conf = kpts_conf is not None
                    for kp_idx in range(len(person_kps)):
                        kp_x, kp_y = int(person_kps[kp_idx][0]), int(person_kps[kp_idx][1])
                        if has_conf:
                            if kpts_conf[i][kp_idx] <= 0.5:
                                continue
                        else:
                            if kp_x == 0 and kp_y == 0:
                                continue
                        cv2.circle(image, (kp_x, kp_y), 5, (255, 0, 0), -1)
                        cv2.putText(image, str(kp_idx), (kp_x + 5, kp_y - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        return image

    def draw_ellipses(self, image=None):
        """
        绘制椭圆中心点和轮廓，区分匹配/未匹配状态。

        绘制内容:
            - 黄色实心圆: 所有椭圆中心
            - 红色轮廓(粗2px): 被关键点匹配到的椭圆
            - 绿色轮廓(细1px): 未被匹配的椭圆

        读取: self._all_ellipses, self._all_centers, self._neighbors（不重新计算）

        参数:
            image: np.ndarray, 可选，在指定图像上绘制；None则使用 self._frame

        返回:
            np.ndarray: 绘制后的图像
        """
        if image is None:
            image = self._frame.copy()
        if self._all_ellipses is None:
            return image

        matched_ellipse_set = set()
        if self._neighbors is not None and self._current_points is not None:
            for kp_idx, kp_x, kp_y in self._current_points:
                nearby = self._neighbors[0].get(kp_idx, [])
                for ecx, ecy in nearby:
                    matched_ellipse_set.add((round(ecx, 1), round(ecy, 1)))

        for i, (centers, ellipses) in enumerate(zip(self._all_centers, self._all_ellipses)):
            for cx, cy, a, b, angle in ellipses:
                key = (round(cx, 1), round(cy, 1))
                cv2.circle(image, (int(cx), int(cy)), 2, (0, 255, 255), -1)
                if key in matched_ellipse_set:
                    cv2.ellipse(image, (int(cx), int(cy)),
                                (int(a), int(b)), int(angle), 0, 360, (0, 0, 255), 2)
                else:
                    cv2.ellipse(image, (int(cx), int(cy)),
                                (int(a), int(b)), int(angle), 0, 360, (0, 255, 0), 1)
        return image

    def draw_concentric_centers(self, image=None):
        """
        绘制同心圆拟合后的圆心位置，用"+"标记。

        绘制内容:
            - 黑色"+"标记: 同心圆拟合中心
            - 黑色文字: 'cc_{kp_idx}' 标注关键点索引

        读取: self._concentric_results（不重新计算）

        参数:
            image: np.ndarray, 可选，在指定图像上绘制；None则使用 self._frame

        返回:
            np.ndarray: 绘制后的图像
        """
        if image is None:
            image = self._frame.copy()
        if self._concentric_results is None:
            return image

        for kp_idx in sorted(self._concentric_results.keys()):
            cc = self._concentric_results[kp_idx]['concentric_center']
            if cc is None:
                continue
            cx_, cy_ = int(cc[0]), int(cc[1])
            cv2.line(image, (cx_ - 6, cy_), (cx_ + 6, cy_), (0, 0, 0), 2)
            cv2.line(image, (cx_, cy_ - 6), (cx_, cy_ + 6), (0, 0, 0), 2)
            cv2.putText(image, f'cc_{kp_idx}', (cx_ + 8, cy_ - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        return image

    def draw_match_lines(self, image=None):
        """
        绘制关键点到匹配椭圆中心的连线。

        绘制内容:
            - 红色线段: 关键点 → 匹配的椭圆中心
            - 红色实心圆: 匹配的椭圆中心
            - 红色轮廓: 匹配到的椭圆轮廓
            - 自动跳过扁椭圆（轴比低于 min_axis_ratio）

        读取: self._neighbors, self._current_points, self._all_ellipses, self._all_centers
        注意: 此方法直接读取已存储的匹配结果，不重新调用 get_keypoint_neighbors_adaptive，
              消除了原 getKeypoints.py 中的重复计算问题。

        参数:
            image: np.ndarray, 可选，在指定图像上绘制；None则使用 self._frame

        返回:
            np.ndarray: 绘制后的图像
        """
        if image is None:
            image = self._frame.copy()
        if self._neighbors is None or self._current_points is None:
            return image
        if self._all_ellipses is None or self._all_centers is None:
            return image

        for i, (centers, ellipses) in enumerate(zip(self._all_centers, self._all_ellipses)):
            if len(centers) == 0:
                continue
            centers_list = centers.tolist() if isinstance(centers, np.ndarray) else centers
            center_to_idx = {(round(cx, 1), round(cy, 1)): idx
                             for idx, (cx, cy) in enumerate(centers_list)}

            for kp_idx, kp_x, kp_y in self._current_points:
                nearby_centers = self._neighbors[0].get(kp_idx, [])
                for ecx, ecy in nearby_centers:
                    key = (round(ecx, 1), round(ecy, 1))
                    # 跳过扁椭圆
                    if key in center_to_idx:
                        _, _, a_e, b_e, _ = ellipses[center_to_idx[key]]
                        if min(a_e, b_e) / max(a_e, b_e) < self._min_axis_ratio:
                            continue
                    cv2.line(image, (int(kp_x), int(kp_y)),
                             (int(ecx), int(ecy)), (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.circle(image, (int(ecx), int(ecy)), 2, (0, 0, 255), 1)
                    if key in center_to_idx:
                        e_idx = center_to_idx[key]
                        _, _, a_e, b_e, angle_e = ellipses[e_idx]
                        cv2.ellipse(image, (int(ecx), int(ecy)),
                                    (int(a_e), int(b_e)), int(angle_e), 0, 360, (0, 0, 255), 1)
        return image

    def draw_pose_axes(self, image=None, length=30):
        """
        绘制 PnP 求解得到的坐标轴（红-X, 绿-Y, 蓝-Z）。

        读取: self._pnp_result, self.K, self.dist（不重新计算）

        参数:
            image: np.ndarray, 可选，在指定图像上绘制；None则使用 self._frame
            length: int, 坐标轴长度（像素），默认30

        返回:
            np.ndarray: 绘制后的图像
        """
        if image is None:
            image = self._frame.copy()
        if self._pnp_result is None:
            return image
        cv2.drawFrameAxes(image, self.K, self.dist,
                          self._pnp_result['rvec'], self._pnp_result['tvec'], length, 3)
        return image

    def draw_all(self, image=None):
        """
        依次调用所有绘制方法，返回完整标注图像。

        绘制顺序: YOLO关键点 → 椭圆 → 同心圆中心 → 匹配连线 → 坐标轴

        参数:
            image: np.ndarray, 可选，在指定图像上绘制；None则使用 self._frame

        返回:
            np.ndarray: 完整标注后的图像
        """
        image = self.draw_yolo_keypoints(image)
        image = self.draw_ellipses(image)
        image = self.draw_concentric_centers(image)
        image = self.draw_match_lines(image)
        image = self.draw_pose_axes(image)
        return image

    def get_annotated_frame(self):
        """
        根据 visual 标志返回标注图像或原图副本。

        返回:
            np.ndarray 或 None
                visual=True: 返回 draw_all() 的结果
                visual=False: 返回 self._frame 的副本
                未处理过帧时: 返回 None
        """
        if self.visual and self._frame is not None:
            return self.draw_all()
        return self._frame.copy() if self._frame is not None else None

    # ==========================================================================
    # 批处理方法
    # ==========================================================================

    def batch_process_to_base(self, img_dir, save_dir=None, coplanar=False):
        """
        遍历图像文件夹及同名 .npy 文件，计算物体在基坐标系下的位姿。

        流程:
            1. 读取每张图像及同名 .npy 文件（末端→基座变换矩阵 M_end2base）
            2. 调用 process_frame 计算相机坐标系下的 PnP 位姿
            3. 通过链式变换 T_obj2base = M_end2base @ M_cam2end @ T_obj2cam 得到基坐标系位姿
            4. 计算相邻帧间的旋转/平移误差（衡量位姿一致性）
            5. 保存位姿结果 (base_poses.json) 和误差结果 (base_errors.json)

        参数:
            img_dir: str, 图像目录路径。目录中每张图像需有同名 .npy 文件，
                     存放 4x4 末端→基座变换矩阵 M_end2base
            save_dir: str, 结果保存目录，默认为 img_dir/base_output
            coplanar: bool, 是否使用共面 PnP 求解，默认 False

        返回:
            all_results: dict, 每张图的位姿结果 {filename: {...} 或 None}
            errors: dict, 相邻帧误差 {filename: {rotation_error_deg, translation_error_mm, ...}}
        """
        M_cam2end = self.M_cam2end

        if save_dir is None:
            save_dir = os.path.join(img_dir, 'base_output')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        img_files = sorted([f for f in os.listdir(img_dir)
                            if f.lower().endswith(('.jpg', '.png', '.bmp'))])
        print(f"共 {len(img_files)} 张图像")

        all_results = {}

        for cnt, fname in enumerate(img_files):
            img_path = os.path.join(img_dir, fname)
            npy_path = os.path.join(img_dir, os.path.splitext(fname)[0] + '.npy')

            frame = cv2.imread(img_path)
            if frame is None:
                print(f"  [{cnt+1}] 跳过无法读取: {fname}")
                continue

            if not os.path.exists(npy_path):
                print(f"  [{cnt+1}] {fname}: 未找到 .npy，跳过")
                continue
            M_end2base = np.load(npy_path)
            if M_end2base.shape != (4, 4):
                print(f"  [{cnt+1}] {fname}: .npy 不是 4x4，跳过")
                continue

            t0 = time.perf_counter()
            pnp_result = self.process_frame(frame, coplanar=coplanar)
            elapsed = (time.perf_counter() - t0) * 1000

            if self.visual:
                vis_img = self.get_annotated_frame()
                cv2.imwrite(os.path.join(save_dir, fname), vis_img)

            if pnp_result is None:
                print(f"  [{cnt+1}] {fname}: PnP 失败, {elapsed:.0f}ms")
                all_results[fname] = None
                continue

            # T_obj2cam
            R, _ = cv2.Rodrigues(pnp_result['rvec'])
            t = pnp_result['tvec'].flatten()
            T_obj2cam = np.eye(4)
            T_obj2cam[:3, :3] = R
            T_obj2cam[:3, 3] = t

            # T_obj2base
            T_obj2base = M_end2base @ M_cam2end @ T_obj2cam
            R_base = T_obj2base[:3, :3]
            t_base = T_obj2base[:3, 3]
            rvec_base, _ = cv2.Rodrigues(R_base)

            all_results[fname] = {
                'rvec': pnp_result['rvec'].flatten().tolist(),
                'tvec': pnp_result['tvec'].flatten().tolist(),
                'reproj_error': float(pnp_result['reproj_error']),
                'per_point_errors': pnp_result.get('per_point_errors', {}),
                'T_obj2base': T_obj2base.tolist(),
            }

            print(f"  [{cnt+1}] {fname}: reproj={pnp_result['reproj_error']:.2f}px, "
                  f"t_base=[{t_base[0]:.1f},{t_base[1]:.1f},{t_base[2]:.1f}], {elapsed:.0f}ms")

        # 保存位姿
        poses_path = os.path.join(save_dir, 'base_poses.json')
        with open(poses_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"基坐标系位姿已保存到: {poses_path}")

        # 计算相邻帧误差
        fnames = sorted([k for k in all_results if all_results[k] is not None])
        errors = {}
        for i in range(1, len(fnames)):
            T_prev = np.array(all_results[fnames[i - 1]]['T_obj2base'], dtype=np.float64)
            T_curr = np.array(all_results[fnames[i]]['T_obj2base'], dtype=np.float64)
            T_rel = np.linalg.inv(T_prev) @ T_curr
            R_rel = T_rel[:3, :3]
            t_rel = T_rel[:3, 3]

            sy = np.sqrt(R_rel[0, 0]**2 + R_rel[1, 0]**2)
            if sy >= 1e-6:
                roll  = np.arctan2(R_rel[2, 1], R_rel[2, 2])
                pitch = np.arctan2(-R_rel[2, 0], sy)
                yaw   = np.arctan2(R_rel[1, 0], R_rel[0, 0])
            else:
                roll  = np.arctan2(-R_rel[1, 2], R_rel[1, 1])
                pitch = np.arctan2(-R_rel[2, 0], sy)
                yaw   = 0.0

            rot_total = np.degrees(np.linalg.norm([roll, pitch, yaw]))
            trans_error = np.linalg.norm(t_rel)

            errors[fnames[i]] = {
                'rotation_error_deg': round(rot_total, 6),
                'rotation_roll_deg': round(np.degrees(roll), 6),
                'rotation_pitch_deg': round(np.degrees(pitch), 6),
                'rotation_yaw_deg': round(np.degrees(yaw), 6),
                'translation_error_mm': round(trans_error, 6),
                'reference_frame': fnames[i - 1],
            }

        errors_path = os.path.join(save_dir, 'base_errors.json')
        with open(errors_path, 'w', encoding='utf-8') as f:
            json.dump(errors, f, indent=2, ensure_ascii=False)
        print(f"误差结果已保存到: {errors_path}")

        if errors:
            rot_vals = [v['rotation_error_deg'] for v in errors.values()]
            trans_vals = [v['translation_error_mm'] for v in errors.values()]
            print(f"\n共 {len(errors)} 对相邻帧:")
            print(f"  旋转误差: mean={np.mean(rot_vals):.4f}°, max={np.max(rot_vals):.4f}°")
            print(f"  平移误差: mean={np.mean(trans_vals):.4f}mm, max={np.max(trans_vals):.4f}mm")

        return all_results, errors

    def batch_detect_and_save(self, img_dir, save_dir=None):
        """
        批量检测图像中同心圆的圆心坐标，保存为 .txt 文件。

        对每张图像执行完整的关键点检测 + 椭圆拟合 + 同心圆拟合流程，
        将每个关键点对应的同心圆中心坐标写入同名 .txt 文件。
        检测失败的关键点坐标记为 "-1 -1"。

        输出格式: 每行一个关键点，坐标以空格分隔，顺序与 keypoint_obj_pts 一致。
        例如: "123.4567 234.5678 -1 -1 345.6789 456.7890 ..."

        参数:
            img_dir: str, 输入图像目录路径
            save_dir: str, 输出 .txt 文件保存目录，默认为 img_dir/txt_output
        """
        if save_dir is None:
            save_dir = os.path.join(img_dir, 'txt_output')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        img_files = sorted([f for f in os.listdir(img_dir)
                            if f.lower().endswith(('.jpg', '.png', '.bmp'))])
        print(f"共 {len(img_files)} 张图像")
        success_count = 0

        for cnt, fname in enumerate(img_files):
            img_path = os.path.join(img_dir, fname)
            frame = cv2.imread(img_path)
            if frame is None:
                continue

            self.process_frame(frame)
            if self._concentric_results is None:
                continue

            num_keypoints = len(self.keypoint_obj_pts)
            coords = []
            for kp_idx in range(num_keypoints):
                cc_data = self._concentric_results.get(kp_idx, {})
                cc = cc_data.get('concentric_center', None)
                if cc is not None:
                    coords.append(f"{cc[0]:.4f} {cc[1]:.4f}")
                else:
                    coords.append("-1 -1")

            basename = os.path.splitext(fname)[0]
            txt_path = os.path.join(save_dir, basename + '.txt')
            with open(txt_path, 'w') as f:
                f.write(' '.join(coords))

            success_count += 1
            matched = sum(1 for v in self._concentric_results.values() if v.get('concentric_center') is not None)
            print(f"  [{cnt+1}] {fname}: 拟合 {matched}/{num_keypoints} 个圆心 -> {basename}.txt")

        print(f"\n完成: {success_count}/{len(img_files)} 张图像成功")

    def run_on_images(self, img_dir, save_dir=None):
        """
        批量处理图像集，逐张显示标注结果并可选保存。

        对每张图像调用 process_frame 计算 PnP 位姿，
        通过 get_annotated_frame 获取可视化结果并在窗口中显示。
        按 ESC 键可提前终止，按其他键继续下一张。

        参数:
            img_dir: str, 输入图像目录路径
            save_dir: str, 可选，标注图像和 PnP 结果保存目录。
                      若提供则保存标注图像和 pnp_results.json

        返回:
            all_results: dict, 每张图的 PnP 结果 {filename: {rvec, tvec, R, reproj_error} 或 None}
        """
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)

        img_files = sorted([f for f in os.listdir(img_dir)
                            if f.lower().endswith(('.jpg', '.png', '.bmp'))])
        all_results = {}

        for cnt, fname in enumerate(img_files):
            frame = cv2.imread(os.path.join(img_dir, fname))
            if frame is None:
                continue

            t0 = time.perf_counter()
            pnp_result = self.process_frame(frame)
            elapsed = (time.perf_counter() - t0) * 1000

            vis_img = self.get_annotated_frame()

            if pnp_result:
                all_results[fname] = {
                    'rvec': pnp_result['rvec'].flatten().tolist(),
                    'tvec': pnp_result['tvec'].flatten().tolist(),
                    'R': pnp_result['R'].tolist(),
                    'reproj_error': float(pnp_result['reproj_error']),
                }
                print(f"[{cnt+1}] {fname}: OK, reproj={pnp_result['reproj_error']:.2f}px, {elapsed:.0f}ms")
            else:
                all_results[fname] = None
                print(f"[{cnt+1}] {fname}: FAIL, {elapsed:.0f}ms")

            if save_dir:
                cv2.imwrite(os.path.join(save_dir, fname), vis_img)

            cv2.imshow("result", vis_img)
            key = cv2.waitKey(0)
            if key == 27:
                break

        cv2.destroyAllWindows()
        if save_dir:
            json_path = os.path.join(save_dir, 'pnp_results.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)

        return all_results

    def run_on_video(self, video_source=0, save_video=None):
        """
        对视频流或摄像头进行持续实时处理。

        逐帧调用 process_frame 计算 PnP 位姿，在画面左上角叠加 FPS 和重投影误差信息，
        窗口实时显示标注结果。按 ESC 键退出。

        参数:
            video_source: int 或 str, 视频源。
                          整数: 摄像头索引（默认0为默认摄像头）
                          字符串: 视频文件路径
            save_video: str, 可选，录制结果保存路径（XVID 编码）
        """
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            print(f"无法打开视频源: {video_source}")
            return

        writer = None
        if save_video:
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            writer = cv2.VideoWriter(save_video, cv2.VideoWriter_fourcc(*'XVID'), fps, (w, h))

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            t0 = time.perf_counter()
            pnp_result = self.process_frame(frame)
            elapsed = (time.perf_counter() - t0) * 1000

            vis_img = self.get_annotated_frame()
            status_text = f"FPS: {1000/elapsed:.1f}" if elapsed > 0 else "FPS: --"
            if pnp_result:
                status_text += f" | reproj: {pnp_result['reproj_error']:.2f}px"
            else:
                status_text += " | PnP: FAIL"
            cv2.putText(vis_img, status_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            if writer:
                writer.write(vis_img)
            cv2.imshow("socket detection", vis_img)
            if cv2.waitKey(1) == 27:
                break

        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        print(f"共处理 {frame_count} 帧")

    @staticmethod
    def evaluate_pnp_errors(json_path, Mc2b, save_path=None):
        """
        读取 PnP 结果 JSON 文件，计算相邻帧间的旋转/平移误差。

        原理: 将每帧的相机坐标系位姿通过手眼标定矩阵 Mc2b 变换到基坐标系，
              然后计算相邻帧间的相对变换 T_rel = inv(T_prev) @ T_curr，
              从 T_rel 中提取旋转角（roll/pitch/yaw）和平移距离作为误差度量。

        参数:
            json_path: str, PnP 结果 JSON 文件路径，格式为 {filename: {rvec, tvec, ...}}
            Mc2b: np.ndarray 或 list, 4x4 手眼标定矩阵（相机→基座）
            save_path: str, 可选，误差结果保存路径，默认为 JSON 同目录下的 pnp_errors.json

        返回:
            errors: dict, 相邻帧误差 {filename: {
                rotation_error_deg: 总旋转误差(度),
                rotation_roll_deg: 横滚角(度),
                rotation_pitch_deg: 俯仰角(度),
                rotation_yaw_deg: 偏航角(度),
                translation_error_mm: 平移误差(mm),
                reference_frame: 参考帧文件名
            }}
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            all_results = json.load(f)

        Mc2b = np.array(Mc2b, dtype=np.float64)
        fnames = sorted([k for k in all_results if all_results[k] is not None])

        base_poses = []
        for fname in fnames:
            res = all_results[fname]
            R, _ = cv2.Rodrigues(np.array(res['rvec'], dtype=np.float64))
            T_c = np.eye(4)
            T_c[:3, :3] = R
            T_c[:3, 3] = np.array(res['tvec'], dtype=np.float64).flatten()
            base_poses.append((fname, Mc2b @ T_c))

        errors = {}
        for i in range(1, len(base_poses)):
            fname_prev, T_prev = base_poses[i - 1]
            fname_curr, T_curr = base_poses[i]
            T_rel = np.linalg.inv(T_prev) @ T_curr
            R_rel = T_rel[:3, :3]
            t_rel = T_rel[:3, 3]

            sy = np.sqrt(R_rel[0, 0]**2 + R_rel[1, 0]**2)
            if sy >= 1e-6:
                roll  = np.arctan2(R_rel[2, 1], R_rel[2, 2])
                pitch = np.arctan2(-R_rel[2, 0], sy)
                yaw   = np.arctan2(R_rel[1, 0], R_rel[0, 0])
            else:
                roll  = np.arctan2(-R_rel[1, 2], R_rel[1, 1])
                pitch = np.arctan2(-R_rel[2, 0], sy)
                yaw   = 0.0

            errors[fname_curr] = {
                'rotation_error_deg': round(np.degrees(np.linalg.norm([roll, pitch, yaw])), 6),
                'rotation_roll_deg': round(np.degrees(roll), 6),
                'rotation_pitch_deg': round(np.degrees(pitch), 6),
                'rotation_yaw_deg': round(np.degrees(yaw), 6),
                'translation_error_mm': round(np.linalg.norm(t_rel), 6),
                'reference_frame': fname_prev,
            }

        if save_path is None:
            save_path = os.path.join(os.path.dirname(json_path), 'pnp_errors.json')
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(errors, f, indent=2, ensure_ascii=False)
        print(f"误差结果已保存到: {save_path}")
        return errors

    # ==========================================================================
    # 辅助方法 (@staticmethod)
    # ==========================================================================

    @staticmethod
    def get_success_keypoints(results, conf_threshold=0.5):
        """
        从 YOLO-Pose 模型输出结果中提取检测成功的关键点索引和坐标。

        对每个目标，遍历所有关键点，根据置信度阈值判断是否检测成功。
        有置信度数据时按阈值过滤，无置信度时坐标非零即视为成功。

        参数:
            results: YOLO模型推理返回的结果对象 (ultralytics.engine.results.Results)
            conf_threshold: float, 置信度阈值，默认0.5

        返回:
            list: 格式为 [[(index, x, y), ...], ...]
                  外层列表每个元素对应画面中的一个目标，
                  内层列表每个元组为 (关键点索引, x坐标, y坐标)
        """
        all_success_points = []
        for result in results:
            if result.keypoints is not None:
                kpts_xy = result.keypoints.xy.cpu().numpy()
                kpts_conf = result.keypoints.conf
                if kpts_conf is not None:
                    kpts_conf = kpts_conf.cpu().numpy()
                for i in range(len(kpts_xy)):
                    person_kps = kpts_xy[i]
                    has_conf = kpts_conf is not None
                    success_points = []
                    for j in range(len(person_kps)):
                        x, y = person_kps[j]
                        if has_conf:
                            if kpts_conf[i][j] > conf_threshold:
                                success_points.append((j, x, y))
                        else:
                            if x != 0 or y != 0:
                                success_points.append((j, x, y))
                    all_success_points.append(success_points)
        return all_success_points

    @staticmethod
    def get_keypoint_neighbors_adaptive(success_points, circle_centers, ellipses_info=None, scale_factor=3.0):
        """
        自适应阈值的邻近匹配：利用椭圆自身尺寸（轴长均值）作为距离阈值基准，
        替代固定像素阈值，使得在不同拍摄距离下都能正确匹配。

        原理: 取所有椭圆轴长中位值 × scale_factor 作为距离阈值，
              距离小于阈值的椭圆中心被视为该关键点的邻近点。

        参数:
            success_points: list, 格式为 [[(kp_idx, x, y), ...], ...]
            circle_centers: np.ndarray, 圆形拟合得到的圆心点集，形状 (M, 2)
            ellipses_info: list, 椭圆信息列表 [(cx, cy, a, b, angle), ...]
                           若为 None，退化为固定阈值 15px
            scale_factor: float, 距离阈值 = scale_factor × 椭圆中位轴长，默认3.0

        返回:
            list: [{kp_idx: [(cx, cy), ...], ...}, ...]
                  每个字典对应一个目标的匹配结果
        """
        if ellipses_info is None or len(ellipses_info) == 0:
            return KeypointDetector.get_keypoint_neighbors(success_points, circle_centers, distance_threshold=15.0)

        centers = np.array(circle_centers)
        all_neighbors = []
        axes = [(e[2] + e[3]) / 2.0 for e in ellipses_info]
        median_axis = np.median(axes) if axes else 5.0
        distance_threshold = max(median_axis * scale_factor, 5.0)

        for target_points in success_points:
            target_neighbors_dict = {}
            for kp_idx, kp_x, kp_y in target_points:
                dists = np.sqrt((centers[:, 0] - kp_x)**2 + (centers[:, 1] - kp_y)**2)
                nearby_indices = np.where(dists < distance_threshold)[0]
                target_neighbors_dict[kp_idx] = centers[nearby_indices].tolist()
            all_neighbors.append(target_neighbors_dict)
        return all_neighbors

    @staticmethod
    def get_keypoint_neighbors(success_points, circle_centers, distance_threshold=10.0):
        """
        固定阈值邻近匹配：将椭圆中心匹配到距离关键点小于固定阈值的所有椭圆。

        原理: 对每个关键点，计算其与所有椭圆中心的欧氏距离，
              距离小于 distance_threshold 的椭圆中心均被视为该关键点的邻近点。
              适用于拍摄距离固定的场景，不适应距离变化较大的情况。

        参数:
            success_points: list, 格式为 [[(kp_idx, x, y), ...], ...]
                           外层列表每个元素对应画面中的一个目标
            circle_centers: np.ndarray, 椭圆中心点集，形状 (M, 2)
            distance_threshold: float, 匹配距离阈值（像素），默认 10.0

        返回:
            list: [{kp_idx: [(cx, cy), ...], ...}, ...]
                  每个字典对应一个目标的匹配结果
        """
        centers = np.array(circle_centers)
        all_neighbors = []
        for target_points in success_points:
            target_neighbors_dict = {}
            for kp_idx, kp_x, kp_y in target_points:
                dists = np.sqrt((centers[:, 0] - kp_x)**2 + (centers[:, 1] - kp_y)**2)
                nearby_indices = np.where(dists < distance_threshold)[0]
                target_neighbors_dict[kp_idx] = centers[nearby_indices].tolist()
            all_neighbors.append(target_neighbors_dict)
        return all_neighbors

    @staticmethod
    def filter_outliers_by_clustering(neighbors_dict, std_multiplier=2.0):
        """
        对每个关键点的邻近点集做精筛选：剔除远离主聚集区的离群点。

        原理: 计算所有点两两间距的最小值作为"典型紧密度"的估计。
        对每个点，如果它离最近点的距离远大于紧密度间距的 std_multiplier 倍，
        则判定为离群点剔除。这种方法对点数少的情况也适用。

        参数:
            neighbors_dict: dict, {kp_idx: [(cx, cy), ...], ...}
            std_multiplier: float, 紧密度间距的倍数阈值，默认 2.0

        返回:
            dict: 过滤后的邻近点集，格式同输入
        """
        filtered = {}
        for kp_idx, pts in neighbors_dict.items():
            if len(pts) < 3:
                filtered[kp_idx] = pts
                continue
            pts_arr = np.array(pts)
            pairwise = np.linalg.norm(pts_arr[:, None] - pts_arr[None, :], axis=2)
            np.fill_diagonal(pairwise, np.inf)
            min_pairwise = np.min(pairwise)
            if min_pairwise == 0 or np.isinf(min_pairwise):
                filtered[kp_idx] = pts
                continue
            min_dists = np.min(pairwise, axis=1)
            threshold = min_pairwise * std_multiplier
            keep_mask = min_dists <= threshold
            filtered[kp_idx] = pts_arr[keep_mask].tolist()
        return filtered

    @staticmethod
    def match_keypoints_hungarian(success_points, circle_centers, ellipses_info=None,
                                   scale_factor=3.0, max_match_distance=None):
        """
        匈牙利算法全局最优一对一匹配：每个关键点最多匹配一个椭圆中心，反之亦然。

        原理: 构建关键点-椭圆中心的代价矩阵（欧氏距离），
              距离超过 max_match_distance 的赋予极大惩罚值，
              使用匈牙利算法（linear_sum_assignment）求解全局最小代价分配，
              确保每个关键点和椭圆中心至多出现在一个匹配对中。

        参数:
            success_points: list, 格式为 [[(kp_idx, x, y), ...], ...]
            circle_centers: np.ndarray, 椭圆中心点集，形状 (M, 2)
            ellipses_info: list, 椭圆信息列表 [(cx, cy, a, b, angle), ...]
                           若提供，则 max_match_distance 默认取中位轴长 × scale_factor
            scale_factor: float, 距离阈值 = scale_factor × 中位轴长，默认 3.0
            max_match_distance: float, 可选，手动指定最大匹配距离（像素）
                               None 时根据 ellipses_info 自动计算，无椭圆信息时默认 15.0

        返回:
            list: [{kp_idx: [[cx, cy], ...], ...}, ...]
                  每个字典对应一个目标的匹配结果（一对一，每个关键点最多一个匹配）
        """
        centers = np.array(circle_centers)
        all_neighbors = []

        if max_match_distance is None:
            if ellipses_info is not None and len(ellipses_info) > 0:
                axes = [(e[2] + e[3]) / 2.0 for e in ellipses_info]
                median_axis = np.median(axes) if axes else 5.0
                max_match_distance = max(median_axis * scale_factor, 5.0)
            else:
                max_match_distance = 15.0

        for target_points in success_points:
            if len(target_points) == 0 or len(centers) == 0:
                all_neighbors.append({kp_idx: [] for kp_idx, _, _ in target_points})
                continue
            kp_coords = np.array([[x, y] for _, x, y in target_points])
            kp_indices = [kp_idx for kp_idx, _, _ in target_points]
            cost_matrix = np.linalg.norm(kp_coords[:, None, :] - centers[None, :, :], axis=2)
            penalty = cost_matrix.max() * 10 + 1000
            cost_matrix[cost_matrix > max_match_distance] = penalty
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            result = {kp_idx: [] for kp_idx in kp_indices}
            for r, c in zip(row_ind, col_ind):
                if cost_matrix[r, c] < penalty:
                    kp_idx = kp_indices[r]
                    cx, cy = centers[c]
                    result[kp_idx] = [[float(cx), float(cy)]]
            all_neighbors.append(result)
        return all_neighbors

    @staticmethod
    def detect_ellipse_centers_by_pyced(image, remain_score=0.8):
        """
        使用 pycylinderedsf (CED) 库进行椭圆检测。

        原理: 调用 CED (Cylinder Edge Drawing) 算法检测图像中的椭圆，
              先进行边缘检测与边缘段提取，再对边缘段进行椭圆拟合，
              最后对拟合结果进行聚类去重。精度较高但需要编译安装 EDSF 库。

        参数:
            image: np.ndarray, 输入图像（BGR 或灰度）
            remain_score: float, 边缘保留得分阈值，默认 0.8。
                          值越大保留的边缘越少，检测越严格

        返回:
            centers: np.ndarray, 椭圆中心坐标，形状 (N, 2)；无检测结果时为 (0, 2) 空数组
            ellipses: list, 椭圆参数列表 [(cx, cy, a, b, angle), ...]
                      cx, cy: 中心坐标; a, b: 半长轴/半短轴; angle: 旋转角度(度)
        """
        if not HAS_PYCED:
            print("警告: pycylinderedsf 未安装")
            return np.empty((0, 2)), []
        detector = pyced.CED(np.ascontiguousarray(image))
        detector.remain_score = remain_score
        detector.minimum_edge_length = 10
        detector.inlier_dis = 1.0
        detector.run_CED()
        rotRects = detector.getEllipsesAfterCluster()
        ellipses = []
        centers = []
        for e in rotRects:
            cx, cy = e.center
            a, b = e.size[0] / 2, e.size[1] / 2
            angle = e.angle
            ellipses.append((cx, cy, a, b, angle))
            centers.append([cx, cy])
        return np.array(centers) if centers else np.empty((0, 2)), ellipses

    @staticmethod
    def detect_ellipse_centers_by_ed(image, gradient_threshold=20, ed_detector=None,
                                     ed_params=None, cached_gradient_threshold=None):
        """
        使用 OpenCV EdgeDrawing (ED) 算法进行椭圆检测。

        原理: 调用 OpenCV ximgproc 模块的 EdgeDrawing 算法，
              通过梯度阈值提取边缘路径，再对路径段进行椭圆拟合。
              使用 NFA (Number of False Alarms) 验证剔除误检。
              无需额外安装，是默认的椭圆检测方法。

        参数:
            image: np.ndarray, 输入图像（BGR 或灰度）
            gradient_threshold: int, 梯度阈值，默认 20。
                                值越大检测越严格，只保留梯度更强的边缘
            ed_detector: cv2.ximgproc_EdgeDrawing, 可选，预创建的 EdgeDrawing 检测器实例。
                         若提供则复用该检测器，避免重复创建开销；
                         若为 None 则内部临时创建。
            ed_params: cv2.ximgproc.EdgeDrawing.Params, 可选，预创建的参数对象。
                       配合 ed_detector 使用，仅当 gradient_threshold 与
                       cached_gradient_threshold 不同时才更新参数，避免重复设置开销。
            cached_gradient_threshold: int, 可选，上次设置时的 gradient_threshold 值，
                                       用于判断是否需要重新 setParams。

        返回:
            centers: np.ndarray, 椭圆中心坐标，形状 (N, 2)；无检测结果时为 (0, 2) 空数组
            ellipses: list, 椭圆参数列表 [(cx, cy, a, b, angle), ...]
                      cx, cy: 中心坐标; a, b: 半长轴/半短轴; angle: 旋转角度(度)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

        if ed_detector is not None and ed_params is not None:
            ed = ed_detector
            if gradient_threshold != cached_gradient_threshold:
                ed_params.GradientThresholdValue = gradient_threshold
                ed.setParams(ed_params)
        else:
            params = cv2.ximgproc.EdgeDrawing.Params()
            params.EdgeDetectionOperator = 0
            params.MinPathLength = 10
            params.PFmode = True
            params.NFAValidation = True
            params.GradientThresholdValue = gradient_threshold
            ed = cv2.ximgproc.createEdgeDrawing()
            ed.setParams(params)

        ed.detectEdges(gray)
        ellipses_raw = ed.detectEllipses()
        ellipses = []
        centers = []
        if ellipses_raw is not None:
            for i in range(len(ellipses_raw)):
                e = ellipses_raw[i][0]
                if e[2] == 0:
                    cx, cy, a, b, angle = e[0], e[1], e[3], e[4], e[5]
                else:
                    cx, cy, a, b, angle = e[0], e[1], e[2], e[2], 0
                ellipses.append((cx, cy, a, b, angle))
                centers.append([cx, cy])
        return np.array(centers) if centers else np.empty((0, 2)), ellipses

    @staticmethod
    def detect_ellipse_centers(image, method='ed', ed_detector=None, ed_params=None,
                               cached_gradient_threshold=None, **kwargs):
        """
        统一椭圆检测接口，根据 method 参数分发到具体的检测算法。

        参数:
            image: np.ndarray, 输入图像
            method: str, 检测方法
                'ed':    使用 OpenCV EdgeDrawing（默认，无需额外安装）
                'pyced': 使用 pycylinderedsf CED（需编译安装 EDSF）
            ed_detector: cv2.ximgproc_EdgeDrawing, 可选，预创建的 EdgeDrawing 检测器实例
            ed_params: cv2.ximgproc.EdgeDrawing.Params, 可选，预创建的参数对象
            cached_gradient_threshold: int, 可选，上次缓存时的 gradient_threshold 值
            **kwargs: 传递给具体检测方法的额外参数
                - 'ed' 方法: gradient_threshold (int, 默认20)
                - 'pyced' 方法: remain_score (float, 默认0.8)

        返回:
            同具体检测方法的返回值: (centers, ellipses)

        异常:
            ValueError: method 不为 'ed' 或 'pyced' 时
        """
        if method == 'pyced':
            return KeypointDetector.detect_ellipse_centers_by_pyced(image, **kwargs)
        elif method == 'ed':
            return KeypointDetector.detect_ellipse_centers_by_ed(
                image, ed_detector=ed_detector, ed_params=ed_params,
                cached_gradient_threshold=cached_gradient_threshold, **kwargs)
        else:
            raise ValueError(f"不支持的检测方法: '{method}'")

    @staticmethod
    def detect_ellipse_centers_in_roi(image, boxes, method='ed', ed_detector=None,
                                      ed_params=None, cached_gradient_threshold=None, **kwargs):
        """
        在检测框 ROI 区域内进行椭圆检测，坐标还原到原图坐标系。

        原理: 对每个检测框裁剪 ROI 子图，在子图内调用椭圆检测，
              然后将检测到的椭圆中心和坐标加上 ROI 偏移量 (x1, y1)，
              还原到原图坐标系。这样可以减少背景干扰、提升检测速度。

        参数:
            image: np.ndarray, 原始输入图像
            boxes: np.ndarray, 检测框坐标，形状 (N, 4)，格式 [x1, y1, x2, y2]
            method: str, 椭圆检测方法，默认 'ed'
            ed_detector: cv2.ximgproc_EdgeDrawing, 可选，预创建的 EdgeDrawing 检测器实例
            ed_params: cv2.ximgproc.EdgeDrawing.Params, 可选，预创建的参数对象
            cached_gradient_threshold: int, 可选，上次缓存时的 gradient_threshold 值
            **kwargs: 传递给 detect_ellipse_centers 的额外参数

        返回:
            all_centers: list, 每个元素为 np.ndarray (M_i, 2)，对应第 i 个检测框内的椭圆中心
            all_ellipses: list, 每个元素为 [(cx, cy, a, b, angle), ...]，
                          椭圆参数已还原到原图坐标系
        """
        all_centers = []
        all_ellipses = []
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            roi = image[y1:y2, x1:x2]
            # 预处理 导向滤波
            gray_f = roi.astype(np.float32)
            smooth = cv2.ximgproc.guidedFilter(gray_f, gray_f, radius=5, eps=150.0)
            roi = np.clip(smooth, 0, 255).astype(np.uint8)
            # Unsharp Mask 锐化：增强边缘
            blurred = cv2.GaussianBlur(roi, (0, 0), 3)
            roi = cv2.addWeighted(roi, 1.5, blurred, -0.5, 0)


            centers, ellipses = KeypointDetector.detect_ellipse_centers(
                roi, method=method, ed_detector=ed_detector, ed_params=ed_params,
                cached_gradient_threshold=cached_gradient_threshold, **kwargs)
            if len(centers) > 0:
                centers[:, 0] += x1
                centers[:, 1] += y1
            ellipses_original = [(cx + x1, cy + y1, a, b, angle)
                                 for cx, cy, a, b, angle in ellipses]
            all_centers.append(centers)
            all_ellipses.append(ellipses_original)
        return all_centers, all_ellipses

    @staticmethod
    def refine_point_centroid(gray, pt, win_size=10):
        """
        重心法亚像素精化：在关键点周围局部窗口内，通过二值化+重心计算实现亚像素定位。

        原理: 以 pt 为中心取 (2*win_size+1) × (2*win_size+1) 的局部窗口，
              对窗口内灰度图使用 Otsu 自动阈值二值化（反色，使暗区域为前景），
              然后计算前景区域的图像矩重心作为精化后的坐标。
              可将整数像素坐标精化到亚像素精度。

        参数:
            gray: np.ndarray, 灰度图像
            pt: tuple 或 array-like, 初始坐标 (x, y)
            win_size: int, 局部窗口半径，默认 10（实际窗口 21×21）

        返回:
            np.ndarray: 精化后的坐标 [x, y]，dtype=float64。
                        若点在图像边缘外或矩为零，返回原始坐标。
        """
        x0, y0 = int(round(pt[0])), int(round(pt[1]))
        h, w = gray.shape[:2]
        r = win_size
        if x0 - r < 0 or x0 + r >= w or y0 - r < 0 or y0 + r >= h:
            return np.array(pt, dtype=np.float64)
        roi = gray[y0 - r:y0 + r + 1, x0 - r:x0 + r + 1]
        _, thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        m = cv2.moments(thresh)
        if m["m00"] == 0:
            return np.array(pt, dtype=np.float64)
        cx = m["m10"] / m["m00"]
        cy = m["m01"] / m["m00"]
        return np.array([(x0 - r) + cx, (y0 - r) + cy], dtype=np.float64)

    @staticmethod
    def _refine_pnp_with_reprojection(rvec, tvec, K, dist, valid_indices,
                                       concentric_results, gray, keypoint_obj_pts,
                                       max_iter=3, search_radius=15.0, verbose=False):
        """
        反投影迭代精化 PnP：利用当前位姿预测的投影位置重新匹配椭圆中心，迭代提升精度。

        原理:
            1. 用当前 rvec/tvec 将 3D 物体点投影到图像，得到预测的 2D 位置
            2. 在预测位置附近（search_radius 范围内）搜索最近的椭圆中心
            3. 对匹配到的椭圆中心调用 refine_point_centroid 做亚像素精化
            4. 用精化后的 2D-3D 对应重新求解 PnP（RANSAC + ITERATIVE）
            5. 若新位姿的重投影误差更低则接受，否则停止迭代
            6. 重复 max_iter 次或收敛后停止

        参数:
            rvec: np.ndarray, 初始旋转向量 (3, 1)
            tvec: np.ndarray, 初始平移向量 (3, 1)
            K: np.ndarray, 相机内参矩阵 (3, 3)
            dist: np.ndarray, 畸变系数 (5,)
            valid_indices: list, 有效的关键点索引列表
            concentric_results: dict, 同心圆拟合结果 {kp_idx: {concentric_center, ...}}
            gray: np.ndarray, 灰度图像（用于亚像素精化）
            keypoint_obj_pts: np.ndarray, 3D 物体坐标模板 (N, 3)
            max_iter: int, 最大迭代次数，默认 3
            search_radius: float, 投影点搜索半径（像素），默认 15.0
            verbose: bool, 是否打印调试信息，默认 False

        返回:
            (rvec, tvec, reproj_error): 精化后的旋转向量、平移向量和平均重投影误差
                                        若迭代过程中 PnP 求解失败，返回 (None, None, None)
        """
        all_cc_pts = []
        for kp_idx in sorted(concentric_results.keys()):
            cc = concentric_results[kp_idx]['concentric_center']
            if cc is not None:
                all_cc_pts.append(np.array(cc, dtype=np.float64))
        if len(all_cc_pts) < 4:
            return None, None, None
        all_cc_pts = np.array(all_cc_pts)

        all_obj_pts = []
        all_kp_indices = []
        for kp_idx in sorted(concentric_results.keys()):
            if kp_idx < len(keypoint_obj_pts):
                all_obj_pts.append(keypoint_obj_pts[kp_idx])
                all_kp_indices.append(kp_idx)
        all_obj_pts = np.array(all_obj_pts, dtype=np.float32)

        current_rvec = rvec.copy()
        current_tvec = tvec.copy()
        initial_cc_map = {kp_idx: all_cc_pts[j].copy() for j, kp_idx in enumerate(all_kp_indices)}

        for iteration in range(max_iter):
            proj_pts, _ = cv2.projectPoints(all_obj_pts, current_rvec, current_tvec, K, dist)
            proj_pts = proj_pts.reshape(-1, 2)
            refined_img_pts = []
            refined_obj_pts = []
            used_cc = set()

            for j, kp_idx in enumerate(all_kp_indices):
                px, py = proj_pts[j]
                dists = np.linalg.norm(all_cc_pts - np.array([px, py]), axis=1)
                sorted_indices = np.argsort(dists)
                matched = False
                for si in sorted_indices:
                    if dists[si] > search_radius:
                        break
                    cc_key = (round(all_cc_pts[si][0], 1), round(all_cc_pts[si][1], 1))
                    if cc_key not in used_cc:
                        refined = KeypointDetector.refine_point_centroid(gray, all_cc_pts[si], win_size=10)
                        refined_img_pts.append(refined if refined is not None else all_cc_pts[si])
                        refined_obj_pts.append(all_obj_pts[j])
                        used_cc.add(cc_key)
                        matched = True
                        break
                if not matched:
                    refined_img_pts.append(initial_cc_map[kp_idx])
                    refined_obj_pts.append(all_obj_pts[j])

            if len(refined_obj_pts) < 4:
                break

            refined_obj_pts = np.array(refined_obj_pts, dtype=np.float32)
            refined_img_pts = np.array(refined_img_pts, dtype=np.float32)
            success, new_rvec, new_tvec, new_inliers = cv2.solvePnPRansac(
                refined_obj_pts, refined_img_pts, K, dist,
                useExtrinsicGuess=False, iterationsCount=100,
                reprojectionError=2.0, confidence=0.99)
            if success and new_inliers is not None and len(new_inliers) >= 6:
                inlier_mask = new_inliers.flatten()
                success, new_rvec, new_tvec = cv2.solvePnP(
                    refined_obj_pts[inlier_mask], refined_img_pts[inlier_mask], K, dist,
                    flags=cv2.SOLVEPNP_ITERATIVE,
                    rvec=new_rvec.copy(), tvec=new_tvec.copy())
            if not success:
                break

            proj_check, _ = cv2.projectPoints(all_obj_pts, new_rvec, new_tvec, K, dist)
            proj_curr, _ = cv2.projectPoints(all_obj_pts, current_rvec, current_tvec, K, dist)
            new_err = np.linalg.norm(all_cc_pts - proj_check.reshape(-1, 2)[:len(all_cc_pts)], axis=1).mean()
            curr_err = np.linalg.norm(all_cc_pts - proj_curr.reshape(-1, 2)[:len(all_cc_pts)], axis=1).mean()

            if new_err < curr_err:
                current_rvec, current_tvec = new_rvec, new_tvec
            else:
                break
            if np.linalg.norm(new_rvec.flatten() - current_rvec.flatten()) < 1e-5 and \
               np.linalg.norm(new_tvec.flatten() - current_tvec.flatten()) < 0.01:
                break

        proj_final, _ = cv2.projectPoints(all_obj_pts, current_rvec, current_tvec, K, dist)
        reproj_error = np.linalg.norm(all_cc_pts - proj_final.reshape(-1, 2)[:len(all_cc_pts)], axis=1).mean()
        return current_rvec, current_tvec, reproj_error


# ==============================================================================
# 示例用法
# ==============================================================================
if __name__ == "__main__":
    # 初始化检测器
    detector = KeypointDetector(
        port_type='slow',       # 'super' 快充9孔 | 'slow' 慢充7孔
        camera='large',         # 'small' 1280x720 | 'large' 2560x1440
        detect_method='ed',     # 'ed' | 'pyced'
        verbose=True,           # 打印
        visual=True,            # 可视化
    )

    # --- 单帧处理 ---
    img_path = "dataset/save_data3/20260511_120244/images/frame_004995.jpg"
    frame = cv2.imread(img_path)
    # 慢充口：coplanar=True, 快充口 coplanar=False
    pnp_result = detector.process_frame(frame, coplanar=True)

    concentric_results = detector._concentric_results
    # for kp_idx in sorted(concentric_results.keys()):
    #     cc = concentric_results[kp_idx]['concentric_center']
    #     if cc is not None:
    #         print(f"关键点 {kp_idx}: 圆心 ({cc[0]:.4f}, {cc[1]:.4f})")
    #     else:
    #         print(f"关键点 {kp_idx}: 未匹配到圆心")

    cc_points = [] #获取到圆心坐标
    cc_indices = [] #获取到圆心对应的关键点索引
    for kp_idx in sorted(concentric_results.keys()):
        cc = concentric_results[kp_idx]['concentric_center']
        if cc is not None:
            cc_points.append(cc)
            cc_indices.append(kp_idx)
    cc_points = np.array(cc_points)

    rvec = pnp_result['rvec']
    tvec = pnp_result['tvec']
    R,_ = cv2.Rodrigues(rvec)
    t = tvec.flatten()
    T_obj2cam = np.eye(4)
    T_obj2cam[:3, :3] = R
    T_obj2cam[:3, 3] = t
    print("T_obj2cam:", T_obj2cam)


    vis_img = detector.get_annotated_frame()
    cv2.imshow("result", vis_img)
    cv2.waitKey(0)
    cv2.imwrite("result.jpg",vis_img)
    # --- 批量处理到基坐标系 ---
    # detector.batch_process_to_base(
    #     img_dir="dataset/save_data3/20260511_120244/images",
    #     save_dir="result/save_data3/20260511_120244/cdd",
    #     coplanar=True,
    # )
