import os
import numpy as np
import cv2
import time
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO
try:
    os.add_dll_directory(r'D:\SDK\opencv4.7.0_Library\x64\vc17\bin')
    import pycylinderedsf as pyced
    HAS_PYCED = True
except (ImportError, OSError):
    HAS_PYCED = False

"""
kp1-> S- : (-12,-21,17.5)
kp2-> CC2 : (0,-21,12.5)
kp3-> S+ : (12,-21,17.5)
kp4-> CC1 : (0,-9,17.5)
kp5-> DC- : (-17,1.2,7.5)
kp6-> DC+ : (17,1.2,7.5)
kp7-> A- : (-14.25,21,17.5)
kp8-> D : (0,21,7.5)
kp9-> A+ : (14.25,21,17.5)
"""
#超充口
# 关键点物理坐标模板：索引0~8 对应 kp1~kp9，格式 (x, y, z)
Super_KEYPOINT_OBJ_PTS = np.array([
    [-12.0, -21.0, 17.5],   # kp1: S-
    [0.0, -21.0, 12.5],      # kp2: CC2
    [12.0, -21.0, 17.5],    # kp3: S+
    [0.0, -9.0, 17.5],      # kp4: CC1
    [-17.0, 1.2, 7.5],      # kp5: DC-
    [17.0, 1.2, 7.5],       # kp6: DC+
    [-14.25, 21.0, 17.5],    # kp7: A-
    [0.0, 21.0, 7.5],       # kp8: D
    [14.25, 21.0,17.5],     # kp9: A+
], dtype=np.float32)



#慢充口
Slow_KEYPOINT_OBJ_PTS = np.array([
    [-8.0, -11.2, 7.5],   # tp0: CP
    [8.0, -11.2, 7.5],      # tp1: CC
    [-25.5, 0.0, 7.5],    # tp2: N
    [0.0, 0.0, 7.5],      # tp3: D
    [25.5, 0.0, 7.5],      # tp4: L1
    [-8.0, 13.9, 7.5],       # tp5: L3
    [8.0, 13.9, 7.5],    # tp6: L2
], dtype=np.float32)


# KEYPOINT_OBJ_PTS = Super_KEYPOINT_OBJ_PTS
KEYPOINT_OBJ_PTS =Slow_KEYPOINT_OBJ_PTS


def get_success_keypoints(results, conf_threshold=0.5):
    """
    从 YOLO-Pose 模型输出结果中提取检测成功的关键点索引和坐标。
    
    参数:
        results: YOLO 模型推理返回的结果对象 (ultralytics.engine.results.Results)
        conf_threshold: 置信度阈值 (float)，默认为 0.5
        
    返回:
        list: 包含所有目标检测成功点的列表。
              格式为: [[(index, x, y), ...], [(index, x, y), ...]]
              外层列表的每个元素对应画面中的一个目标。
    """
    all_success_points = []
    
    # 遍历每一个检测到的目标
    for result in results:
        if result.keypoints is not None:
            # 提取坐标 (N, K, 2)
            kpts_xy = result.keypoints.xy.cpu().numpy()
            # 置信度 (N, K) — 部分模型不输出conf，此时视为全部检测成功
            kpts_conf = result.keypoints.conf
            if kpts_conf is not None:
                print("kpts have conf")
                kpts_conf = kpts_conf.cpu().numpy()

            # 遍历当前画面中的每一个目标
            for i in range(len(kpts_xy)):
                person_kps = kpts_xy[i]
                has_conf = kpts_conf is not None

                # 准备一个列表，存放当前目标检测成功的点
                success_points = []

                # 遍历该目标的所有关键点（例如你的9个点）
                for j in range(len(person_kps)):
                    x, y = person_kps[j]
                    # 如果没有conf数据，只要坐标非零就视为检测成功
                    if has_conf:
                        if kpts_conf[i][j] > conf_threshold:
                            success_points.append((j, x, y))
                    else:
                        if x != 0 or y != 0:
                            success_points.append((j, x, y))

                # 将当前目标的成功点加入总列表
                all_success_points.append(success_points)
                
    return all_success_points

def visualize_keypoints(image, results):
    """
    在图像上仅绘制检测框和关键点坐标。
    
    参数:
        image: 原始图像 (numpy array)
        results: YOLO 模型推理返回的结果对象
    """
    # 复制一份图像用于绘制，避免修改原图
    vis_img = image.copy()
    
    # 遍历每一个检测到的目标
    for result in results:
        # 1. 绘制检测框 (绿色)
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy().astype(int)
            for box in boxes:
                x1, y1, x2, y2 = box
                cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # 2. 绘制关键点 (蓝色) 及坐标/索引
        if result.keypoints is not None:
            kpts_xy = result.keypoints.xy.cpu().numpy()
            kpts_conf = result.keypoints.conf
            if kpts_conf is not None:
                kpts_conf = kpts_conf.cpu().numpy()

            # 遍历当前图片中的每一个目标
            for i in range(len(kpts_xy)):
                person_kps = kpts_xy[i]
                has_conf = kpts_conf is not None

                # 遍历该目标的所有关键点
                for kp_idx in range(len(person_kps)):
                    kp_x, kp_y = int(person_kps[kp_idx][0]), int(person_kps[kp_idx][1])
                    # 判断是否绘制：有conf时按阈值过滤，无conf时坐标非零即绘制
                    if has_conf:
                        if kpts_conf[i][kp_idx] <= 0.5:
                            continue
                    else:
                        if kp_x == 0 and kp_y == 0:
                            continue

                    # 绘制关键点本身（蓝色实心圆）
                    cv2.circle(vis_img, (kp_x, kp_y), 5, (255, 0, 0), -1)

                    # 标注关键点索引
                    cv2.putText(vis_img, str(kp_idx), (kp_x + 5, kp_y - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                            
    return vis_img


def visualize_results(image, results, neighbors_result, distance_threshold=10.0):
    """
    在图像上绘制检测框、关键点以及匹配到的邻近圆心点集。
    
    参数:
        image: 原始图像 (numpy array)
        results: YOLO 模型推理返回的结果对象
        neighbors_result: get_keypoint_neighbors 函数返回的邻近点集结果
        distance_threshold: 距离阈值（用于绘制红色警戒圈，辅助观察）
    """
    # 复制一份图像用于绘制，避免修改原图
    vis_img = image.copy()
    
    # 遍历每一个检测到的目标
    for i, result in enumerate(results):
        if result.boxes is not None:
            # 1. 绘制检测框 (绿色)
            boxes = result.boxes.xyxy.cpu().numpy().astype(int)
            for box in boxes:
                x1, y1, x2, y2 = box
                cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # 获取当前目标的关键点和邻近点数据
        # 注意：neighbors_result 的索引顺序与 results 遍历顺序一致
        if i < len(neighbors_result):
            target_neighbors_dict = neighbors_result[i]
            
            if result.keypoints is not None:
                kpts_xy = result.keypoints.xy.cpu().numpy()
                kpts_conf = result.keypoints.conf
                if kpts_conf is not None:
                    kpts_conf = kpts_conf.cpu().numpy()
                person_kps = kpts_xy[i]
                has_conf = kpts_conf is not None

                # 2. 遍历并绘制关键点 (蓝色) 和 邻近圆心 (红色)
                for kp_idx in range(len(person_kps)):
                    kp_x, kp_y = int(person_kps[kp_idx][0]), int(person_kps[kp_idx][1])
                    # 判断是否绘制：有conf时按阈值过滤，无conf时坐标非零即绘制
                    if has_conf:
                        if kpts_conf[i][kp_idx] <= 0.5:
                            continue
                    else:
                        if kp_x == 0 and kp_y == 0:
                            continue

                    # 绘制关键点本身（蓝色实心圆）
                    cv2.circle(vis_img, (kp_x, kp_y), 5, (255, 0, 0), -1)
                    # 标注关键点索引
                    cv2.putText(vis_img, str(kp_idx), (kp_x + 5, kp_y - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                    # 绘制该关键点匹配到的邻近圆心点集（红色空心圆）
                    # 字典的 value 是一个包含多个 (cx, cy) 的列表
                    nearby_centers = target_neighbors_dict.get(kp_idx, [])
                    for cx, cy in nearby_centers:
                        cx, cy = int(cx), int(cy)
                        cv2.circle(vis_img, (cx, cy), 8, (0, 0, 255), 2)
                            
    return vis_img



def get_keypoint_neighbors_adaptive(success_points, circle_centers, ellipses_info=None, scale_factor=3.0):
    """
    自适应阈值的邻近匹配：利用椭圆自身尺寸（轴长均值）作为距离阈值基准，
    替代固定像素阈值，使得在不同拍摄距离下都能正确匹配。

    参数:
        success_points: list，格式为 [[(kp_idx, x, y), ...], ...]
        circle_centers: np.ndarray，圆形拟合得到的圆心点集，形状为 (M, 2)。
        ellipses_info: list，椭圆信息列表 [(cx, cy, a, b, angle), ...]，
                       用于提取每个椭圆的尺寸。若为 None，退化为固定阈值 15px。
        scale_factor: float，距离阈值 = scale_factor * 椭圆的平均轴长，默认 3.0

    返回:
        list: 包含字典的列表，格式同 get_keypoint_neighbors
    """
    if ellipses_info is None or len(ellipses_info) == 0:
        print("not use adaptive, use fixed distance_th 15")
        return get_keypoint_neighbors(success_points, circle_centers, distance_threshold=15.0)

    centers = np.array(circle_centers)
    all_neighbors = []

    # 计算自适应阈值：取所有椭圆轴长中位值 * scale_factor
    axes = [(e[2] + e[3]) / 2.0 for e in ellipses_info]  # (a+b)/2 作为每个椭圆的特征尺寸
    median_axis = np.median(axes) if axes else 5.0
    print(" ellipses axis median :", median_axis)
    distance_threshold = median_axis * scale_factor
    distance_threshold = max(distance_threshold, 5.0)  # 最小不低于 5px
    print("distance_th: ",distance_threshold)

    # 遍历每一个目标的关键点集
    for target_points in success_points:
        target_neighbors_dict = {}

        for kp_idx, kp_x, kp_y in target_points:
            dists = np.sqrt((centers[:, 0] - kp_x)**2 + (centers[:, 1] - kp_y)**2)
            nearby_indices = np.where(dists < distance_threshold)[0]
            nearby_centers = centers[nearby_indices].tolist()
            target_neighbors_dict[kp_idx] = nearby_centers

        all_neighbors.append(target_neighbors_dict)

    return all_neighbors


def get_keypoint_neighbors(success_points, circle_centers, distance_threshold=10.0):
    """
    获取每个关键点在圆形拟合点集中的邻近点（固定阈值版本）。
    推荐使用 get_keypoint_neighbors_adaptive 替代。
    
    参数:
        success_points: list，格式为 [[(kp_idx, x, y), ...], ...]
        circle_centers: np.ndarray，圆形拟合得到的圆心点集，形状为 (M, 2)。
        distance_threshold: float，距离阈值，默认 10.0 像素。
        
    返回:
        list: 包含字典的列表。每个字典代表一个目标的匹配结果。
              字典格式: {关键点索引: [(cx1, cy1), ...], ...}
              若某关键点没有邻近点，其对应的值为空列表 []。
    """
    centers = np.array(circle_centers)
    all_neighbors = []
    
    # 遍历每一个目标的关键点集
    for target_points in success_points:
        # 使用字典来存储当前目标的匹配结果，key为关键点索引
        target_neighbors_dict = {}
        
        # 遍历当前目标的每一个成功检测的关键点
        for kp_idx, kp_x, kp_y in target_points:
            # 向量化计算当前关键点到所有圆心的欧氏距离
            dists = np.sqrt((centers[:, 0] - kp_x)**2 + (centers[:, 1] - kp_y)**2)
            
            # 找出距离小于阈值的圆心的索引
            nearby_indices = np.where(dists < distance_threshold)[0]
            
            # 提取邻近圆心的坐标
            nearby_centers = centers[nearby_indices].tolist()
            
            # 【核心改进】将关键点索引作为字典的键，存入结果（即使为空列表也会存入）
            target_neighbors_dict[kp_idx] = nearby_centers
            
        all_neighbors.append(target_neighbors_dict)
        
    return all_neighbors


def filter_outliers_by_clustering(neighbors_dict, std_multiplier=2.0):
    """
    对每个关键点的邻近点集做精筛选：剔除远离主聚集区的离群点。

    原理：计算所有点两两间距的最小值作为"典型紧密度"的估计。
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

        # 计算两两距离
        pairwise = np.linalg.norm(pts_arr[:, None] - pts_arr[None, :], axis=2)
        np.fill_diagonal(pairwise, np.inf)

        # 取最小两两距离作为"典型紧密度"（代表最近的两个点之间的距离）
        min_pairwise = np.min(pairwise)
        if min_pairwise == 0 or np.isinf(min_pairwise):
            filtered[kp_idx] = pts
            continue

        # 对每个点，找它离最近点的距离
        min_dists = np.min(pairwise, axis=1)

        # 如果某个点离最近点的距离 > 典型紧密度 * multiplier，视为离群
        threshold = min_pairwise * std_multiplier
        keep_mask = min_dists <= threshold
        filtered[kp_idx] = pts_arr[keep_mask].tolist()

    return filtered


def match_keypoints_hungarian(success_points, circle_centers, ellipses_info=None,
                              scale_factor=3.0, max_match_distance=None):
    """
    使用匈牙利算法进行关键点与椭圆中心的全局最优一对一匹配。

    与 get_keypoint_neighbors_adaptive + filter_outliers_by_clustering 的区别：
    - 后者是贪心匹配，每个关键点独立找邻近点，可能出现一对多或多对一
    - 本方法构建代价矩阵，用匈牙利算法求全局最优一对一分配，避免匹配冲突

    返回格式与 filter_outliers_by_clustering 一致，可直接替换使用。

    参数:
        success_points: list，格式为 [[(kp_idx, x, y), ...], ...]
        circle_centers: np.ndarray，椭圆中心点集，形状为 (M, 2)
        ellipses_info: list，椭圆信息 [(cx, cy, a, b, angle), ...]，用于自适应阈值
        scale_factor: float，距离阈值 = scale_factor * 椭圆平均轴长（仅用于过滤明显不合理的匹配）
        max_match_distance: float，最大匹配距离阈值，若为 None 则使用自适应阈值

    返回:
        list: 包含字典的列表，格式同 get_keypoint_neighbors_adaptive
              [{kp_idx: [(cx, cy), ...], ...}, ...]
    """
    centers = np.array(circle_centers)
    all_neighbors = []

    # 计算匹配距离上限
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

        # 构建代价矩阵：(N_keypoints × M_centers)
        kp_coords = np.array([[x, y] for _, x, y in target_points])
        kp_indices = [kp_idx for kp_idx, _, _ in target_points]

        # 距离矩阵
        cost_matrix = np.linalg.norm(kp_coords[:, None, :] - centers[None, :, :], axis=2)

        # 对超过最大匹配距离的代价设为极大值，防止错误匹配
        penalty = cost_matrix.max() * 10 + 1000
        cost_matrix[cost_matrix > max_match_distance] = penalty

        # 匈牙利算法求解最优一对一分配
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # 构建结果字典，格式与原有接口一致
        result = {kp_idx: [] for kp_idx in kp_indices}
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < penalty:
                kp_idx = kp_indices[r]
                cx, cy = centers[c]
                result[kp_idx] = [[float(cx), float(cy)]]

        all_neighbors.append(result)

    return all_neighbors



def detect_ellipse_centers_by_pyced(image, remain_score=0.8):
    """
    使用 pyced (CED) 对图像进行椭圆检测，返回椭圆中心坐标列表。

    参数:
        image: BGR 图像 (numpy array)
        remain_score: CED 检测的保留分数阈值，默认 0.8

    返回:
        np.ndarray: 椭圆中心坐标数组，形状为 (N, 2)
        list: 完整椭圆信息列表 [(cx, cy, a, b, angle), ...]
    """
    if not HAS_PYCED:
        print("警告: pycylinderedsf 未安装，无法使用 pyced 检测。请先编译安装 EDSF 项目。")
        return np.empty((0, 2)), []
    detector = pyced.CED(np.ascontiguousarray(image))
    detector.remain_score = 0.6 #remain_score  #0.5~0.8 之间尝试，漏检时降低，噪点多时升高
    detector.minimum_edge_length = 10    # 保留更短的弧段
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


def detect_ellipse_centers_by_ed(image, gradient_threshold=15):
    """
    使用 OpenCV EdgeDrawing 对图像进行椭圆检测，返回椭圆中心坐标列表。

    参数:
        image: BGR 图像 (numpy array)
        gradient_threshold: 梯度阈值，默认 20

    返回:
        np.ndarray: 椭圆中心坐标数组，形状为 (N, 2)
        list: 完整椭圆信息列表 [(cx, cy, a, b, angle), ...]
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

    params = cv2.ximgproc.EdgeDrawing.Params()
    params.EdgeDetectionOperator = 0  # 0=SOBEL
    params.MinPathLength = 10  # 默认值为 10。增大该值可以过滤掉细小的噪声边缘
    params.PFmode = True  # 概率森林模式 (PFmode)：启用后可以提高边缘检测准确性，但会增加计算时间
    params.NFAValidation = True  # 默认为 True。设置为 False 时能检测到更多的圆或直线，但可能包含更多噪点
    params.GradientThresholdValue = gradient_threshold # 默认值为 20。如果图像对比度低，可以尝试降低该值（如 10-15）；高对比度可适当提高

    ed = cv2.ximgproc.createEdgeDrawing()
    ed.setParams(params)
    ed.detectEdges(gray)
    ellipses_raw = ed.detectEllipses()

    ellipses = []
    centers = []
    if ellipses_raw is not None:
        for i in range(len(ellipses_raw)):
            e = ellipses_raw[i][0]
            # e[2] == 0 表示圆形（半径在e[3],e[4]），否则为椭圆（半径在e[2]）
            if e[2] == 0:
                cx, cy, a, b, angle = e[0], e[1], e[3], e[4], e[5]
            else:
                cx, cy, a, b, angle = e[0], e[1], e[2], e[2], 0
            ellipses.append((cx, cy, a, b, angle))
            centers.append([cx, cy])

    return np.array(centers) if centers else np.empty((0, 2)), ellipses


def detect_ellipse_centers(image, method='ed', **kwargs):
    """
    统一椭圆检测接口，支持选择检测方法。

    参数:
        image: BGR 图像 (numpy array)
        method: 检测方法，'pyced' 或 'ed'（默认 'ed'）
        **kwargs: 传递给对应检测方法的额外参数
            pyced: remain_score (默认 0.8)
            ed: gradient_threshold (默认 20)

    返回:
        np.ndarray: 椭圆中心坐标数组，形状为 (N, 2)
        list: 完整椭圆信息列表 [(cx, cy, a, b, angle), ...]
    """
    if method == 'pyced':
        return detect_ellipse_centers_by_pyced(image, **kwargs)
    elif method == 'ed':
        return detect_ellipse_centers_by_ed(image, **kwargs)
    else:
        raise ValueError(f"不支持的检测方法: '{method}'，请选择 'pyced' 或 'ed'")


def refine_point_centroid(gray, pt, win_size=10):
    """
    重心法亚像素精化：Otsu 阈值分割 + 图像矩求质心。

    原理：充电孔是深色圆孔在浅色面板上，Otsu 自动分割出孔洞区域，
    用图像矩计算孔洞的灰度质心，精度可达亚像素级。

    参数:
        gray: 灰度图像
        pt: 初始点坐标 (x, y)
        win_size: 搜索窗口半径，默认 10

    返回:
        精化后的坐标 numpy 数组 (2,)，失败返回 None
    """
    x0, y0 = int(round(pt[0])), int(round(pt[1]))
    h, w = gray.shape[:2]
    r = win_size

    if x0 - r < 0 or x0 + r >= w or y0 - r < 0 or y0 + r >= h:
        return np.array(pt, dtype=np.float64)

    roi = gray[y0 - r:y0 + r + 1, x0 - r:x0 + r + 1]

    # 反转颜色（针对深色孔洞）并 Otsu 阈值化
    _, thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    m = cv2.moments(thresh)
    if m["m00"] == 0:
        return np.array(pt, dtype=np.float64)

    # 阈值化区域质心（相对于 ROI 左上角）
    cx = m["m10"] / m["m00"]
    cy = m["m01"] / m["m00"]

    # 转换回原图坐标
    new_x = (x0 - r) + cx
    new_y = (y0 - r) + cy

    return np.array([new_x, new_y], dtype=np.float64)


def detect_ellipse_centers_in_roi(image, boxes, method='ed', **kwargs):
    """
    在 YOLO 检测框内进行椭圆检测，将坐标还原到原图。

    参数:
        image: 原始 BGR 图像
        boxes: YOLO 检测框，形状 (N, 4)，格式 [x1, y1, x2, y2]
        method: 检测方法，'pyced' 或 'ed'（默认 'ed'）
        **kwargs: 传递给检测方法的额外参数

    返回:
        list: 每个检测框内的椭圆中心坐标（原图坐标），元素为 np.ndarray (M, 2)
        list: 每个检测框内的完整椭圆信息（原图坐标），元素为 list of (cx, cy, a, b, angle)
    """
    all_centers = []
    all_ellipses = []

    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        roi = image[y1:y2, x1:x2]

        centers, ellipses = detect_ellipse_centers(roi, method=method, **kwargs)

        # 将 ROI 坐标还原到原图坐标
        if len(centers) > 0:
            centers[:, 0] += x1
            centers[:, 1] += y1

        ellipses_original = [(cx + x1, cy + y1, a, b, angle)
                             for cx, cy, a, b, angle in ellipses]

        all_centers.append(centers)
        all_ellipses.append(ellipses_original)

    return all_centers, all_ellipses




def _refine_pnp_with_reprojection(rvec, tvec, K, dist, valid_indices,
                                   concentric_results, gray, max_iter=3,
                                   search_radius=15.0, verbose=False):
    """
    反投影迭代精化 PnP：将初始 PnP 结果反投影到 2D，以反投影点为锚
    找最近的拟合圆心，重新求解 PnP，迭代直到收敛。

    参数:
        rvec, tvec: 初始 PnP 结果
        K, dist: 相机内参和畸变系数
        valid_indices: 有效的关键点索引列表
        concentric_results: 同心圆拟合结果字典
        gray: 灰度图像（用于重心法精化）
        max_iter: 最大迭代次数，默认 3
        search_radius: 反投影点搜索最近圆心的最大距离（像素），默认 15
        verbose: 是否打印详细信息

    返回:
        (rvec, tvec, reproj_error) 或 (None, None, None)
    """
    # 收集所有拟合圆心作为候选点池
    all_cc_pts = []
    for kp_idx in sorted(concentric_results.keys()):
        cc = concentric_results[kp_idx]['concentric_center']
        if cc is not None:
            all_cc_pts.append(np.array(cc, dtype=np.float64))
    if len(all_cc_pts) < 4:
        return None, None, None
    all_cc_pts = np.array(all_cc_pts)

    # 构建所有关键点对应的 3D 物体点
    all_obj_pts = []
    all_kp_indices = []
    for kp_idx in sorted(concentric_results.keys()):
        if kp_idx < len(KEYPOINT_OBJ_PTS):
            all_obj_pts.append(KEYPOINT_OBJ_PTS[kp_idx])
            all_kp_indices.append(kp_idx)
    all_obj_pts = np.array(all_obj_pts, dtype=np.float32)

    current_rvec = rvec.copy()
    current_tvec = tvec.copy()

    # 记录每个 kp_idx 初始对应的圆心坐标（反投影匹配失败时回退用）
    initial_cc_map = {}
    for j, kp_idx in enumerate(all_kp_indices):
        initial_cc_map[kp_idx] = all_cc_pts[j].copy()

    for iteration in range(max_iter):
        # 反投影所有 3D 点到 2D
        proj_pts, _ = cv2.projectPoints(all_obj_pts, current_rvec, current_tvec, K, dist)
        proj_pts = proj_pts.reshape(-1, 2)

        # 对每个反投影点，找最近的拟合圆心
        refined_img_pts = []
        refined_obj_pts = []
        refined_indices = []
        used_cc = set()  # 防止多个反投影点匹配到同一个圆心

        for j, kp_idx in enumerate(all_kp_indices):
            px, py = proj_pts[j]
            dists = np.linalg.norm(all_cc_pts - np.array([px, py]), axis=1)

            # 找最近的、未被占用的圆心
            sorted_indices = np.argsort(dists)
            matched = False
            for si in sorted_indices:
                if dists[si] > search_radius:
                    break
                cc_key = (round(all_cc_pts[si][0], 1), round(all_cc_pts[si][1], 1))
                if cc_key not in used_cc:
                    # 重心法亚像素精化
                    refined = refine_point_centroid(gray, all_cc_pts[si], win_size=10)
                    if refined is not None:
                        refined_img_pts.append(refined)
                    else:
                        refined_img_pts.append(all_cc_pts[si])
                    refined_obj_pts.append(all_obj_pts[j])
                    refined_indices.append(kp_idx)
                    used_cc.add(cc_key)
                    matched = True
                    if verbose:
                        print(f"    kp{kp_idx}: 反投影=({px:.4f},{py:.4f}), "
                              f"匹配圆心=({all_cc_pts[si][0]:.4f},{all_cc_pts[si][1]:.4f}), "
                              f"距离={dists[si]:.4f}px")
                    break

            if not matched:
                # 反投影匹配失败时，回退到初始圆心坐标，保证点数不减少
                fallback = initial_cc_map[kp_idx]
                refined_img_pts.append(fallback)
                refined_obj_pts.append(all_obj_pts[j])
                refined_indices.append(kp_idx)
                if verbose:
                    print(f"    kp{kp_idx}: 反投影=({px:.4f},{py:.4f}), "
                          f"未匹配,回退初始圆心=({fallback[0]:.4f},{fallback[1]:.4f})")

        if len(refined_obj_pts) < 4:
            if verbose:
                print(f"  迭代 {iteration+1}: 匹配点不足({len(refined_obj_pts)})，终止")
            break

        refined_obj_pts = np.array(refined_obj_pts, dtype=np.float32)
        refined_img_pts = np.array(refined_img_pts, dtype=np.float32)

        # 重新求解 PnP（RANSAC + ITERATIVE 两步法）
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

        # 计算本轮重投影误差（用全部点评估，不只是内点）
        proj_check, _ = cv2.projectPoints(all_obj_pts, new_rvec, new_tvec, K, dist)
        proj_check = proj_check.reshape(-1, 2)
        new_err = np.linalg.norm(all_cc_pts - proj_check[:len(all_cc_pts)], axis=1).mean()

        # 计算当前重投影误差
        proj_curr, _ = cv2.projectPoints(all_obj_pts, current_rvec, current_tvec, K, dist)
        proj_curr = proj_curr.reshape(-1, 2)
        curr_err = np.linalg.norm(all_cc_pts - proj_curr[:len(all_cc_pts)], axis=1).mean()

        rvec_diff = np.linalg.norm(new_rvec.flatten() - current_rvec.flatten())
        tvec_diff = np.linalg.norm(new_tvec.flatten() - current_tvec.flatten())

        if verbose:
            print(f"  迭代 {iteration+1}: reproj={new_err:.2f}px (prev={curr_err:.2f}px), "
                  f"rvec_diff={rvec_diff:.6f}, tvec_diff={tvec_diff:.4f}")

        # 只在误差下降时更新，否则保留上一轮结果并终止
        if new_err < curr_err:
            current_rvec = new_rvec
            current_tvec = new_tvec
        else:
            if verbose:
                print(f"  迭代 {iteration+1}: 误差未下降，终止迭代")
            break

        # 收敛则提前退出
        if rvec_diff < 1e-5 and tvec_diff < 0.01:
            if verbose:
                print(f"  迭代收敛于第 {iteration+1} 次")
            break

    # 最终重投影误差
    proj_final, _ = cv2.projectPoints(all_obj_pts, current_rvec, current_tvec, K, dist)
    proj_final = proj_final.reshape(-1, 2)
    reproj_error = np.linalg.norm(all_cc_pts - proj_final[:len(all_cc_pts)], axis=1).mean()

    return current_rvec, current_tvec, reproj_error


def process_frame(model, frame, detect_method='ed', K=None, dist=None, verbose=False, coplanar=False):
    """
    对单帧图像执行完整的关键点检测 + 椭圆拟合 + PnP 流程。

    参数:
        model: YOLO 模型实例
        frame: BGR 图像 (numpy array)
        detect_method: 椭圆检测方法，'ed' 或 'pyced'
        K: 相机内参矩阵，若为 None 则使用默认值
        dist: 畸变系数，若为 None 则使用默认值
        verbose: 是否打印详细调试信息

    返回:
        vis_img: 可视化结果图像
        pnp_result: dict, 包含 rvec, tvec, R, reproj_error 等；失败时为 None
    """
    # 相机内参（默认值）
    if K is None:
        K = np.array([[2674.7629874104787, 0., 1279.5],
                       [0., 2674.7629874104787, 719.5],
                       [0., 0., 1.]])
    if dist is None:
        dist = np.array([-0.11744968686298927, 0.27089153364253454,
                          0.0012180578884344092, 0.00067320963008635703,
                          -0.078845410108757258])

    vis_img = frame.copy()
    pnp_result = None
    concentric_results = {}

    # 1. YOLO 关键点检测
    results = model(frame, verbose=False)
    boxes = results[0].boxes
    print("box :",boxes)
    if boxes is None or len(boxes) == 0:
        if verbose: print("  未检测到目标物")
        return vis_img, None

    # 2. 选择关键点最多的目标
    detected_points = get_success_keypoints(results, 0.6)
    best_idx, best_num = 0, 0
    for i, pts in enumerate(detected_points):
        if len(pts) > best_num:
            best_num = len(pts)
            best_idx = i

    if best_num < 4:
        if verbose: print(f"  关键点不足: {best_num}")
        vis_img = visualize_keypoints(frame, results)
        return vis_img, None

    current_points = detected_points[best_idx]

    # 3. 在目标框内进行椭圆检测
    method = detect_method
    if method == 'pyced' and not HAS_PYCED:
        method = 'ed'
    print("use detect_method: ",method)
    boxes_xyxy = boxes.xyxy.cpu().numpy()
    target_box = boxes_xyxy[best_idx:best_idx+1]
    # 计算 box 宽高的最大值
    box_w = target_box[0][2] - target_box[0][0]
    box_h = target_box[0][3] - target_box[0][1]
    box_max_dim = max(box_w, box_h)
    max_diameter = box_max_dim / 2  # 椭圆直径上限

    all_centers, all_ellipses = detect_ellipse_centers_in_roi(frame, target_box, method=method)

    # 过滤椭圆：直径过大 或 长短轴比过小（太扁的）
    min_axis_ratio = 0.8  # 短轴/长轴 最小比值，低于此视为太扁而过滤
    circle_axis_ratio = 0.99
    for i in range(len(all_ellipses)):
        filtered_ellipses = []
        filtered_centers = []
        for j, (cx, cy, a, b, angle) in enumerate(all_ellipses[i]):
            diameter = max(a, b) * 2  # 椭圆长轴直径
            if diameter >= max_diameter:
                continue
            if min(a, b) / max(a, b) < min_axis_ratio:
                continue
            filtered_ellipses.append((cx, cy, a, b, angle))
            filtered_centers.append([cx, cy])
        all_ellipses[i] = filtered_ellipses
        all_centers[i] = np.array(filtered_centers) if filtered_centers else np.empty((0, 2))
    # match_method='hungarian' #'adaptive'  #
    # 4. 邻近匹配 + 同心圆拟合
    for i, (centers, ellipses) in enumerate(zip(all_centers, all_ellipses)):
        if len(centers) == 0:
            continue

        # if match_method == 'hungarian':
        #     # 匈牙利全局最优一对一匹配
        #     neighbors = match_keypoints_hungarian(
        #         [current_points], centers, ellipses, scale_factor=0.80)
        # else:
        #     # 原有贪心邻近匹配 + 离群点过滤
        #     neighbors = get_keypoint_neighbors_adaptive(
        #         [current_points], centers, ellipses, scale_factor=0.8)
        #     neighbors[0] = filter_outliers_by_clustering(neighbors[0], std_multiplier=4.0)

        neighbors = get_keypoint_neighbors_adaptive(
                [current_points], centers, ellipses, scale_factor=0.8)
        neighbors[0] = filter_outliers_by_clustering(neighbors[0], std_multiplier=2.0)

        # 对邻近点集再次筛选：若某关键点的邻近点超过3个，去除面积最大的椭圆
        for kp_idx in list(neighbors[0].keys()):
            nearby = neighbors[0][kp_idx]
            if len(nearby) >= 3:
                # 找邻近点中面积最大的椭圆并移除
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

        if verbose:
            print("  关键点-椭圆中心匹配结果:")
            for kp_idx, kp_x, kp_y in current_points:
                nearby = neighbors[0].get(kp_idx, [])
                print(f"    关键点 {kp_idx} ({kp_x:.4f},{kp_y:.4f}): {len(nearby)} 个邻近点")
                for n_idx, (ncx, ncy) in enumerate(nearby):
                    kp_dist = np.sqrt((ncx - kp_x) ** 2 + (ncy - kp_y) ** 2)
                    print(f"      邻近点 {n_idx}: ({ncx:.4f},{ncy:.4f}), 距离={kp_dist:.4f}")

        for kp_idx, kp_x, kp_y in current_points:
            nearby = neighbors[0].get(kp_idx, [])
            if len(nearby) >= 2:
                # # 取内环（最小面积）椭圆中心
                best_area = float('inf')
                # 取外环（最大面积）椭圆中心
                # best_area = -1
                concentric_center = None
                for ecx, ecy in nearby:
                    for (ex, ey, ea, eb, _) in ellipses:
                        if abs(ex - ecx) < 1.5 and abs(ey - ecy) < 1.5:
                            # 跳过扁椭圆
                            if min(ea, eb) / max(ea, eb) < circle_axis_ratio:
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

            # # 重心法亚像素精化（Otsu + 图像矩）
            # if concentric_center is not None:
            #     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
            #     refined = refine_point_centroid(gray, concentric_center, win_size=10)
            #     if refined is not None:
            #         concentric_center = refined

            concentric_results[kp_idx] = {
                'concentric_center': concentric_center.tolist() if concentric_center is not None else None,
                'raw_nearby_count': len(nearby),
                'raw_nearby': nearby
            }

    # 5. 可视化
    vis_img = visualize_keypoints(frame, results)
    # 收集被匹配到的椭圆中心（round坐标作为key）
    matched_ellipse_set = set()
    for kp_idx, kp_x, kp_y in current_points:
        nearby = neighbors[0].get(kp_idx, [])
        for ecx, ecy in nearby:
            matched_ellipse_set.add((round(ecx, 1), round(ecy, 1)))
    for i, (centers, ellipses) in enumerate(zip(all_centers, all_ellipses)):
        for cx, cy, a, b, angle in ellipses:
            key = (round(cx, 1), round(cy, 1))
            if key in matched_ellipse_set:
                # 匹配到的：黄色中心 + 红色轮廓
                cv2.circle(vis_img, (int(cx), int(cy)), 4, (0, 255, 255), -1)
                cv2.ellipse(vis_img, (int(cx), int(cy)),
                            (int(a), int(b)), int(angle), 0, 360, (0, 0, 255), 2)
            else:
                # 未匹配的：黄色中心 + 绿色轮廓
                cv2.circle(vis_img, (int(cx), int(cy)), 4, (0, 255, 255), -1)
                cv2.ellipse(vis_img, (int(cx), int(cy)),
                            (int(a), int(b)), int(angle), 0, 360, (0, 255, 0), 1)

        # 绘制拟合同心圆坐标 "+"
        for kp_idx in sorted(concentric_results.keys()):
            cc = concentric_results[kp_idx]['concentric_center']
            if cc is None:
                continue
            cx_, cy_ = int(cc[0]), int(cc[1])
            cv2.line(vis_img, (cx_ - 6, cy_), (cx_ + 6, cy_), (0, 0, 0), 2)
            cv2.line(vis_img, (cx_, cy_ - 6), (cx_, cy_ + 6), (0, 0, 0), 2)
            cv2.putText(vis_img, f'cc_{kp_idx}', (cx_ + 8, cy_ - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # 绘制匹配连线
        if i < len(detected_points) and len(centers) > 0:
            centers_list = centers.tolist() if isinstance(centers, np.ndarray) else centers
            center_to_idx = {(round(cx, 1), round(cy, 1)): idx for idx, (cx, cy) in enumerate(centers_list)}

            target_neighbors = get_keypoint_neighbors_adaptive([current_points], centers, ellipses, scale_factor=0.8)
            current_neighbors = filter_outliers_by_clustering(target_neighbors[0], std_multiplier=3.0)
            # 同步：邻近点超过3个时去除面积最大的
            for kp_idx in list(current_neighbors.keys()):
                nearby = current_neighbors[kp_idx]
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
                    current_neighbors[kp_idx].pop(max_area_idx)
            matched_ellipse_indices = set()

            for kp_idx, kp_x, kp_y in current_points:
                nearby_centers = current_neighbors.get(kp_idx, [])
                for ecx, ecy in nearby_centers:
                    cv2.line(vis_img, (int(kp_x), int(kp_y)),
                             (int(ecx), int(ecy)), (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.circle(vis_img, (int(ecx), int(ecy)), 2, (0, 0, 255), 2)
                    key = (round(ecx, 1), round(ecy, 1))
                    if key in center_to_idx:
                        e_idx = center_to_idx[key]
                        if e_idx not in matched_ellipse_indices:
                            _, _, a_e, b_e, angle_e = ellipses[e_idx]
                            cv2.ellipse(vis_img, (int(ecx), int(ecy)),
                                        (int(a_e), int(b_e)), int(angle_e), 0, 360, (0, 0, 255), 2)
                            matched_ellipse_indices.add(e_idx)

    # 6. solvePnP（含反投影迭代精化）
    if concentric_results:
        obj_pts_list = []
        img_pts_list = []
        valid_indices = []

        for kp_idx in sorted(concentric_results.keys()):
            cc = concentric_results[kp_idx]['concentric_center']
            if cc is not None and kp_idx < len(KEYPOINT_OBJ_PTS):
                obj_pts_list.append(KEYPOINT_OBJ_PTS[kp_idx])
                img_pts_list.append(cc)
                valid_indices.append(kp_idx)

        if len(obj_pts_list) >= 4:
            obj_pts = np.array(obj_pts_list, dtype=np.float32)
            img_pts = np.array(img_pts_list, dtype=np.float32)

            # 手动筛选参与PnP的点索引（按 valid_indices 中的顺序）
            select = [0,1,3,5,6]  # 取 valid_indices 中第0和第1个点，可按需修改
            obj_pts = obj_pts[select]
            img_pts = img_pts[select]
            valid_indices = [valid_indices[i] for i in select]

            # PnP 求解
            print(f"  DEBUG: K type={type(K)}, dtype={K.dtype if hasattr(K,'dtype') else 'N/A'}, shape={K.shape if hasattr(K,'shape') else 'N/A'}")
            print(f"  DEBUG: dist type={type(dist)}, dtype={dist.dtype if hasattr(dist,'dtype') else 'N/A'}, shape={dist.shape if hasattr(dist,'shape') else 'N/A'}")
            print(f"  DEBUG: obj_pts type={type(obj_pts)}, dtype={obj_pts.dtype}, shape={obj_pts.shape}")
            print(f"  DEBUG: img_pts type={type(img_pts)}, dtype={img_pts.dtype}, shape={img_pts.shape}")

            if coplanar:
                # 共面点：先用 RANSAC(ITERATIVE) 过滤离群点，再用 IPPE 精化
                print("solvePnPRansac (ITERATIVE for filtering)....")
                success, rvec, tvec, inliers = cv2.solvePnPRansac(
                    obj_pts, img_pts, K, dist,
                    useExtrinsicGuess=False, reprojectionError=10.0, confidence=0.99,
                    flags=cv2.SOLVEPNP_ITERATIVE)
                print("inliers:", len(inliers) if inliers is not None else 0)
                if success and inliers is not None and len(inliers) >= 4:
                    print("solvePnP (IPPE refine)....")
                    inlier_mask = inliers.flatten()
                    success, rvec, tvec = cv2.solvePnP(
                        obj_pts[inlier_mask], img_pts[inlier_mask], K, dist,
                        flags=cv2.SOLVEPNP_IPPE, rvec=rvec, tvec=tvec)
                elif not success or inliers is None or len(inliers) < 4:
                    # RANSAC 失败时，直接用全部点 IPPE 求解
                    print("RANSAC failed, fallback to IPPE with all points....")
                    success, rvec, tvec = cv2.solvePnP(
                        obj_pts, img_pts, K, dist,
                        flags=cv2.SOLVEPNP_IPPE)
            else:
                # 非共面点：RANSAC + ITERATIVE 两步法
                print("solvePnPRansac....")
                success, rvec, tvec, inliers = cv2.solvePnPRansac(
                    obj_pts, img_pts, K, dist,
                    useExtrinsicGuess=False,reprojectionError=3.0, confidence=0.99, flags=cv2.SOLVEPNP_ITERATIVE)
                print("inliers:", len(inliers) if inliers is not None else 0)
                if success and inliers is not None and len(inliers) >= 6:
                    print("solvePnPRansac success, solvePnP....")
                    inlier_mask = inliers.flatten()
                    success, rvec, tvec = cv2.solvePnP(
                        obj_pts[inlier_mask], img_pts[inlier_mask], K, dist,
                        flags=cv2.SOLVEPNP_ITERATIVE, rvec=rvec, tvec=tvec)

            if success:
                # 第一步：基于初始解识别并剔除误匹配点
                proj_pts, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, dist)
                proj_pts = proj_pts.reshape(-1, 2)
                per_point_errors = np.linalg.norm(img_pts - proj_pts, axis=1)
                # 用中位数+MAD 判断离群点，比固定阈值更鲁棒
                median_err = np.median(per_point_errors)
                mad = np.median(np.abs(per_point_errors - median_err))
                outlier_threshold = median_err + 3.0 * max(mad, 1.0)
                good_mask = per_point_errors <= outlier_threshold

                if verbose:
                    print(f"  初始误差: median={median_err:.2f}, MAD={mad:.2f}, 阈值={outlier_threshold:.2f}")
                    for j, kp_idx in enumerate(valid_indices):
                        tag = "OUTLIER" if not good_mask[j] else ""
                        print(f"    kp{kp_idx}: {per_point_errors[j]:.2f}px {tag}")

                # 用好点重新求解 + LM 精化
                if np.sum(good_mask) >= 4 and np.sum(~good_mask) > 0:
                    good_obj = obj_pts[good_mask]
                    good_img = img_pts[good_mask]
                    good_indices = [valid_indices[j] for j in range(len(valid_indices)) if good_mask[j]]

                    print(f"  剔除{np.sum(~good_mask)}个离群点后重解+LM精化....")
                    success2, rvec2, tvec2 = cv2.solvePnP(
                        good_obj, good_img, K, dist,
                        flags=cv2.SOLVEPNP_ITERATIVE)
                    if success2:
                        rvec2, tvec2 = cv2.solvePnPRefineLM(good_obj, good_img, K, dist, rvec2, tvec2)
                        # 比较剔除前后在全部点上的误差
                        proj2, _ = cv2.projectPoints(obj_pts, rvec2, tvec2, K, dist)
                        proj2 = proj2.reshape(-1, 2)
                        err2 = np.linalg.norm(img_pts - proj2, axis=1)
                        print("剔除后,err2: ",err2)
                        if err2[good_mask].mean() < per_point_errors[good_mask].mean():
                            rvec, tvec = rvec2, tvec2
                            if verbose:
                                print(f"  剔除后好点误差: {per_point_errors[good_mask].mean():.2f} -> {err2[good_mask].mean():.2f}px")
                    else:
                        print("剔除后,cv2.solvePnP失败...")
                else:
                    # 所有点都好，直接 LM 精化
                    print("solvePnPRefineLM....")
                    rvec, tvec = cv2.solvePnPRefineLM(obj_pts, img_pts, K, dist, rvec, tvec)

                R, _ = cv2.Rodrigues(rvec)
                proj_pts, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, dist)
                proj_pts = proj_pts.reshape(-1, 2)
                per_point_errors = np.linalg.norm(img_pts - proj_pts, axis=1)
                reproj_error = per_point_errors.mean()

                # 逐点重投影误差
                if verbose:
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

                # 绘制坐标轴
                cv2.drawFrameAxes(vis_img, K, dist, rvec, tvec, 30, 3)

                if verbose:
                    print(f"  PnP: rvec={rvec.flatten()}, tvec={tvec.flatten()}, reproj={reproj_error:.2f}px")

    return vis_img, pnp_result


def batch_detect_and_save(model, img_dir, detect_method='ed', save_dir=None):
    """
    对图像集执行关键点检测 + 邻近圆匹配 + 同心圆拟合，
    将拟合得到的圆心坐标按关键点索引顺序保存到 .txt 文件中。

    每张图像生成一个同名 .txt 文件，格式为：
    x1 y1 x2 y2 x3 y3 ...（按关键点索引 0,1,2,... 排列）
    若某个关键点未拟合到圆心，对应位置写入 -1 -1

    参数:
        model: YOLO 模型实例
        img_dir: 图像目录路径
        detect_method: 椭圆检测方法，'ed' 或 'pyced'
        save_dir: 结果保存目录，若为 None 则保存到 img_dir 下的 txt_output 子目录
    """
    import os
    if save_dir is None:
        save_dir = os.path.join(img_dir, 'txt_output')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    img_files = [f for f in os.listdir(img_dir)
                 if f.lower().endswith(('.jpg', '.png', '.bmp'))]
    print(f"共 {len(img_files)} 张图像，结果保存到: {save_dir}")

    success_count = 0

    for cnt, fname in enumerate(img_files):
        img_path = os.path.join(img_dir, fname)
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"  [{cnt+1}] 跳过无法读取: {fname}")
            continue

        # 1. YOLO 关键点检测
        results = model(frame, verbose=False)
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            print(f"  [{cnt+1}] {fname}: 未检测到目标物")
            continue

        detected_points = get_success_keypoints(results, 0.6)

        # 2. 选择关键点最多的目标
        best_idx, best_num = 0, 0
        for i, pts in enumerate(detected_points):
            if len(pts) > best_num:
                best_num = len(pts)
                best_idx = i

        if best_num < 4:
            print(f"  [{cnt+1}] {fname}: 关键点不足 ({best_num})")
            continue

        current_points = detected_points[best_idx]

        # 3. 在目标框内进行椭圆检测
        method = detect_method
        if method == 'pyced' and not HAS_PYCED:
            method = 'ed'

        boxes_xyxy = boxes.xyxy.cpu().numpy()
        target_box = boxes_xyxy[best_idx:best_idx+1]
        all_centers, all_ellipses = detect_ellipse_centers_in_roi(frame, target_box, method=method)

        # 4. 邻近匹配 + 同心圆拟合
        concentric_results = {}
        for i, (centers, ellipses) in enumerate(zip(all_centers, all_ellipses)):
            if len(centers) == 0:
                continue

            neighbors = get_keypoint_neighbors_adaptive([current_points], centers, ellipses, scale_factor=0.8)
            neighbors[0] = filter_outliers_by_clustering(neighbors[0], std_multiplier=4.0)

            for kp_idx, kp_x, kp_y in current_points:
                nearby = neighbors[0].get(kp_idx, [])
                if len(nearby) >= 2:
                    concentric_center = np.mean(np.array(nearby), axis=0)
                elif len(nearby) == 1:
                    concentric_center = np.array(nearby[0])
                else:
                    concentric_center = None

                concentric_results[kp_idx] = concentric_center.tolist() if concentric_center is not None else None

        # 5. 按关键点索引顺序写入 .txt 文件
        # 获取当前使用的 KEYPOINT_OBJ_PTS 长度作为索引范围
        num_keypoints = len(KEYPOINT_OBJ_PTS)
        coords = []
        for kp_idx in range(num_keypoints):
            cc = concentric_results.get(kp_idx, None)
            if cc is not None:
                coords.append(f"{cc[0]:.4f} {cc[1]:.4f}")
            else:
                coords.append("-1 -1")

        # 构造输出文件名：图像名去掉扩展名 + .txt
        basename = os.path.splitext(fname)[0]
        txt_path = os.path.join(save_dir, basename + '.txt')
        with open(txt_path, 'w') as f:
            f.write(' '.join(coords))

        success_count += 1
        matched = sum(1 for v in concentric_results.values() if v is not None)
        print(f"  [{cnt+1}] {fname}: 拟合 {matched}/{num_keypoints} 个圆心 -> {basename}.txt")

    print(f"\n完成: {success_count}/{len(img_files)} 张图像成功拟合并保存")


def process_frame_with_template(model, frame, detect_method='ed', K=None, dist=None, verbose=False):
    """
    使用 gemiEd.UltimateSocketMatcher 的模板匹配流程进行 PnP 求解。

    与 process_frame 的区别：
    - process_frame: YOLO关键点 → 椭圆邻近匹配 → 同心圆拟合 → PnP
    - 本函数: YOLO检测框 → ROI内椭圆检测 → 模板匹配(UltimateSocketMatcher) → PnP

    适用于：慢充7孔插座（UltimateSocketMatcher内置7孔模板）

    参数:
        model: YOLO 模型实例（仅需检测框，不需要关键点）
        frame: BGR 图像
        detect_method: 椭圆检测方法
        K, dist: 相机内参和畸变系数
        verbose: 是否打印详细信息

    返回:
        vis_img: 可视化图像
        pnp_result: dict 或 None
    """
    try:
        from gemiEd import UltimateSocketMatcher, draw_ellipse
    except ImportError as e:
        print(f"  导入 gemiEd 失败: {e}")
        vis_img = frame
        return vis_img, None

    # 相机内参（默认值）
    if K is None:
        K = np.array([[2674.7629874104787, 0., 1279.5],
                       [0., 2674.7629874104787, 719.5],
                       [0., 0., 1.]])
    if dist is None:
        dist = np.array([-0.11744968686298927, 0.27089153364253454,
                          0.0012180578884344092, 0.00067320963008635703,
                          -0.078845410108757258])

    vis_img = frame.copy()
    pnp_result = None

    # 1. YOLO 检测框
    results = model(frame, verbose=False)
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        if verbose: print("  未检测到目标物")
        return vis_img, None

    boxes_xyxy = boxes.xyxy.cpu().numpy()
    # 选最大的框
    areas = (boxes_xyxy[:, 2] - boxes_xyxy[:, 0]) * (boxes_xyxy[:, 3] - boxes_xyxy[:, 1])
    best_idx = np.argmax(areas)
    box = boxes_xyxy[best_idx]
    x1, y1, x2, y2 = map(int, box)

    # 2. ROI 裁剪 + 椭圆检测
    method = detect_method
    if method == 'pyced' and not HAS_PYCED:
        method = 'ed'

    roi = frame[y1:y2, x1:x2]
    centers_roi, ellipses_roi = detect_ellipse_centers(roi, method=method)

    # 还原到原图坐标
    ellipses_original = [(cx + x1, cy + y1, a, b, angle)
                         for cx, cy, a, b, angle in ellipses_roi]

    if verbose:
        print(f"  检测框: [{x1},{y1},{x2},{y2}]")
        print(f"  ROI内检测到 {len(ellipses_original)} 个椭圆")

    # 3. UltimateSocketMatcher 模板匹配
    rect = [x1, y1, x2 - x1, y2 - y1]  # tl_x, tl_y, w, h
    matcher = UltimateSocketMatcher()
    # 替换 matcher 的内参为当前相机参数
    matcher.K = np.array(K, dtype=np.float32)
    matcher.dist = np.array(dist, dtype=np.float32)

    final_pts, status, matched_centers = matcher.solve(ellipses_original, rect)

    if final_pts is None:
        if verbose: print("  模板匹配失败")
        # 绘制检测框
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        return vis_img, None

    if verbose:
        print(f"  模板匹配得分: {status}")

    # 4. PnP 求解
    rvec, tvec, proj_back = matcher.estimate_pose(matched_centers, None)

    if rvec is not None:
        R, _ = cv2.Rodrigues(rvec)
        # 计算重投影误差
        reprojected_pts, _ = cv2.projectPoints(matcher.obj_pts, rvec, tvec, K, dist)
        reprojected_pts = reprojected_pts.squeeze()
        img_pts = np.array(matched_centers, dtype=np.float32) + np.array([x1, y1], dtype=np.float32)
        per_point_errors = np.linalg.norm(img_pts - reprojected_pts, axis=1)
        reproj_error = per_point_errors.mean()

        if verbose:
            print(f"  PnP: rvec={rvec.flatten()}, tvec={tvec.flatten()}")
            print(f"  重投影误差: {reproj_error:.2f}px")
            print("  逐点重投影误差:")
            for j in range(len(per_point_errors)):
                print(f"    pt{j}: {per_point_errors[j]:.2f}px")

        pnp_result = {
            'rvec': rvec, 'tvec': tvec, 'R': R,
            'reproj_error': reproj_error,
            'per_point_errors': {str(j): round(float(per_point_errors[j]), 4)
                                 for j in range(len(per_point_errors))},
            'num_matched': len(matched_centers),
        }

        # 绘制
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # 绘制匹配到的椭圆中心
        for cx, cy in matched_centers:
            cx_full, cy_full = int(cx + x1), int(cy + y1)
            cv2.circle(vis_img, (cx_full, cy_full), 5, (0, 255, 255), -1)
        # 绘制坐标轴
        cv2.drawFrameAxes(vis_img, K, dist, rvec, tvec, 30, 3)

    return vis_img, pnp_result


def run_on_images(model, img_dir, detect_method='ed', K=None, dist=None, save_dir=None):
    """
    对图像集进行批量处理。

    参数:
        model: YOLO 模型实例
        img_dir: 图像目录路径
        detect_method: 椭圆检测方法
        K, dist: 相机内参和畸变系数
        save_dir: 结果保存目录，若为 None 则不保存
    """
    import os
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    img_files = [f for f in os.listdir(img_dir)
                 if f.lower().endswith(('.jpg', '.png', '.bmp'))]
    print(f"共 {len(img_files)} 张图像")

    all_results = {}

    for cnt, fname in enumerate(img_files):
        img_path = os.path.join(img_dir, fname)
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"  跳过无法读取: {fname}")
            continue

        t0 = time.perf_counter()
        vis_img, pnp_result = process_frame(model, frame, detect_method, K, dist, verbose=True)
        elapsed = (time.perf_counter() - t0) * 1000

        status = "OK" if pnp_result else "FAIL"
        if pnp_result:
            print(f"[{cnt+1}/{len(img_files)}] {fname}: {status}, "
                  f"reproj={pnp_result['reproj_error']:.2f}px, {elapsed:.0f}ms")
            all_results[fname] = {
                'rvec': pnp_result['rvec'].flatten().tolist(),
                'tvec': pnp_result['tvec'].flatten().tolist(),
                'R': pnp_result['R'].tolist(),
                'reproj_error': float(pnp_result['reproj_error']),
                'per_point_errors': pnp_result.get('per_point_errors', {}),
            }
        else:
            print(f"[{cnt+1}/{len(img_files)}] {fname}: {status}, {elapsed:.0f}ms")
            all_results[fname] = None

        if save_dir:
            cv2.imwrite(os.path.join(save_dir, fname), vis_img)

        cv2.imshow("result", vis_img)
        key = cv2.waitKey(0)
        if key == 27:  # ESC 退出
            break

    cv2.destroyAllWindows()

    # 保存 PnP 结果到 JSON
    if save_dir:
        import json
        json_path = os.path.join(save_dir, 'pnp_results.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"PnP 结果已保存到: {json_path}")


def evaluate_pnp_errors(json_path, Mc2b, save_path=None):
    """
    读取 PnP 结果 JSON，将每张图的 rvec/tvec 转为 4x4 齐次矩阵，
    乘以 Mc2b 转到基坐标系，计算相邻帧之间的旋转误差和平移误差。

    参数:
        json_path: pnp_results.json 文件路径
        Mc2b: 4x4 numpy 数组，相机坐标系到机械臂基坐标系的变换矩阵
        save_path: 误差结果保存路径，若为 None 则保存到与 json_path 同目录

    输出 JSON 格式:
        {
            "fname_i": {
                "rotation_error_deg": ...,
                "translation_error_mm": ...
            },
            ...
        }
    """
    import json

    with open(json_path, 'r', encoding='utf-8') as f:
        all_results = json.load(f)

    Mc2b = np.array(Mc2b, dtype=np.float64)
    if Mc2b.shape != (4, 4):
        raise ValueError(f"Mc2b 必须是 4x4 矩阵，当前形状: {Mc2b.shape}")

    # 按 fname 排序，保证顺序一致
    fnames = sorted([k for k in all_results if all_results[k] is not None])

    # 将每帧的 rvec, tvec -> 4x4 齐次矩阵 -> 乘 Mc2b 得到基坐标系下的位姿
    base_poses = []
    for fname in fnames:
        res = all_results[fname]
        rvec = np.array(res['rvec'], dtype=np.float64)
        tvec = np.array(res['tvec'], dtype=np.float64)
        R, _ = cv2.Rodrigues(rvec)
        # 构建 4x4 齐次变换矩阵 (camera-to-object -> 实际是 object 在 camera 下的位姿)
        T_c = np.eye(4)
        T_c[:3, :3] = R
        T_c[:3, 3] = tvec.flatten()
        # 转到基坐标系
        T_b = Mc2b @ T_c
        base_poses.append((fname, T_b))

    # 计算相邻帧之间的旋转误差和平移误差
    errors = {}
    for i in range(1, len(base_poses)):
        fname_prev, T_prev = base_poses[i - 1]
        fname_curr, T_curr = base_poses[i]

        # 相对变换: T_rel = T_prev^{-1} @ T_curr
        T_prev_inv = np.linalg.inv(T_prev)
        T_rel = T_prev_inv @ T_curr

        R_rel = T_rel[:3, :3]
        t_rel = T_rel[:3, 3]

        # 旋转误差: 分解为 ZYX 欧拉角 (yaw, pitch, roll)
        # R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
        sy = np.sqrt(R_rel[0, 0]**2 + R_rel[1, 0]**2)
        singular = sy < 1e-6
        if not singular:
            roll  = np.arctan2(R_rel[2, 1], R_rel[2, 2])   # 绕 X
            pitch = np.arctan2(-R_rel[2, 0], sy)             # 绕 Y
            yaw   = np.arctan2(R_rel[1, 0], R_rel[0, 0])    # 绕 Z
        else:
            roll  = np.arctan2(-R_rel[1, 2], R_rel[1, 1])
            pitch = np.arctan2(-R_rel[2, 0], sy)
            yaw   = 0.0

        roll_deg  = np.degrees(roll)
        pitch_deg = np.degrees(pitch)
        yaw_deg   = np.degrees(yaw)
        rot_total_deg = np.degrees(np.linalg.norm([roll, pitch, yaw]))

        # 平移误差: 相对平移向量的范数
        trans_error = np.linalg.norm(t_rel)

        errors[fname_curr] = {
            'rotation_error_deg': round(rot_total_deg, 6),
            'rotation_roll_deg': round(roll_deg, 6),
            'rotation_pitch_deg': round(pitch_deg, 6),
            'rotation_yaw_deg': round(yaw_deg, 6),
            'translation_error_mm': round(trans_error, 6),
            'reference_frame': fname_prev,
        }

    # 保存结果
    if save_path is None:
        save_dir = os.path.dirname(json_path)
        save_path = os.path.join(save_dir, 'pnp_errors.json')

    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(errors, f, indent=2, ensure_ascii=False)
    print(f"误差结果已保存到: {save_path}")

    # 打印摘要
    if errors:
        rot_vals   = [v['rotation_error_deg'] for v in errors.values()]
        roll_vals  = [v['rotation_roll_deg'] for v in errors.values()]
        pitch_vals = [v['rotation_pitch_deg'] for v in errors.values()]
        yaw_vals   = [v['rotation_yaw_deg'] for v in errors.values()]
        trans_vals = [v['translation_error_mm'] for v in errors.values()]
        print(f"共 {len(errors)} 对相邻帧:")
        print(f"  旋转误差(总):  mean={np.mean(rot_vals):.4f}°, max={np.max(rot_vals):.4f}°, min={np.min(rot_vals):.4f}°")
        print(f"  Roll  (X轴):   mean={np.mean(roll_vals):.4f}°, max={np.max(roll_vals):.4f}°, min={np.min(roll_vals):.4f}°")
        print(f"  Pitch (Y轴):   mean={np.mean(pitch_vals):.4f}°, max={np.max(pitch_vals):.4f}°, min={np.min(pitch_vals):.4f}°")
        print(f"  Yaw   (Z轴):   mean={np.mean(yaw_vals):.4f}°, max={np.max(yaw_vals):.4f}°, min={np.min(yaw_vals):.4f}°")
        print(f"  平移误差:      mean={np.mean(trans_vals):.4f}mm, max={np.max(trans_vals):.4f}mm, min={np.min(trans_vals):.4f}mm")

    return errors


def batch_process_to_base(model, img_dir, M_cam2end, detect_method='ed', K=None, dist=None,
                          verbose=False, coplanar=False, save_dir=None):
    """
    遍历图像文件夹及对应的 .npy 文件，执行 process_frame 得到 PnP 结果，
    结合手眼标定矩阵和机械臂位姿(从 .npy 读取)计算物体在基坐标系下的位姿，
    保存到 JSON 文件，并计算相邻帧间的旋转/平移误差。

    参数:
        model: YOLO 模型实例
        img_dir: 图像目录，其中每张图像需有同名 .npy 文件（4x4 矩阵，M_end2base）
        M_cam2end: 4x4 numpy 数组，相机到末端的手眼标定矩阵
        detect_method: 椭圆检测方法
        K, dist: 相机内参和畸变系数
        verbose: 是否打印详细信息
        coplanar: 是否使用共面 PnP 方法
        save_dir: 结果保存目录，若为 None 则保存到 img_dir 下的 base_output 子目录

    输出:
        base_poses.json: 每帧在基坐标系下的位姿
        base_errors.json: 相邻帧间的旋转/平移误差
    """
    import json

    M_cam2end = np.array(M_cam2end, dtype=np.float64)
    if M_cam2end.shape != (4, 4):
        raise ValueError(f"M_cam2end 必须是 4x4 矩阵，当前形状: {M_cam2end.shape}")

    if save_dir is None:
        save_dir = os.path.join(img_dir, 'base_output')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    img_files = sorted([f for f in os.listdir(img_dir)
                        if f.lower().endswith(('.jpg', '.png', '.bmp'))])
    print(f"共 {len(img_files)} 张图像")

    all_base_poses = {}
    all_results = {}

    for cnt, fname in enumerate(img_files):
        img_path = os.path.join(img_dir, fname)
        npy_path = os.path.join(img_dir, os.path.splitext(fname)[0] + '.npy')

        frame = cv2.imread(img_path)
        if frame is None:
            print(f"  [{cnt+1}] 跳过无法读取: {fname}")
            continue

        # 读取 M_end2base
        if not os.path.exists(npy_path):
            print(f"  [{cnt+1}] {fname}: 未找到对应 .npy 文件，跳过")
            continue
        M_end2base = np.load(npy_path)
        if M_end2base.shape != (4, 4):
            print(f"  [{cnt+1}] {fname}: .npy 不是 4x4 矩阵(shape={M_end2base.shape})，跳过")
            continue

        t0 = time.perf_counter()
        vis_img, pnp_result = process_frame(model, frame, detect_method, K, dist, verbose, coplanar)
        elapsed = (time.perf_counter() - t0) * 1000
        cv2.imshow("PnP_img",vis_img)
        cv2.waitKey(0)
        # 保存可视化图像
        cv2.imwrite(os.path.join(save_dir, fname), vis_img)

        if pnp_result is None:
            print(f"  [{cnt+1}] {fname}: PnP 失败, {elapsed:.0f}ms")
            all_results[fname] = None
            continue

        # 1. T_obj2cam
        R, _ = cv2.Rodrigues(pnp_result['rvec'])
        t = pnp_result['tvec'].flatten()
        T_obj2cam = np.eye(4)
        T_obj2cam[:3, :3] = R
        T_obj2cam[:3, 3] = t

        # 2. T_obj2base
        T_obj2base = M_end2base @ M_cam2end @ T_obj2cam

        R_base = T_obj2base[:3, :3]
        t_base = T_obj2base[:3, 3]
        rvec_base, _ = cv2.Rodrigues(R_base)

        all_base_poses[fname] = {
            'rvec_base': rvec_base.flatten().tolist(),
            'tvec_base': t_base.tolist(),
            'T_obj2base': T_obj2base.tolist(),
        }
        all_results[fname] = {
            'rvec': pnp_result['rvec'].flatten().tolist(),
            'tvec': pnp_result['tvec'].flatten().tolist(),
            'reproj_error': float(pnp_result['reproj_error']),
            'per_point_errors': pnp_result.get('per_point_errors', {}),
            'T_obj2base': T_obj2base.tolist(),
        }

        print(f"  [{cnt+1}] {fname}: reproj={pnp_result['reproj_error']:.2f}px, "
              f"t_base=[{t_base[0]:.1f},{t_base[1]:.1f},{t_base[2]:.1f}], {elapsed:.0f}ms")

    # 保存基坐标系位姿到 JSON
    poses_path = os.path.join(save_dir, 'base_poses.json')
    with open(poses_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"基坐标系位姿已保存到: {poses_path}")

    # 计算相邻帧误差
    fnames = sorted([k for k in all_results if all_results[k] is not None])
    errors = {}
    for i in range(1, len(fnames)):
        fname_prev = fnames[i - 1]
        fname_curr = fnames[i]

        T_prev = np.array(all_results[fname_prev]['T_obj2base'], dtype=np.float64)
        T_curr = np.array(all_results[fname_curr]['T_obj2base'], dtype=np.float64)

        T_rel = np.linalg.inv(T_prev) @ T_curr
        R_rel = T_rel[:3, :3]
        t_rel = T_rel[:3, 3]

        # ZYX 欧拉角分解
        sy = np.sqrt(R_rel[0, 0]**2 + R_rel[1, 0]**2)
        singular = sy < 1e-6
        if not singular:
            roll  = np.arctan2(R_rel[2, 1], R_rel[2, 2])
            pitch = np.arctan2(-R_rel[2, 0], sy)
            yaw   = np.arctan2(R_rel[1, 0], R_rel[0, 0])
        else:
            roll  = np.arctan2(-R_rel[1, 2], R_rel[1, 1])
            pitch = np.arctan2(-R_rel[2, 0], sy)
            yaw   = 0.0

        roll_deg  = np.degrees(roll)
        pitch_deg = np.degrees(pitch)
        yaw_deg   = np.degrees(yaw)
        rot_total_deg = np.degrees(np.linalg.norm([roll, pitch, yaw]))
        trans_error = np.linalg.norm(t_rel)

        errors[fname_curr] = {
            'rotation_error_deg': round(rot_total_deg, 6),
            'rotation_roll_deg': round(roll_deg, 6),
            'rotation_pitch_deg': round(pitch_deg, 6),
            'rotation_yaw_deg': round(yaw_deg, 6),
            'translation_error_mm': round(trans_error, 6),
            'reference_frame': fname_prev,
        }

    # 保存误差结果
    errors_path = os.path.join(save_dir, 'base_errors.json')
    with open(errors_path, 'w', encoding='utf-8') as f:
        json.dump(errors, f, indent=2, ensure_ascii=False)
    print(f"误差结果已保存到: {errors_path}")

    # 打印摘要
    if errors:
        rot_vals   = [v['rotation_error_deg'] for v in errors.values()]
        roll_vals  = [v['rotation_roll_deg'] for v in errors.values()]
        pitch_vals = [v['rotation_pitch_deg'] for v in errors.values()]
        yaw_vals   = [v['rotation_yaw_deg'] for v in errors.values()]
        trans_vals = [v['translation_error_mm'] for v in errors.values()]
        print(f"\n共 {len(errors)} 对相邻帧:")
        print(f"  旋转误差(总):  mean={np.mean(rot_vals):.4f}°, max={np.max(rot_vals):.4f}°, min={np.min(rot_vals):.4f}°")
        print(f"  Roll  (X轴):   mean={np.mean(roll_vals):.4f}°, max={np.max(roll_vals):.4f}°, min={np.min(roll_vals):.4f}°")
        print(f"  Pitch (Y轴):   mean={np.mean(pitch_vals):.4f}°, max={np.max(pitch_vals):.4f}°, min={np.min(pitch_vals):.4f}°")
        print(f"  Yaw   (Z轴):   mean={np.mean(yaw_vals):.4f}°, max={np.max(yaw_vals):.4f}°, min={np.min(yaw_vals):.4f}°")
        print(f"  平移误差:      mean={np.mean(trans_vals):.4f}mm, max={np.max(trans_vals):.4f}mm, min={np.min(trans_vals):.4f}mm")

    return all_results, errors


def run_on_video(model, video_source=0, detect_method='ed', K=None, dist=None, save_video=None):
    """
    对视频流或摄像头进行持续处理。

    参数:
        model: YOLO 模型实例
        video_source: 视频路径或摄像头编号（默认 0）
        detect_method: 椭圆检测方法
        K, dist: 相机内参和畸变系数
        save_video: 保存视频路径，若为 None 则不保存
    """
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"无法打开视频源: {video_source}")
        return

    # 视频写入器
    writer = None
    if save_video:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        writer = cv2.VideoWriter(save_video, fourcc, fps, (w, h))

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("视频结束或读取失败")
            break

        frame_count += 1
        t0 = time.perf_counter()
        vis_img, pnp_result = process_frame(model, frame, detect_method, K, dist, verbose=False)
        elapsed = (time.perf_counter() - t0) * 1000

        # 在画面上显示状态信息
        status_text = f"FPS: {1000/elapsed:.1f}" if elapsed > 0 else "FPS: --"
        if pnp_result:
            status_text += f" | reproj: {pnp_result['reproj_error']:.2f}px"
            tvec = pnp_result['tvec'].flatten()
            status_text += f" | t: [{tvec[0]:.1f}, {tvec[1]:.1f}, {tvec[2]:.1f}]"
        else:
            status_text += " | PnP: FAIL"

        cv2.putText(vis_img, status_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if writer:
            writer.write(vis_img)

        cv2.imshow("socket detection", vis_img)
        key = cv2.waitKey(1)
        if key == 27:  # ESC 退出
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print(f"共处理 {frame_count} 帧")


if __name__ =="__main__":





    # ===== 选择椭圆检测方法 =====
    # 'ed'    : 使用 OpenCV EdgeDrawing
    # 'pyced' : 使用 pycylinderedsf CED
    DETECT_METHOD = 'ed'
    # DETECT_METHOD = 'pyced'


    img_dir = "D:\Data\manchong\data"
    model = YOLO("./slowport_yolo_Pose.pt")
    # model = YOLO("./superport_yolo_Pose.pt")

    # 批量处理图像集
    #batch_detect_and_save(model,img_dir)


    #处理单帧 (model, frame, detect_method='ed', K=None, dist=None, verbose=False
    img_path = "./2026-05-11_10_47_33_027.png"
    # img_path = "./2026-04-22_15_24_12_519859201974349.png" 
    src_mat = cv2.imread(img_path)

#小相机1280-720
    K = np.array([
        [1015.445938660267, 0., 638.51741890470555],
        [0., 1015.445938660267, 386.838616473841],
        [0., 0., 1.]
        ], dtype=np.float64)

    dist = np.array([
    0.11753195467413819, -0.19301774104640848,
    0.00016793575097772418, -0.00061144051421409198, 0.072260521199194336
    ], dtype=np.float64)

    M_cam2end = np.array([
        [-6.9855857e-01,  7.1512282e-01,  2.4804471e-02, -5.1826664e+01],
        [-7.1555281e-01, -6.9815123e-01, -2.3854841e-02,  5.5274796e+01],
        [ 2.5813223e-04, -3.4412913e-02,  9.9940765e-01,  9.5362617e+01],
        [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  1.0000000e+00]
        ], dtype=np.float64)



#大相机 2560-1440
    # K = np.array([[2674.7629874104787,0.,1279.5],[0.,2674.7629874104787,719.5],[0.,0.,1.]], dtype=np.float64)
  
    # dist = np.array([-0.11744968686298927,0.27089153364253454,0.0012180578884344092,0.00067320963008635703,-0.078845410108757258], dtype=np.float64)

    # M_cam2end = np.array([[-7.2267956e-01,  6.9102561e-01, -1.4759262e-02 ,-5.1758522e+01],
    #     [-6.9116789e-01, -7.2264087e-01,  8.7790741e-03,  6.0040222e+01],
    #     [-4.5990809e-03,  1.6545586e-02,  9.9985254e-01,  9.7955963e+01],
    #     [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  1.0000000e+00]], dtype=np.float64)



    # 计算图像集的误差
    #batch_process_to_base(model, img_dir, M_cam2end, detect_method='ed', K=None, dist=None,
    #                      verbose=False, coplanar=False, save_dir=None):

    imgs_dir = "D:\Data\manchong\save_data5"
    save_dir = "D:\\Vscode\\code\\result_data5"
    batch_process_to_base(model,imgs_dir,M_cam2end,DETECT_METHOD,K,dist,True,True,save_dir)

    # ===== 方式1：YOLO-Pose 关键点流程（快充9孔） =====
    # vis_img, pnp_result = process_frame(model, src_mat, DETECT_METHOD, K, dist, True, coplanar=False)

    # # ===== 方式2：UltimateSocketMatcher 模板匹配流程（慢充7孔） =====
    # # 注意：需要使用对应的 YOLO 检测模型（仅需检测框），如 yolo_pose_port513.pt
    # # model_slow = YOLO("./checkpoint/best.pt")
    # # vis_img, pnp_result = process_frame_with_template(
    # #     model_slow, src_mat, 'ed', K, dist, verbose=True)

    # cv2.imshow("vis image", vis_img)
    # cv2.waitKey(0)
    # cv2.imwrite("vis_image.jpg", vis_img)



    # # 计算装配姿态：物体坐标系 → 机械臂基坐标系
    # if pnp_result is not None:
    #     # 1. rvec, tvec → 4x4 齐次变换矩阵 T_obj2cam
    #     R, _ = cv2.Rodrigues(pnp_result['rvec'])
    #     t = pnp_result['tvec'].flatten()
    #     T_obj2cam = np.eye(4)
    #     T_obj2cam[:3, :3] = R
    #     T_obj2cam[:3, 3] = t
    #     print(f"T_obj2cam:\n{T_obj2cam}")

    #     # 2. 通过手眼标定矩阵，转到基坐标系
    #     #    M_cam2end: 相机到末端变换矩阵 (手眼标定结果)
    #     #    M_end2base: 末端到基座变换矩阵 (机械臂当前位姿，从示教器/SDK读取)
    #     #
    #     #    链式变换: T_obj2base = M_end2base @ M_cam2end @ T_obj2cam
    #     #
        

    #     M_end2base = np.eye(4)  # ← 替换为机械臂当前位姿
    #     T_obj2base = M_end2base @ M_cam2end @ T_obj2cam
    #     print(f"T_obj2base:\n{T_obj2base}")
    #     #
    #     # # 提取基坐标系下的位姿
    #     # R_base = T_obj2base[:3, :3]
    #     # t_base = T_obj2base[:3, 3]
    #     # rvec_base, _ = cv2.Rodrigues(R_base)
    #     # print(f"装配姿态 rvec(base): {rvec_base.flatten()}")
    #     # print(f"装配位置 tvec(base): {t_base}")
    # else:
    #     print("PnP 求解失败，无法计算装配姿态")

 
    # img_path = "./2026-05-11_10_47_33_027.png"
    # src_mat = cv2.imread(img_path)
    # results = model(img_path)
    # print("results shape",results)
   
    # # 判断关键点检测是否成功：至少检测到 1 个目标且关键点 > 4
    # detected_points = get_success_keypoints(results, 0.6)
    # has_valid_detection = len(detected_points) > 0 and len(detected_points[0]) > 4

    # print(f"keypoints size: {len(detected_points)}, 有效目标: {has_valid_detection}")
   
    # if not has_valid_detection:
    #     print("警告: 未检测到有效目标或关键点不足，跳过椭圆检测")

    # for i, points in enumerate(detected_points):
    #     print(f"第 {i+1} 个目标检测成功点（索引，x,y）:")
    #     for idx, x, y in points:
    #         print(f"  索引:{idx}, 坐标: ({x:.2f},{y:.2f})")

    # concentric_results = {}  # 存储拟合圆心结果，供后续 PnP 使用

    # # --- 在 YOLO 检测框内进行椭圆检测 ---
    # if not has_valid_detection:
    #     vis_img = visualize_keypoints(src_mat, results)
    # elif len(results) > 0 and results[0].boxes is not None:
    #     boxes = results[0].boxes.xyxy.cpu().numpy()

    #     if DETECT_METHOD == 'pyced' and not HAS_PYCED:
    #         print("pyced 未安装，自动切换为 OpenCV EdgeDrawing")
    #         DETECT_METHOD = 'ed'

    #     all_centers, all_ellipses = detect_ellipse_centers_in_roi(
    #         src_mat, boxes, method=DETECT_METHOD
    #     )

    #     for i, (centers, ellipses) in enumerate(zip(all_centers, all_ellipses)):
    #         print(f"\n第 {i+1} 个 ROI 内检测到 {len(centers)} 个椭圆 (method={DETECT_METHOD}):")
    #         for cx, cy, a, b, angle in ellipses:
    #             print(f"  中心:({cx:.1f},{cy:.1f}), 轴:({a:.1f},{b:.1f}), 角度:{angle:.1f}")

    #         # 将 YOLO 关键点与椭圆中心做邻近匹配
    #         if i < len(detected_points) and len(centers) > 0:
    #             neighbors = get_keypoint_neighbors_adaptive([detected_points[i]], centers, ellipses, scale_factor=0.8)
    #             # 精筛选：剔除远离主聚集区的离群点（如个别偏远同心圆杂点）
    #             neighbors[0] = filter_outliers_by_clustering(neighbors[0], std_multiplier=4.0)

    #             # 对每个关键点的邻近椭圆集做同心圆拟合，得到拟合后的同心圆坐标
    #             # concentric_results = {}
    #             for kp_idx, kp_x, kp_y in detected_points[i]:
    #                 nearby = neighbors[0].get(kp_idx, [])
    #                 if len(nearby) >= 2:
    #                     pts_arr = np.array(nearby)
    #                     concentric_center = np.mean(pts_arr, axis=0)
    #                 elif len(nearby) == 1:
    #                     concentric_center = np.array(nearby[0])
    #                 else:
    #                     concentric_center = None

    #                 concentric_results[kp_idx] = {
    #                     'concentric_center': concentric_center.tolist() if concentric_center is not None else None,
    #                     'raw_nearby_count': len(nearby),
    #                     'raw_nearby': nearby
    #                 }

    #             print(f"  关键点-椭圆中心匹配结果:")
    #             for kp_idx, kp_x, kp_y in detected_points[i]:
    #                 nearby = neighbors[0].get(kp_idx, [])
    #                 print(f"    关键点 {kp_idx} ({kp_x:.1f},{kp_y:.1f}): {len(nearby)} 个邻近点")
    #                 for n_idx, (ncx, ncy) in enumerate(nearby):
    #                     dist = np.sqrt((ncx - kp_x) ** 2 + (ncy - kp_y) ** 2)
    #                     print(f"      邻近点 {n_idx}: ({ncx:.1f},{ncy:.1f}), 距离={dist:.2f}")

    #     # 可视化：关键点 + 椭圆中心 + 匹配连线
    #     vis_img = visualize_keypoints(src_mat, results)
    #     # 绘制椭圆中心和椭圆轮廓
    #     for i, (centers, ellipses) in enumerate(zip(all_centers, all_ellipses)):
    #         for cx, cy, a, b, angle in ellipses:
    #             cv2.circle(vis_img, (int(cx), int(cy)), 4, (0, 255, 255), -1)  # 黄色：椭圆中心
    #             cv2.ellipse(vis_img, (int(cx), int(cy)),
    #                         (int(a), int(b)), int(angle), 0, 360, (0, 255, 0), 1)  # 绿色：椭圆轮廓

    #         # 用 "+" 标记绘制拟合后的同心圆坐标
    #         if i < len(detected_points) and len(centers) > 0:
    #             _neighbors = get_keypoint_neighbors_adaptive([detected_points[i]], centers, ellipses, scale_factor=0.8)
    #             _neighbors[0] = filter_outliers_by_clustering(_neighbors[0], std_multiplier=4.0)

    #             for kp_idx, kp_x, kp_y in detected_points[i]:
    #                 nearby = _neighbors[0].get(kp_idx, [])
    #                 if not nearby:
    #                     continue
    #                 pts_arr = np.array(nearby)
    #                 cc = np.mean(pts_arr, axis=0)
    #                 cx_, cy_ = int(cc[0]), int(cc[1])
    #                 line_len = 6
    #                 thickness = 2
    #                 # 品红色 "+" 标记
    #                 cv2.line(vis_img, (cx_ - line_len, cy_), (cx_ + line_len, cy_), (0, 0, 0), thickness)
    #                 cv2.line(vis_img, (cx_, cy_ - line_len), (cx_, cy_ + line_len), (0, 0, 0), thickness)
    #                 cv2.putText(vis_img, f'cc_{kp_idx}', (cx_ + 8, cy_ - 8),
    #                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    #         # 绘制关键点与邻近椭圆中心的匹配连线
    #         if i < len(detected_points) and len(centers) > 0:
    #             # 用索引匹配：构建 centers 坐标 → 索引 的查找表
    #             centers_list = centers.tolist() if isinstance(centers, np.ndarray) else centers
    #             center_to_idx = {(round(cx, 1), round(cy, 1)): idx for idx, (cx, cy) in enumerate(centers_list)}

    #             # target_neighbors = get_keypoint_neighbors([detected_points[i]], centers, distance_threshold=15.0)
    #             target_neighbors = get_keypoint_neighbors_adaptive([detected_points[i]], centers, ellipses, scale_factor=0.8)
    #             current_neighbors = filter_outliers_by_clustering(target_neighbors[0], std_multiplier=4.0)  # 当前目标的邻近匹配字典（已剔除离群点）
    #             matched_ellipse_indices = set()  # 已匹配的椭圆索引

    #             for kp_idx, kp_x, kp_y in detected_points[i]:
    #                 nearby_centers = current_neighbors.get(kp_idx, [])  # 该关键点的全部邻近椭圆中心
    #                 for ecx, ecy in nearby_centers:
    #                     # 红色连线：关键点 → 匹配的椭圆中心
    #                     cv2.line(vis_img, (int(kp_x), int(kp_y)),
    #                              (int(ecx), int(ecy)), (0, 0, 255), 1, cv2.LINE_AA)
    #                     # 红色圆环：匹配到的椭圆中心
    #                     cv2.circle(vis_img, (int(ecx), int(ecy)), 2, (0, 0, 255), 2)
    #                     # 通过四舍五入坐标查找索引，绘制对应的椭圆轮廓
    #                     key = (round(ecx, 1), round(ecy, 1))
    #                     if key in center_to_idx:
    #                         e_idx = center_to_idx[key]
    #                         if e_idx not in matched_ellipse_indices:
    #                             _, _, a_e, b_e, angle_e = ellipses[e_idx]
    #                             cv2.ellipse(vis_img, (int(ecx), int(ecy)),
    #                                         (int(a_e), int(b_e)), int(angle_e), 0, 360, (0, 0, 255), 2)
    #                             matched_ellipse_indices.add(e_idx)
    # else:
    #     vis_img = visualize_keypoints(src_mat, results)

    # cv2.imshow("keypoints + ellipse centers", vis_img)
    # cv2.waitKey(0)
    # cv2.imwrite("vis_outputimg.jpg", vis_img)

    # # 使用 cv2.solvePnP() 求解位姿：根据拟合圆心与 KEYPOINT_OBJ_PTS 的对应关系
    # print(f"\n--- solvePnP 调试信息 ---")
    # print(f"  concentric_results: {concentric_results}")

    # if concentric_results:
    #     obj_pts_list = []  # 三维物点
    #     img_pts_list = []  # 对应的二维图像点（拟合圆心）
    #     valid_indices = []  # 有效匹配的关键点索引

    #     for kp_idx in sorted(concentric_results.keys()):
    #         cc = concentric_results[kp_idx]['concentric_center']
    #         if cc is not None and kp_idx < len(KEYPOINT_OBJ_PTS):
    #             obj_pts_list.append(KEYPOINT_OBJ_PTS[kp_idx])
    #             img_pts_list.append(cc)
    #             valid_indices.append(kp_idx)

    #     print(f"  有效匹配点数: {len(obj_pts_list)}, 索引: {valid_indices}")
    #     for idx, (obj, img) in enumerate(zip(obj_pts_list, img_pts_list)):
    #         print(f"    kp{valid_indices[idx]+1}: obj={obj}, img=({img[0]:.1f},{img[1]:.1f})")

    #     if len(obj_pts_list) >= 4:
    #         obj_pts = np.array(obj_pts_list, dtype=np.float32)
    #         img_pts = np.array(img_pts_list, dtype=np.float32)

    #         # 相机内参矩阵
    #         K = np.array([[2674.7629874104787, 0., 1279.5],
    #                       [0., 2674.7629874104787, 719.5],
    #                       [0., 0., 1.]])
    #         dist = np.array([-0.11744968686298927, 0.27089153364253454,
    #                          0.0012180578884344092, 0.00067320963008635703,
    #                          -0.078845410108757258])

    #         print(f"  img_pts range: x=[{img_pts[:,0].min():.1f}, {img_pts[:,0].max():.1f}], y=[{img_pts[:,1].min():.1f}, {img_pts[:,1].max():.1f}]")
    #         print(f"  image size: {src_mat.shape[:2]}")

    #         # 先用 RANSAC 剔除可能的误匹配外点，再用 ITERATIVE 精化
    #         success, rvec, tvec, inliers = cv2.solvePnPRansac(obj_pts, img_pts, K, dist,
    #                                                            confidence=0.99, reprojectionError=8.0)
    #         if success and inliers is not None and len(inliers) >= 4:
    #             # 用内点重新做 ITERATIVE 精化，得到更稳定的结果
    #             inlier_mask = inliers.flatten()
    #             success, rvec, tvec = cv2.solvePnP(
    #                 obj_pts[inlier_mask], img_pts[inlier_mask], K, dist,
    #                 flags=cv2.SOLVEPNP_ITERATIVE,
    #                 rvec=rvec, tvec=tvec  # 用 RANSAC 结果作为初始值
    #             )
    #             print(f"  RANSAC 内点数: {len(inliers)}/{len(obj_pts)}")

    #         if success:
    #             R, _ = cv2.Rodrigues(rvec)
    #             print(f"\nsolvePnP 成功:")
    #             print(f"  有效匹配点数: {len(obj_pts_list)}, 索引: {valid_indices}")
    #             print(f"  旋转向量 rvec:\n{rvec.flatten()}")
    #             print(f"  平移向量 tvec:\n{tvec.flatten()}")
    #             print(f"  旋转矩阵 R:\n{R}")

    #             # 计算重投影误差
    #             proj_pts, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, dist)
    #             proj_pts = proj_pts.reshape(-1, 2)
    #             error = np.linalg.norm(img_pts - proj_pts, axis=1)
    #             print(f"  平均重投影误差: {error.mean():.4f} px, 最大: {error.max():.4f} px")

    #             # 在图像上绘制坐标轴验证
    #             # r-x g-y b-z
    #             cv2.drawFrameAxes(vis_img, K, dist, rvec, tvec, 30, 3)
    #             cv2.imshow("PnP result", vis_img)
    #             cv2.waitKey(0)
    #             cv2.imwrite("vis_outputimg.jpg", vis_img)
    #         else:
    #             print("solvePnP 失败")
    #     else:
    #         print(f"有效匹配点不足: {len(obj_pts_list)} < 4, 无法求解PnP")


    


