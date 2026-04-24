import time

import cv2
import numpy as np
# from tomlkit import inline_table


class SocketDetectorPostProcessor:
    def __init__(self):
        # 定义标准模板坐标 (根据插座标准几何参数)
        # 2个小孔 (CC, CP), 5个大孔 (L1, N, PE, L2, L3)
        self.template_points = np.array([
            [-8.0, 11.2], [8.0, 11.2],  # 上方小孔
            [-16.0, 0.0], [0.0, 0.0], [16.0, 0.0],  # 中间大孔
            [-8.0, -13.9], [8.0, -13.9]  # 下方大孔
        ], dtype=np.float32)

        self.num_points = len(self.template_points)

    def process(self, detected_ellipses, image_gray):
        """
        detected_ellipses: list of (x, y) 坐标，由 EdgeDrawing 提取
        image_gray: 原始灰度图，用于精化边缘
        """
        if len(detected_ellipses) < 3:
            return None, "检测到的特征点太少，无法重建模板"

        # 1. 尝试匹配已检测点到模板 (使用 RANSAC 寻找最佳变换矩阵)
        # 这里模拟从大量检测结果中找最符合 7 孔结构的子集
        best_transform = None
        max_inliers = 0

        # 转换输入格式
        detected_pts = np.array(detected_ellipses, dtype=np.float32)

        # 工业场景通常使用仿射变换 (包含旋转、缩放、平移)
        # 实际操作中可以使用 cv2.estimateAffinePartial2D 处理刚性变换
        for _ in range(500):  # 简单的 RANSAC 逻辑
            sample_idx = np.random.choice(len(detected_pts), 3, replace=False)
            temp_idx = np.random.choice(self.num_points, 3, replace=False)

            M = cv2.getAffineTransform(self.template_points[temp_idx], detected_pts[sample_idx])

            # 投影所有模板点看匹配度
            projected = cv2.transform(self.template_points.reshape(-1, 1, 2), M).reshape(-1, 2)

            # 计算距离，统计内点
            dists = np.linalg.norm(projected[:, None] - detected_pts, axis=2)
            min_dists = np.min(dists, axis=1)
            inliers = np.sum(min_dists < 5.0)  # 阈值 5 像素

            if inliers > max_inliers:
                max_inliers = inliers
                best_transform = M

        if best_transform is None:
            return None, "几何校正失败"

        # 2. 补全漏检点 & 排序
        print(f'max inliner:{max_inliers}')
        final_points = cv2.transform(self.template_points.reshape(-1, 1, 2), best_transform).reshape(-1, 2)
        color = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)
        vis2 = visualize(color, final_points)
        cv2.imshow("vis2", vis2)
        # 3. 边缘再精化 (Sub-pixel Refinement)
        refined_points = []
        for pt in final_points:
            # 在预测点周围进行局部重检测 (例如重心法或亚像素边缘提取)
            # 这里演示使用角点精化思路，实际可用径向梯度搜索
            refined_pt = self._refine_center(image_gray, pt)
            refined_points.append(refined_pt)
        pts = []
        color = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)

        vis3 = visualize(color, refined_points)
        cv2.imshow("refine_vis3", vis3)
        return np.array(refined_points), "Success"

    def _refine_center(self, img, pt, win_size=10):
        """局部重心法精化坐标"""
        x, y = int(pt[0]), int(pt[1])
        if y - win_size < 0 or y + win_size >= img.shape[0] or x - win_size < 0 or x + win_size >= img.shape[1]:
            return pt

        roi = img[y - win_size:y + win_size, x - win_size:x + win_size]
        # 反转颜色（针对深色孔洞）并计算质心
        _, thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        cv2.imshow('roi',roi)
        m = cv2.moments(thresh)
        if m["m00"] == 0: return pt

        new_x = (x - win_size) + m["m10"] / m["m00"]
        new_y = (y - win_size) + m["m01"] / m["m00"]
        return [new_x, new_y]


# from itertools import combinations
from scipy.optimize import linear_sum_assignment
from itertools import permutations, combinations


class RobustSocketMatcher:
    def __init__(self,img):
        # 1. 定义标准模板坐标 (x, y) - 这里的坐标需根据您的实际 CAD 数据严格校准
        # 顺序：CC, CP, L1, N, PE, L2, L3
        self.template_pts = np.array([
            [-8.0, 11.2], [8.0, 11.2],  # 上方小孔
            [-16.0, 0.0], [0.0, 0.0], [16.0, 0.0],  # 中间大孔
            [-8.0, -13.9], [8.0, -13.9]  # 下方大孔
        ], dtype=np.float32)
        self.area_min = np.pi*5*5
        self.area_max=  np.pi*40*40
        self.DIST_THRESHOLD = 5  # 判定为同心圆的距离阈值
        self.num_target = len(self.template_pts)
        self.img = img
        self.template_types = np.array([0, 0, 1, 1, 1, 1, 1])
    def _merge_concentric(self, ellipses):
        if not ellipses: return []

        # 按中心坐标排序，方便聚合
        ellipses.sort(key=lambda x: (x[0], x[1]))
        merged = []
        used = [False] * len(ellipses)

        for i in range(len(ellipses)):
            if used[i]: continue

            curr_group = [ellipses[i]]
            used[i] = True

            for j in range(i + 1, len(ellipses)):
                if used[j]: continue
                # 计算两个椭圆中心的欧氏距离
                dist = np.sqrt((ellipses[i][0] - ellipses[j][0]) ** 2 +
                               (ellipses[i][1] - ellipses[j][1]) ** 2)

                if dist < self.DIST_THRESHOLD:
                    curr_group.append(ellipses[j])
                    used[j] = True

            # 取组内最大面积的椭圆（通常是外径）作为特征点位置
            best_ellipse = max(curr_group, key=lambda e: e[2] * e[3])
            merged.append(best_ellipse)

        return merged
    def filter_and_match(self, raw_ellipses):
        """
        第一阶段：先验几何过滤
        """
        candidates = []
        for e in raw_ellipses:
            x, y, a, b, ang = e
            # 过滤面积与长短轴比 (轴比 > 0.8)
            if self.area_min < (np.pi * a * b) < self.area_max and min(a, b) / max(a, b) > 0.8:
                candidates.append([x, y])

        # 同心圆合并（略，参考前述实现）
        # unique_pts = self._merge_concentric(candidates)
        unique_pts = self._merge_close_points(np.array(candidates))
        if len(unique_pts) < 3:
            return None, "特征点不足"

        vis1 = visualize(self.img, unique_pts)
        cv2.imshow('vis1',vis1)
        # 第二阶段：确定性匹配 (寻找全局最优变换矩阵)
        best_M = self._find_best_initial_transform(unique_pts)

        if best_M is None:
            return None, "无法建立初始拓扑关系"
        best_pts = cv2.transform(self.template_pts.reshape(-1, 1, 2),best_M).reshape(-1, 2)
        vis3 = visualize(self.img, best_pts)
        cv2.imshow('vis3',vis3)
        # 第三阶段：基于匈牙利算法的全局精化
        final_pts, refined_M = self._global_refinement2(unique_pts, best_M)
        vis2 = visualize(self.img, final_pts)
        cv2.imshow('vis2',vis2)
        return final_pts, "Success"

    def _find_best_initial_transform(self, detected_pts):
        """
        穷举所有可能的 3 点组合，通过最小化重投影误差找到最优解
        """
        best_score = float('inf')
        best_M = None

        # 限制采样范围以提高效率：从检测点中选 3 个，从模板中选对应的 3 个
        # 工业场景下通常可以直接遍历，因为点数很少
        sample_indices = list(combinations(range(len(detected_pts)), 3))
        template_indices = list(combinations(range(self.num_target), 3))

        for det_idx in sample_indices:
            src_tri = detected_pts[list(det_idx)].astype(np.float32)
            for tmp_idx in template_indices:
                dst_tri = self.template_pts[list(tmp_idx)].astype(np.float32)

                # 计算亲和变换矩阵 (仿射变换)
                M = cv2.getAffineTransform(dst_tri, src_tri)

                # 评估模型：计算所有模板点投影后的距离和
                proj = cv2.transform(self.template_pts.reshape(-1, 1, 2), M).reshape(-1, 2)

                # 寻找每个投影点最近的检测点距离
                dists = np.linalg.norm(proj[:, np.newaxis] - detected_pts, axis=2)
                min_dists = np.min(dists, axis=1)

                # 评分函数：内点数量优先，距离和次之
                inliers = np.sum(min_dists < 8.0)
                score = np.sum(min_dists[min_dists < 8.0]) - (inliers * 100)  # 惩罚项

                if score < best_score:
                    best_score = score
                    best_M = M

        return best_M

    def _global_refinement2(self, detected_ellipses, M):
        """
        detected_ellipses: 现在传入带尺寸信息的列表 [(x, y, area), ...]
        """
        # 1. 提取检测点的坐标和面积
        det_pts = np.array([[e[0], e[1]] for e in detected_ellipses])
        det_areas = np.array([e[2] for e in detected_ellipses])

        # 2. 预测模板投影位置
        proj_template = cv2.transform(self.template_pts.reshape(-1, 1, 2), M).reshape(-1, 2)

        # 3. 构建“尺寸感知”的代价矩阵
        num_det = len(det_pts)
        num_tmp = len(self.template_pts)
        cost_matrix = np.zeros((num_det, num_tmp))

        # 自动获取当前图像中“大孔”和“小孔”的面积基准（利用中位数）
        # 或者根据先验：小孔面积通常是大孔的一半以下
        area_threshold = np.median(det_areas)

        for j in range(num_tmp):  # 模板点索引
            is_template_small = (self.template_types[j] == 0)
            for i in range(num_det):  # 检测点索引
                dist = np.linalg.norm(det_pts[i] - proj_template[j])

                # 尺寸不匹配惩罚
                is_detected_small = (det_areas[i] < area_threshold)
                size_penalty = 0
                if is_template_small != is_detected_small:
                    size_penalty = 500  # 给一个巨大的惩罚值，防止误匹配

                cost_matrix[i, j] = dist + size_penalty

        # 4. 执行匈牙利算法
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # 4. 筛选有效匹配并重新进行最小二乘拟合
        valid_src = []
        valid_dst = []
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < 10.0:  # 匹配阈值
                valid_src.append(self.template_pts[c])
                valid_dst.append(det_pts[r])
        print(f'templ: {col_ind}')
        print(f'det: {row_ind}')
        if len(valid_src) >= 3:
            # 使用所有匹配点重新估算更精确的变换矩阵
            refined_M, _ = cv2.estimateAffinePartial2D(np.array(valid_src), np.array(valid_dst))
            final_projected = cv2.transform(self.template_pts.reshape(-1, 1, 2), refined_M).reshape(-1, 2)
            return final_projected, refined_M
    def _global_refinement(self, detected_pts, M):
        """
        使用匈牙利算法进行全局分配，消除局部最优，补全漏检
        """
        # 1. 投影模板
        proj_template = cv2.transform(self.template_pts.reshape(-1, 1, 2), M).reshape(-1, 2)

        # 2. 构建代价矩阵 (距离矩阵)
        cost_matrix = np.linalg.norm(detected_pts[:, np.newaxis] - proj_template, axis=2)

        # 3. 匈牙利算法求最小权重匹配
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # 4. 筛选有效匹配并重新进行最小二乘拟合
        valid_src = []
        valid_dst = []
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < 10.0:  # 匹配阈值
                valid_src.append(self.template_pts[c])
                valid_dst.append(detected_pts[r])
        print(f'templ: {col_ind}')
        print(f'det: {row_ind}')
        if len(valid_src) >= 3:
            # 使用所有匹配点重新估算更精确的变换矩阵
            refined_M, _ = cv2.estimateAffinePartial2D(np.array(valid_src), np.array(valid_dst))
            final_projected = cv2.transform(self.template_pts.reshape(-1, 1, 2), refined_M).reshape(-1, 2)
            return final_projected, refined_M

        return proj_template, M

    def _merge_close_points(self, pts, threshold=10):
        if len(pts) == 0: return pts
        merged = []
        while len(pts) > 0:
            p = pts[0]
            dists = np.linalg.norm(pts - p, axis=1)
            group = pts[dists < threshold]
            merged.append(np.mean(group, axis=0))
            pts = pts[dists >= threshold]
        return np.array(merged)



class EllipsePreFilter:
    def __init__(self):
        # 预设参数（根据实际物理尺寸及相机分辨率调整）
        self.MIN_AREA = 50  # 最小面积
        self.MAX_AREA = 4800  # 最大面积
        self.AXIS_RATIO_LIMIT = 0.75  # 轴比（短轴/长轴），接近1为正圆
        self.DIST_THRESHOLD = 3  # 判定为同心圆的距离阈值

    def filter_candidates(self, raw_ellipses):
        """
        raw_ellipses: EdgeDrawing输出的原始列表 [(x, y, a, b, angle), ...]
        """
        valid_ellipses = []

        for e in raw_ellipses:
            x, y, a, b, angle = e
            area = np.pi * a * b
            ratio = min(a, b) / max(a, b)

            # 1. 面积约束 & 2. 轴比约束 (过滤掉条状干扰)
            if self.MIN_AREA < area < self.MAX_AREA and ratio > self.AXIS_RATIO_LIMIT:
                valid_ellipses.append(e)

        # 3. 包含关系/同心圆合并 (处理内径和外径)
        # 工业检测中，我们通常只需要孔位的中心点
        merged_ellipses = self._merge_concentric(valid_ellipses)

        return merged_ellipses

    def _merge_concentric(self, ellipses):
        if not ellipses: return []

        # 按中心坐标排序，方便聚合
        ellipses.sort(key=lambda x: (x[0], x[1]))
        merged = []
        used = [False] * len(ellipses)

        for i in range(len(ellipses)):
            if used[i]: continue

            curr_group = [ellipses[i]]
            used[i] = True

            for j in range(i + 1, len(ellipses)):
                if used[j]: continue
                # 计算两个椭圆中心的欧氏距离
                dist = np.sqrt((ellipses[i][0] - ellipses[j][0]) ** 2 +
                               (ellipses[i][1] - ellipses[j][1]) ** 2)

                if dist < self.DIST_THRESHOLD:
                    curr_group.append(ellipses[j])
                    used[j] = True

            # 取组内最大面积的椭圆（通常是外径）作为特征点位置
            best_ellipse = max(curr_group, key=lambda e: e[2] * e[3])
            merged.append(best_ellipse)

        return merged


class Type2SocketFinalProcessor:
    def __init__(self,img):
        # 1. 定义标准模板坐标 (x, y) 及 类型 (0: 小孔, 1: 大孔)
        # 坐标基于您提供的几何数据
        self.template_data = [
            {'pos': [-8.0, 11.2], 'type': 0},  # CC
            {'pos': [8.0, 11.2], 'type': 0},  # CP
            {'pos': [-16.0, 0.0], 'type': 1},  # L1
            {'pos': [0.0, 0.0], 'type': 1},  # N
            {'pos': [16.0, 0.0], 'type': 1},  # PE
            {'pos': [-8.0, -13.9], 'type': 1},  # L2
            {'pos': [8.0, -13.9], 'type': 1}  # L3
        ]
        self.template_pts = np.array([d['pos'] for d in self.template_data], dtype=np.float32)
        self.template_types = np.array([d['type'] for d in self.template_data])
        self.img = img
    def solve(self, raw_ellipses, image_gray=None):
        """
        主入口函数
        raw_ellipses: List of (x, y, axis_a, axis_b, angle)
        """
        # --- 步骤 1: 几何初筛 ---
        candidates = self._filter_and_merge(raw_ellipses)
        if len(candidates) < 3:
            return None, "有效特征点不足，无法匹配"


        # --- 步骤 2: 初始变换估计 (穷举法) ---
        # 此时只拿坐标去暴力碰撞一个粗略的 M
        best_M = self._estimate_initial_pose(candidates)
        if best_M is None:
            return None, "几何拓扑对齐失败"

        best_pts = cv2.transform(self.template_pts.reshape(-1, 1, 2),best_M).reshape(-1, 2)
        vis3 = visualize(self.img, best_pts)
        cv2.imshow('vis3',vis3)
        print(f'best_M:{best_M}')
        # --- 步骤 3: 尺寸感知型匈牙利匹配 (核心：解决翻转问题) ---
        final_points, refined_M = self._size_aware_matching(candidates, best_M)
        print(f'refined_M:{refined_M}')
        return final_points, "Success"

    def _filter_and_merge(self, ellipses, dist_thresh=10):
        """筛选、去重并提取属性"""
        filtered = []
        for e in ellipses:
            x, y, a, b, ang = e
            area = np.pi * a * b
            ratio = min(a, b) / max(a, b)
            # 过滤条状噪声及极小面积
            if ratio > 0.75 and (np.pi*5*5 < area<np.pi*40*40):
                filtered.append({'pos': [x, y], 'area': area})

        if not filtered: return []

        # 合并同心圆 (例如同一个孔检测到了内径和外径)
        merged = []
        used = [False] * len(filtered)
        for i in range(len(filtered)):
            if used[i]: continue
            group = [filtered[i]]
            used[i] = True
            for j in range(i + 1, len(filtered)):
                d = np.linalg.norm(np.array(filtered[i]['pos']) - np.array(filtered[j]['pos']))
                if d < dist_thresh:
                    group.append(filtered[j])
                    used[j] = True
            # 取组内面积最大的作为代表
            merged.append(max(group, key=lambda x: x['area']))
        return merged

    def _estimate_initial_pose(self, candidates):
        """穷举3点组合寻找初始 M 矩阵"""
        det_pts = np.array([c['pos'] for c in candidates], dtype=np.float32)
        img = self.img.copy()
        vis1 = visualize(img,det_pts)
        cv2.imshow('img',vis1)
        best_score = float('inf')
        best_M = None

        # # 为了速度，只在检测点中选 3 个，模板中固定选 3 个代表性点(如CC, CP, PE)
        # # 若需要更强鲁棒性，可以增加模板采样组
        # tmp_idx_ref = [0, 1, 4]  # CC, CP, PE
        # src_tri = self.template_pts[tmp_idx_ref]
        #
        # for combo in combinations(range(len(det_pts)), 3):
        #     dst_tri = det_pts[list(combo)]
        #     M = cv2.getAffineTransform(src_tri, dst_tri)
        #
        #     # 简易评分：投影所有模板点，看有多少能对上
        #     proj = cv2.transform(self.template_pts.reshape(-1, 1, 2), M).reshape(-1, 2)
        #     dists = np.linalg.norm(proj[:, np.newaxis] - det_pts, axis=2)
        #     min_dists = np.min(dists, axis=1)
        #
        #     score = np.sum(np.clip(min_dists, 0, 20))  # 越小越好
        #     if score < best_score:
        #         best_score = score
        #         best_M = M

        best_score = float('inf')
        best_M = None

        # 限制采样范围以提高效率：从检测点中选 3 个，从模板中选对应的 3 个
        # 工业场景下通常可以直接遍历，因为点数很少
        sample_indices = list(combinations(range(len(det_pts)), 3))
        template_indices = list(combinations(range(len(self.template_pts)), 3))

        for det_idx in sample_indices:
            src_tri = det_pts[list(det_idx)].astype(np.float32)
            for tmp_idx in template_indices:
                dst_tri = self.template_pts[list(tmp_idx)].astype(np.float32)

                # 计算亲和变换矩阵 (仿射变换)
                M = cv2.getAffineTransform(dst_tri, src_tri)

                # 评估模型：计算所有模板点投影后的距离和
                proj = cv2.transform(self.template_pts.reshape(-1, 1, 2), M).reshape(-1, 2)

                # 寻找每个投影点最近的检测点距离
                dists = np.linalg.norm(proj[:, np.newaxis] - det_pts, axis=2)
                min_dists = np.min(dists, axis=1)

                # 评分函数：内点数量优先，距离和次之
                inliers = np.sum(min_dists < 8.0)
                score = np.sum(min_dists[min_dists < 8.0]) - (inliers * 100)  # 惩罚项

                if score < best_score:
                    best_score = score
                    best_M = M

        # return best_M


        return best_M

    def _size_aware_matching(self, candidates, M):
        """核心：结合面积权重的全局配准"""
        det_pts = np.array([c['pos'] for c in candidates])
        det_areas = np.array([c['area'] for c in candidates])

        # 判定检测点中的“大孔”和“小孔”阈值
        # 动态取中位数，工业场景中大孔(5个)多于小孔(2个)，中位数通常落在大孔范围
        area_median = np.median(det_areas)

        proj_template = cv2.transform(self.template_pts.reshape(-1, 1, 2), M).reshape(-1, 2)

        # 构建代价矩阵
        num_det = len(det_pts)
        num_tmp = len(self.template_pts)
        cost_matrix = np.zeros((num_det, num_tmp))

        for j in range(num_tmp):  # 模板索引
            is_tmp_small = (self.template_types[j] == 0)
            for i in range(num_det):  # 检测索引
                dist = np.linalg.norm(det_pts[i] - proj_template[j])
                is_det_small = (det_areas[i] < area_median * 0.7)  # 面积判定逻辑

                # 惩罚项：如果模板要求小孔但检测到大孔，或反之
                penalty = 0
                if is_tmp_small != is_det_small:
                    penalty = 1000  # 极大的代价，强制排除

                cost_matrix[i, j] = dist + penalty

        # 匈牙利分配
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        print(f'templ: {col_ind}')
        print(f'det: {row_ind}')
        # 提取有效匹配点对进行最小二乘拟合精化
        valid_src, valid_dst = [], []
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < 500:  # 排除掉惩罚项后的距离阈值
                valid_src.append(self.template_pts[c])
                valid_dst.append(det_pts[r])

        if len(valid_src) >= 3:
            refined_M, _ = cv2.estimateAffinePartial2D(np.array(valid_src), np.array(valid_dst))
            final_res = cv2.transform(self.template_pts.reshape(-1, 1, 2), refined_M).reshape(-1, 2)
            return final_res, refined_M

        return proj_template, M

def integrated_detection_pipeline(image_gray, raw_ed_ellipses):
    # 第一步：几何过滤（剔除背景杂波、合并内外圈）
    filter_tool = EllipsePreFilter()
    filtered = filter_tool.filter_candidates(raw_ed_ellipses)
    pts = []
    color = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)
    for e in filtered:
        pts.append([e[0],e[1]])
    vis1 = visualize(color,pts)
    cv2.imshow("vis1", vis1)
    # 获取过滤后的中心点
    candidate_pts = [(e[0], e[1]) for e in filtered]
    pts = []
    color = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)
    for e in filtered:
        pts.append([e[0],e[1]])
    vis2 = visualize(color,pts)
    cv2.imshow("vis2", vis2)
    # 第二步：模板匹配与补全
    # 只要过滤后剩下 >3 个点，就能靠模板推算出所有孔位
    processor = SocketDetectorPostProcessor()
    final_results, status = processor.process(candidate_pts, image_gray)

    return final_results, status

# =========================
# 可视化
# =========================
def visualize(img, pts):

    vis = img.copy()
    if len(vis.shape) == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    for i, (x, y) in enumerate(pts):
        cv2.circle(vis, (int(x), int(y)), 5, (0,255,0), -1)
        cv2.putText(vis, str(i), (int(x)+5,int(y)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
    return vis


class PerspectiveSocketProcessor:
    def __init__(self,img):
        # 模板点及类型 (0: 小孔, 1: 大孔)
        self.template_data = [
            {'pos': [-8.0, 11.2], 'type': 0},  # 0: CC
            {'pos': [8.0, 11.2], 'type': 0},  # 1: CP
            {'pos': [-16.0, 0.0], 'type': 1},  # 2: L1
            {'pos': [0.0, 0.0], 'type': 1},  # 3: N
            {'pos': [16.0, 0.0], 'type': 1},  # 4: PE
            {'pos': [-8.0, -13.9], 'type': 1},  # 5: L2
            {'pos': [8.0, -13.9], 'type': 1}  # 6: L3
        ]
        self.img = img
        self.template_pts = np.array([d['pos'] for d in self.template_data], dtype=np.float32)
        self.template_types = np.array([d['type'] for d in self.template_data])
    def _filter_and_merge(self, ellipses, dist_thresh=10):
        """筛选、去重并提取属性"""
        filtered = []
        for e in ellipses:
            x, y, a, b, ang = e
            area = np.pi * a * b
            ratio = min(a, b) / max(a, b)
            # 过滤条状噪声及极小面积
            if ratio > 0.75 and (np.pi*5*5 < area<np.pi*40*40):
                filtered.append({'pos': [x, y], 'area': area})

        if not filtered: return []

        # 合并同心圆 (例如同一个孔检测到了内径和外径)
        merged = []
        used = [False] * len(filtered)
        for i in range(len(filtered)):
            if used[i]: continue
            group = [filtered[i]]
            used[i] = True
            for j in range(i + 1, len(filtered)):
                d = np.linalg.norm(np.array(filtered[i]['pos']) - np.array(filtered[j]['pos']))
                if d < dist_thresh:
                    group.append(filtered[j])
                    used[j] = True
            # 取组内面积最大的作为代表
            merged.append(max(group, key=lambda x: x['area']))
        return merged
    def solve(self, raw_ellipses):
        # 1. 预过滤 (同前)
        candidates = self._filter_and_merge(raw_ellipses)
        if len(candidates) < 4: return None, "点数不足以进行透视投影"

        # 2. 初始估计：使用更强大的 estimateAffine2D (6自由度)
        # 它可以处理拉伸和挤压，解决初步失真
        initial_M, inlier_indices = self._feature_guided_initial_guess(candidates)
        print(f'inlier:{inlier_indices}')
        if initial_M is None: return None, "初始对齐失败"
        best_pts = cv2.transform(self.template_pts.reshape(-1, 1, 2),initial_M).reshape(-1, 2)
        vis3 = visualize(self.img, best_pts)
        cv2.imshow('vis3',vis3)
        # 3. 锁定拓扑的精化 (不再允许翻转)
        # 使用 findHomography (8自由度) 解决倾斜失真
        final_pts, H = self._refine_with_homography(candidates, initial_M)

        return final_pts, H

    def _feature_guided_initial_guess(self, candidates):
        """
        不再盲目采样，而是利用大小孔的分类信息进行匹配
        """
        det_pts = np.array([c['pos'] for c in candidates], dtype=np.float32)
        det_areas = np.array([c['area'] for c in candidates])

        # 假设前几个是小孔（即使有噪点，我们也遍历前 4 个最小的）
        small_indices = list(range(min(len(candidates), 4)))
        large_indices = list(range(max(0, len(candidates) - 5), len(candidates)))

        best_M = None
        max_score = -1

        # 核心策略：尝试匹配 (检测点i, 检测点j, 检测点k) 到 (模板0, 模板1, 模板3)
        # 模板 0,1 是小孔，模板 3 是中心大孔。这个三角形是不对称的，能唯一确定姿态。
        for i, j in permutations(small_indices, 2):
            for k in large_indices:
                if k == i or k == j: continue

                # 建立 3 对点的映射
                src_tri = self.template_pts[[0, 1, 3]]  # 模板: CC, CP, N
                dst_tri = det_pts[[i, j, k]]  # 检测: 假设的小1, 小2, 大1

                # 几何校验：小孔间距与到大孔距离的比值应在一定范围内
                d_ij = np.linalg.norm(dst_tri[0] - dst_tri[1])
                d_ik = np.linalg.norm(dst_tri[0] - dst_tri[2])
                if d_ij == 0 or not (0.4 < d_ij / d_ik < 1.5): continue

                M = cv2.getAffineTransform(src_tri, dst_tri)

                # 评估该变换下的内点数
                proj = cv2.transform(self.template_pts.reshape(-1, 1, 2), M).reshape(-1, 2)
                dists = np.linalg.norm(proj[:, None] - det_pts, axis=2)
                min_dists = np.min(dists, axis=1)

                # 只有当面积类型也匹配时才计入内点
                inliers = 0
                for idx, d in enumerate(min_dists):
                    if d < 15:  # 像素阈值
                        # 检查匹配到的检测点面积是否符合模板类型
                        matched_det_idx = np.argmin(dists[idx])
                        is_det_small = (matched_det_idx in small_indices)
                        is_tmp_small = (self.template_types[idx] == 0)
                        if is_det_small == is_tmp_small:
                            inliers += 1

                if inliers > max_score:
                    max_score = inliers
                    best_M = M
                    if inliers == 7: return M,inliers     # 完美匹配直接退出
        return best_M,inliers

    def _refine_with_homography(self, candidates, initial_M):
        """
        使用单应性矩阵解决透视变形，并严格限制匹配邻域以防止翻转
        """
        det_pts = np.array([c['pos'] for c in candidates])

        # 使用初次估计投影模板
        proj_init = cv2.transform(self.template_pts.reshape(-1, 1, 2), initial_M).reshape(-1, 2)

        valid_src = []
        valid_dst = []

        # 关键改进：局部邻域搜索，而非全局匈牙利
        # 这确保了：模板点 0 只会在初次投影点 0 的附近找检测点，绝不会跳到点 5 去
        for i in range(len(self.template_pts)):
            # 在初次投影点周围 15 像素范围内寻找最近的检测点
            dists = np.linalg.norm(det_pts - proj_init[i], axis=1)
            nearest_idx = np.argmin(dists)

            if dists[nearest_idx] < 20:  # 搜索阈值
                valid_src.append(self.template_pts[i])
                valid_dst.append(det_pts[nearest_idx])

        if len(valid_src) < 4:
            return proj_init, None  # 如果精化失败，退回初次估计

        # 使用单应性变换 (8自由度)，解决倾斜导致的失真
        H, _ = cv2.findHomography(np.array(valid_src), np.array(valid_dst))

        # 投影最终结果
        final_res = cv2.perspectiveTransform(self.template_pts.reshape(-1, 1, 2), H).reshape(-1, 2)
        return final_res, H

import cv2
import numpy as np
from itertools import combinations
from scipy.optimize import linear_sum_assignment


class RobustSocketMatcher2:
    def __init__(self):
        self.tmp_pts = np.array([
            [-8.0, 11.2], [8.0, 11.2],
            [-16.0, 0.0], [0.0, 0.0], [16.0, 0.0],
            [-8.0, -13.9], [8.0, -13.9]
        ], dtype=np.float32)

    def solve(self, raw_ellipses):
        det_pts = np.array([[e[0], e[1]] for e in raw_ellipses], dtype=np.float32)

        if len(det_pts) < 4:
            return None, "点不足"

        best_H = None
        best_score = float('inf')

        # 🔴 关键：4点组合（而不是3点）
        for d_idx in combinations(range(len(det_pts)), 4):
            dst_quad = det_pts[list(d_idx)]

            # 模板也选4点（固定结构更稳）
            for t_idx in combinations(range(7), 4):
                src_quad = self.tmp_pts[list(t_idx)]

                H, mask = cv2.findHomography(src_quad, dst_quad, 0)
                if H is None:
                    continue

                # ================= 投影 =================
                proj = cv2.perspectiveTransform(
                    self.tmp_pts.reshape(-1, 1, 2), H
                ).reshape(-1, 2)

                # ================= 结构约束 =================
                if not self._structure_ok(proj):
                    continue

                # ================= 一对一匹配 =================
                cost, assignment = self._hungarian_match(proj, det_pts)

                if cost < best_score:
                    best_score = cost
                    best_H = self._refine_H(assignment, det_pts)

        if best_H is None:
            return None, "匹配失败"

        final_pts = cv2.perspectiveTransform(
            self.tmp_pts.reshape(-1, 1, 2), best_H
        ).reshape(-1, 2)

        return final_pts, best_H
    def _structure_ok(self, proj):
        # 上中下顺序
        y_top = np.mean(proj[[0, 1], 1])
        y_mid = np.mean(proj[[2, 3, 4], 1])
        y_bot = np.mean(proj[[5, 6], 1])

        if not (y_top < y_mid < y_bot):
            return False

        # 左右顺序
        if not (proj[0, 0] < proj[1, 0]):
            return False
        if not (proj[2, 0] < proj[3, 0] < proj[4, 0]):
            return False
        if not (proj[5, 0] < proj[6, 0]):
            return False

        return True
    def _hungarian_match(self, proj, det_pts):
        cost_matrix = np.linalg.norm(
            proj[:, None, :] - det_pts[None, :, :],
            axis=2
        )

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        total_cost = cost_matrix[row_ind, col_ind].sum()

        return total_cost, dict(zip(row_ind, col_ind))

    def _refine_H(self, assignment, det_pts):
        src, dst = [], []

        for i in assignment:
            src.append(self.tmp_pts[i])
            dst.append(det_pts[assignment[i]])

        if len(src) < 4:
            return None

        src = np.array(src, dtype=np.float32)
        dst = np.array(dst, dtype=np.float32)

        H, _ = cv2.findHomography(src, dst, cv2.RANSAC)
        return H


class IndustrialSocketMatcher:
    def __init__(self):
        self.tmp_pts = np.array([
            [-8.0, 11.2], [8.0, 11.2],
            [-16.0, 0.0], [0.0, 0.0], [16.0, 0.0],
            [-8.0, -13.9], [8.0, -13.9]
        ], dtype=np.float32)

    # ================= 主入口 =================
    def solve(self, raw_ellipses):
        det_pts = np.array([[e[0], e[1]] for e in raw_ellipses], dtype=np.float32)

        if len(det_pts) < 4:
            return None, "点不足"

        # Step1: 预过滤
        det_pts = self._filter_points(det_pts)
        if len(det_pts) < 4:
            return None, "有效点不足"

        # Step2: PCA对齐
        norm_pts, R, mean = self._normalize(det_pts)

        # Step3: 分行（自适应）
        candidates = self._split_rows_adaptive(norm_pts)

        best_H = None
        best_cost = float('inf')

        for rows in candidates:
            # Step4: 结构评分
            if self._score_structure(norm_pts, rows) > 50:
                continue

            # Step5: 构造模板行
            tmp_rows = ([0,1], [2,3,4], [5,6])

            # Step6: 初始仿射（用中间行）
            if len(rows[1]) < 2:
                continue

            src = self.tmp_pts[tmp_rows[1][:len(rows[1])]]
            dst = det_pts[rows[1][:len(rows[1])]]

            if len(src) < 2:
                continue

            M, _ = cv2.estimateAffinePartial2D(src, dst)
            if M is None:
                continue

            proj = cv2.transform(self.tmp_pts.reshape(-1,1,2), M).reshape(-1,2)

            # Step7: 行内匹配
            assignment, cost = self._match_all(proj, det_pts, tmp_rows, rows)
            if assignment is None:
                continue

            # Step8: Homography refine
            H = self._refine_H(assignment, det_pts)
            if H is None:
                continue

            # Step9: 投影验证（关键）
            proj_final = cv2.perspectiveTransform(
                self.tmp_pts.reshape(-1,1,2), H
            ).reshape(-1,2)

            if not self._final_check(proj_final):
                continue

            if cost < best_cost:
                best_cost = cost
                best_H = H

        if best_H is None:
            return None, "匹配失败"

        final_pts = cv2.perspectiveTransform(
            self.tmp_pts.reshape(-1,1,2), best_H
        ).reshape(-1,2)

        return final_pts, best_H

    # ================= 工具函数 =================

    def _filter_points(self, pts):
        keep = []
        for i, p in enumerate(pts):
            d = np.linalg.norm(pts - p, axis=1)
            d = np.sort(d)[1:4]
            if 5 < np.mean(d) < 80:
                keep.append(i)
        return pts[keep]

    def _normalize(self, pts):
        mean = pts.mean(axis=0)
        pts_c = pts - mean
        U, S, Vt = np.linalg.svd(pts_c)
        R = Vt[:2].T
        return pts_c @ R, R, mean

    def _split_rows_adaptive(self, pts):
        idx = np.argsort(pts[:,1])
        n = len(pts)

        candidates = []
        for i in range(1, n-1):
            for j in range(i+1, n):
                top = idx[:i]
                mid = idx[i:j]
                bot = idx[j:]

                if len(top) > 3 or len(mid) > 4 or len(bot) > 3:
                    continue

                candidates.append((top, mid, bot))

        return candidates

    def _score_structure(self, pts, rows):
        top, mid, bot = rows

        score = 0
        score += abs(len(top)-2)*10
        score += abs(len(mid)-3)*10
        score += abs(len(bot)-2)*10

        def my(r):
            return np.mean(pts[r,1]) if len(r)>0 else 0

        y_top, y_mid, y_bot = my(top), my(mid), my(bot)

        if not (y_top < y_mid < y_bot):
            return 1e6

        score += abs((y_mid - y_top) - (y_bot - y_mid))
        return score

    def _match_all(self, proj, det_pts, tmp_rows, det_rows):
        total_cost = 0
        assignment = {}

        for tr, dr in zip(tmp_rows, det_rows):
            if len(tr) == 0 or len(dr) == 0:
                continue

            proj_row = proj[tr]
            det_row = det_pts[dr]

            if len(det_row) < len(proj_row):
                return None, 1e6

            cost = np.linalg.norm(
                proj_row[:,None,:] - det_row[None,:,:],
                axis=2
            )

            row_ind, col_ind = linear_sum_assignment(cost)

            total_cost += cost[row_ind, col_ind].sum()

            for i, j in zip(np.array(tr)[row_ind], np.array(dr)[col_ind]):
                assignment[i] = j

        return assignment, total_cost

    def _refine_H(self, assignment, det_pts):
        src, dst = [], []

        for i in assignment:
            src.append(self.tmp_pts[i])
            dst.append(det_pts[assignment[i]])

        if len(src) < 4:
            return None

        src = np.array(src, np.float32)
        dst = np.array(dst, np.float32)

        H, _ = cv2.findHomography(src, dst, cv2.RANSAC)
        return H

    def _final_check(self, proj):
        y_top = np.mean(proj[[0,1],1])
        y_mid = np.mean(proj[[2,3,4],1])
        y_bot = np.mean(proj[[5,6],1])

        if not (y_top < y_mid < y_bot):
            return False

        if not (proj[0,0] < proj[1,0]):
            return False
        if not (proj[2,0] < proj[3,0] < proj[4,0]):
            return False
        if not (proj[5,0] < proj[6,0]):
            return False

        return True

def get_signed_area(pts):
    """计算前三个点构成的三角形带符号面积"""
    p0, p1, p2 = pts[0], pts[1], pts[2]
    return (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0])
class RobustSocketMatcher:
    def __init__(self):
        # 模板坐标与类型 (0: 小孔, 1: 大孔)
        self.template = [
            {'p': [-8.0, 11.2], 't': 0},  # 0: CC
            {'p': [8.0, 11.2], 't': 0},  # 1: CP
            {'p': [-16.0, 0.0], 't': 1},  # 2: L1
            {'p': [0.0, 0.0], 't': 1},  # 3: N
            {'p': [16.0, 0.0], 't': 1},  # 4: PE
            {'p': [-8.0, -13.9], 't': 1},  # 5: L2
            {'p': [8.0, -13.9], 't': 1}  # 6: L3
        ]
        self.tmp_pts = np.array([d['p'] for d in self.template], dtype=np.float32)
        self.tmp_types = np.array([d['t'] for d in self.template])



    def solve(self, raw_ellipses):
        # 1. 提取检测点并打上“大小”标签
        det_data = self.preprocess_candidates(raw_ellipses)
        if len(det_data) < 4: return None, "点数不足4个，无法定姿"


        # 第二步：基于尺寸初步分类 (大小孔)
        # 根据你提供的物理参数：大孔显著大于小孔
        det_data.sort(key=lambda x: x['size'])
        sizes = [c['size'] for c in det_data]

        # 2. 计算相邻两个点之间的尺寸增长率
        gaps = []
        for i in range(len(sizes) - 1):
            # 计算比例增长：例如从 9px 到 13px 增长了 44%
            gap = sizes[i + 1] / sizes[i]
            gaps.append(gap)

        # 3. 找到增长最剧烈的那个索引（即小孔与大孔的分界线）
        # 理论上这个跳跃应该出现在索引 1 和 2 之间（即第2个和第3个点之间）
        split_idx = np.argmax(gaps)

        # 4. 阈值设定为跳跃点的中间值
        threshold = (sizes[split_idx] + sizes[split_idx + 1]) / 2

        for c in det_data:
            c['t'] = 1 if c['size'] > threshold else 0
        best_H = None
        max_score = -1

        # 2. 从模板中选 4 个不共线的代表性点 (例如: CC, L1, PE, L2)
        # 选这四个点是因为它们分布最广，计算出的 H 最稳
        tmp_subset_idx = [0,  4, 5,2 ]
        src_quad = self.tmp_pts[tmp_subset_idx]
        src_types = self.tmp_types[tmp_subset_idx]
        src_area_sign = np.sign(get_signed_area(src_quad))

        # 3. 遍历检测点中所有 4 个点的组合
        num_det = len(det_data)
        det_indices = list(range(num_det))

        for d_combo in combinations(det_indices, 4):
            det_subset = [det_data[i] for i in d_combo]

            # 4. 尝试这 4 个点的所有排列
            for p_idx in permutations(range(4)):
                p_det = [det_subset[i] for i in p_idx]
                p_det_pts = np.array([d['p'] for d in p_det], dtype=np.float32)
                p_det_types = np.array([d['t'] for d in p_det])
                # if not np.array_equal(src_types, [d['t'] for d in p_det]):
                #     continue
                p_pts = np.array([d['p'] for d in p_det], dtype=np.float32)
                # 约束 2：核心手性校验！
                # 如果检测点的环绕方向与模板相反，直接判定为镜像解，跳过
                if np.sign(get_signed_area(p_pts)) != src_area_sign:
                    continue
                # --- 核心剪枝 1: 属性匹配 ---
                # 如果 4 个点的“大小”属性与模板不符，直接跳过
                if not np.array_equal(src_types, p_det_types):
                    continue

                # # --- 核心剪枝 2: 凸包/手性匹配 ---
                # # 检查 4 点构成的多边形方向，防止镜像翻转
                # if self._get_polygon_area(src_quad) * self._get_polygon_area(p_det_pts) < 0:
                #     continue

                # 5. 计算单应性矩阵 H
                H, _ = cv2.findHomography(src_quad, p_det_pts)
                if H is None: continue

                # 6. 全局验证 (评分机制)
                score, refined_pts = self._evaluate_structure(H, det_data)

                if score > max_score:
                    print(f'H:{H}')
                    print(f'p_det:{p_det_pts}')
                    print(f'idx:{p_idx}')
                    print(f'src_area_sign:{src_area_sign}')
                    print(np.sign(get_signed_area(p_pts)))
                    max_score = score
                    best_H = H
                    print(H)
                    final_pts = refined_pts
                    if score >= 7: break  # 完美匹配
        print(f'final_pts:{final_pts}')
        print(f'max_score:{max_score}')
        return final_pts, max_score

    def preprocess_candidates(self, ellipses, dist_thresh=10):
        """
        合并同心圆：将中心距离小于阈值的椭圆归为一个物理孔
        """
        if not ellipses: return []

        # 1. 转换为基础信息
        nodes = []
        for e in ellipses:
            # e: (cx, cy, a, b, angle)
            nodes.append({'c': np.array([e[0], e[1]]), 'd': (e[2] + e[3])})

        merged = []
        used = [False] * len(nodes)

        for i in range(len(nodes)):
            if used[i]: continue

            # 寻找与当前点同心的所有椭圆
            cluster = [nodes[i]]
            used[i] = True
            for j in range(i + 1, len(nodes)):
                if not used[j]:
                    dist = np.linalg.norm(nodes[i]['c'] - nodes[j]['c'])
                    if dist < dist_thresh:
                        cluster.append(nodes[j])
                        used[j] = True

            # 2. 合并特征：取平均中心，记录最大直径（代表外径）
            avg_c = np.mean([n['c'] for n in cluster], axis=0)
            max_d = max([n['d'] for n in cluster])
            is_double = len(cluster) >= 2  # 是否具有内外圈结构

            merged.append({'p': avg_c, 'size': max_d, 'is_double': is_double})

        # 3. 按尺寸再次过滤明显非孔物体
        # 假设小孔外径在图像中至少有一定像素宽度
        merged = [m for m in merged if m['size'] > 10 and m['size']< 100]
        return merged

    def _get_polygon_area(self, pts):
        """计算带符号的多边形面积，用于判断点序方向"""
        return 0.5 * np.sum(pts[:, 0] * np.roll(pts[:, 1], 1) - pts[:, 1] * np.roll(pts[:, 0], 1))

    def _evaluate_structure(self, H, det_data, thresh=15):
        """用 H 投影所有点，计算有多少个能对上"""
        proj = cv2.perspectiveTransform(self.tmp_pts.reshape(-1, 1, 2), H).reshape(-1, 2)
        det_pts = np.array([d['p'] for d in det_data])

        inliers = 0
        for i in range(7):
            dists = np.linalg.norm(det_pts - proj[i], axis=1)
            min_idx = np.argmin(dists)
            if dists[min_idx] < thresh:
                # 进一步校验面积类型是否匹配
                if det_data[min_idx]['t'] == self.tmp_types[i]:
                    inliers += 1
        return inliers, proj


class UltimateSocketMatcher:
    def __init__(self):
        # 模板定义
        # self.tmp_pts = np.array([
        #     [-8.0, 11.2], [8.0, 11.2],  # 0,1: Small
        #     [-16.0, 0.0], [0.0, 0.0], [16.0, 0.0],  # 2,3,4: Large
        #     [-8.0, -13.9], [8.0, -13.9]  # 5,6: Large
        # ], dtype=np.float32)
        self.obj_pts = np.array([
            [-8.0, 11.2, 0.0], [8.0, 11.2, 0.0],  # CC, CP
            [-16.0, 0.0, 0.0], [0.0, 0.0, 0.0], [16.0, 0.0, 0.0],  # L1, N, PE
            [-8.0, -13.9, 0.0], [8.0, -13.9, 0.0]  # L2, L3
        ], dtype=np.float32)
        self.tmp_types = [0, 0, 1, 1, 1, 1, 1]

        self.K =     np.array([[1015.445938660267,0.,638.51741890470555],[0.,1015.445938660267,386.838616473841],[0.,0.,1.]])
        print(f'cameraMatrix:',self.K)
        # distCoeffs = np.array([0.11753195467413819,-0.19301774104640848,0.00016793575097772418,-.00061144051421409198,0.072260521199194336])
        self.dist = np.array([0.11753195467413819,-0.19301774104640848,0.00016793575097772418,-.00061144051421409198,0.072260521199194336])
    def _clean_and_classify(self, ellipses, dist_thresh=10):
        """
        合并同心圆：将中心距离小于阈值的椭圆归为一个物理孔
        """
        if not ellipses: return []

        # 1. 转换为基础信息
        nodes = []
        for e in ellipses:
            # e: (cx, cy, a, b, angle)
            if(0.9<e[2]/e[3]<1.1 and e[2]+e[3]<80) :
                nodes.append({'c': np.array([e[0], e[1]]), 'd': (e[2] + e[3])})

        merged = []
        used = [False] * len(nodes)

        for i in range(len(nodes)):
            if used[i]: continue

            # 寻找与当前点同心的所有椭圆
            cluster = [nodes[i]]
            used[i] = True
            for j in range(i + 1, len(nodes)):
                if not used[j]:
                    dist = np.linalg.norm(nodes[i]['c'] - nodes[j]['c'])
                    if dist < dist_thresh:
                        cluster.append(nodes[j])
                        used[j] = True

            # 2. 合并特征：取平均中心，记录最大直径（代表外径）
            avg_c = np.mean([n['c'] for n in cluster], axis=0)
            max_d = max([n['d'] for n in cluster])
            is_double = len(cluster) >= 2  # 是否具有内外圈结构

            merged.append({'p': avg_c, 'size': max_d, 'is_double': is_double})

        # 3. 按尺寸再次过滤明显非孔物体
        # 假设小孔外径在图像中至少有一定像素宽度
        merged = [m for m in merged if m['size'] > 10 and m['size']< 100]
        return merged
    def solve(self, raw_ellipses):
        # 1. 预处理：合并同心圆 + 间隙法分类 (Gap Method)
        candidates = self._clean_and_classify(raw_ellipses)
        if len(candidates) < 4: return None, 0
        print('candidate points: ',candidates)

        # 第二步：基于尺寸初步分类 (大小孔)
        # 根据你提供的物理参数：大孔显著大于小孔
        candidates.sort(key=lambda x: x['size'])
        sizes = [c['size'] for c in candidates]

        # 2. 计算相邻两个点之间的尺寸增长率
        gaps = []
        for i in range(len(sizes) - 1):
            # 计算比例增长：例如从 9px 到 13px 增长了 44%
            gap = sizes[i + 1] / sizes[i]
            gaps.append(gap)

        # 3. 找到增长最剧烈的那个索引（即小孔与大孔的分界线）
        # 理论上这个跳跃应该出现在索引 1 和 2 之间（即第2个和第3个点之间）
        split_idx = np.argmax(gaps)

        # 4. 阈值设定为跳跃点的中间值
        threshold = (sizes[split_idx] + sizes[split_idx + 1]) / 2

        for c in candidates:
            c['t'] = 1 if c['size'] > threshold else 0

        best_H, max_score = None, 0

        # 2. 获取模板中所有可能的4点组合及其类型签名
        # 例如: (0,2,3,4) 的类型签名是 (0,1,1,1)
        tmp_combos = []
        for indices in combinations(range(7), 4):
            types = tuple(sorted([self.tmp_types[i] for i in indices]))
            tmp_combos.append({'idx': indices, 'types': types})

        # 3. 遍历检测点的 4 点组合
        det_indices = list(range(len(candidates)))
        for d_idx_tuple in combinations(det_indices, 4):
            d_subset = [candidates[i] for i in d_idx_tuple]
            d_types_signature = tuple(sorted([d['t'] for d in d_subset]))

            # 4. 类型匹配：只尝试类型分布一致的模板组合
            for t_combo in tmp_combos:
                # if t_combo['types'] != d_types_signature:
                #     continue

                # 5. 确定了 4 对 4，开始排列检测点以对齐模板类型
                src_pts = self.obj_pts[list(t_combo['idx'])][:,:2]
                src_types = [self.tmp_types[i] for i in t_combo['idx']]
                src_area_sign = get_signed_area(src_pts)
                for p_d_subset in permutations(d_subset):
                    # if [d['t'] for d in p_d_subset] != src_types:
                    #     continue

                    dst_pts = np.array([d['p'] for d in p_d_subset], dtype=np.float32)

                    if np.sign(get_signed_area(dst_pts)) == src_area_sign:
                        continue

                    # 6. 计算 H 并进行“反镜像”校验
                    H, _ = cv2.findHomography(src_pts, dst_pts)
                    if H is None: continue

                    # 关键逻辑：检测线性变换的行列式
                    # 若 det < 0, 说明发生了镜像翻转
                    det_sign = np.linalg.det(H[:2, :2])
                    if det_sign > 0: continue

                    # 7. 全局一致性验证
                    score, proj = self.evaluate_refined(H, self.obj_pts[:,:2], candidates)
                    if score > max_score:
                        print(f'score:{score},det_sign:{det_sign}, H:{H},det_pts:{dst_pts},src_idx:{t_combo}')
                        max_score = score
                        best_H = H
                        final_res = proj
                        # if score == 7: return proj, 7
        return (final_res, max_score) if best_H is not None else (None, 0)

    def estimate_pose(self, p_img, tl):
        """
        执行 solvePnP 得到 3D 位姿
        """
        # if len(matched_pairs) < 4:
        #     return None, None, None

        # 提取匹配好的 3D 和 2D 点
        # p_obj = []
        # p_img = []
        # for r_idx, c_idx in matched_pairs:
        #     p_obj.append(self.obj_pts[r_idx])
        #     p_img.append(candidates[c_idx]['p'])
        #
        # p_obj = np.array(p_obj, dtype=np.float32)
        p_img = np.array(p_img, dtype=np.float32)
        if tl is not None:
            p_img +=np.array(tl)
        # 使用迭代法或 SQPnP (如果有的话) 求解
        # rvec: 旋转向量, tvec: 平移向量
        success, rvec, tvec = cv2.solvePnP(self.obj_pts, p_img, self.K, self.dist, flags=cv2.SOLVEPNP_ITERATIVE)

        if success:
            # 计算重投影验证误差
            proj_back, _ = cv2.projectPoints(self.obj_pts, rvec, tvec, self.K, self.dist)
            proj_back = proj_back.reshape(-1, 2)
            return rvec, tvec, proj_back
        return None, None, None
    # def _evaluate(self, H, candidates, thresh=15):
    #     # 计算投影后，对比检测点，返回内点数
    #     proj = cv2.perspectiveTransform(self.tmp_pts.reshape(-1, 1, 2), H).reshape(-1, 2)
    #     det_pts = np.array([c['p'] for c in candidates])
    #     count = 0
    #     for p in proj:
    #         dists = np.linalg.norm(det_pts - p, axis=1)
    #         if np.min(dists) < thresh: count += 1
    #     return count, proj


    def evaluate_refined(self,H, template_pts, candidates, dist_thresh=15):
        """
        使用最优分配算法计算重投影评分
        """
        # 1. 投影模板点
        proj = cv2.perspectiveTransform(template_pts.reshape(-1, 1, 2), H).reshape(-1, 2)
        det_pts = np.array([c['p'] for c in candidates])

        # 2. 构建距离矩阵 (N_template x M_candidates)
        # 计算每一对点之间的欧氏距离
        diff = proj[:, np.newaxis, :] - det_pts[np.newaxis, :, :]
        dist_matrix = np.linalg.norm(diff, axis=2)

        # 3. 使用匈牙利算法求解最优一比一匹配
        row_ind, col_ind = linear_sum_assignment(dist_matrix)

        # 4. 统计有效匹配（在距离阈值内）
        valid_errors = []
        matched_indices = []

        for r, c in zip(row_ind, col_ind):
            d = dist_matrix[r, c]
            if d < dist_thresh:
                valid_errors.append(d)
                matched_indices.append((r, c))  # 模板索引 r 匹配到 检测索引 c

        inlier_count = len(valid_errors)
        if inlier_count == 0:
            return -9999, proj  # 无匹配点，评分极低

        # 5. 计算综合得分
        # RMSE (均方根误差)
        rmse = np.sqrt(np.mean(np.square(valid_errors)))

        # 最终分 = 内点权重 + 精度权重
        # 减去 rmse 是为了让误差越小的解总分越高
        score = (inlier_count * 1000) - rmse

        return score, proj

import skimage,scipy
def minimum_of_directional_tophat_bottomhat(im_np, size, method='tophat'):  # method='tophat', 'bottomhat'
    x = list(range(0, size)) + [size - 1] * size
    y = [0] * size + list(range(0, size))

    fims_np = []
    for i in range(len(x)):
        se_np = np.zeros((size, size), dtype=bool)
        rr, cc = skimage.draw.line(y[i], x[i], size - 1 - y[i], size - 1 - x[i])
        se_np[rr, cc] = True

        # imageio.imsave('tmp/' + str(i) + '.png', se_np.astype(float))

        filtered_np = np.zeros(im_np.shape)
        for j in range(im_np.shape[2]):
            if (method == 'tophat'):
                filtered_np[:, :, j] = im_np[:, :, j] - scipy.ndimage.grey_opening(im_np[:, :, j], size=(size, size),
                                                                                   footprint=se_np)
            else:
                filtered_np[:, :, j] = scipy.ndimage.grey_closing(im_np[:, :, j], size=(size, size),
                                                                  footprint=se_np) - im_np[:, :, j]
        #        tifffile.imsave('tmp/f_' + str(i) + '.tif', filtered_np)
        fims_np.append(filtered_np)

    fims_np = np.array(fims_np)
    fims_np = np.min(fims_np, axis=0)

    fims_grey_np = np.min(fims_np, axis=2)
    # tifffile.imsave('out_grey.tif', fims_np)

    return fims_np, fims_grey_np

# from Devernay import DevernayEdges
from PIL import Image
# from __future__ import division
# from pyransac.ransac import RansacFeature
# from pyransac.features import Circle

def draw_ellipse(img,ellipses):
    vis = img.copy()
    for ellipse in ellipses:
        center = (int(ellipse[0]), int(ellipse[1]))
        axes = (int(ellipse[2]) , int(ellipse[3]))#, int(ellipse[2]) + int(ellipse[4]))
        angle = ellipse[4]
        color = (0, 0, 255)
        # if ellipse[2] == 0:
        #     color = (0, 255, 0)
        cv2.ellipse(vis, center, axes, angle, 0, 360, color, 1, cv2.LINE_AA)
    return vis

def get_ellipse(ed,color_roi):
    gray = cv2.cvtColor(color_roi, cv2.COLOR_BGR2GRAY)
    ed.detectEdges(gray)
    # cv2.imshow('gray',gray)
    ellipses = ed.detectEllipses()
    if ellipses is not None:  # Check if circles and ellipses have been found and only then iterate over these and add them to the image
        ellipses_ = []
        for i in range(len(ellipses)):
            if ellipses[i][0][2] == 0:
                ellipses_.append([ellipses[i][0][0],ellipses[i][0][1],ellipses[i][0][3],ellipses[i][0][4],ellipses[i][0][5]])
            else:
                ellipses_.append((ellipses[i][0][0], ellipses[i][0][1], ellipses[i][0][2],ellipses[i][0][2],0))
    return ellipses_


def postprocess_ed(circles, gray):
    matcher = UltimateSocketMatcher()
    t = time.perf_counter_ns()
    final_pts,status = matcher.solve(circles)
    print(f'find {len(final_pts)} points')
    print('ellipse fileter time: ',(time.perf_counter_ns()-t)/1e6)
    return final_pts
        # pts = postprocess_ed(circles, gray)
        # final_pts,status = integrated_detection_pipeline(gray,ellipses_)
        # matcher = Type2SocketFinalProcessor(img)
        # matcher = UltimateSocketMatcher()
        # t = time.perf_counter_ns()
        # final_pts,status = matcher.solve(ellipses_)
        # return final_pts
        # d = np.linalg.norm(final_pts[3] -final_pts[2])
        # r_max = d*13.6/16
        # r_min = d*9/16
        # print(f'd:{d},r_max:{r_max},r_min:{r_min}')
        # refined = []
        # for i,pt in enumerate(final_pts):
        #     if i in [0,1]:
        #         r = r_min
        #     else:
        #         r = r_max
        #     roi = gray[int(pt[1]-r/2-3):int(pt[1]+r/2+3),int(pt[0]-r/2-3):int(pt[0]+r/2+3)]
        #     mask = np.full(roi.shape,255,dtype=np.uint8)
        #     mask = cv2.circle(mask,(int(r/2+3),int(r/2+3)),int(r/2-9),0,-1,cv2.LINE_AA)
        #     cv2.imwrite('mask.png',mask)
        #     cv2.imwrite(f'{i}_th.png',roi)
        #     dev = DevernayEdges(Image.fromarray(roi),2.0,10,0,mask=mask)
        #     (xs,ys) = dev.run()
        #     for x,y in zip(xs,ys):
        #         cv2.drawMarker(color_,(int(x+pt[0]-r/2-3),int(y+pt[1]-r/2-3)),(0,0,255),cv2.MARKER_CROSS,3,1,cv2.LINE_AA)
        #     ransac_process = RansacFeature(Circle, max_it=1E3, inliers_percent=0.3, dst=3, threshold=100)
        #     coord = [[x,y] for x in xs for y in ys]
        #     dc, percent = ransac_process.detect_feature(np.array(coord))
        #     print(f'percent:{percent}')
        #     cv2.circle(color_, (int(dc.xc+pt[0]-r/2-3), int(dc.yc+pt[1]-r/2-3)), int(dc.radius), (255, 255, 0), 1, cv2.LINE_AA)
        # rvec,tvec,_ = matcher.estimate_pose(final_pts,tl)
        # print(rvec,tvec)
        # cv2.drawFrameAxes(image,matcher.K,matcher.dist,rvec,tvec,50,3)
        
        # vis = visualize(color_, final_pts)
        # cv2.imshow("vis", vis)

        # cv2.imshow('img',image)
        # cv2.imwrite('sample.png',gray)
        # print(f'iter {cnt}')
        # cv2.imshow('color',color_)
        # cv2.waitKey(0)


if __name__=='__main__':
    from scipy import stats

    Params = cv2.ximgproc.EdgeDrawing.Params()
    ed = cv2.ximgproc.createEdgeDrawing()
    Params.EdgeDetectionOperator = 0
    Params.MinPathLength = 10
    Params.PFmode = 0
    Params.NFAValidation = True
    Params.GradientThresholdValue = 20
    ed.setParams(Params)
    n = 1
    # IMG_DIR = "20260408"
    IMG_DIR = "../dataset/images"
    import os,json

    for fname in os.listdir(IMG_DIR):
        if not fname.endswith((".jpg", ".png")):
            continue
        print(f'fnname:{fname}')
        img_path = os.path.join(IMG_DIR, fname)
        image = cv2.imread(img_path)
        tl, tr, br, bl = [0,0],[image.shape[1],0],[image.shape[1],image.shape[0]],[0,image.shape[0]]
        rect = image[int(tl[1]):int(br[1]), int(tl[0]):int(tr[0]), :]
        rect = image
        color_ = rect.copy()
        gray = cv2.cvtColor(color_, cv2.COLOR_BGR2GRAY)
        # img_ = Image.fromarray(gray)

        # np1,np2 = minimum_of_directional_tophat_bottomhat(color_,15)
        ed.detectEdges(gray)
        cv2.imshow('gray',gray)
        ellipses = ed.detectEllipses()
        if ellipses is not None:
            circles = []
            ellipses_ = []
            for i in range(len(ellipses)):
                center = (int(ellipses[i][0][0]), int(ellipses[i][0][1]))
                axes = (int(ellipses[i][0][2]) + int(ellipses[i][0][3]), int(ellipses[i][0][2]) + int(ellipses[i][0][4]))
                angle = ellipses[i][0][5]
                color = (0, 0, 255)
                if ellipses[i][0][2] == 0:
                    color = (0, 255, 0)
                    ellipses_.append([ellipses[i][0][0],ellipses[i][0][1],ellipses[i][0][3],ellipses[i][0][4],ellipses[i][0][5]])
                else:
                    ellipses_.append((ellipses[i][0][0], ellipses[i][0][1], ellipses[i][0][2],ellipses[i][0][2],0))
                cv2.ellipse(color_, center, axes, angle, 0, 360, color, 1, cv2.LINE_AA)

            # pts = postprocess_ed(circles, gray)
            # final_pts,status = integrated_detection_pipeline(gray,ellipses_)
            # matcher = Type2SocketFinalProcessor(img)
            matcher = UltimateSocketMatcher()
            t = time.perf_counter_ns()
            final_pts,status = matcher.solve(ellipses_)

            # d = np.linalg.norm(final_pts[3] -final_pts[2])
            # r_max = d*13.6/16
            # r_min = d*9/16
            # print(f'd:{d},r_max:{r_max},r_min:{r_min}')
            # refined = []
            # for i,pt in enumerate(final_pts):
            #     if i in [0,1]:
            #         r = r_min
            #     else:
            #         r = r_max
            #     roi = gray[int(pt[1]-r/2-3):int(pt[1]+r/2+3),int(pt[0]-r/2-3):int(pt[0]+r/2+3)]
            #     mask = np.full(roi.shape,255,dtype=np.uint8)
            #     mask = cv2.circle(mask,(int(r/2+3),int(r/2+3)),int(r/2-9),0,-1,cv2.LINE_AA)
            #     cv2.imwrite('mask.png',mask)
            #     cv2.imwrite(f'{i}_th.png',roi)
            #     dev = DevernayEdges(Image.fromarray(roi),2.0,10,0,mask=mask)
            #     (xs,ys) = dev.run()
            #     for x,y in zip(xs,ys):
            #         cv2.drawMarker(color_,(int(x+pt[0]-r/2-3),int(y+pt[1]-r/2-3)),(0,0,255),cv2.MARKER_CROSS,3,1,cv2.LINE_AA)
            #     ransac_process = RansacFeature(Circle, max_it=1E3, inliers_percent=0.3, dst=3, threshold=100)
            #     coord = [[x,y] for x in xs for y in ys]
            #     dc, percent = ransac_process.detect_feature(np.array(coord))
            #     print(f'percent:{percent}')
            #     cv2.circle(color_, (int(dc.xc+pt[0]-r/2-3), int(dc.yc+pt[1]-r/2-3)), int(dc.radius), (255, 255, 0), 1, cv2.LINE_AA)
            rvec,tvec,_ = matcher.estimate_pose(final_pts,tl)
            print(rvec,tvec)
            cv2.drawFrameAxes(image,matcher.K,matcher.dist,rvec,tvec,50,3)
            print((time.perf_counter_ns()-t)/1e6)
            vis = visualize(color_, final_pts)
            cv2.imshow("vis", vis)

            cv2.imshow('img',image)
            # cv2.imwrite('sample.png',gray)
            print(f'iter {fname}')
            cv2.imshow('color',color_)
            cv2.waitKey(0)
    cv2.destroyAllWindows()
# 使用示例
# processor = SocketDetectorPostProcessor()
# final_pts, msg = processor.process(ed_ellipses_centers, gray_img)