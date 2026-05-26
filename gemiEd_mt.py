import time
from scipy.spatial.transform import Rotation
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from itertools import permutations, combinations
IMGH = 1440

eMc=np.array(
[[-7.2849429e-01,  6.8505180e-01, -3.0797155e-04, -6.3927837e+01],
 [-6.8492085e-01, -7.2834611e-01,  1.9882789e-02,  6.0054520e+01],
 [ 1.3396430e-02 , 1.4695434e-02,  9.9980229e-01, -1.7391582e+02],
 [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  1.0000000e+00]])

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


def get_signed_area(pts):
    """计算前三个点构成的三角形带符号面积"""
    p0, p1, p2 = pts[0], pts[1], pts[2]
    return (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0])

def filter_concentric_ellipses(ellipse_list, dist_threshold=8.0):
    """
    针对字典列表进行同心圆筛选
    输入示例: [{'p': (100, 105), 'd': 20.5}, ...]
    返回示例: [{'p': (100.2, 105.1), 'd': 21.0}, ...]
    """
    if len(ellipse_list) < 2:
        return ellipse_list

    # 1. 提取坐标和直径到 numpy 数组便于计算
    pts = np.array([item['c'] for item in ellipse_list])
    diams = np.array([item['d'] for item in ellipse_list])
    ids = [item['id'] for item in ellipse_list]
    n = len(pts)
    
    # 2. 计算距离矩阵
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_matrix = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist_matrix, np.inf) # 排除自身
    
    final_ellipses = []
    used_indices = set()
    final= None
    for i in range(n):
        if i in used_indices:
            continue
            
        # 3. 找到点 i 的最近邻 j
        j = np.argmin(dist_matrix[i])
        min_dist = dist_matrix[i, j]
        
        # 4. 相互最近邻逻辑 (Mutual Nearest Neighbor)

        if min_dist < dist_threshold:
            if np.argmin(dist_matrix[j]) == i:
                # 命中内外环点对，计算合并后的属性
                # min_d = min(diams[i],diams[j])
                # max_d = max(diams[i],diams[j])
                # if (min_d/max_d>0.8):#大概率同一环不同极性
                #     # used_indices.add(i)
                #     used_indices.add(j)
                #     dist_matrix[j][i] = np.inf
                #     continue
                merged_p = tuple((pts[i] + pts[j]) / 2.0) #取平均坐标
                # merged_p = pts[i] if diams[i]>diams[j] else pts[j] #取较大的圆心坐标
                # 直径通常取平均值，或者取较大值（外环）取决于你的应用
                # merged_d = float((diams[i] + diams[j]) / 2.0)
                merged_d = max(diams[i], diams[j])
                idx_i = i if diams[i]>diams[j] else j
                idx_j = j if diams[i]>diams[j] else i
                
                # final_ellipses.append({
                #     'c': merged_p,
                #     'd': merged_d,
                #     'id_i': ids[i],
                #     'id_j': ids[j]
                    
                # })
                final = {                    'c': merged_p,
                    'd': merged_d,
                    'id_i': ids[idx_i],
                    'id_j': ids[idx_j]}
                used_indices.add(i)
                used_indices.add(j)
                dist_threshold = min_dist
    final_ellipses.append(final)
    # 注意：这里默认丢弃了所有无法成对的“孤立环”（干扰项）
    return final_ellipses


def solveRects2(raw_ellipses):
    # 1. 转换为基础信息
    nodes = []
    for e_ in raw_ellipses:
        # e: (cx, cy, a, b, angle)
        # s = 0.15-(d-150)/IMGH/2
        s = 0.1
        e = (*e_.center,*e_.size,e_.angle)
        # if(1-s<e[2]/e[3]<1+s and e[2]+e[3]<2*d/3) :
        if(1-s<e[2]/e[3]<1+s) :
            nodes.append({'c': np.array([e[0], e[1]]), 'd': (e[2] + e[3]),'r': min(e[2:4])/max(e[2:4]),'a':e[2],'b':e[3],'angle':e[4]})        
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
                # if dist < dist_thresh:
                if dist < max([nodes[i]['d']/4,nodes[j]['d']/4]):
                    cluster.append(nodes[j])
                    used[j] = True

        # 2. 合并特征：取平均中心，记录最大直径（代表外径）
        # avg_c = np.mean([n['c'] for n in cluster], axis=0)
        max_id = np.argmax([n['d'] for n in cluster])
        if len(cluster)==2:
            dist = np.linalg.norm(cluster[0]['c'] - cluster[1]['c'])
            if dist > 3: #非同心环
                max_id = np.argmax([n['r'] for n in cluster])
        # avg_c = cluster[max_id]['c'] #取大圆坐标
        avg_c = np.mean([n['c'] for n in cluster], axis=0) #取平均坐标
        max_d = max([n['d'] for n in cluster])
        a_id = cluster[max_id]['a']
        b_id = cluster[max_id]['b']
        angle_id = cluster[max_id]['angle']
        out_c = cluster[max_id]['c']
        min_d = min([n['d'] for n in cluster])
        min_id = np.argmin([n['d'] for n in cluster])
        in_c = cluster[min_id]['c']
        

        is_double = len(cluster) >= 2  # 是否具有内外圈结构
        
        if len(cluster)>2:
                cluster_ = []
                for i,e in enumerate(cluster):
                    cluster_.append({'c': e['c'],'d':e['d'],'r':e['r'],'a':e['a'],'b':e['b'],'id':i,'angle':e['angle']})
                filter_dict = filter_concentric_ellipses(cluster_,3)
                # sort_cluster = sorted(cluster, key=lambda x: x['d'])
                id_list = [filter_dict[0]['id_i'],filter_dict[0]['id_j']]
                itemij = [x for x in cluster_ if x['id'] in id_list]

                min_d = min([itemij[0]['d'],itemij[1]['d']])
                max_d = max([itemij[0]['d'],itemij[1]['d']])
                if min_d/max_d>0.75: #同一椭圆接近
                    cluster = [x for x in cluster_ if x['id']!= filter_dict[0]['id_i']]
                    filter_dict = filter_concentric_ellipses(cluster,2)
                if filter_dict[0] is not None:
                    avg_c = filter_dict[0]['c']
                    max_d = filter_dict[0]['d']
                    a_id = cluster_[filter_dict[0]['id_i']]['a']
                    b_id = cluster_[filter_dict[0]['id_i']]['b']
                    angle_id = cluster_[filter_dict[0]['id_i']]['angle']
                    out_c = cluster_[filter_dict[0]['id_i']]['c']
                    in_c = cluster_[filter_dict[0]['id_j']]['c']
                    min_d = cluster_[filter_dict[0]['id_j']]['d']
                        
                else:#剩余的椭圆距离远选择初始
                    candidata =  [x for x in cluster_ if x['id']==id_list[1]][0]
                    avg_c  =candidata['c']
                    max_d = candidata['d']
                    min_d = candidata['d']
                    a_id = candidata['a']
                    b_id = candidata['b']
                    angle_id = candidata['angle']
                    out_c=in_c = avg_c
        merged.append({'p': avg_c, 'size': max_d, 'is_double': is_double,'clusters':cluster,'a':a_id,'b':b_id,'angle':angle_id,'out':out_c,'in':in_c,'min_d':min_d})
    sort_ = sorted(merged,key=lambda x: x['size'])
    # biggest = sort_[-2]['size']
    # merged = [x for x in merged if x['size']> biggest/2.8]#过滤三个注塑口
    # 3. 按尺寸再次过滤明显非孔物体
    # 假设小孔外径在图像中至少有一定像素宽度
    # merged = [m for m in merged if m['size'] > 20 ]
    return sort_[-1]


def process_keypoint(args):
    import pycylinderedsf as pyced
    i, img,tl = args

    detector = pyced.CED(img)

    detector.run_CED()

    rotRects = detector.getEllipsesAfterCluster()

    candidate = solveRects2(
        rotRects
    )

    candidate['p'] += tl

    return i, candidate

class UltimateSocketMatcher:
    def __init__(self,p720=False):
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
        self.p720 = p720
        if not self.p720:
            self.scale = 1/2
            # self.scale = 1.0
        # self.scale = 1/2
        # self.r_idx = None
        # self.c_idx = None
            self.K = np.array([[2719.5208537233339,0.,1281.0886668428632],[0.,2719.5208537233339,723.91283147536694],[0.,0.,1.]])
            self.dist = np.array([-0.1144987327252607,0.28272638050601662,0.0025612432593659007,-0.00044510622008947845,-0.074291650761231273])
            
        else:
            self.scale=1.0
            self.K = np.array([[1359.1199645944478,0.,640.54132556823811],[0.,1359.1199645944478,362.00605351844041],[0.,0.,1.]])
            self.dist = np.array([-0.11507587466685387,0.28954640142800997,0.002523795531233719,-0.0003497505689869382,-0.093769042874787809])
            # self.candidates = None
        print(f'cameraMatrix:',self.K)
        
    def _clean_and_classify(self, ellipses, d=400,dist_thresh=15):
        """
        合并同心圆：将中心距离小于阈值的椭圆归为一个物理孔
        """
        if not ellipses: return []

        # 1. 转换为基础信息
        nodes = []
        for e in ellipses:
            # e: (cx, cy, a, b, angle)
            # s = 0.15-(d-150)/IMGH/2
            s = 0.1
            # if(1-s<e[2]/e[3]<1+s and e[2]+e[3]<2*d/3) :
            if(1-s<e[2]/e[3]<1+s and e[2]+e[3]<2*d/3) :
                nodes.append({'c': np.array([e[0], e[1]]), 'd': (e[2] + e[3]),'r': min(e[2:4])/max(e[2:4]),'a':e[2],'b':e[3],'angle':e[4]})

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
                    # if dist < dist_thresh:
                    if dist < max([nodes[i]['d']/4,nodes[j]['d']/4]):
                        cluster.append(nodes[j])
                        used[j] = True

            # 2. 合并特征：取平均中心，记录最大直径（代表外径）
            # avg_c = np.mean([n['c'] for n in cluster], axis=0)
            max_id = np.argmax([n['d'] for n in cluster])
            if len(cluster)==2:
                dist = np.linalg.norm(cluster[0]['c'] - cluster[1]['c'])
                if dist > 3: #非同心环
                    max_id = np.argmax([n['r'] for n in cluster])
            # avg_c = cluster[max_id]['c'] #取大圆坐标
            avg_c = np.mean([n['c'] for n in cluster], axis=0) #取平均坐标
            max_d = max([n['d'] for n in cluster])
            a_id = cluster[max_id]['a']
            b_id = cluster[max_id]['b']
            angle_id = cluster[max_id]['angle']
            out_c = cluster[max_id]['c']
            min_d = min([n['d'] for n in cluster])
            min_id = np.argmin([n['d'] for n in cluster])
            in_c = cluster[min_id]['c']
            

            is_double = len(cluster) >= 2  # 是否具有内外圈结构
            
            if len(cluster)>2:
                    cluster_ = []
                    for i,e in enumerate(cluster):
                        cluster_.append({'c': e['c'],'d':e['d'],'r':e['r'],'a':e['a'],'b':e['b'],'id':i,'angle':e['angle']})
                    filter_dict = filter_concentric_ellipses(cluster_,3)
                    # sort_cluster = sorted(cluster, key=lambda x: x['d'])
                    id_list = [filter_dict[0]['id_i'],filter_dict[0]['id_j']]
                    itemij = [x for x in cluster_ if x['id'] in id_list]

                    min_d = min([itemij[0]['d'],itemij[1]['d']])
                    max_d = max([itemij[0]['d'],itemij[1]['d']])
                    if min_d/max_d>0.75: #同一椭圆接近
                        cluster = [x for x in cluster_ if x['id']!= filter_dict[0]['id_i']]
                        filter_dict = filter_concentric_ellipses(cluster,2)
                    if filter_dict[0] is not None:
                        avg_c = filter_dict[0]['c']
                        max_d = filter_dict[0]['d']
                        a_id = cluster_[filter_dict[0]['id_i']]['a']
                        b_id = cluster_[filter_dict[0]['id_i']]['b']
                        angle_id = cluster_[filter_dict[0]['id_i']]['angle']
                        out_c = cluster_[filter_dict[0]['id_i']]['c']
                        in_c = cluster_[filter_dict[0]['id_j']]['c']
                        min_d = cluster_[filter_dict[0]['id_j']]['d']
                         
                    else:#剩余的椭圆距离远选择初始
                        candidata =  [x for x in cluster_ if x['id']==id_list[1]][0]
                        avg_c  =candidata['c']
                        max_d = candidata['d']
                        min_d = candidata['d']
                        a_id = candidata['a']
                        b_id = candidata['b']
                        angle_id = candidata['angle']
                        out_c=in_c = avg_c



            merged.append({'p': avg_c, 'size': max_d, 'is_double': is_double,'clusters':cluster,'a':a_id,'b':b_id,'angle':angle_id,'out':out_c,'in':in_c,'min_d':min_d})
        sort_ = sorted(merged,key=lambda x: x['size'])
        biggest = sort_[-2]['size']
        merged = [x for x in merged if x['size']> biggest/2.8]#过滤三个注塑口
        # 3. 按尺寸再次过滤明显非孔物体
        # 假设小孔外径在图像中至少有一定像素宽度
        # merged = [m for m in merged if m['size'] > 20 ]
        return merged


    def solve(self, raw_ellipses,rect,img=None,keypoints=None):#rect tl x,y ,w,h
        # 1. 预处理：合并同心圆 + 间隙法分类 (Gap Method)
        t0 = time.perf_counter_ns()
        candidates = self._clean_and_classify(raw_ellipses,min(rect[2:]),20)
        e_fl_max = [[c['out'][0],c['out'][1],c['size']/4,c['size']/4,0] for c in candidates]
        e_fl_min = [[c['in'][0],c['in'][1],c['min_d']/4,c['min_d']/4,0] for c in candidates]
        # e_fl_min = [[c['p'][0],c['p'][1],c['a']/2,c['b']/2,c['angle']] for c in candidates]
        e_fl = e_fl_max+e_fl_min
        # if img is not None:
        #     el_draw = draw_ellipse(img,e_fl)
        #     cv2.imwrite('ellipse_filter.png',el_draw)
        if len(candidates) < 7: return None, 0,None
        # print('candidate points: ',candidates)


        # 第二步：基于尺寸初步分类 (大小孔)
        # 根据你提供的物理参数：大孔显著大于小孔
        candidates.sort(key=lambda x: x['size'])
        sizes = [c['size'] for c in candidates]
        centers = np.array([ca['p'] for ca in candidates])

        if keypoints is not None:
            # 2. 构建距离矩阵 (N_template x M_candidates)
            # 计算每一对点之间的欧氏距离
            final_res = keypoints
            best_H = np.eye(3)
            max_score = 0
        else:

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
                    if t_combo['types'] != d_types_signature:
                        continue

                    # 5. 确定了 4 对 4，开始排列检测点以对齐模板类型
                    src_pts = self.obj_pts[list(t_combo['idx'])][:,:2]
                    src_types = [self.tmp_types[i] for i in t_combo['idx']]
                    src_area_sign = get_signed_area(src_pts)
                    for p_d_subset in permutations(d_subset):
                        if [d['t'] for d in p_d_subset] != src_types:
                            continue

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
                            # print(f'score:{score},det_sign:{det_sign}, H:{H},det_pts:{dst_pts},src_idx:{t_combo}')
                            max_score = score
                            best_H = H
                            final_res = proj

                        # if score == 7: return proj, 7
        

        # 2. 构建距离矩阵 (N_template x M_candidates)
        # 计算每一对点之间的欧氏距离
        diff = final_res[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist_matrix = np.linalg.norm(diff, axis=2)

        # 3. 使用匈牙利算法求解最优一比一匹配
        row_ind, col_ind = linear_sum_assignment(dist_matrix)
        # self.r_idx = row_ind
        # self.c_idx = col_ind

        # 4. 统计有效匹配（在距离阈值内）
        valid_errors = []
        matched_indices = []

        for r, c in zip(row_ind, col_ind):
            d = dist_matrix[r, c]
            if d < 20:
                valid_errors.append(d)
                matched_indices.append((r, c))  # 模板索引 r 匹配到 检测索引 c
        # self.r_idx = [x[0] for x in matched_indices]
        c_idx = [x[1] for x in matched_indices]
        self.candidates = [candidates[i] for i in c_idx]
        self.e_fl = [e_fl[i] for i in c_idx] + [e_fl[i+7] for i in c_idx]
        print(f'solve time: {(time.perf_counter_ns()-t0)/1e6}')
        return (final_res, max_score,centers[c_idx]/self.scale+np.array(rect[:2])) if best_H is not None else (None, 0,None)

    def estimate_pose(self, p_img, tl):
        """
        执行 solvePnP 得到 3D 位姿
        """
        p_img = np.array(p_img, dtype=np.float32)
        if tl is not None:
            p_img +=np.array(tl)
        p_img = p_img
        print(f'points pnp:{p_img}')
        # 使用迭代法或 SQPnP (如果有的话) 求解
        # rvec: 旋转向量, tvec: 平移向量
        t0 = time.perf_counter_ns()
        success, rvec, tvec = cv2.solvePnP(self.obj_pts, p_img, self.K, self.dist, flags=cv2.SOLVEPNP_IPPE)
        print(f'pnp time: {(time.perf_counter_ns()-t0)/1e6}')
        if success:
            # 计算重投影验证误差
            proj_back, _ = cv2.projectPoints(self.obj_pts, rvec, tvec, self.K, self.dist)
            proj_back = proj_back.reshape(-1, 2)
            error = np.linalg.norm(p_img - proj_back, axis=1).mean()
            print(f"pnp平均重投影误差: {error:.4f} 像素")
            print(f'rvec:{rvec}   tvec:{tvec}')
            success_, rvec_, tvec_,inlier = cv2.solvePnPRansac(self.obj_pts, p_img, self.K, self.dist,rvec=rvec, tvec=tvec, useExtrinsicGuess=True, reprojectionError=1.0,flags=cv2.SOLVEPNP_IPPE)
            if success_:
                proj_back, _ = cv2.projectPoints(self.obj_pts, rvec_, tvec_, self.K, self.dist)
                proj_back = proj_back.reshape(-1, 2)
                error_ = np.linalg.norm(p_img[inlier] - proj_back[inlier], axis=1).mean()
                print(f"Ransac pnp平均重投影误差: {error_:.4f} 像素")
                print(f'inliers:{inlier}')
                print(f'rvec:{rvec_}   tvec:{tvec_}')
                if error_ < error:
                    return rvec_, tvec_, proj_back
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
        self.r_idx = row_ind
        self.c_idx = col_ind

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


def getInferResult(model,img):
    results= model(img)
    if(len(results)==0):
        return []
    keypoints = results[0].keypoints.xy.cpu().numpy().squeeze()
    return results[0].boxes.xyxy.cpu().numpy(),keypoints

if __name__ == '__main__':
    
    from ultralytics import YOLO
    import pycylinderedsf as pyced
    from concurrent.futures import ThreadPoolExecutor
    cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)
# print('he',cap.get(cv2.CAP_PROP))
# print('wi',cap.get(cv2.CAP_PROP_GIGA_FRAME_WIDTH_MAX))
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1440)
    if not cap.isOpened():
        print("camera open failed")
        exit()
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_BRIGHTNESS,100)
        model = YOLO("checkpoint/best.pt")  # load an official model
        ret, img = cap.read()
        result,keypoints = getInferResult(model, img)
        tl = result[0][:2].astype(np.uint16)
        br = result[0][2:].astype(np.uint16)
        rect = img[tl[1]:br[1],tl[0]:br[0]]

        matcher = UltimateSocketMatcher(True)

        # wh = br-tl

        rect_s = np.linalg.norm(keypoints[1]-keypoints[0])
        rect_l = np.linalg.norm(keypoints[6]-keypoints[5])
        candidates = [None]*7
        imgs = []
        tls = []
        for i,keypoint in enumerate(keypoints):
            wh = rect_l
            if i<2:
                wh = rect_s
            tl = (keypoint - wh/2).astype(np.int32)
            br = (keypoint + wh/2).astype(np.int32)
            wh = br-tl
            # cv2.imshow('roi',image[tl[1]:br[1],tl[0]:br[0]])
            # cv2.waitKey(0)
            img = image[tl[1]:br[1],tl[0]:br[0]]
            tls.append(tl)
            imgs.append(np.ascontiguousarray(img))
        t0 = time.perf_counter_ns()

        args_list = [
            (i, img,tl)
            for i, (img,tl) in enumerate(zip(imgs,tls))]
        
        with ThreadPoolExecutor(max_workers=8) as executor:
            results = executor.map(process_keypoint, args_list)
        for i, candidate in results:
            candidates[i] = candidate
        centers = [cand['p'] for cand in candidates]
        print(f'pyd time:{(time.perf_counter_ns()-t0)/1e6}')
        if len(centers)==7:
            rvec, tvec, proj_back = matcher.estimate_pose(centers,None)
            cMo = np.eye(4,dtype=np.float32)
            cMo[:3,:3] = Rotation.from_rotvec(rvec[:3,0]).as_matrix()
            cMo[:3,3] = tvec[:3,0]
            bMo = bMe@eMc@cMo
            print(f'bMo curr:{bMe@eMc@cMo}')
            print(f'bMo euler:{Rotation.from_matrix(bMo[:3,:3]).as_euler("xyz",degrees=True)}')