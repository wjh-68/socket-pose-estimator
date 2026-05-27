import numpy as np

def detect_and_refine_ellipses(image):
    """Detect and refine ellipses in the given image.
    This function first detects ellipses using a detection method (e.g., pycylinderedsf), 
    then refines the detected ellipses by fitting them more accurately to the image data.
    """
    # 1. Detect raw ellipses
    raw_ellipses = detect_ellipses(image)

    # 2. Refine detected ellipses
    refined_ellipse = refine_ellipse(raw_ellipses)

    return refined_ellipse
def detect_ellipses(image):
    """Detect ellipses in one given image using pycylinderedsf.
    """
    import pycylinderedsf as pyced
    detector = pyced.CED(image)
    detector.run_CED()
    ellipses = detector.getEllipsesAfterCluster()
    return ellipses

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

def refine_ellipse(raw_ellipses):
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
