import cv2 as cv
from ultralytics import YOLO
import time
from gemiEd import *
from scipy.spatial.transform import Rotation
import os
import numpy as np
from static_pose_optimizer import StaticPoseOptimizer, pose_to_euler_tvec

# Load a model
model = YOLO("checkpoint/best.pt")  # load an official model

# ============ Config ============
DATA_DIR = "dataset/save_data2"
RESULT_DIR = "result/save_data2"
SLIDING_WINDOW_SIZE = 8
# Camera on robot end-effector (eye-to-hand extrinsic)
eMc = np.array([
    [-7.2267956e-01,  6.9102561e-01, -1.4759262e-02 ,-5.1758522e+01],
    [-6.9116789e-01, -7.2264087e-01,  8.7790741e-03,  6.0040222e+01],
    [-4.5990809e-03,  1.6545586e-02,  9.9985254e-01,  9.7955963e+01],
    [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  1.0000000e+00]
    ], dtype=np.float64)
# camera intrinsics
K = np.array([
        [2674.7629874104787,0.,1279.5],
        [0.,2674.7629874104787,719.5],
        [0.,0.,1.]
        ], dtype=np.float64)
dist = np.array([-0.11744968686298927,0.27089153364253454,0.0012180578884344092,0.00067320963008635703,-0.078845410108757258], dtype=np.float64)
# 3D object points in object frame (charge port keypoints)
obj_pts = np.array([
            [-8.0, 11.2, 0.0], [8.0, 11.2, 0.0],
            [-16.0, 0.0, 0.0], [0.0, 0.0, 0.0], [16.0, 0.0, 0.0],
            [-8.0, -13.9, 0.0], [8.0, -13.9, 0.0]
        ], dtype=np.float64)
# Transform from object frame to model frame (if needed)        
oMo = np.eye(4,dtype=np.float32)

def solvePnP_IPPE(pts2d, pts3d, K, dist):
    """Wrapper for cv2.solvePnP with IPPE method and validity checks"""
    success, rvec, tvec = cv2.solvePnP(
        pts3d, pts2d, K, dist,flags=cv2.SOLVEPNP_IPPE)
    if not success:
        return None, None, False

    rvec = rvec.flatten()
    tvec = tvec.flatten()
    # Check cMo validity: object should be in front of camera (z > 0)
    cMo = np.eye(4)
    cMo[:3, :3] = Rotation.from_rotvec(rvec).as_matrix()
    cMo[:3, 3] = tvec

    valid, reason = validate_cMo(cMo)
    if not valid:
        print(f"  [WARN] PnP result invalid: {reason}")
        return None, None, False

    return rvec, tvec, True


def compute_reproj_error(pts3d, rvec, tvec, pts2d, K, dist):
        proj, _ = cv2.projectPoints(pts3d, rvec, tvec, K, dist)
        return np.linalg.norm(proj.reshape(-1,2) - pts2d, axis=1).mean()   


def validate_cMo(cMo):
    """Check if cMo is physically valid"""
    # Check rotation matrix determinant (should be +1, not -1 for reflection)
    R = cMo[:3, :3]
    det_R = np.linalg.det(R)
    if abs(det_R - 1.0) > 1e-6:
        return False, f"Rotation det={det_R:.4f} (reflection/flip)"

    # Check translation: object origin should be in front of camera (z > 0)
    tvec = cMo[:3, 3]
    if tvec[2] <= 0:
        return False, f"Object behind camera (z={tvec[2]:.2f})"

    # Sanity check: object should be within reasonable distance (50-1000mm)
    dist = np.linalg.norm(tvec)
    if dist < 50 or dist > 3000:
        return False, f"Object distance={dist:.1f}mm (unreasonable)"

    return True, "ok"

# def init_ed():
#     Params = cv2.ximgproc.EdgeDrawing.Params()
#     ed = cv2.ximgproc.createEdgeDrawing()
#     Params.EdgeDetectionOperator = 1
#     Params.MinPathLength = 45
#     Params.PFmode = 0
#     Params.NFAValidation = True
#     Params.GradientThresholdValue = 30
#     ed.setParams(Params)
#     return ed


def getInferResult(model,img):
    results= model(img)
    if(len(results)==0):
        return []
    return results[0].boxes.xyxy.cpu().numpy()

# from pyAAMED import pyAAMED
if __name__=='__main__':

    # Define a static pose optimizer instance
    optimizer = StaticPoseOptimizer(K, dist)
    optimizer.set_extrinsics(eMc)

    # Traverse dataset and feed frames to optimizer
    import re                                                                                                          
                                                                                                                     
    # 数字排序                                                                                                                     
    jpg_files = sorted(                                                                                                
        [f for f in os.listdir(DATA_DIR) if f.endswith('.jpg') and f != 'temp'],
        key=lambda x: int(re.search(r'(\d+)', x).group(1))
    )                                                                                                                  
    # 字符串排序有误
    # jpg_files = sorted([f for f in os.listdir(DATA_DIR) 
    #                     if f.endswith('.jpg') and f != 'temp'])
    print(f"jpg_files: {jpg_files}")
    npy_files = {f.replace('.npy', '') : f for f in os.listdir(DATA_DIR) 
                 if f.endswith('.npy')}

    frame_id = 0
    last_bMo = None
    for jpg_file in jpg_files:
        ts = jpg_file.replace('.jpg', '')
        ts = ts.replace('img_','')
        npy_name = 'pose_' + ts
        if npy_name not in npy_files:
            continue
        print(f"Processing frame {frame_id}: {jpg_file}, {npy_name}")

        img_path = os.path.join(DATA_DIR, jpg_file)
        img = cv2.imread(img_path)

        
        robot_pose_path = os.path.join(DATA_DIR, npy_files[npy_name])
        robot_pose = np.load(robot_pose_path)

        t0 = time.perf_counter_ns()

        img_float = img.astype(np.float32)
        img_bright = img_float -50

        # 限制范围并转回 uint8
        img_bright = np.clip(
            img_bright, 0, 255).astype(np.uint8)
        result = getInferResult(model, img_bright)
        if result.shape[0]==0 or result.shape[1]==0:
            continue
        
        roi = img[int(result[0][1]):int(result[0][3]),
                  int(result[0][0]):int(result[0][2])]

        roi_x_min = int(result[0][0])
        roi_y_min = int(result[0][1])
        roi_x_max = int(result[0][2])
        roi_y_max = int(result[0][3])

        # aamed = pyAAMED(721, 1281)
        # aamed.setParameters(3.1415926/3, 3.4,0.77)
        gray_roi = cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
        # ed = init_ed()
        # ellipses_ = get_ellipse(ed,roi)

        detector = pyced.CED(np.ascontiguousarray(roi))
        detector.run_CED()
        rotRects = detector.getEllipsesAfterCluster()
        ellipses_ = []

        for e in rotRects:
            ellipses_.append((*e.center,e.size[0]/2,e.size[1]/2,e.angle))

        # res = aamed.run_AAMED(gray_roi)
        # ellipses_ = []
        # for ret in res:
        #     y,x,w,h,angle,score = ret
        #     ellipses_.append([x,y,w/2,h/2,angle])
        t1 = time.perf_counter_ns()
        # final_pts = postprocess_ed(ellipse,roi)
        matcher = UltimateSocketMatcher()
        matcher.obj_pts = obj_pts
        matcher.K = K
        matcher.dist = dist
        matcher.eMc = eMc

        vis_ellipse = draw_ellipse(roi,ellipses_)
        # cv.imwrite("roi1.png", vis_ellipse)
        
        final_pts,status,centers = matcher.solve(ellipses_,[*(result[0][:2]),*(result[0][2:]-result[0][:2])])
        print(f'find {len(final_pts)} points')

        t2 = time.perf_counter_ns()
        if final_pts is not None and centers.shape[0] >= 4: # need at least 4 points for solvePnP_IPPE
            
            ## Get cMo by SOLVEPNP_ITERATIVE
            # rvec, tvec, proj_back = matcher.estimate_pose(centers,None)

            # Get cMo by SOLVEPNP_EPNP
            pts3d = matcher.obj_pts[matcher.r_idx]
            pts2d = centers
            rvec, tvec, valid = solvePnP_IPPE(pts2d, pts3d, K, dist)
            if not valid:
                print(f"Frame:{frame_id} Timestamp:{ts}: PnP failed, skipping pose estimation")
                continue
            t3 = time.perf_counter_ns()

            # Build cMo (object in camera frame)
            cMo = np.eye(4)
            cMo[:3, :3] = Rotation.from_rotvec(rvec).as_matrix()
            cMo[:3, 3] = tvec

           
            # Compute initial bMo from first frame
            # bMo = bMe @ eMc @ cMo
            bMo_init = robot_pose @ eMc @ cMo
            if not optimizer.is_initialized():
                optimizer.set_initial_pose(bMo_init)

            max_error = None
            if last_bMo is not None:
                # print(f"last bMo:{last_bMo}")
                cMo_before = np.linalg.inv(eMc) @ np.linalg.inv(robot_pose) @ last_bMo
                rvec_before = Rotation.from_matrix(cMo_before[:3,:3]).as_rotvec()
                tvec_before = cMo_before[:3,3] 
                print(f"cMo_before: {pose_to_euler_tvec(cMo_before)}")
                proj, _ = cv2.projectPoints(pts3d, rvec_before, tvec_before, K, dist)
                proj_errors = proj.reshape(-1,2) - pts2d
                max_error = np.linalg.norm(proj_errors, axis=1).max()
                print(f"cMo_before reprojection error per point:\n {proj_errors} \n mean error: {np.linalg.norm(proj_errors, axis=1).mean():.4f}")
                # cMo_before_error = compute_reproj_error(pts3d, rvec_before, tvec_before, pts2d, K, dist)
                # print(f"cMo_before error: {cMo_before_error:.4f}")

                # find the largest 2 reprojection errors
                max_error_idx = np.argsort(np.linalg.norm(proj_errors, axis=1))[-2:]
                print(f"Indices of 2 largest reprojection errors: {max_error_idx}, errors: {np.linalg.norm(proj_errors[max_error_idx], axis=1)}")

                # visulize all the reprojection errors on the image and mark the largest two red
                vis_error = img.copy()
                for i, (x, y) in enumerate(pts2d):
                    color = (0,255,0) if i not in max_error_idx else (0,0,255)
                    cv2.circle(vis_error, (int(x), int(y)), 3, color, -1)
                    x_proj, y_proj = proj[i][0]
                    cv2.circle(vis_error, (int(x_proj), int(y_proj)), 3, (255,0,0), -1)
                    cv2.line(vis_error, (int(x), int(y)), (int(x_proj), int(y_proj)), (0,255,0), 1)
                # clip the image to the roi and visualize the reprojection error there
                vis_error = vis_error[roi_y_min:roi_y_max, roi_x_min:roi_x_max]
                # 放大2倍显示
                vis_error = cv2.resize(vis_error, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
                cv.imshow("reprojection_error", vis_error)
                vis_error_path = os.path.join(RESULT_DIR, f"{ts}_vis_error.png")
                cv.imwrite(vis_error_path, vis_error)
                # 保存图片
                # vis_error_path = os.path.join(RESULT_DIR, f"{frame_id}_reproj_error.png")
                # cv.imwrite(vis_error_path, vis_error)

            if max_error is None or max_error < 5.0:
                if optimizer.get_frame_count() >= SLIDING_WINDOW_SIZE:
                    optimizer.remove_frame(frame_id - SLIDING_WINDOW_SIZE+1)
                t4 = time.perf_counter_ns()

                optimizer.add_frame(frame_id, robot_pose, pts2d, pts3d)
                t5 = time.perf_counter_ns()

                result_optimized = optimizer.optimize()
                t6 = time.perf_counter_ns()

                print('ellipse fileter time: ',(t1-t0)/1e6)
                print('matcher solve time: ',(t2-t1)/1e6)
                print(f'cost time(ms):  pnp={(t3-t2)/1e6}, remove_frame={(t4-t3)/1e6}, add_frame={(t5-t4)/1e6}, optimize={(t6-t5)/1e6}')
                print('optimization total time: ',(time.perf_counter_ns()-t2)/1e6)

                now_error, _ = optimizer.get_frame_error(frame_id)
                print(f"Now Optimized reprojection error: {now_error:.4f}")
                ave_error = optimizer.get_average_error()
                print(f"Optimized:\n Average reprojection error: {ave_error:.4f}")
            

                frame_id += 1
            else:
                print(f"Timestamp:{ts}: Max reprojection error {max_error:.2f} exceeds threshold, skipping optimization update")
           

            bMo_optimized = optimizer.get_pose()
            if last_bMo is None:
                last_bMo = bMo_optimized
            cMo_optimized = optimizer.compute_cMo(robot_pose)
            # print(f'robot_pose: {robot_pose}')

            success, rvec_ba, tvec_ba = cv2.solvePnP(pts3d, pts2d, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
            cMo_ba = np.eye(4)
            cMo_ba[:3,:3] = Rotation.from_rotvec(rvec_ba.flatten()).as_matrix()
            cMo_ba[:3,3] = tvec_ba.flatten()
            bMo_ba = robot_pose @ eMc @ cMo_ba
            print(f"bMo_optimized: {pose_to_euler_tvec(bMo_optimized)}")
            print(f"bMo_pnp: {pose_to_euler_tvec(bMo_init)}")
            print(f"bMo_ba: {pose_to_euler_tvec(bMo_ba)}")

            print(f"cMo_optimized: {pose_to_euler_tvec(cMo_optimized)}")
            print(f"cMo_pnp: {pose_to_euler_tvec(cMo)}")
            print(f"cMo_ba: {pose_to_euler_tvec(cMo_ba)}")

            pnp_error = compute_reproj_error(pts3d, rvec, tvec, pts2d, K, dist)
            print(f"PnP reprojection error: {pnp_error:.4f}")


            cv2.drawFrameAxes(img,K,dist,cMo_optimized[:3,:3],cMo_optimized[:3,3:],10,3)
            cv2.drawFrameAxes(img,K,dist,cMo[:3,:3],cMo[:3,3:],20,1)

            
        # visualize centers
        if centers is not None:
            for i, (x, y) in enumerate(centers):
                cv2.circle(img, (int(x), int(y)), 2, (0,255,0), -1)
        # vis_points = visualize(roi,final_pts)
        # cv.imshow("vis_points",vis_points)
        # cv.imshow("vis_ellipse",vis_ellipse)
        # 裁切到roi显示，并放大2倍
        vis_result = img[roi_y_min:roi_y_max, roi_x_min:roi_x_max]
        vis_result = cv2.resize(vis_result, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
        cv.imshow("vis_result", vis_result)
        # 保存图片
        vis_result_path = os.path.join(RESULT_DIR, f"{ts}_vis_result.png")
        cv.imwrite(vis_result_path, vis_result)
        # 固定窗口大小显示
        # cv.namedWindow('image', cv.WINDOW_NORMAL)
        # cv.resizeWindow('image', 1280, 720)
        # cv.imshow('image',img)
        cv.waitKey(1)
    cv.destroyAllWindows()