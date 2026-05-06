import cv2 as cv
import threading
from ultralytics import YOLO
import time
from gemiEd import *
from scipy.spatial.transform import Rotation
import os
import numpy as np
from static_pose_optimizer import StaticPoseOptimizer, pose_to_euler_tvec

# Load a model
model = YOLO("checkpoint/best.pt")  # load an official model

# ======== Sensor Config =========
ROBOT_IP = "192.168.1.20"
ROBOT_PORT = 30004
CAMERA_ID = 0

# ============ Config ============
DATA_DIR = "dataset/save_data"
SLIDING_WINDOW_SIZE = 10
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
oMo[:3,:3] = Rotation.from_rotvec(np.array([1,0,0])*np.pi).as_matrix()

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

def get_robot_pose(robot_rpc_client, robot_name):
    """Get robot TCP pose from rpc client, convert to 4x4 matrix."""
    tcp_pose = robot_rpc_client.getRobotInterface(robot_name).getRobotState().getTcpPose()
    r = Rotation.from_euler('xyz', tcp_pose[3:])
    t = np.array(tcp_pose[:3]).reshape((3, 1))
    robot_pose = np.eye(4)
    robot_pose[:3, :3] = r.as_matrix()
    robot_pose[:3, 3] = t.flatten() * 1000  # mm
    return robot_pose, tcp_pose

# ===================== 线程安全：最新帧缓存 =====================
latest_frame = None
latest_pose = None
frame_lock = threading.Lock()

# robot params
robot_rpc_client = None
robot_name = None
# ===================== 独立线程：同时读取 相机和机械臂 =====================
def read_camera_robot():
    
    global latest_frame, latest_pose
    global robot_name, robot_rpc_client
    cap = cv2.VideoCapture(CAMERA_ID)
    
    # 你的摄像头参数
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1440)
    cap.set(cv2.CAP_PROP_BRIGHTNESS, 128)
    
    while True:
        ret, frame = cap.read()
        robot_pose, raw_tcp = get_robot_pose(robot_rpc_client, robot_name)

        if not ret:
            break
        
        # 只保存最新一帧，旧帧直接丢弃！解决缓冲区延迟
        with frame_lock:
            latest_frame = frame.copy()
            latest_pose = robot_pose
    
    cap.release()


# ===================== 独立线程：只读取最新帧 =====================
def read_camera():
    
    global latest_frame
    cap = cv2.VideoCapture(CAMERA_ID)
    
    # 你的摄像头参数
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1440)
    cap.set(cv2.CAP_PROP_BRIGHTNESS, 128)
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # 只保存最新一帧，旧帧直接丢弃！解决缓冲区延迟
        with frame_lock:
            latest_frame = frame.copy()
            
    
    cap.release()



def main():
    import pyaubo_sdk
    global latest_pose,latest_frame
    global robot_rpc_client,robot_name
    
    # Define a static pose optimizer instance
    optimizer = StaticPoseOptimizer(K, dist)
    optimizer.set_extrinsics(eMc)

    # # Camera
    # cap = cv2.VideoCapture(CAMERA_ID)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1440)
    # cap.set(cv2.CAP_PROP_BRIGHTNESS, 128)

    # Robot connection
    robot_rpc_client = pyaubo_sdk.RpcClient()
    robot_rpc_client.connect(ROBOT_IP, ROBOT_PORT)
    if not robot_rpc_client.hasConnected():
        print("Failed to connect to robot")
        return
    robot_rpc_client.login("aubo", "123456")
    if not robot_rpc_client.hasLogined():
        print("Failed to login to robot")
        return
    robot_name = robot_rpc_client.getRobotNames()[0]
    print(f"Connected to robot: {robot_name}")

    # 启动摄像头和机械臂拿取线程（后台一直跑）
    frame_thread = threading.Thread(target=read_camera_robot, daemon=True)
    frame_thread.start()

    frame_id = 0
    while True:
        # Get camera frame
        # 1. 获取【最新实时帧】
        with frame_lock:
            if latest_frame is None or latest_pose is None:
                continue
            img = latest_frame  # 永远是当前最新画面！
            robot_pose = latest_pose
        # ret, img = cap.read()
        # if not ret:
        #     print("Failed to grab frame")
        #     break

        # Get robot pose
        # 机械臂末端在 base 坐标系上位姿
        # robot_pose, raw_tcp = get_robot_pose(robot_rpc_client, robot_name)

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
        cv.imwrite("roi1.png", vis_ellipse)
        
        final_pts,status,centers = matcher.solve(ellipses_,[*(result[0][:2]),*(result[0][2:]-result[0][:2])])
        t2 = time.perf_counter_ns()
        if final_pts is not None and centers.shape[0] >= 4: # need at least 4 points for solvePnP_IPPE
            
            ## Get cMo by SOLVEPNP_ITERATIVE
            # rvec, tvec, proj_back = matcher.estimate_pose(centers,None)

            # Get cMo by SOLVEPNP_EPNP
            pts3d = matcher.obj_pts[matcher.r_idx]
            pts2d = centers
            rvec, tvec, valid = solvePnP_IPPE(pts2d, pts3d, K, dist)
            if not valid:
                print(f"{frame_id} Frame: PnP failed, skipping pose estimation")
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

            if optimizer.get_frame_count() >= SLIDING_WINDOW_SIZE:
                optimizer.remove_frame(frame_id - SLIDING_WINDOW_SIZE+1)
            t4 = time.perf_counter_ns()

            optimizer.add_frame(frame_id, robot_pose, pts2d, pts3d)
            t5 = time.perf_counter_ns()

            frame_id += 1
            result = optimizer.optimize()
            t6 = time.perf_counter_ns()

            bMo_optimized = optimizer.get_pose()
            cMo_optimized = optimizer.compute_cMo(robot_pose)
            # Transform optimized poses to model frame for visulization
            # bMo_optimized = bMo_optimized @ oMo
            # cMo_optimized = cMo_optimized @ oMo
            euler_bMo, trans_bMo = pose_to_euler_tvec(bMo_optimized)
            euler_cMo, trans_cMo = pose_to_euler_tvec(cMo_optimized)
            print(f'robot_pose: {robot_pose}')

            print(f"bMo: {euler_bMo, trans_bMo}")
            print(f"cMo: {euler_cMo, trans_cMo}")
            ave_error = optimizer.get_average_error()
            print(f"Average reprojection error: {ave_error:.4f}")

            cv2.drawFrameAxes(img,K,dist,cMo_optimized[:3,:3],cMo_optimized[:3,3:],20,3)
            t7 = time.perf_counter_ns()

            print(f'find {len(final_pts)} points')
        print('ellipse fileter time: ',(t1-t0)/1e6)
        print('matcher solve time: ',(t2-t1)/1e6)
        print(f'cost time(ms):  pnp={(t3-t2)/1e6}, remove_frame={(t4-t3)/1e6}, add_frame={(t5-t4)/1e6}, optimize={(t6-t5)/1e6}, draw={(t7-t6)/1e6}')
        print('optimization total time: ',(time.perf_counter_ns()-t2)/1e6)
        # vis_points = visualize(roi,final_pts)
        # cv.imshow("vis_points",vis_points)

        # save image with name of frame_id
        img_path = os.path.join(DATA_DIR, f"img_{frame_id}.jpg")
        
        # 保存
        cv2.imwrite(img_path, img)
        
        cv.imshow("vis_ellipse",vis_ellipse)
        cv.imshow('image',img)
        cv.waitKey(1)
    cv.destroyAllWindows()

# from pyAAMED import pyAAMED
if __name__=='__main__':
    main()