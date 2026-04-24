import cv2 as cv
from ultralytics import YOLO
import time
from gemiEd import *
from scipy.spatial.transform import Rotation
# Load a model
model = YOLO("checkpoint/best.pt")  # load an official model

import pyaubo_sdk
t = time.perf_counter_ns()
# Predict with the model
results = model("000652_0000_1776327529268139_1776327529319782_Color_1920x1080.png")  # predict on an image
print('infer time: ',(time.perf_counter_ns()-t)/1e6)
# Access the results
robot_ip = "192.168.1.20"  # 服务器 IP 地址
robot_port = 30004  # 端口号
M_PI = 3.14159265358979323846
robot_rpc_client = pyaubo_sdk.RpcClient()
# from handeyecalib import get_robot_pose
def get_robot_pose(robot_name):
    tcp_pose = robot_rpc_client.getRobotInterface(robot_name).getRobotState().getTcpPose()
    print("TCP的位姿:", tcp_pose)
    r = Rotation.from_euler('xyz', tcp_pose[3:])
    t = np.array(tcp_pose[:3]).reshape((3,1))
    robot_pose = np.eye(4)
    robot_pose[:3,:3] = r.as_matrix()
    robot_pose[:3,3] = t.flatten()*1000
    print('robot pose euler:{}'.format(Rotation.from_matrix(robot_pose[:3,:3]).as_euler('xyz',True)))
    print('robot translation:{}'.format(robot_pose[:3,3]))
    return robot_pose,tcp_pose
def init_ed():
    Params = cv2.ximgproc.EdgeDrawing.Params()
    ed = cv2.ximgproc.createEdgeDrawing()
    Params.EdgeDetectionOperator = 1
    Params.MinPathLength = 45
    Params.PFmode = 0
    Params.NFAValidation = True
    Params.GradientThresholdValue = 30
    ed.setParams(Params)
    return ed


def getInferResult(model,img):
    results= model(img)
    if(len(results)==0):
        return []
    return results[0].boxes.xyxy.cpu().numpy()
eMc = np.array([[-6.9855857e-01,  7.1512282e-01,  2.4804471e-02, -5.1826664e+01],
       [-7.1555281e-01, -6.9815123e-01, -2.3854841e-02,  5.5274796e+01],
       [ 2.5813223e-04, -3.4412913e-02,  9.9940765e-01,  9.5362617e+01],
       [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  1.0000000e+00]],
      dtype=np.float32)
if __name__ == '__main__':
    obj_t = np.array([     434.33 ,    -510.15  ,    523.57])
    obj_euler = np.array([    -80.346,    -0.32487,      179.95])
    bMo_t = np.eye(4,dtype=np.float32)
    bMo_t[:3,:3] = Rotation.from_euler('xyz',obj_euler,True).as_matrix()
    bMo_t[:3,3] = obj_t
    print(f'bMo_t:{bMo_t}')
    gripper_pose = np.eye(4,dtype=np.float32)
    gripper_pose[:3,:3] = Rotation.from_euler('xyz',[1.658,0.806,0.057]).as_matrix()
    gripper_pose[:3,3] = np.array([428.38,-231.39,553.56])
    print(f'gripper pose:{gripper_pose}')
    gMo = np.linalg.inv(gripper_pose)@bMo_t
    print(f'gMo:{gMo}')
    robot_name = None
    robot_rpc_client.connect(robot_ip, robot_port)  # 接口调用: 连接 RPC 服务
    if robot_rpc_client.hasConnected():
        print("Robot rcp_client connected successfully!")
        robot_rpc_client.login("aubo", "123456")  # 接口调用: 机械臂登录
        if robot_rpc_client.hasLogined():
            print("Robot rcp_client logined successfully!")
            robot_name = robot_rpc_client.getRobotNames()[0]  # 接口调用: 获取机器人的名字
    img = cv.imread("Camera Feed_screenshot_20.04.2026.png")
    cap = cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv.CAP_PROP_BRIGHTNESS,128)
    ret,img = cap.read()
    s=time.strftime('%Y-%m-%d_%H_%M_%S', time.localtime())
    rnd = np.random.randint(100)
    cv.imwrite(f'{s}_{rnd:03d}.png',img)
    if not ret:
        print('error read img')
    result = getInferResult(model, img)

    roi = img[int(result[0][1]):int(result[0][3]),int(result[0][0]):int(result[0][2])]
    cv.imwrite("roi1.png", roi)
    ed = init_ed()
    ellipse = get_ellipse(ed,roi)
    # final_pts = postprocess_ed(ellipse,roi)
    matcher = UltimateSocketMatcher()
    t = time.perf_counter_ns()
    final_pts,status = matcher.solve(ellipse)
    if final_pts is not None:
        # 插座在相机坐标系下的位姿
        rvec, tvec, proj_back = matcher.estimate_pose(final_pts,result[0][:2])
        print(f'rvec:{rvec}   tvec:{tvec}')
        cMo = np.eye(4,dtype=np.float32)
        cMo[:3,:3] = Rotation.from_rotvec(rvec[:3,0]).as_matrix()
        cMo[:3,3] = tvec[:3,0]
        oMo = np.eye(4,dtype=np.float32)
        oMo[:3,:3] = Rotation.from_rotvec(np.array([1,0,0])*np.pi).as_matrix()
        cMo_ = cMo@oMo
        # 机械臂末端在 base 坐标系上位姿
        robot_pose,raw = get_robot_pose(robot_name)
        bMo = robot_pose@eMc@cMo_
        # bMo_ = robot_pose@eMc@cMo
        # print(f'obj pose:{bMo[:3,3]}')
        # eu = Rotation.from_matrix(bMo[:3,:3]).as_euler('xyz',True)
        # print(f'obj euler:{eu}')
        # cMd = np.eye(4,dtype=np.float32)
        # cMd[:3,3] = np.array([0,0,250])
        # bMe_ = bMo@np.linalg.inv(eMc@cMd)
        # bMo = robot_pose@eMc@cMo
        print(f'bMo curr:{bMo}')
        bMe_n = bMo@np.linalg.inv(gMo)
        print('translate: ',bMe_n[:3,3])
        print('Rotation degree: ',Rotation.from_matrix(bMe_n[:3,:3]).as_euler('xyz',True))
        print('Rotation radians: ',Rotation.from_matrix(bMe_n[:3,:3]).as_euler('xyz'))
        cv2.drawFrameAxes(img,matcher.K,matcher.dist,cMo[:3,:3],cMo[:3,3:],50,3)
        print(f'find {len(final_pts)} points')
    print('ellipse fileter time: ',(time.perf_counter_ns()-t)/1e6)
    vis_points = visualize(roi,final_pts)
    vis_ellipse = draw_ellipse(roi,ellipse)
    cv.imshow("vis_points",vis_points)
    cv.imshow("vis_ellipse",vis_ellipse)
    cv.imshow('image',img)
    cv.waitKey(0)
    cv.destroyAllWindows()
    
