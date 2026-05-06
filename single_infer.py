import cv2 as cv
from ultralytics import YOLO
import time
from gemiEd import *
from scipy.spatial.transform import Rotation
import os
# Load a model
model = YOLO("checkpoint/best.pt")  # load an official model

# from post_refine import getInferResult

# ============ Config ============
DATA_DIR = "dataset/save_data9"
eMc = np.array([[-7.2267956e-01,  6.9102561e-01, -1.4759262e-02 ,-5.1758522e+01],
                [-6.9116789e-01, -7.2264087e-01,  8.7790741e-03,  6.0040222e+01],
                [-4.5990809e-03,  1.6545586e-02,  9.9985254e-01,  9.7955963e+01],
                [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  1.0000000e+00]],
               dtype=np.float64)

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

# from pyAAMED import pyAAMED
if __name__=='__main__':
     # Collect timestamped files
    png_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.png') and f != 'temp'])
    npy_files = {f.replace('.npy', ''): f for f in os.listdir(DATA_DIR) if f.endswith('.npy')}

    for png_file in png_files:
        ts = png_file.replace('.png', '')

        img_path = os.path.join(DATA_DIR, png_file)
        img = cv2.imread(img_path)

        if ts not in npy_files:
            continue
        robot_pose_path = os.path.join(DATA_DIR, npy_files[ts])
        robot_pose = np.load(robot_pose_path)

    # for f in ['2026-04-23_11_26_18_591986505651862.png','2026-04-23_09_25_20_584726976776803.png','2026-04-22_15_23_56_519842711615772.png','2026-04-23_11_26_47_592014566825021.png',
    #           '2026-04-23_11_26_32_592000466579094.png']:
        
        # img =cv.imread(f'test_data9/{f}')
        # s=time.strftime('%Y-%m-%d_%H_%M_%S', time.localtime())
        # rnd = np.random.randint(100)
        # fn = f'{s}_{rnd:03d}.png'
        # cv.imwrite(fn,img)
        # print('img write: ',fn)

        img_float = img.astype(np.float32)
        img_bright = img_float -50

        # 限制范围并转回 uint8
        img_bright = np.clip(img_bright, 0, 255).astype(np.uint8)
        result = getInferResult(model, img_bright)

        roi = img[int(result[0][1]):int(result[0][3]),int(result[0][0]):int(result[0][2])]
        

        # aamed = pyAAMED(721, 1281)
        # aamed.setParameters(3.1415926/3, 3.4,0.77)
        gray_roi = cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
        # ed = init_ed()
        # ellipses_ = get_ellipse(ed,roi)

        detector = pyced.CED(np.ascontiguousarray(roi))
        detector.run_CED()
        rotRects = detector.getEllipsesAfterCluster()
        ellipses_ = []
        # save data for ba


        for e in rotRects:
            ellipses_.append((*e.center,e.size[0]/2,e.size[1]/2,e.angle))


        # res = aamed.run_AAMED(gray_roi)
        # ellipses_ = []
        # for ret in res:
        #     y,x,w,h,angle,score = ret
        #     ellipses_.append([x,y,w/2,h/2,angle])

        # final_pts = postprocess_ed(ellipse,roi)
        matcher = UltimateSocketMatcher()
        vis_ellipse = draw_ellipse(roi,ellipses_)
        cv.imwrite("roi1.png", vis_ellipse)
        t = time.perf_counter_ns()
        final_pts,status,centers = matcher.solve(ellipses_,[*(result[0][:2]),*(result[0][2:]-result[0][:2])])

        if final_pts is not None:
            rvec, tvec, proj_back = matcher.estimate_pose(centers,None)
            # print(f'rvec:{180.0/np.pi*rvec.flatten()}   tvec:{tvec.flatten()}')
            cMo = np.eye(4,dtype=np.float32)
            cMo[:3,:3] = Rotation.from_rotvec(rvec[:3,0]).as_matrix()
            cMo[:3,3] = tvec[:3,0]
            oMo = np.eye(4,dtype=np.float32)
            oMo[:3,:3] = Rotation.from_rotvec(np.array([0,0,0])*np.pi).as_matrix()
            cMo_ = cMo@oMo
            bMo = robot_pose @ eMc @ cMo_

            def compute_reproj_error(obj_pts, rvec, tvec, pts2d, K, dist):
                proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, dist)
                return np.linalg.norm(proj.reshape(-1,2) - pts2d, axis=1).mean()                                        
            
            err1 = compute_reproj_error(matcher.obj_pts[matcher.r_idx], rvec, tvec, centers, matcher.K, matcher.dist)
            euler_out = Rotation.from_matrix(bMo[:3, :3]).as_euler('xyz', True)
            print(
                f"{bMo[0,3]:>10.2f} {bMo[1,3]:>10.2f} {bMo[2,3]:>10.2f} "
                f"{euler_out[0]:>10.2f} {euler_out[1]:>10.2f} {euler_out[2]:>10.2f} "
                f"{err1:>8.2f}")
            # print(f'bMo: rvec:{180.0/np.pi*Rotation.from_matrix(bMo[:3,:3]).as_rotvec()}   tvec:{bMo[:3,3]}')
            # print(f'bMo: rvec:{180.0/np.pi*rvec.flatten()}   tvec:{tvec.flatten()}')

            # print(f'bMo curr:{bMo}')

            cv2.drawFrameAxes(img,matcher.K,matcher.dist,cMo[:3,:3],cMo[:3,3:],50,3)
            print(f'find {len(final_pts)} points')
        print('ellipse fileter time: ',(time.perf_counter_ns()-t)/1e6)
        # vis_points = visualize(roi,final_pts)
        # cv.imshow("vis_points",vis_points)
        cv.imshow("vis_ellipse",vis_ellipse)
        cv.imshow('image',img)
        cv.waitKey(1)
    cv.destroyAllWindows()