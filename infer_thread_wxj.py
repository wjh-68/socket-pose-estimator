import cv2
import numpy as np
import time
import queue
import threading
import numpy as np
# import pyced
from gemiEd import *
from trt_pose_inf import *
import os

# from ultralytics import YOLO
# import pycylinderedsf as pyced

from concurrent.futures import ThreadPoolExecutor

# =========================================================
# TensorRT 推理线程
# =========================================================
class TRTWorker(threading.Thread):
    # img_path = "2026-05-15_11_53_04_181925111773603_720.jpg"
    # class_names = ["object"]
    # conf_th = 0.5
    # iou_th = 0.45
    # engine_path = "best.engine"
    # num_keypoints = 7
    def __init__(self,
                 engine_path,
                 frame_source,
                 det_queue,
                 stop_event,
                 class_names=['obj'],
                 conf_th=0.5,
                 iou_th=0.45,
                 num_keypoints=7):

        super().__init__(daemon=True)

        self.trt_model = YOLOTRTposeInference(engine_path, class_names, num_keypoints, conf_th, iou_th)
        # self.trt_model = YOLO('checkpoint/best.pt')
        self.frame_source = frame_source
        self.det_queue = det_queue
        self.stop_event = stop_event

    def run(self):

        while not self.stop_event.is_set():

            # ---------------------------------
            # 读取图像
            # ---------------------------------
            image = self.frame_source.read()

            if image is None:
                continue

            # ---------------------------------
            # TensorRT 推理
            # ---------------------------------
            t0 = time.time()

            try:
                boxes, keypoints = getInferResults(self.trt_model, image)
            except Exception:
                import traceback
                traceback.print_exc()
                continue
            if boxes.size==0 or keypoints.size==0:
                continue
            det_result = {
                'image': image,
                'keypoints': keypoints,
                'bboxes': boxes
            }
            # ---------------------------------
            # 实时系统：队列满则丢旧帧
            # ---------------------------------
            if self.det_queue.full():

                try:
                    self.det_queue.get_nowait()
                except queue.Empty:
                    pass

            self.det_queue.put(det_result)

            print(f"[TRT] {(time.time()-t0)*1000:.2f} ms")


# =========================================================
# 椭圆检测线程
# =========================================================
class EllipseWorker(threading.Thread):

    def __init__(self,
                 det_queue,
                 result_queue,
                 stop_event):

        super().__init__(daemon=True)
        self.det_queue = det_queue
        self.result_queue = result_queue

        self.stop_event = stop_event

    def run(self):

        while not self.stop_event.is_set():

            try:
                det_result = self.det_queue.get(timeout=0.1)

            except queue.Empty:
                continue

            image = det_result['image'][0]
            keypoints = det_result['keypoints'][0]
            t0 = time.time()
            
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
                img = image[tl[1]:br[1],tl[0]:br[0]]
                tls.append(tl)
                imgs.append(np.ascontiguousarray(img))
            t0 = time.perf_counter_ns()

            args_list = [
                (i, img,tl)
                for i, (img,tl) in enumerate(zip(imgs,tls))]
            
            with ThreadPoolExecutor(max_workers=8) as executor:
                results = executor.map(process_keypoint, args_list)
            centers = []
            for i, candidate in results:
                candidates[i] = candidate
                centers.append(candidate['p'])
            print(f'pyd time:{(time.perf_counter_ns()-t0)/1e6}')

            final_result = {
                'image': image,
                'keypoints': keypoints,
                'centers': centers
            }

            # ---------------------------------
            # 实时系统：队列满则丢旧结果
            # ---------------------------------
            if self.result_queue.full():

                try:
                    self.result_queue.get_nowait()
                except queue.Empty:
                    pass

            self.result_queue.put(final_result)

            print(f"[Ellipse] {(time.time()-t0)*1000:.2f} ms")


# =========================================================
# 示例 Camera
# =========================================================
class Camera:

    def __init__(self, cam_id=0):

        self.cap = cv2.VideoCapture(cam_id)
    def read(self):

        ret, frame = self.cap.read()

        if not ret:
            return None

        return frame

# =========================================================
# 主程序
# =========================================================
def main():

    # cv2.setNumThreads(1)

    # ---------------------------------
    # 通信队列
    # ---------------------------------
    det_queue = queue.Queue(maxsize=2)

    result_queue = queue.Queue(maxsize=2)

    stop_event = threading.Event()

    # ---------------------------------
    # 初始化模块
    # ---------------------------------

    camera = Camera(0)


    # ---------------------------------
    # 创建线程
    # ---------------------------------
    trt_thread = TRTWorker(
        'checkpoint/best.engine',
        camera,
        det_queue,
        stop_event
    )

    ellipse_thread = EllipseWorker(
        det_queue,
        result_queue,
        stop_event
    )

    # ---------------------------------
    # 启动线程
    # ---------------------------------
    trt_thread.start()

    ellipse_thread.start()

    # =====================================================
    # 主循环
    # =====================================================
    while True:

        try:
            result = result_queue.get(timeout=1.0)

        except queue.Empty:
            continue

        image = result['image']

        centers = result['centers']

        # ---------------------------------
        # 可视化
        # ---------------------------------
        for p in centers:

            cv2.circle(
                image,
                tuple(p.astype(np.int32)),
                5,
                (0, 255, 0),
                -1
            )
        cv2.imshow("result", image)

        key = cv2.waitKey(1)

        if key == 27:
            break

    # ---------------------------------
    # 退出
    # ---------------------------------
    stop_event.set()

    trt_thread.join()

    ellipse_thread.join()

    cv2.destroyAllWindows()


if __name__ == "__main__":

    main()

