import argparse
import time

import cv2
import numpy as np
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda

# COCO 17个关键点的标准骨骼连接关系 (起点索引, 终点索引)
# COCO17_SKELETON = [
#     (0, 1), (0, 2), (1, 3), (2, 4),
#     (0, 5), (0, 6), (5, 7), (7, 9),
#     (6, 8), (8, 10), (5, 6), (5, 11),
#     (6, 12), (11, 12), (11, 13), (13, 15),
#     (12, 14), (14, 16),
# ]

KEYPOINT_NAMES = [
    "kp1", "kp2", "kp3", "kp4", "kp5",
    "kp6", "kp7"
]

class YOLOTRTposeInference:
    def __init__(self, engine_path, class_names=None, num_keypoints=7, conf_threshold=0.25, iou_threshold=0.45):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.num_keypoints = num_keypoints
        self.class_names = class_names or ["person"]

        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()
        # self.input_binding_idx = self._find_binding_index(is_input=True)
        # self.output_binding_idx = self._find_binding_index(is_input=False)

        self.input_name = self.engine.get_tensor_name(0)
        self.output_name = self.engine.get_tensor_name(1)   


        # self.input_shape = tuple(self.engine.get_binding_shape(self.input_binding_idx))
        # self.output_shape = tuple(self.engine.get_binding_shape(self.output_binding_idx))

        self.input_shape = tuple(self.engine.get_tensor_shape(self.input_name))
        self.output_shape = tuple(self.engine.get_tensor_shape(self.output_name))



        self.input_h = int(self.input_shape[2])
        self.input_w = int(self.input_shape[3])

        self._allocate_buffers()

    def _find_binding_index(self, is_input):
        for idx in range(self.engine.num_bindings):
            if self.engine.binding_is_input(idx) == is_input:
                return idx
        raise ValueError("Unable to find matching engine binding.")

    def _allocate_buffers(self):
        if any(dim <= 0 for dim in self.input_shape):
            raise ValueError("Engine input shape must be fixed.")
        if any(dim <= 0 for dim in self.output_shape):
            raise ValueError("Engine output shape must be fixed.")

        self.d_input = cuda.mem_alloc(trt.volume(self.input_shape) * np.dtype(np.float32).itemsize)
        self.d_output = cuda.mem_alloc(trt.volume(self.output_shape) * np.dtype(np.float32).itemsize)
        self.stream = cuda.Stream()

    def preprocess(self, image):
        orig_h, orig_w = image.shape[:2]
        scale = min(self.input_w / orig_w, self.input_h / orig_h)
        new_w, new_h = int(orig_w * scale), int(orig_h * scale)

        resized = cv2.resize(image, (new_w, new_h))
        padded = np.full((self.input_h, self.input_w, 3), 114, dtype=np.uint8)

        pad_w = (self.input_w - new_w) // 2
        pad_h = (self.input_h - new_h) // 2
        padded[pad_h : pad_h + new_h, pad_w : pad_w + new_w] = resized

        processed = padded.astype(np.float32) / 255.0
        processed = processed[..., ::-1].transpose(2, 0, 1)
        processed = np.ascontiguousarray(processed)

        return processed, scale, pad_w, pad_h, orig_w, orig_h

    def postprocess(self, output, scale, pad_w, pad_h, orig_w, orig_h):
        output = np.ascontiguousarray(output).reshape(self.output_shape)
        if output.ndim == 3 and output.shape[1] < output.shape[2]:
            output = output.transpose(0, 2, 1)

        predictions = output[0]
        total_dim = predictions.shape[1]

        # 输出维度
        # 4 (xywh) + 1 (objectness) + num_keypoints*2 (x,y)
        #或者 4 (xywh) + 1 (objectness) + num_keypoints*3 (x,y,v)
        # expected_pose_dim = 5 + self.num_keypoints * 2
        expected_pose_dim = 5 + self.num_keypoints * 3
        if total_dim < expected_pose_dim:
            raise ValueError(
                f"Unexpected pose output shape {predictions.shape}. "
                f"Expected at least {expected_pose_dim + 1} columns."
            )

        xywh = predictions[:, :4]
        objectness = predictions[:, 4]
        # keypoints = predictions[:, 5 : 5 + self.num_keypoints * 2].reshape(-1, self.num_keypoints, 2)
        # class_scores = predictions[:, 5 + self.num_keypoints * 2 :]

        keypoints = predictions[:, 5 : 5 + self.num_keypoints * 3].reshape(-1, self.num_keypoints, 3)
        class_scores = predictions[:, 5 + self.num_keypoints * 3 :]
        # 【核心修复点】增加空数组判断，防止 argmax 报错
        if class_scores.size == 0:
            # 如果没有分类分支（即总维度刚好等于26），置信度直接等于 objectness
            confidences = objectness
            class_ids = np.zeros(len(objectness), dtype=int) # 默认为第0类（人）
        else:
            # 如果有分类分支，正常计算 argmax 和最终置信度
            class_ids = np.argmax(class_scores, axis=1)
            class_confidences = class_scores[np.arange(len(class_scores)), class_ids]
            confidences = objectness * class_confidences

        mask = confidences >= self.conf_threshold
        if not np.any(mask):
            return []

        xywh = xywh[mask]
        keypoints = keypoints[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]

        x, y, w, h = xywh.T
        x1 = (x - w / 2 - pad_w) / scale
        y1 = (y - h / 2 - pad_h) / scale
        x2 = (x + w / 2 - pad_w) / scale
        y2 = (y + h / 2 - pad_h) / scale

        boxes = np.stack([x1, y1, x2, y2], axis=1)
        boxes[:, 0] = np.clip(boxes[:, 0], 0, orig_w - 1)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, orig_h - 1)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, orig_w - 1)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, orig_h - 1)

        keypoints[:, :, 0] = np.clip((keypoints[:, :, 0] - pad_w) / scale, 0, orig_w - 1)
        keypoints[:, :, 1] = np.clip((keypoints[:, :, 1] - pad_h) / scale, 0, orig_h - 1)

        keep = self.nms(boxes, confidences, self.iou_threshold)

        results = []
        for idx in keep:
            results.append(
                {
                    "box": boxes[idx].astype(int).tolist(),
                    "score": float(confidences[idx]),
                    "class_id": int(class_ids[idx]),
                    "keypoints": keypoints[idx].astype(int).tolist(),
                }
            )

        return results

    @staticmethod
    def nms(boxes, scores, iou_threshold):
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        keep = []

        while order.size > 0:
            i = order[0]
            keep.append(i)
            if order.size == 1:
                break

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            order = order[1:][iou <= iou_threshold]

        return keep

    def infer(self, image):
        if image is None:
            raise ValueError("输入图像无效")

        processed, scale, pad_w, pad_h, orig_w, orig_h = self.preprocess(image)

        cuda.memcpy_htod_async(self.d_input, processed, self.stream)

        # self.context.set_binding_shape(self.input_binding_idx, self.input_shape)
        # self.context.set_tensor_address(self.input_binding_idx, int(self.d_input))
        # self.context.set_tensor_address(self.output_binding_idx, int(self.d_output))
        # self.context.execute_async_v3(self.stream.handle)

        self.context.set_input_shape(self.input_name, self.input_shape)
        self.context.set_tensor_address(self.input_name, int(self.d_input))
        self.context.set_tensor_address(self.output_name, int(self.d_output))
        self.context.execute_async_v3(self.stream.handle)

        output = np.empty(self.output_shape, dtype=np.float32)
        cuda.memcpy_dtoh_async(output, self.d_output, self.stream)
        self.stream.synchronize()

        return self.postprocess(output, scale, pad_w, pad_h, orig_w, orig_h)

    def show_results(self, image, results):
        for res in results:
            x1, y1, x2, y2 = res["box"]
            score = res["score"]
            class_id = res["class_id"]
            label = self.class_names[class_id] if class_id < len(self.class_names) else str(class_id)

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f"{label}: {score:.2f}", (x1, max(y1 - 10, 0)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)



            kpts = res["keypoints"]

            for idx, kp in enumerate(kpts):
                x,y = int(kp[0]), int(kp[1])
                cv2.circle(image, (x, y), 3, (0, 0, 255), -1)
                if idx < len(KEYPOINT_NAMES):
                    cv2.putText(image, KEYPOINT_NAMES[idx], (x + 5, y - 5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

            

            # for connection in COCO17_SKELETON:
            #     idx1, idx2 = connection
            #     if idx1 >= len(res["keypoints"]) or idx2 >= len(res["keypoints"]):
            #         continue
            #     pt1 = tuple(res["keypoints"][idx1])
            #     pt2 = tuple(res["keypoints"][idx2])
            #     cv2.line(image, pt1, pt2, (255, 0, 0), 2)

        return image


def load_class_names(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]



def getInferResults(model, image):
    results = model.infer(image)

    # 如果没有检测到任何目标，直接返回两个空的 numpy 数组
    if len(results) == 0:
        return np.array([]), np.array([])
    
    # 遍历字典列表，分别提取 box 和 keypoints
    boxes_list = []
    keypoints_list = []
    
    for res in results:
        boxes_list.append(res["box"]) 
        keypoints_list.append(res["keypoints"]) 
    
    
    return np.array(boxes_list), np.array(keypoints_list)


def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv8 pose TensorRT inference")
    parser.add_argument("--engine", required=True, help="TensorRT engine file path")
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--classes", default=None, help="Optional class names txt file")
    parser.add_argument("--keypoints", type=int, default=17, help="Number of pose keypoints")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--output", default="output_pose.jpg", help="Output image path")
    return parser.parse_args()





def main():
    # args = parse_args()
    img_path = "2026-05-15_11_53_04_181925111773603_720.jpg"
    class_names = ["object"]  
    conf_th = 0.5
    iou_th = 0.45
    engine_path = "best.engine"
    num_keypoints = 7
    model = YOLOTRTposeInference(
        engine_path,
        class_names=class_names,
        num_keypoints=num_keypoints,
        conf_threshold=conf_th,
        iou_threshold=iou_th,
    )

    image = cv2.imread(img_path)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")


    # 调用封装的函数获取检测结果
    boxes, keypoints = getInferResults(model, image)
    print("检测框坐标:", boxes[0])
    print("关键点", keypoints.shape)

    # 原始检测调用结果输出 
    start_time = time.time()
    results = model.infer(image)
    elapsed = time.time() - start_time
    print(f"Inference time: {elapsed * 1000:.1f} ms, poses: {len(results)}")

    output_image = model.show_results(image.copy(), results)
    cv2.imwrite("output.jpg", output_image)
    print(f"Saved result to: output.jpg")


if __name__ == "__main__":
    main()
