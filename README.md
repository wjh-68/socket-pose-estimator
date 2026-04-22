# Socket Pose Estimator

基于滑动窗口和光束平差(Bundle Adjustment)优化的插座姿态估计系统。

## 概述

该项目实现了一个7孔插座姿态估计Pipeline:
1. 使用YOLOv8检测插座区域
2. 使用EdgeDrawing椭圆检测提取插座孔位
3. 模板匹配确定7个关键点
4. IPPE(SolvePnP)求解初始位姿
5. 滑动窗口BA联合优化多帧位姿

## 核心文件

- `socket_pose_estimator.py` - 主要实现代码
- `checkpoint/best.pt` - YOLOv8检测模型
- `dataset/images/` - 测试图片目录

## 依赖

```bash
pip install opencv-python opencv-contrib-python scipy ultralytics numpy
```

## 使用方法

```bash
python socket_pose_estimator.py
```

VISUALIZE=True 时会显示检测结果，VISUALIZE_DELAY 控制每帧停留时间(ms)。

## 算法流程

1. **单帧检测**: IPPE (SolvePnP with SOLVEPNP_IPPE) 从2D-3D对应关系求解相机位姿
2. **滑动窗口**: 维护N帧的窗口，对所有位姿进行联合优化
3. **BA优化**: 使用Levenberg-Marquardt最小化重投影误差

## 模型文件

模型文件(`.pt`, `.onnx`)较大，不在版本控制内。如需使用:
1. 下载best.pt到checkpoint/目录
2. 或修改代码中的模型路径