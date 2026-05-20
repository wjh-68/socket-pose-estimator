# Socket Pose Estimator

基于滑动窗口与光束平差（Bundle Adjustment）的插座姿态估计系统。

## 项目概述

本仓库实现了一个7孔插座（charge port）姿态估计流程，适用于机器人视觉定位场景。核心流程包括：

- 使用 `YOLOv8` 检测插座 ROI
- 使用 EdgeDrawing / 椭圆拟合提取孔位候选
- 使用 `UltimateSocketMatcher` 进行7点匹配
- 使用 `cv2.solvePnP(..., flags=SOLVEPNP_IPPE)` 进行初始单帧位姿求解
- 使用滑动窗口 BA 进行多帧联合优化，提升稳定性与准确度

## 重点文件

- `chargeport_pose_estimator_ba.py`
  - 这是主程序，负责读取数据集、执行YOLO检测、孔位提取、PnP求解、滑动窗口 BA 优化、可视化与结果保存。

- `static_pose_optimizer_ba.py`
  - 定义 `StaticPoseOptimizer` 类，提供基于全局物体位姿加每帧 SE(3) 扰动的 BA 优化器。
  - 可作为模块使用，也包含一个简单的脚本级测试入口。

- `checkpoint/best.pt`
  - YOLOv8 目标检测模型权重（需放置到 `checkpoint/` 目录）。

- `dataset/save_data3/` 等
  - 测试/记录数据集目录，包含图像与 `metadata.json`。

## 环境依赖

使用 `conda activate cv48` 激活环境, 并在仓库根目录激活使用 `venv310` 虚拟环境。

```bash
source venv310/bin/activate
pip install opencv-python opencv-contrib-python scipy ultralytics numpy pandas matplotlib
```

> `pandas` 和 `matplotlib` 用于结果保存与绘图，如果只想运行核心估计流程，也可以先安装最少依赖。

## 运行 `chargeport_pose_estimator_ba.py`

```bash
python chargeport_pose_estimator_ba.py
```

### 主要配置项

在文件顶部可以调整以下常量：

- `DATA_DIR` - 数据集根目录，程序会读取 `metadata.json` 和图像文件
- `RESULT_DIR` - 结果输出目录，程序会保存 JSON/CSV/PNG
- `SLIDING_WINDOW_SIZE` - 滑动窗口帧数
- `MAX_FRAMES` - 最大处理帧数，`-1` 表示使用全部帧
- `BEGIN_FRAME_ID` - 数据集中跳过比该帧号更小的帧
- `PROCESSED_INTERVAL` - 帧时间间隔筛选阈值

### 关键流程说明

1. 读取 `metadata.json` 中的相机图像路径、时间戳和机器人末端位姿。
2. 使用 `YOLOv8` 对图像进行插座 ROI 检测，并裁剪 ROI 进行后续处理。
3. 使用 `gemiEd.py` 中的 EdgeDrawing 椭圆拟合提取孔位候选。将结果传入 `UltimateSocketMatcher` 进行7个孔位的匹配。
4. 对匹配点使用 `two_round_pnp(...)`：
   - 第一轮使用 `SOLVEPNP_IPPE` 求解位姿
   - 计算每点重投影误差并使用自适应阈值剔除离群点
   - 如果满足最少 4 个内点，则再次求解精化位姿
5. 将当前帧添加到 `StaticPoseOptimizer` 的滑动窗口中，并执行 BA 优化。
   - 若 PnP 结果与优化器估计的位姿差距过大, 则不加入窗口(排除可能的点序检测出错情况)
6. 输出当前帧的 PnP 结果、优化结果、重投影误差和可视化图像。

### 输出内容

结果会保存在 `RESULT_DIR` 下，主要文件包括：

- `frame_records.json` - 每帧完整记录（PnP、优化结果、2D/3D 点、误差等）
- `pnp_records.json` / `pnp_results.csv` - 单帧 PnP 结果记录
- `optimize_records.json` / `optimize_results.csv` - BA 优化结果记录
- `frame_******_vis_result.png` - 帧级可视化结果图
- `pose_components_vs_frame.png` - bMo/cMo/bMe 平移随帧变化图
- `bmo_euler_vs_frame.png`, `cmo_euler_vs_frame.png` - Euler 角随帧变化图
- `error_vs_frame.png` - 重投影误差曲线图

### 运行结果说明

`chargeport_pose_estimator_ba.py` 会打印：

- 每帧是否检测成功
- PnP 是否成功，以及是否发生大位姿差导致帧被拒绝
- 初始 PnP、BA 和滑动窗口优化后的位姿比较
- 当前帧误差与平均误差

程序结束后，会输出统计信息，例如：

- 无检测帧数 (`No Detection`)
- 检测点数不足 7 个的帧数
- PnP 失败帧数
- 大位姿差拒绝帧数

## `static_pose_optimizer_ba.py` 说明

### `StaticPoseOptimizer`

该类实现了一个静态物体的多帧联合优化器，核心思想是：

- 全局标定一个共享的物体位姿 `bMo`
- 为每个帧引入一个可优化的 SE(3) 扰动变量，用于吸收机器人末端与相机时序/标定误差
- 通过 `scipy.optimize.least_squares` 最小化重投影误差
- 支持 per-point 观测噪声 `point_sigmas` 和基于 PnP 误差的动态权重调整

### 主要接口

- `set_extrinsics(eMc)` - 设置相机到末端执行器的外参
- `set_object_pts(obj_pts)` - 设置物体 3D 点
- `set_point_sigmas(point_sigmas)` - 设置每个观察点的像素不确定性
- `set_initial_pose(pose)` - 设置 BA 优化的初始全局物体位姿
- `add_frame(frame_index, robot_pose, pts2d, pts3d=None, per_point_errors_pnp=None)`
- `optimize(pose_init=None)` - 运行优化
- `get_pose()` / `get_pose_euler()` / `get_average_error()`

### 典型使用

`chargeport_pose_estimator_ba.py` 中的用法：

```python
optimizer = StaticPoseOptimizer(K, dist, prior_sigma=PRIOR_SIGMA, point_sigmas=POINT_SIGMAS)
optimizer.set_extrinsics(eMc)
optimizer.set_object_pts(obj_pts)
optimizer.set_initial_pose(bMo_init)
optimizer.add_frame(frame_id, robot_pose, pts2d, pts3d, per_point_errors_pnp=per_point_errors_pnp)
optimizer.optimize()
```

### 模块测试入口

如果直接运行 `static_pose_optimizer_ba.py`，会执行一个简单示例：

```bash
python static_pose_optimizer_ba.py
```

该示例会创建 5 个随机帧、执行优化并打印最终位姿与平均误差。

## 模型与数据准备

- `checkpoint/best.pt`：YOLOv8 权重，如果不在仓库中请自行下载并放到该目录。
- `DATA_DIR` 下应包含 `metadata.json` 和对应图像路径。
- `metadata.json` 格式需包含 `records` 列表，记录 `frame_id`、`image_path`、`pose_matrix_4x4`、`camera_timestamp_ns` 等字段。

## 运行建议

- 先确认 `DATA_DIR` 对应的数据集路径、`checkpoint/best.pt` 是否存在。
- 若想快速调试，可将 `MAX_FRAMES` 设为小于 20 的正整数。
- 若想验证优化效果，可对比输出的 `pnp_results.csv` 与 `optimize_results.csv`。

## 其他说明

- `chargeport_pose_estimator_ba.py` 中的 YOLO ROI 检测结果用于进一步椭圆提取和匹配，并不会直接输出最终位姿。
- 优化结果的坐标系转换依赖 `robot_pose @ eMc @ cMo`，其中 `robot_pose` 为基座到末端执行器的位姿，`eMc` 为执行器到相机外参，`cMo` 为相机到物体位姿。
- 目前程序默认使用 `SOLVEPNP_IPPE` 作为初始求解方法，并在 `two_round_pnp` 中使用自适应阈值处理异常点。
