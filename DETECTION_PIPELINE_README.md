# Detection Pipeline Visualization Tool

## 概述

`test_detection_pipeline.py` 是一个用于测试和可视化 socket pose 检测算法各个步骤的工具。该脚本会自动处理数据集中的所有图像，并生成详细的可视化结果，帮助开发者分析算法在每个处理阶段的表现。

## 功能特性

- **全自动处理**：无需手动干预，自动处理所有数据集图像
- **7步详细可视化**：从原始图像到最终分类结果的完整流程
- **算法调试**：帮助识别算法瓶颈和优化点
- **批量分析**：一次性分析整个数据集的表现

## 环境要求

### Python 版本
- Python 3.10.6 或更高版本

### 依赖包
```bash
pip install opencv-contrib-python numpy scipy ultralytics torch torchvision
```

### 数据集结构
```
dataset/
└── savedata4/
    ├── 2026-04-22_15_20_17_519624670540908.png
    ├── 2026-04-22_15_20_17_519624670540908.npy
    ├── 2026-04-22_15_20_19_519626077220539.png
    └── ...
```

### 模型文件
```
checkpoint/
└── best.pt  # YOLO 模型权重文件
```

## 使用方法

### 基本运行
```bash
# 使用系统 Python
python test_detection_pipeline.py

# 或使用虚拟环境 Python
.\.venv\Scripts\python.exe test_detection_pipeline.py
```

### 运行输出
```
Initializing YOLO model...
Initializing EdgeDrawing...
Initializing SocketPoseEstimator...
Found 35 PNG files to process

Processing frame 0: 2026-04-22_15_20_17_519624670540908.png
0: 384x640 1 port, 43.3ms
Speed: 1.5ms preprocess, 43.3ms inference, 0.9ms postprocess per image at shape (1, 3, 384, 640)
Saved visualization for frame 0

Processing frame 1: 2026-04-22_15_20_19_519626077220539.png
...
```

## 输出结果

### 目录结构
```
debug_pipeline/
├── frame_0000/
│   ├── 01_original.png
│   ├── 02_yolo_roi.png
│   ├── 03_roi_region.png
│   ├── 04_initial_ellipses.png
│   ├── 05_used_discarded_ellipses.png
│   ├── 06_cleaned_candidates.png
│   └── 07_classified_candidates.png
├── frame_0001/
└── ...
```

### 可视化步骤详解

#### 1. 01_original.png - 原始图像
- 显示未经处理的原始输入图像
- 用于对比后续处理效果

#### 2. 02_yolo_roi.png - YOLO ROI 检测
- 显示 YOLO 模型检测到的插座区域
- 绿色矩形框标出检测到的 ROI
- 如果未检测到 ROI，该帧将被跳过

#### 3. 03_roi_region.png - ROI 区域裁剪
- 显示从原始图像中裁剪出的 ROI 区域
- 这是后续椭圆检测的输入图像

#### 4. 04_initial_ellipses.png - 初始椭圆检测
- 显示 EdgeDrawing 算法检测到的所有椭圆
- 蓝色轮廓线标出检测到的椭圆/圆形
- 数字标出椭圆的索引号

#### 5. 05_used_discarded_ellipses.png - 使用 vs 抛弃椭圆
- 显示 clean_and_classify 步骤中椭圆的筛选结果
- **绿色椭圆**：被使用的椭圆（通过长宽比和尺寸过滤）
- **红色椭圆**：被抛弃的椭圆（未通过过滤）
- **灰色椭圆**：所有椭圆的背景参考
- 右上角显示统计信息：`Used: X | Discarded: Y`

#### 6. 06_cleaned_candidates.png - 清洁后的候选点
- 显示经过聚类合并和尺寸过滤后的候选点
- 彩色圆圈标出候选点位置
- 显示候选点的尺寸信息

#### 7. 07_classified_candidates.png - 分类后的候选点
- 显示经过 gap_method_threshold 分类的候选点
- **绿色候选点**：大尺寸类型 (L)
- **红色候选点**：小尺寸类型 (S)
- 用于模板匹配的最终候选点

## 算法参数

### EdgeDrawing 参数
```python
EdgeDetectionOperator = 1
MinPathLength = 45
PFmode = 0
NFAValidation = True
GradientThresholdValue = 30
```

### 椭圆过滤条件
```python
# 长宽比过滤
0.9 < e[2] / e[3] < 1.1

# 尺寸过滤
e[2] + e[3] < 80

# 最终尺寸过滤
10 < m['size'] < 100
```

### 聚类参数
```python
dist_thresh = 10  # 聚类距离阈值
```

## 故障排除

### 常见问题

#### 1. ModuleNotFoundError
```
ModuleNotFoundError: No module named 'ultralytics'
```
**解决方案**：
```bash
pip install ultralytics
```

#### 2. 模型文件未找到
```
FileNotFoundError: checkpoint/best.pt
```
**解决方案**：
- 确保模型文件存在于 `checkpoint/best.pt`
- 检查文件路径是否正确

#### 3. 数据集未找到
```
FileNotFoundError: dataset/savedata4/
```
**解决方案**：
- 确保数据集目录存在
- 检查目录结构是否正确

#### 4. 没有检测到 ROI
如果某些帧显示 "No ROI detected, skipping ellipse detection"，说明：
- YOLO 模型未检测到插座
- 可以调整 YOLO 的置信度阈值

### 性能优化

#### 减少处理时间
- 调整 `conf` 参数降低 YOLO 检测阈值
- 减少数据集大小进行测试

#### 提高检测准确性
- 调整 EdgeDrawing 参数
- 修改椭圆过滤条件
- 优化聚类距离阈值

## 示例分析

### 正常处理流程
1. YOLO 成功检测到插座 ROI
2. EdgeDrawing 检测到多个椭圆
3. 大部分椭圆通过过滤（绿色）
4. 少量椭圆被抛弃（红色）
5. 聚类后得到合适数量的候选点
6. 分类出大尺寸和小尺寸候选点

### 问题识别
- **大量红色椭圆**：过滤条件过于严格
- **候选点过少**：过滤条件过于宽松
- **分类不均**：gap threshold 需要调整

## 扩展使用

### 自定义参数
可以修改脚本中的参数来自定义处理：
```python
DATA_DIR = "your_dataset_path"
DEBUG_DIR = "your_output_path"
```

### 集成到其他脚本
可以将可视化函数导入到其他脚本中使用：
```python
from test_detection_pipeline import (
    draw_used_ellipses,
    draw_cleaned_candidates,
    # ... 其他函数
)
```

## 贡献

如需改进算法或添加新功能，请：
1. 修改 `test_detection_pipeline.py`
2. 测试新功能
3. 更新此文档
4. 提交 Pull Request

## 版本历史

- **v1.0** (2026-04-26): 初始版本，支持7步可视化流程
- **v0.5** (2026-04-26): 添加使用vs抛弃椭圆可视化
- **v0.1** (2026-04-26): 基础可视化框架</content>
<parameter name="filePath">d:\WJH\Documents\work\socket-pose-estimator\DETECTION_PIPELINE_README.md