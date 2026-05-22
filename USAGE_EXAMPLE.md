# Chargeport Pose Estimator - 使用示例

## 概述

`chargeport_pose_estimator_ba.py` 现已支持两种数据获取模式：
- **离线模式（offline）**：从文件读取图像和机械臂位姿
- **在线模式（online）**：实时从相机和机械臂获取数据

## 系统架构

### SensorDataManager 类

统一管理数据源的类，支持两种模式的自动切换。

```
SensorDataManager
├── 离线模式 (offline)
│   ├── 从文件系统加载图像
│   ├── 加载对应的位姿数据 (.npy)
│   └── 按顺序迭代读取
│
└── 在线模式 (online)
    ├── 相机线程（read_camera_loop）
    ├── 机械臂线程（read_robot_loop）
    └── 同步获取（get_synced_frame_pose）
```

### ChargeportPoseEstimator 类

主处理器类，现已集成 SensorDataManager。

```
ChargeportPoseEstimator
├── 初始化 data_source_mode
├── 创建 SensorDataManager
├── run() - 自动选择在线或离线处理
├── _run_online() - 在线处理流程
└── _run_offline() - 离线处理流程
```

## 使用示例

### 1. 离线模式（从文件读取）

```python
from chargeport_pose_estimator_ba import ChargeportPoseEstimator

# 创建估计器（离线模式）
estimator = ChargeportPoseEstimator(
    data_source_mode='offline',
    data_dir='dataset/0515',
    result_dir='result/0515',
    max_frames=100  # 处理前100帧
)

# 运行
estimator.run()
```

### 2. 在线模式（实时硬件）

```python
from chargeport_pose_estimator_ba import ChargeportPoseEstimator

# 创建估计器（在线模式）
estimator = ChargeportPoseEstimator(
    data_source_mode='online',
    result_dir='result/online_session',
    robot_ip='192.168.1.20',
    robot_port=30004,
    camera_id=0,
    camera_width=2560,
    camera_height=1440,
    camera_brightness=128,
    sync_tolerance_ns=20_000_000,  # 20ms
    max_frames=-1  # 无限处理，直到用户中断
)

# 运行（按 'q' 退出）
try:
    estimator.run()
except KeyboardInterrupt:
    print("Session stopped")
```

### 3. 配置参数

#### 全局配置
```python
# 数据源模式
DATA_SOURCE_MODE = "offline"  # 或 "online"

# 离线模式参数
DATA_DIR = "dataset/save_data3/20260511_120244"
RESULT_DIR = "result/0515"

# 在线模式参数
ROBOT_IP = "192.168.1.20"
ROBOT_PORT = 30004
CAMERA_ID = 0
CAMERA_WIDTH = 2560
CAMERA_HEIGHT = 1440
CAMERA_BRIGHTNESS = 128
FRAME_POSE_SYNC_TOLERANCE_NS = 20_000_000  # 20ms
```

#### 处理参数
```python
SLIDING_WINDOW_SIZE = 8
MAX_FRAMES = -1  # -1 为无限，正数为限制

# PnP 阈值配置
USE_ADAPTIVE_THRESHOLD = True
FIXED_ERROR_THRESHOLD = 1.0
ADAPTIVE_MULTIPLIER = 2.0
```

## 数据流

### 离线模式数据流
```
文件系统
  ├── 图像文件 (.jpg)
  └── 位姿文件 (.npy)
        ↓
SensorDataManager (offline mode)
  └── get_next_frame_pose()
        ↓
ChargeportPoseEstimator._process_one_frame()
  ├── 检测套字
  ├── 配点匹配
  ├── PnP 求解
  ├── 优化器处理
  └── 保存结果
```

### 在线模式数据流
```
硬件设备
  ├── 相机（线程）
  │   └── read_camera_loop()
  └── 机械臂（线程）
      └── read_robot_loop()
          ↓
      线程安全队列（locks）
          ↓
  SensorDataManager (online mode)
    └── get_synced_frame_pose()
      （允许 ±20ms 时间差）
          ↓
ChargeportPoseEstimator._process_one_frame()
  ├── 检测套字
  ├── 配点匹配
  ├── PnP 求解
  ├── 优化器处理
  ├── 实时显示
  └── 保存结果
```

## 线程安全设计

在线模式使用独立的锁管理多线程安全：

- `frame_lock`: 保护最新帧数据
- `pose_lock`: 保护最新位姿数据

这允许相机和机械臂独立采集，避免相互阻塞。

## 同步机制

```python
# 在线模式同步容差设置
FRAME_POSE_SYNC_TOLERANCE_NS = 20_000_000  # 20 毫秒

# 如果时间差超过容差
if time_diff > tolerance:
    # 跳过此帧，等待更好的同步
    continue
```

## 输出结果

无论在线还是离线模式，结果保存结构相同：

```
RESULT_DIR/
├── frame_records.json      # 每帧的完整数据
├── pnp_records.json        # PnP 求解结果
├── optimize_records.json   # 优化器结果
├── frame_*.csv             # CSV 格式数据
├── pose_components_vs_frame.png
├── euler_angles_vs_frame.png
└── reprojection_error_vs_frame.png
```

## 常见使用场景

### 场景 1：离线验证和调试
```python
estimator = ChargeportPoseEstimator(
    data_source_mode='offline',
    data_dir='dataset/0515',
    result_dir='result/debug',
    max_frames=10  # 只处理前10帧用于快速验证
)
estimator.run()
```

### 场景 2：在线实时测试
```python
estimator = ChargeportPoseEstimator(
    data_source_mode='online',
    robot_ip='192.168.1.20',
    robot_port=30004,
)
estimator.run()  # 按 Ctrl+C 退出
```

### 场景 3：批量离线处理
```python
for dataset_name in ['0515', '0516', '0517']:
    estimator = ChargeportPoseEstimator(
        data_source_mode='offline',
        data_dir=f'dataset/{dataset_name}',
        result_dir=f'result/{dataset_name}',
    )
    estimator.run()
```

## 错误处理

### 在线模式

- **机械臂连接失败**：会捕获异常并打印错误信息
- **数据不同步**：自动跳过时间差超过阈值的帧
- **用户中断**：按 'q' 或 Ctrl+C 优雅退出

### 离线模式

- **文件不存在**：自动跳过缺失的文件
- **加载失败**：打印警告但继续处理

## 性能考虑

### 在线模式
- CPU 使用率：~30-50%（取决于图像处理速度）
- 内存使用：相对稳定（滑动窗口管理）
- 延迟：取决于 `SLIDING_WINDOW_SIZE` 和优化器性能

### 离线模式
- IO 受限：受磁盘读取速度影响
- 可以处理任意数量的帧
- 易于并行化处理多个数据集

## 扩展建议

1. **支持更多数据源**：扩展 SensorDataManager 以支持点云、深度图等
2. **云端存储**：添加 S3/OSS 支持用于远程数据抓取
3. **可视化改进**：集成实时 3D 可视化
4. **分布式处理**：支持多机协同处理
