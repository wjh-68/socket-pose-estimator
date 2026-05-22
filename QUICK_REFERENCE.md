# 快速参考 - 在线获取功能整合

## 🎯 核心问题解决

**问题**：需要统一管理离线（文件）和在线（硬件）两种数据源模式

**解决方案**：
```python
# 新增 SensorDataManager 类
# 统一接口处理两种数据源
# 自动线程安全管理
```

## 📊 系统架构

```
chargeport_pose_estimator_ba.py
│
├─ Config (行 15-35)
│  └─ DATA_SOURCE_MODE = "offline"|"online"
│
├─ Utilities (行 204-213)
│  └─ get_robot_pose_from_rpc()
│
├─ SensorDataManager (214-420)
│  ├─ 离线模式：文件读取
│  ├─ 在线模式：实时采集
│  ├─ 线程安全：frame_lock, pose_lock
│  └─ Interface: get_next_frame_pose()
│
└─ ChargeportPoseEstimator (423+)
   ├─ __init__(): 创建 data_source
   ├─ run(): 自动选择执行方式
   ├─ _run_online(): 在线处理
   └─ _run_offline(): 离线处理
```

## 🚀 快速开始

### 离线模式
```python
from chargeport_pose_estimator_ba import ChargeportPoseEstimator

# 方法 1：使用全局配置
estimator = ChargeportPoseEstimator()  # 使用 DATA_SOURCE_MODE='offline'
estimator.run()

# 方法 2：覆盖配置参数
estimator = ChargeportPoseEstimator(
    data_source_mode='offline',
    data_dir='dataset/my_data',
    result_dir='result/my_result'
)
estimator.run()
```

### 在线模式
```python
estimator = ChargeportPoseEstimator(
    data_source_mode='online',
    robot_ip='192.168.1.20',
    robot_port=30004
)
estimator.run()  # 按 'q' 退出
```

### 批量处理
```python
for dataset in ['0515', '0516', '0517']:
    estimator = ChargeportPoseEstimator(
        data_source_mode='offline',
        data_dir=f'dataset/{dataset}',
        result_dir=f'result/{dataset}'
    )
    estimator.run()
```

## 📝 关键方法

### SensorDataManager

| 方法 | 说明 | 返回值 |
|------|------|--------|
| `__init__(mode, **kwargs)` | 初始化 | - |
| `get_next_frame_pose()` | 获取数据 | `(img, pose, info, success)` |
| `start_online()` | 启动线程 | - |
| `stop_online()` | 停止线程 | - |
| `reset_offline()` | 重置计数 | - |

### ChargeportPoseEstimator

| 方法 | 说明 |
|------|------|
| `__init__()` | 初始化 |
| `run()` | 执行主流程 |
| `_run_offline()` | 离线处理 |
| `_run_online()` | 在线处理 |

## ⚙️ 配置参数完整列表

### 全局配置指导
```python
# 数据源选择
DATA_SOURCE_MODE = "offline"  # 或 "online"

# 通用参数
SLIDING_WINDOW_SIZE = 8
MAX_FRAMES = -1

# 离线模式
DATA_DIR = "dataset/save_data3/20260511_120244"
RESULT_DIR = "result/0515"

# 在线模式
ROBOT_IP = "192.168.1.20"
ROBOT_PORT = 30004
CAMERA_ID = 0
CAMERA_WIDTH = 2560
CAMERA_HEIGHT = 1440
CAMERA_BRIGHTNESS = 128
FRAME_POSE_SYNC_TOLERANCE_NS = 20_000_000

# 处理参数
USE_ADAPTIVE_THRESHOLD = True
FIXED_ERROR_THRESHOLD = 1.0
ADAPTIVE_MULTIPLIER = 2.0
```

## 🔄 数据流对比

### 离线模式
```
[文件系统] → SensorDataManager → [缓冲区] → 处理 → [结果保存]
  (同步)        (逐帧读取)        (单帧)
```

### 在线模式
```
[相机线程]┐
          ├→ SensorDataManager → [同步缓冲] → 处理 → [结果保存]
[机械臂线程]┘   (20ms容差)       (同步帧对)
(background)
```

## 🔒 线程安全机制（在线模式）

```python
# 两个独立的锁，避免阻塞
frame_lock = threading.Lock()    # 保护: latest_frame, latest_frame_ts
pose_lock = threading.Lock()     # 保护: latest_pose, latest_pose_ts

# 后台线程独立运行
_read_camera_loop()              # 相机采集
_read_robot_loop()               # 机械臂获取

# 主线程定期拉取同步数据
get_next_frame_pose()            # 同时获取 frame 和 pose
                                 # 自动检查时间同步
```

## 📊 返回值说明

### 离线模式
```python
frame, pose, timestamp_ns, success = data_source.get_next_frame_pose()
# frame: numpy.ndarray (图像)
# pose: numpy.ndarray (4x4, 位姿) 或 None
# timestamp_ns: int (时间戳)
# success: bool (是否成功加载)
```

### 在线模式
```python
frame, pose, time_diff_ns, synchronized = data_source.get_next_frame_pose()
# frame: numpy.ndarray (最新相机帧)
# pose: numpy.ndarray (4x4, 对应的机械臂位姿)
# time_diff_ns: int (frame 和 pose 的时间差)
# synchronized: bool (是否在容差范围内同步)
```

## 🎓 学习路径

1. **入门**：运行离线模式，理解基本流程
2. **进阶**：修改 `MAX_FRAMES` 参数，测试部分处理
3. **高级**：切换到在线模式，观察线程行为
4. **定制**：继承 `SensorDataManager` 添加自定义数据源

## 🐛 常见问题

### Q1: 如何切换离线和在线模式？
```python
# 方案 A：修改全局配置
DATA_SOURCE_MODE = "online"

# 方案 B：初始化时覆盖
estimator = ChargeportPoseEstimator(data_source_mode='online')
```

### Q2: 在线模式时间同步失败怎么办？
```python
# 增大容差（允许更大的时间差）
estimator = ChargeportPoseEstimator(
    data_source_mode='online',
    sync_tolerance_ns=50_000_000  # 增加到 50ms
)
```

### Q3: 如何只处理有限帧数进行测试？
```python
estimator = ChargeportPoseEstimator(
    max_frames=10  # 仅处理前10帧
)
estimator.run()
```

### Q4: 如何在线保存实时处理结果？
```python
estimator = ChargeportPoseEstimator(data_source_mode='online')
estimator.run()  # 自动保存到 result_dir
```

## 📈 性能指标

| 指标 | 离线模式 | 在线模式 |
|------|---------|---------|
| CPU | 低 (IO 受限) | 中等 (30-50%) |
| 内存 | 稳定 | 相对稳定 |
| 延迟 | N/A | ~20-100ms |
| 可扩展性 | ⭐⭐⭐ | ⭐⭐ |

## 📦 依赖库

**必需**：
- numpy
- opencv-python
- ultralytics
- scipy

**可选**（仅在线模式）：
- pyaubo_sdk (机械臂)

**可选**（可视化）：
- pandas
- matplotlib

## 🔧 调试建议

### 离线模式调试
```python
estimator = ChargeportPoseEstimator(
    data_source_mode='offline',
    max_frames=1  # 先处理1帧看是否有错误
)
try:
    estimator.run()
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
```

### 在线模式调试
```python
estimator = ChargeportPoseEstimator(data_source_mode='online')

# 检查连接
try:
    estimator.data_source.start_online()
    print("✓ 硬件连接成功")
except Exception as e:
    print(f"✗ 连接失败: {e}")

# 检查数据同步
for i in range(10):
    img, pose, diff, ok = estimator.data_source.get_next_frame_pose()
    print(f"Frame {i}: sync_diff={diff/1e6:.1f}ms, OK={ok}")

estimator.data_source.stop_online()
```

## 🎯 实用模板

### 模板 1：快速单次测试
```python
from chargeport_pose_estimator_ba import ChargeportPoseEstimator

estimator = ChargeportPoseEstimator(max_frames=5)
estimator.run()
```

### 模板 2：带监控的离线处理
```python
estimator = ChargeportPoseEstimator(
    data_source_mode='offline',
    data_dir='dataset/test'
)
estimator.run()
print(f"Processed {len(estimator.frame_records)} frames")
```

### 模板 3：在线实时处理
```python
estimator = ChargeportPoseEstimator(data_source_mode='online')
try:
    estimator.run()  # Ctrl+C 或按 'q' 停止
except KeyboardInterrupt:
    print("Session ended")
finally:
    print(f"Processed {len(estimator.frame_records)} frames")
```

## 📚 注意事项

1. **线程安全**：在线模式使用了 `threading.Lock()`，无需用户额外管理
2. **资源清理**：在线模式会自动清理线程和资源，但建议使用 try-finally
3. **帧率**：在线模式受 PnP 求解速度限制（通常 5-10 FPS）
4. **存储空间**：离线模式的结果会保存JSON和图表，需要足够的磁盘空间
5. **系统角度**：在线模式会占用额外的 CPU 核心（2个线程），可能影响其他应用

---

**更多信息**：查看 `USAGE_EXAMPLE.md` 和 `INTEGRATION_SUMMARY.md`
