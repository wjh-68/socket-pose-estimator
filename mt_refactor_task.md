# 重构 Charging Pose Estimator（多线程 Pipeline 架构）

该项目实现：
**实时运动相机对静态目标的位姿估计。**

现需彻底重构项目，采用：

* Producer-Consumer 多线程模型
* Queue 线程间通信
* Pipeline 数据流架构
* 模块化设计
* 可扩展离线/在线数据源

目标：

* 提升系统工程化程度
* 降低模块耦合
* 提升实时性与可维护性
* 为后续滑窗优化、多传感器扩展、在线部署做准备

---

# 一、总体架构

系统采用：

```text
DataReader
    ↓
Detector
    ↓
Matcher
    ↓
Optimizer
    ↓
Visualizer / Saver
```

线程之间：

* 使用 `Queue`
* 禁止使用复杂全局共享变量
* 禁止模块间直接调用内部状态

系统采用：

# 数据流（Data Flow）驱动

而不是：

# 共享状态（Shared State）驱动

---

# 二、核心数据结构（最重要）

所有线程之间：

# 统一传递 `FramePacket`

禁止使用 tuple/list 临时拼接数据。

---

## FramePacket

建议：

```python
@dataclass
class FramePacket:

    # -------------------------
    # basic info
    # -------------------------
    frame_id: int
    timestamp: float

    # -------------------------
    # raw sensor data
    # -------------------------
    image: np.ndarray
    robot_pose: np.ndarray

    # -------------------------
    # detector outputs
    # -------------------------
    roi = None
    keypoints = None

    # -------------------------
    # matcher outputs
    # -------------------------
    refined_pts2d = None
    valid_mask = None

    # -------------------------
    # optimizer outputs
    # -------------------------
    optimized_pose = None

    # -------------------------
    # visualization
    # -------------------------
    debug_image = None

    # -------------------------
    # profiling
    # -------------------------
    timing = {}

    # -------------------------
    # status
    # -------------------------
    valid = True
```

说明：

* 所有线程均输入/输出 `FramePacket`
* 各模块仅补充和修改自身负责字段
* 禁止重新构造新的数据结构替代 packet

---

# 三、线程与 Queue 架构

推荐：

```text
                ┌────────────────┐
                │ Data Reader    │
                └──────┬─────────┘
                       ↓
                 raw_queue

                ┌────────────────┐
                │ Detector       │
                └──────┬─────────┘
                       ↓
              detection_queue

                ┌────────────────┐
                │ Matcher        │
                └──────┬─────────┘
                       ↓
                match_queue

                ┌────────────────┐
                │ Optimizer      │
                │ sliding window │
                └──────┬─────────┘
                       ↓
                result_queue

                ┌────────────────┐
                │ Visualizer     │
                │ Saver          │
                └────────────────┘
```

---

# 四、Queue 使用要求

所有 Queue：

```python
Queue(maxsize=N)
```

禁止：

```python
Queue()
```

避免：

* queue 无限增长
* 内存泄漏
* 实时系统延迟爆炸

---

## 推荐 Queue 大小

```python
raw_queue         = Queue(maxsize=10)
detection_queue   = Queue(maxsize=10)
match_queue       = Queue(maxsize=10)
result_queue      = Queue(maxsize=10)
```

---

# 五、实时模式与离线模式

系统需支持：

---

## 1. 实时模式（在线）

目标：

* 低延迟
* 允许丢帧

当 queue 满时：

允许丢弃旧帧：

```python
if queue.full():
    queue.get_nowait()
```

优先保留最新数据。

---

## 2. 离线模式（数据集）

目标：

* 不丢帧
* 保证完整处理

queue 满时：

允许阻塞等待。

---

# 六、DataReader 线程

实现统一数据管理模块：

```python
DataManager
```

支持：

* 离线数据集
* 在线传感器

并输出统一 `FramePacket`

---

# 七、离线数据集读取

以：

```text
dataset/0515
```

为例。

包含：

* 图片
* robot pose `.npy`

---

## 要求

### 1. 多图片格式支持

支持：

* png
* jpg
* jpeg

等。

---

### 2. 时间戳检查

检查：

```text
image timestamp
robot pose timestamp
```

是否一致。

初版可先适配：

```text
dataset/0515
```

格式。

---

### 3. 数据集接口抽象

需设计：

```python
BaseDatasetLoader
```

支持后续扩展不同格式。

例如：

```python
class BaseDatasetLoader:
    def load_next(self):
        pass
```

---

### 4. 离线模式统一 timestamp

禁止仅按文件顺序读取。

必须统一：

```python
packet.timestamp
```

保证与在线模式逻辑一致。

---

# 八、在线传感器数据读取

支持：

* Camera
* Robot pose

参考：

```text
chargeport_pose_estimator_ba.py
```

中的接口。

---

## 要求

### 1. 读取频率可配置

例如：

```yaml
camera_fps: 30
robot_pose_fps: 100
```

---

### 2. 时间同步

必须采用：

# timestamp-based synchronization

禁止：

```python
camera.read()
robot.read()
```

后默认认为同步。

---

## 推荐结构

```text
CameraThread
RobotPoseThread
       ↓
  SyncManager
       ↓
 FramePacket
```

---

## 同步方式

推荐：

### nearest timestamp matching

后续可扩展：

### interpolation

---

# 九、Detector 线程

负责：

* TRT 推理
* ROI 检测
* keypoints 提取

参考：

* `trt_pose_inf.py`
* `getInferResults`

---

## 输入

```python
packet.image
```

---

## 输出

写入：

```python
packet.roi
packet.keypoints
```

然后：

```python
output_queue.put(packet)
```

---

# 十、Matcher（圆心 refine）线程

负责：

* 基于 keypoints 和图像
* 精确化圆心位置

参考：

```text
_match_socket_mt()
```

---

## 输入

```python
packet.image
packet.keypoints
```

---

## 输出

```python
packet.refined_pts2d
packet.valid_mask
```

---

# 十一、Matcher 内部并行

Matcher 内部：

允许使用：

```python
ThreadPoolExecutor
```

并行处理多个点。

---

## 禁止

禁止：

```python
每个点创建一个 Thread
```

---

## 推荐

```python
ThreadPoolExecutor(max_workers=8)
```

---

# 十二、valid_mask（非常重要）

由于：

```text
pts2d
pts3d
```

可能不完全对应。

禁止：

* 删除 pts3d
* 修改 object_points 顺序

---

## 必须采用

```python
valid_mask
```

例如：

```python
valid_mask = [1,1,0,1,0]
```

表示：

该 frame 中：

* 第 3 个点缺失
* 第 5 个点缺失

优化器内部：

基于 mask 跳过无效点。

---

# 十三、Optimizer 线程

参考：

* `chargeport_pose_estimator_ba.py`
* `static_pose_optimizer_ba.py`

---

# 十四、Optimizer 设计要求

优化线程：

# 必须为状态型线程（Stateful Thread）

而不是：

```text
输入一帧
输出一帧
```

---

## Optimizer 内部维护

```python
self.sliding_window_packets
```

---

## 输入

```python
packet.refined_pts2d
packet.robot_pose
packet.valid_mask
```

---

## 输出

```python
packet.optimized_pose
```

---

# 十五、修改 static_pose_optimizer_ba.py

---

## object_points

必须：

* 初始化时输入
* 后续每帧不再重复输入

---

## pts2d 缺失点处理

优化器：

必须基于：

```python
valid_mask
```

处理。

禁止：

动态修改 object_points 拓扑。

---

## 参数接口要求

区分：

---

### 高频配置参数

例如：

```python
window_size
reprojection_sigma
robust_kernel
```

---

### 保持默认参数

避免：

```python
初始化接口过长
```

---

# 十六、Visualizer / Saver 线程

负责：

* 结果可视化
* 图片保存
* 数据保存
* profiling统计展示

参考：

```text
chargeport_pose_estimator_ba.py
```

原逻辑。

---

# 十七、必须功能

---

## 1. 结果图像可视化

例如：

* ROI
* keypoints
* refined circle
* reprojection
* optimization result

---

## 2. 图片保存

必须支持：

```yaml
save_raw_image: true
save_debug_image: true
save_result_image: true
```

---

## 3. 数据保存

默认开启。

保存：

* timestamp
* optimized pose
* reprojection error
* timing
* valid points count

去除冗余数据。

---

# 十八、日志系统

必须使用：

```python
logging
```

禁止：

```python
print()
```

---

## 日志等级

支持：

```python
DEBUG
INFO
WARNING
ERROR
```

---

# 十九、程序性能分析（Profiling）

必须支持：

```yaml
enable_profiling: true
```

---

## 推荐方式

每模块记录：

```python
packet.timing["detector"]
packet.timing["matcher"]
packet.timing["optimizer"]
```

单位：

```python
ms
```

---

## 最终统计

Visualizer/Saver：

统一输出：

* 平均耗时
* FPS
* queue delay

---

# 二十、程序退出机制（非常重要）

禁止：

```python
while True:
```

---

## 必须使用

```python
threading.Event()
```

例如：

```python
stop_event.is_set()
```

实现：

* 优雅退出
* 安全 shutdown

---

# 二十一、异常处理

任意线程异常：

必须：

* 记录 traceback
* 设置 stop_event
* 通知主线程退出

禁止：

线程 silently crash。

---

# 二十二、配置管理

建议：

```yaml
config.yaml
```

统一管理：

* dataset path
* queue size
* fps
* save options
* profiling
* visualization
* optimizer params

---

# 二十三、开发要求

---

## 1. 在新文件上开发

禁止直接修改旧版本核心文件。

---

## 2. Conda 环境

使用：

```text
yolov8-trt
```

测试。

---

## 3. 测试数据

使用：

```text
dataset/0515
```

完成：

* 离线模式测试
* pipeline 完整验证

---

# 二十四、推荐目录结构

```text
project/

├── main.py

├── config/
│   └── config.yaml

├── core/
│   ├── packet.py
│   ├── queues.py
│   ├── logger.py
│   └── profiler.py

├── data/
│   ├── base_loader.py
│   ├── offline_loader.py
│   ├── online_loader.py
│   └── sync_manager.py

├── detector/
│   └── detector.py

├── matcher/
│   └── matcher.py

├── optimizer/
│   ├── optimizer_thread.py
│   └── static_pose_optimizer_ba.py

├── visualization/
│   └── visualizer.py

├── utils/
│   └── ...

└── output/
```

---

# 二十五、最终目标

最终系统应具备：

* 清晰线程边界
* 模块解耦
* 可扩展 pipeline
* 实时/离线双模式
* 滑窗优化能力
* 工程级日志/profiling
* 可维护性
* 后续 ROS2 / 在线部署扩展能力

这是本次重构的核心目标。
