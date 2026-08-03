# LIMO Person Following System

基于 LIMO 机器人平台的**行人自动追踪系统**。检测→跟踪→控制三步流水线，在 Jetson Nano 上用 MobileNet-SSD 做实时行人检测，结合深度相机做距离估计，通过解耦 P 控制器输出 `/cmd_vel` 速度指令。

## 硬件要求

- LIMO 机器人（高配版，自带 Jetson Nano）
- Intel RealSense D435 或奥比中光 DaBai 深度相机（已接线）
- 建议使用四轮差速模式（物理开关：插销插入，车灯黄色）

## 软件依赖

- Ubuntu 18.04
- ROS Melodic
- Python 3.6+
- OpenCV ≥ 4.x (with DNN module, CUDA backend optional)
- numpy

LIMO 自带的系统已预装以上依赖。

## 快速开始

### 1. 下载检测模型

```bash
cd limo_person_following
bash scripts/download_model.sh
```

### 2. 启动 LIMO 底盘

```bash
roslaunch limo_base limo_base.launch
```

### 3. 启动深度相机

```bash
# RealSense D435
roslaunch realsense2_camera rs_camera.launch align_depth:=true

# 或奥比中光 DaBai
roslaunch astra_camera dabai_u3.launch
```

### 4. 启动行人追踪

```bash
roslaunch limo_person_following person_following.launch
```

带 RViz 可视化：
```bash
roslaunch limo_person_following person_following.launch use_rviz:=true
```

### 5. 停止

在启动追踪的终端中按 `Ctrl+C`，小车会立即停止。

## 架构

```
/camera/color/image_raw ──→ [person_detector] ──→ /person_detections
/camera/aligned_depth_to_color/image_raw ──┘              │
                                                 ┌────────┘
                                                 ▼
                                     [person_tracker] ──→ /target_person
                                                                  │
                                                                  ▼
                                                  [person_controller] ──→ /cmd_vel
```

### 三个节点

| 节点 | 功能 | 输入 | 输出 |
|---|---|---|---|
| `person_detector` | MobileNet-SSD 行人检测 + 深度采样 + 2D→3D 投影 | RGB图像、深度图像、相机内参 | `/person_detections` |
| `person_tracker` | IoU 状态机选择并维持一个跟踪目标 | `/person_detections` | `/target_person`, `/tracking_status` |
| `person_controller` | 解耦 P 控制器 + 安全区 + 看门狗 | `/target_person` | `/cmd_vel` |

## 参数配置

所有可调参数集中在 `config/person_following.yaml`：

### 主要控制参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `controller/target_distance` | 1.2 m | 目标跟随距离 |
| `controller/min_distance` | 0.6 m | 太近，小车后退 |
| `controller/max_distance` | 3.5 m | 太远/丢失，停车 |
| `controller/kp_linear` | 0.6 | 线速度 P 增益 |
| `controller/kp_angular` | 1.2 | 角速度 P 增益 |
| `controller/max_linear_speed` | 0.5 m/s | 最大前进速度 |
| `controller/max_angular_speed` | 1.0 rad/s | 最大转向速度 |
| `controller/max_linear_accel` | 0.3 m/s² | 线加速度限幅 |

### 检测参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `detector/confidence_threshold` | 0.5 | 行人检测最低置信度 |
| `detector/detection_rate` | 8.0 Hz | 检测频率 |
| `detector/use_gpu` | true | 使用 OpenCV CUDA 后端 |

### 跟踪参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `tracker/iou_match_threshold` | 0.3 | 身份匹配最低 IoU |
| `tracker/max_disappeared_frames` | 15 | 丢失前容忍帧数（约 1.9s @8Hz） |
| `tracker/reacquire_timeout` | 3.0 s | 重新识别超时 |

## 调试

### 单独测试检测器

```bash
roslaunch limo_person_following test_detector.launch
```

这会启动检测器 + RViz + image_view，可以看到标注了检测框和距离的图像。

### 查看话题

```bash
# 行人检测结果
rostopic echo /person_detections

# 当前跟踪目标
rostopic echo /target_person

# 跟踪状态
rostopic echo /tracking_status

# 速度指令
rostopic echo /cmd_vel

# 检测帧率
rostopic hz /person_detections
```

### 运行测试

```bash
# 追踪器单元测试（无需 ROS）
python3 tests/test_tracker.py

# 控制器单元测试（无需 ROS）
python3 tests/test_controller.py
```

## 安全机制

1. **紧急后退**：距离 < 0.6m 时后退
2. **超距停车**：距离 > 3.5m 时停车
3. **通信看门狗**：1 秒无目标消息即停车
4. **检测覆写**：任何检测到的行人在安全区域内，立即停车（不管是否是跟踪目标）
5. **加速度限幅**：防止急启急停

## 目录结构

```
limo_person_following/
├── CMakeLists.txt
├── package.xml
├── setup.py
├── launch/
│   ├── person_following.launch
│   └── test_detector.launch
├── config/
│   └── person_following.yaml
├── models/                          # 运行 download_model.sh 后生成
│   ├── MobileNetSSD_deploy.prototxt
│   └── MobileNetSSD_deploy.caffemodel
├── msg/
│   ├── PersonDetection.msg
│   ├── PersonDetections.msg
│   └── TargetPerson.msg
├── src/person_following/
│   ├── detector.py                  # 检测逻辑（可复用）
│   ├── detector_node.py             # 检测 ROS 节点
│   ├── tracker.py                   # 状态机跟踪逻辑（纯 Python）
│   ├── tracker_node.py              # 跟踪 ROS 节点
│   ├── controller_node.py           # 控制器 ROS 节点
│   └── utils.py                     # 共享工具
├── scripts/
│   ├── download_model.sh
│   └── wait_for_camera.sh
└── tests/
    ├── test_detector.py
    ├── test_tracker.py
    └── test_controller.py
```

## License

MIT
