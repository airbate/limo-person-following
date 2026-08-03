# LIMO Person Following System

基于 [松灵机器人 LIMO](https://github.com/agilexrobotics/limo-doc) 平台的行人自动追踪系统。摄像头看到你，小车就跟着你走。

---

## 目录

- [1. 项目简介](#1-项目简介)
- [2. 系统架构](#2-系统架构)
- [3. 环境要求](#3-环境要求)
- [4. 完整部署指南](#4-完整部署指南)
- [5. 使用说明](#5-使用说明)
- [6. 参数调优](#6-参数调优)
- [7. 调试与诊断](#7-调试与诊断)
- [8. 目录结构](#8-目录结构)
- [9. 常见问题](#9-常见问题)
- [10. License](#10-license)

---

## 1. 项目简介

### 1.1 这是什么

一个运行在 LIMO 机器人上的**自主行人跟随系统**。你走到哪，小车自动跟到哪，始终与你保持约 1.2 米的安全距离。

### 1.2 应用场景

| 场景 | 说明 |
|---|---|
| 科研教学 | ROS 开发入门、计算机视觉课程、机器人控制实验 |
| 服务机器人 | 酒店行李跟随、超市购物跟随、机场引导 |
| 仓储物流 | "跟我来"模式的拣货辅助 |
| 个人项目 | 智能行李箱、自动跟拍摄影 |

### 1.3 核心能力

- **实时行人检测**：用深度学习的 MobileNet-SSD 模型在 Jetson Nano 上跑 5-10 FPS 的行人检测
- **智能目标锁定**：自动锁定最近的行人，短暂遮挡后能重新找回
- **3D 空间定位**：结合深度相机（RealSense / 奥比中光），把 2D 检测框投影到 3D 空间坐标
- **平滑运动控制**：解耦 P 控制器 + 死区 + 加速度限幅，不会急启急停
- **多重安全保护**：紧急后退、超距停车、通信看门狗、检测覆写

### 1.4 性能参数

| 指标 | 数值 |
|---|---|
| 检测帧率 | 5-10 FPS (Jetson Nano CPU) |
| 控制频率 | 20 Hz |
| 最小跟随距离 | 0.6 米（再近就后退） |
| 目标跟随距离 | 1.2 米（可调） |
| 最大跟随距离 | 3.5 米（超距停车） |
| 遮挡容忍时间 | ~1.9 秒 |
| 内存占用 | ~200 MB |

---

## 2. 系统架构

### 2.1 数据流

```
┌──────────────────────────────────────────────────────────────────────┐
│                          LIMO 机器人                                  │
│                                                                      │
│  ┌──────────┐    ┌──────────┐                                       │
│  │ RGB相机   │    │ 深度相机  │                                       │
│  │ 30fps    │    │ 30fps    │                                       │
│  └────┬─────┘    └────┬─────┘                                       │
│       │   /camera/    │  /camera/aligned_depth_to_color/             │
│       │  color/image_raw  image_raw                                  │
│       │               │                                              │
│       └───────┬───────┘                                              │
│               ▼                                                      │
│       ┌───────────────┐                                              │
│       │ person_detector│  MobileNet-SSD 检测 + 深度采样 + 2D→3D     │
│       │   (8Hz 定频)   │                                              │
│       └───────┬───────┘                                              │
│               │ /person_detections                                   │
│               ▼                                                      │
│       ┌───────────────┐                                              │
│       │ person_tracker │  IoU 状态机：锁定/跟踪/丢失                 │
│       │   (按检测频率)  │                                              │
│       └───────┬───────┘                                              │
│               │ /target_person                                       │
│               ▼                                                      │
│       ┌───────────────┐                                              │
│       │person_controller│  P控制器 + 安全区 + 加速度限幅               │
│       │   (20Hz 定频)   │                                              │
│       └───────┬───────┘                                              │
│               │ /cmd_vel                                             │
│               ▼                                                      │
│       ┌───────────────┐                                              │
│       │  limo_base     │  底盘驱动 → 电机                             │
│       └───────────────┘                                              │
└──────────────────────────────────────────────────────────────────────┘
```

### 2.2 节点详解

#### 节点 1：person_detector — 行人检测 + 3D 定位

这是整个系统最核心的节点。

**检测流程：**

1. 用 `message_filters.ApproximateTimeSynchronizer` 把 RGB 和深度帧对齐（时间容差 0.1 秒）
2. 每 125ms（8 Hz）定时触发一次检测循环
3. RGB 图像缩放到 300×300，送入 MobileNet-SSD 网络
4. 前向推理，过滤出 class_id=15（person）且置信度 > 0.5 的检测框
5. 对每个检测框，取框中心 30% 区域的深度中值，滤除空洞和异常值
6. 用相机内参矩阵 K 把 2D 像素 + 深度投影到 3D 相机坐标系
7. 高度过滤：抛弃 y < -1.5m 或 y > 2.0m 的点（地板反光、天花板误检）
8. 发布 `PersonDetections` 数组

**为什么用 MobileNet-SSD：**

| 候选方案 | FPS (Jetson Nano) | 模型大小 | 优点 | 缺点 |
|---|---|---|---|---|
| MobileNet-SSD ✔ | 5-10 | 25 MB | 轻量、OpenCV DNN 原生支持 | 精度略低于 YOLO |
| YOLOv8n | 3-5 (ONNX) | 6 MB | 精度高 | 需要 TensorRT 加速，集成复杂 |
| HOG+SVM | 8-15 | 0 | 零依赖 | 只能检测站立行人，易漏检 |
| YOLOv3-tiny | 3-6 | 34 MB | LIMO 已预装 darknet_ros | 对 Jetson Nano 偏重 |

MobileNet-SSD 是性能和部署复杂度的最佳平衡点。如果后续需要更高精度，可升级为 YOLOv8n + TensorRT。

**为什么用中心区域中值采样取深度：**

- 深度相机在物体边缘会产生跳变（前景-背景混合）
- 单像素采样可能恰好命中深度空洞（值为 0）
- 取框中心 30% 区域的中值，既避免了边缘噪声，又不受个别空洞影响

#### 节点 2：person_tracker — 目标选择与跟踪

用一个轻量的 **三态状态机 + IoU 匹配** 来维持对同一个人的锁定。

**状态转换：**

```
                        ┌──────────┐
          有检测 → 选最近的人      │
                        │  ACQUIRING │
                        └─────┬────┘
                              │ 锁定成功
                              ▼
                        ┌──────────┐
              IoU匹配成功 │          │ IoU匹配失败 × N 帧
              ◄──────────│ TRACKING │──────────┐
              │          │          │          │
              │          └──────────┘          ▼
              │                          ┌──────────┐
              │        3秒内重新匹配成功  │          │
              └──────────────────────────│   LOST   │
                                         │          │
                                         └─────┬────┘
                                               │ 3秒超时
                                               ▼
                                         回到 ACQUIRING
                                         （重新选目标）
```

**为什么用 IoU 而不是 ReID：**

- ReID（行人重识别）需要额外的深度学习模型，Jetson Nano 跑不动
- IoU 匹配零计算开销，在室内场景下足够用
- 1.9 秒的遮挡容忍时间覆盖了大部分短暂遮挡场景（走过桌子/柱子/其他人）

#### 节点 3：person_controller — 运动控制

以 20Hz 定频运行的**解耦 P 控制器**。

**控制逻辑（精简版）：**

```
if distance < 0.6m  → 后退 -0.15 m/s（紧急）
if distance > 3.5m  → 停车（丢失）
if 无目标 > 1.0s    → 停车（看门狗）

linear_error  = distance - 1.2    ← 距离误差
angular_error = atan2(x, z)       ← 方位角误差

linear_cmd  = 0.6 × linear_error    （死区 ±0.15m）
angular_cmd = 1.2 × angular_error   （死区 ±0.08rad）

# 交叉限幅：拐弯时减速，近处精细转向
linear_cmd  ×= max(0.3, 1 - |angular_error|)
angular_cmd ×= min(1.0, distance / 1.2)

# 加速度限幅：防止急启急停
linear_cmd  = ramp(linear_cmd,  ±0.3 m/s² × dt)
angular_cmd = ramp(angular_cmd, ±0.8 rad/s² × dt)
```

**为什么不用 PID：**

| 项 | 作用 | 为什么不用 |
|---|---|---|
| P | 按误差比例输出 ✅ | 使用 |
| I | 消除稳态误差 | 人不是匀速运动的，积分项会持续累积，人停下后小车还会"冲过头" |
| D | 抑制振荡 | 深度相机有噪声，微分项会放大噪声导致抖动 |

在实际测试中，纯 P + 加速度限幅比 PID 更平稳。

### 2.3 ROS 话题一览

| 话题 | 类型 | 频率 | 方向 | 说明 |
|---|---|---|---|---|
| `/camera/color/image_raw` | Image | 30 Hz | 输入 | RGB 彩色图像 |
| `/camera/aligned_depth_to_color/image_raw` | Image | 30 Hz | 输入 | 对齐后的深度图（mm） |
| `/camera/color/camera_info` | CameraInfo | 10 Hz | 输入 | 相机内参 |
| `/person_detections` | PersonDetections | ~8 Hz | 中间 | 所有行人检测结果 |
| `/person_detections_image` | Image | ~8 Hz | 调试 | 标注了检测框的图像 |
| `/target_person` | TargetPerson | ~8 Hz | 中间 | 当前跟踪目标 |
| `/tracking_status` | String | 按需 | 调试 | ACQUIRING / TRACKING / LOST |
| `/target_marker` | Marker | ~8 Hz | 调试 | RViz 中的目标 3D 标记 |
| `/cmd_vel` | Twist | 20 Hz | 输出 | 速度指令 → limo_base |

### 2.4 自定义消息

**PersonDetection.msg**

```
float64 x y z            # 相机坐标系下 3D 位置（米）
float64 confidence       # 检测置信度 [0, 1]
float64 bbox_center_x    # 归一化 bbox 中心 [0, 1]
float64 bbox_center_y
float64 bbox_width
float64 bbox_height
```

**TargetPerson.msg**

```
std_msgs/Header header
geometry_msgs/Point position   # 3D 位置
float64 distance               # 水平距离 (m)
float64 angle                  # 水平偏移角 (rad)
bool is_tracking               # 是否正在跟踪
float64 confidence             # 当前匹配置信度
int32 tracking_id              # 目标 ID（每次重新锁定 +1）
```

---

## 3. 环境要求

### 3.1 硬件

| 项目 | 说明 |
|---|---|
| 机器人 | 松灵 LIMO 高配版（带 NVIDIA Jetson Nano 4GB） |
| 深度相机 | Intel RealSense D435 或 奥比中光 DaBai |
| 底盘模式 | **四轮差速**（物理开关：插销插入，车灯黄色） |

> LIMO Lite（纯底盘版，无 Jetson Nano）**不能**直接使用本系统，需要额外配主控。

### 3.2 软件

| 项目 | 版本 | 说明 |
|---|---|---|
| Ubuntu | 18.04 | LIMO 出厂预装 |
| ROS | Melodic | LIMO 出厂预装 |
| Python | 3.6+ | LIMO 出厂预装 |
| OpenCV | ≥ 4.1 | 需包含 DNN 模块 |
| numpy | 任意 | 深度数据处理 |

---

## 4. 完整部署指南

### 4.1 前置准备

**步骤 1：确认 LIMO 已连上 WiFi**

LIMO 开机后，通过 NoMachine 远程桌面或 SSH 连接：

```bash
# 从你的电脑 SSH 到 LIMO
ssh agilex@<limo-ip-address>
# 默认密码：agx
```

**步骤 2：确认底盘和摄像头工作正常**

```bash
# 启动底盘
roslaunch limo_base limo_base.launch
# 新终端：
rostopic echo /odom    # 有数据输出 → 底盘 OK
```

```bash
# 启动深度相机（二选一）
roslaunch realsense2_camera rs_camera.launch align_depth:=true
# 或
roslaunch astra_camera dabai_u3.launch

# 新终端：
rostopic list | grep camera   # 看到 /camera/color/image_raw 等话题 → 相机 OK
```

**步骤 3：确认底盘在四轮差速模式**

检查小车前面的插销：插入状态，车灯黄色。如果不是，按照 [LIMO 手册](https://github.com/agilexrobotics/limo-doc) 切换。

### 4.2 安装本包

```bash
# 进入 LIMO 的 ROS 工作空间
cd ~/agilex_ws/src

# 克隆本仓库
git clone https://github.com/airbate/limo-person-following.git

# 回到工作空间根目录，编译
cd ~/agilex_ws
catkin_make

# 加载环境
source devel/setup.bash
```

### 4.3 下载检测模型

```bash
cd ~/agilex_ws/src/limo-person-following
bash scripts/download_model.sh
```

这个脚本会从 GitHub 下载两个文件到 `models/`：
- `MobileNetSSD_deploy.prototxt`（网络结构，约 10 KB）
- `MobileNetSSD_deploy.caffemodel`（预训练权重，约 22 MB）

### 4.4 验证安装

```bash
# 检查是否能找到包
rospack find limo_person_following
# 应输出：/home/agilex/agilex_ws/src/limo-person-following

# 运行离线测试
cd ~/agilex_ws/src/limo-person-following
python3 tests/test_tracker.py
python3 tests/test_controller.py
# 两套测试都应全部通过
```

### 4.5 首次运行

建议**先把 LIMO 架起来（轮子离地）**，确认控制逻辑正确后再落地。

**终端 1 — 底盘：**
```bash
roslaunch limo_base limo_base.launch
```

**终端 2 — 摄像头：**
```bash
roslaunch realsense2_camera rs_camera.launch align_depth:=true
```

**终端 3 — 行人追踪：**
```bash
roslaunch limo_person_following person_following.launch
```

此时站到摄像头前，你应该看到：
- 终端输出 `[INFO] Tracker: ACQUIRED target id=1`
- 轮子开始随你的位置转动
- 如果你走近 < 0.6m，小车后退
- 如果你走出视野 > 2 秒，小车停车

> 架车测试通过后，放下来正常行走即可。

---

## 5. 使用说明

### 5.1 基本操作

```bash
# 启动所有节点（无需 RViz）
roslaunch limo_person_following person_following.launch

# 启动 + RViz 可视化
roslaunch limo_person_following person_following.launch use_rviz:=true
```

启动后，系统自动进入 **ACQUIRING** 状态。第一个人走进摄像头视野时，系统锁定目标并开始跟随。

### 5.2 行为说明

| 情况 | 小车行为 |
|---|---|
| 人在 1.2m 正前方 | 静止不动 |
| 人往前走（距离 > 1.2m） | 向前跟随 |
| 人往后走（距离 < 1.2m） | 慢速后退（保持距离） |
| 人往左/右走 | 转向跟随 |
| 人走到 < 0.6m | 后退避让 |
| 人走到 > 3.5m | 停车等待 |
| 人走出视野 < 1.9 秒 | 保持锁定，继续跟踪 |
| 人走出视野 > 1.9 秒 | 停车，进入 LOST 状态 |
| 人走出视野 > 3 秒 | 重新扫描，锁定下一个进入视野的人 |
| 任何时候 Ctrl+C | 停车 |

### 5.3 停止追踪

在启动追踪的终端中按 `Ctrl+C`。所有节点停止，小车立即停车。

---

## 6. 参数调优

所有参数在 `config/person_following.yaml` 中，修改后重新 `roslaunch` 即生效。

### 6.1 改变跟随距离

```yaml
controller:
  target_distance: 1.0   # 改成 1.0 米（跟得更近）
  # 或
  target_distance: 1.5   # 改成 1.5 米（跟得更远）
```

### 6.2 改变跟随速度

```yaml
controller:
  kp_linear: 0.4         # 降低 P 增益 → 更平缓但响应慢
  max_linear_speed: 0.8  # 提高最大速度 → 能跟跑步的人
```

> **调参经验：** 从保守参数开始（kp_linear=0.3, max_linear=0.3），逐步加大。震荡了就降回来。

### 6.3 提高检测精度（牺牲速度）

```yaml
detector:
  confidence_threshold: 0.7   # 从 0.5 提高到 0.7 → 更少误检，但可能漏检
  detection_rate: 10.0        # 从 8.0 Hz 提高到 10 Hz → 更流畅，但 CPU 更高
```

### 6.4 多人场景

```yaml
tracker:
  initial_target_policy: "closest"   # 锁定最近的人（默认）
  # 或
  initial_target_policy: "largest_bbox"  # 锁定画面中最大的人
```

---

## 7. 调试与诊断

### 7.1 单独测试检测器

```bash
roslaunch limo_person_following test_detector.launch
```

这会打开 RViz + 图像窗口，可以看到：
- 黄色框 + 置信度 → raw MobileNet-SSD 检测结果
- 绿色文字 → 有有效深度的检测（"d=1.5m"）

### 7.2 监控话题

```bash
# 检测帧率
rostopic hz /person_detections

# 跟踪目标
rostopic echo /target_person

# 跟踪状态变化
rostopic echo /tracking_status

# 实际速度指令
rostopic echo /cmd_vel

# 查看话题拓扑图
rqt_graph
```

### 7.3 录制和回放数据包

在 LIMO 上录制数据，传回 PC 做离线调试：

```bash
# 录制
rosbag record /camera/color/image_raw \
              /camera/aligned_depth_to_color/image_raw \
              /camera/color/camera_info \
              -O test_person_following.bag

# 回放（在 PC 上）
rosbag play test_person_following.bag --loop
roslaunch limo_person_following person_following.launch
```

### 7.4 常见故障排查

| 现象 | 可能原因 | 排查方法 |
|---|---|---|
| 检测帧率 < 2 FPS | OpenCV 没用 CUDA | `python3 -c "import cv2; print(cv2.dnn.DNN_BACKEND_CUDA)"` 看是否输出 200 |
| 一直显示 "Waiting for camera_info" | 相机未启动 | `rostopic list | grep camera_info` 确认话题存在 |
| 深度总是 0 | 深度图像话题名不对 | RealSense 对齐后的深度话题是 `/camera/aligned_depth_to_color/image_raw`，非对齐的 `/camera/depth/image_rect_raw` 不能用 |
| 小车左右摇摆 | kp_angular 太大 | 降到 0.8，加大 dead_zone_angle |
| 小车冲得太猛 | 加速度限幅太松 | 减小 `max_linear_accel`（如 0.2） |
| 跟踪总是锁错人 | 多人场景太复杂 | 提高 `initial_target_policy: "closest"` 或让目标人站最近 |
| 短暂遮挡就丢锁 | max_disappeared 太小 | 加大到 20-25 帧 |

---

## 8. 目录结构

```
limo_person_following/
├── CMakeLists.txt                    # catkin 编译规则（消息生成）
├── package.xml                       # ROS 包元信息
├── setup.py                          # Python 包安装
├── README.md                         # 本文件
│
├── launch/
│   ├── person_following.launch       # 主启动（三个节点 + 可选 RViz）
│   └── test_detector.launch          # 单独测试检测器
│
├── config/
│   └── person_following.yaml         # 全部可调参数（80+ 项）
│
├── models/
│   └── .gitignore                    # 排除 .caffemodel（需手动下载）
│
├── msg/
│   ├── PersonDetection.msg           # 单次检测（2D bbox + 3D 位置 + 置信度）
│   ├── PersonDetections.msg          # 检测数组
│   └── TargetPerson.msg              # 当前跟踪目标
│
├── src/person_following/
│   ├── __init__.py
│   ├── detector.py                   # MobileNet-SSD 检测类（可复用）
│   ├── detector_node.py              # 检测 ROS 节点
│   ├── tracker.py                    # 三态状态机（纯 Python，无需 ROS）
│   ├── tracker_node.py               # 跟踪 ROS 节点
│   ├── controller_node.py            # P 控制器 + 安全看门狗
│   └── utils.py                      # 深度采样、3D 投影、IoU、加速度限幅
│
├── scripts/
│   ├── download_model.sh             # 下载 MobileNetSSD 模型文件
│   └── wait_for_camera.sh            # 等待摄像头话题就绪
│
└── tests/
    ├── test_tracker.py               # 状态机 5 项测试（无 ROS 依赖）
    ├── test_controller.py            # 控制律 7 项测试（无 ROS 依赖）
    └── test_detector.py              # 检测器冒烟测试
```

---

## 9. 常见问题

### Q: 可以跟跑步的人吗？

可以，把 `max_linear_speed` 调到 0.8 m/s。但 Jetson Nano 的 5-10 FPS 检测率在高速场景下可能跟不上，建议升级到 Jetson Orin NX 并换 YOLOv8n + TensorRT。

### Q: 能在室外用吗？

深度相机（RealSense D435）在室外强光下表现不佳（红外被阳光淹没）。阴天或傍晚可以，大晴天不行。如果要在室外用，建议换成激光雷达加腿检测（leg_detector）。

### Q: 能同时跟多个人吗？

当前版本只支持跟随一个人。多人场景下会锁定最近的人，忽略其他人。

### Q: 能换别的检测模型吗？

可以。修改 `detector.py` 的 `PersonDetector` 类，换成 YOLO / EfficientDet / NanoDet 等任何模型，只要输入输出接口兼容。`detector_node.py` 和其余管道不受影响。

### Q: 支持 LIMO Pro（ROS2）吗？

当前版本为 ROS1 Melodic。ROS2 版本需要把 `rospy` 改为 `rclpy`，消息类型改为 `sensor_msgs.msg.Image` 等，消息定义改为 `.idl`。有需要可以提 Issue。

---

## 10. License

MIT © 2026

---

## 参考

- [LIMO 官方文档](https://github.com/agilexrobotics/limo-doc)
- [LIMO ROS 源码](https://github.com/agilexrobotics/limo_ros)
- [MobileNet-SSD 论文](https://arxiv.org/abs/1704.04861)
- [OpenCV DNN 文档](https://docs.opencv.org/4.x/d2/d58/tutorial_table_of_content_dnn.html)
