# DyWA 真机双端最小链路说明

本目录当前只保留两份核心代码：

- 服务端：`dywa/src/control/dywa_policy_server.py`
- 真机端：`franka_dywa_inference_node.py`

以及一键启动脚本：

- `start_dywa_policy_server.sh`

---

## 1. 总体数据流

1. 真机节点采集侧边 RealSense 的彩色+深度（深度对齐到彩色）。
2. 真机节点读取 `/franka/ee_states`，组合成一次请求发送给服务端。
3. 服务端运行 LINEMOD + 深度反投影得到 `T_cam_obj` 与目标点云 `pc_cam`。
4. 服务端用 `T_base_cam` 将点云与目标位姿变换到 base/world 坐标系。
5. 服务端把 `partial_cloud`、`hand_state`、`robot_state` 喂给 `DywaStudentPolicyInterface`。
6. 服务端返回 7 维动作 chunk（`[x,y,z,roll,pitch,yaw,gripper]`）。
7. 真机节点发布到 `/franka/action_command`，由控制节点执行。

---

## 2. 服务端接口与函数说明

文件：`dywa/src/control/dywa_policy_server.py`

### 2.1 配置结构

- `ServerConfig`
  - `host` / `port`：WebSocket 监听地址
  - `scene_root`：GT 场景 json 目录
  - `template_db`：LINEMOD 模板目录
  - `block_assets_dir`：DyWA 目标物体网格目录
  - `dywa_ckpt` / `dywa_device`：模型与设备
  - `chunk_size`：每次返回动作数
  - `grasp_tol_m`：平移误差阈值，低于该值触发抓取动作
  - `place_tol_m`：抓取后低于该阈值切下一块

### 2.2 关键函数

- `_decode_rgb_depth(req)`  
  输入请求中的 `rgb_jpeg_b64` 与 `depth_zlib_b64`，输出 `rgb(bgr)` 与 `depth(uint16)`。

- `_inference_one(req, block_index)`  
  单次推理核心：读取 GT 当前块 -> LINEMOD+点云 -> DyWA -> 生成动作 chunk。

- `handle_connection(ws)`  
  WebSocket 循环，维护服务器状态机：
  - `ADJUST`：夹爪强制保持打开（`gripper=0`）
  - 当 `pos_err_block_m < grasp_tol_m`：在末尾追加抓取动作（`gripper=1`）
  - `GRASPED`：当 `pos_err_block_m < place_tol_m`：切到下一块并重置 episode

- `run()`  
  启动 WebSocket 服务。

### 2.3 请求/响应协议

#### 请求字段（必填）

- `scene_id: int`
- `ee_state: float[7]`
- `image_width, image_height: int`
- `fx, fy, cx, cy: float`
- `depth_scale: float`
- `T_base_cam: float[16]`（4x4 行主序）
- `rgb_jpeg_b64: str`
- `depth_zlib_b64: str`

#### 响应字段

- `error: str`（空字符串表示成功）
- `actions: List[List[float]]`
- `scene_done: bool`
- `block_index: int`
- `phase: str`（`ADJUST` / `GRASPED`）
- `pos_err_block_m: float`

---

## 3. 真机 inference 节点接口与函数说明

文件：`franka_dywa_inference_node.py`

### 3.1 配置结构

- `Args`
  - 服务器：`policy_server_host`, `policy_server_port`, `scene_id`
  - 手眼外参：`t_base_cam_csv`
  - 相机：`cam_side_serial`, `image_width`, `image_height`, `camera_fps`, `jpeg_quality`
  - 推理循环：`inference_frequency`, `max_actions_to_publish`, `action_publish_interval`, `max_state_age`
  - ROS 话题：`ee_states_topic`, `action_topic`, `queue_status_topic`, `allow_inference_topic`

### 3.2 关键函数

- `_parse_t_base_cam(csv)`  
  解析并校验 16 维外参矩阵参数。

- `_init_camera()` / `_camera_loop()`  
  启动 RealSense、深度对齐彩色、编码 RGB-D。

- `ee_cb(msg)`  
  缓存最新 7 维末端状态。

- `queue_cb(msg)` 与 `allow_cb(msg)`  
  根据控制节点信号决定当前是否允许推理。

- `_build_request()`  
  组装并返回一次完整请求（含 RGB-D、内参、外参、EE）。

- `_request_server(req)`  
  WebSocket 发送请求并获取响应。

- `_publish_actions(actions)`  
  发布动作到 `/franka/action_command`。

- `inference_cb()`  
  节点定时主逻辑（条件检查 -> 请求服务端 -> 发布动作 -> 更新 block/phase）。

---

## 4. 手眼标定矩阵更新位置

### 4.1 更新入口

在真机启动命令里更新 `--t-base-cam-csv`，这是唯一有效入口。

示例：

```bash
python franka_dywa_inference_node.py \
  --policy-server-host 192.168.1.10 \
  --policy-server-port 33060 \
  --scene-id 0 \
  --t-base-cam-csv "0.958094665,-0.286412601,-0.004736603,0.504903000,0.265095759,0.880279130,0.393488108,-0.332732000,-0.108530420,-0.378254510,0.919317504,0.277523000,0,0,0,1"
```

### 4.2 坐标定义

- `T_base_cam`：**相机(OpenCV)坐标系 -> 机器人 base**。
- 如果方向写反（给了 `T_cam_base`），点云会整体错位，阈值逻辑会失真。

---

## 5. 启动方式

### 5.1 启动服务端

```bash
./start_dywa_policy_server.sh
```

可选环境变量：

- `HOST`（默认 `0.0.0.0`）
- `PORT`（默认 `33060`）
- `DEVICE`（默认 `cuda:0`）
- `CHUNK_SIZE`（默认 `20`）
- `GRASP_TOL_M`（默认 `0.02`）
- `PLACE_TOL_M`（默认 `0.02`）

### 5.2 启动真机 inference node

```bash
python franka_dywa_inference_node.py \
  --policy-server-host <A机IP> \
  --policy-server-port 33060 \
  --scene-id 0 \
  --t-base-cam-csv "<16 floats>"
```

---

## 6. 最小排障清单

- 服务端报 `LINEMOD/点云检测失败`：检查模板库、光照、目标是否在视野中。
- `scene_done=True` 但任务未结束：检查 `scene_id` 的 GT 是否正确。
- 机械臂不抓取：检查控制节点对 `action[6]` 的阈值语义（应为 `>=0.5` 抓取）。
- 误差长期无法下降：优先复核 `T_base_cam` 方向与数值。
