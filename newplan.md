# DyWA 仿真部署计划（同真机流程，最短路径）

## 1. 原始目标与边界

### 1.1 原始目标

在**不改服务端业务逻辑**的前提下，把当前真机链路完整迁移到仿真环境，得到与真机一致的状态机推进结果：

1. 读取 GT 目标块序列  
2. 输入 RGB-D + EE 状态 + `T_base_cam`  
3. Server 侧 LINEMOD + 点云得到 `current_T_world_block` / `partial_cloud_world`  
4. DyWA Student 输出调整动作 chunk  
5. 几何抓取模块生成 `pre/grasp/lift`  
6. `ADJUST -> GRASPED -> next block` 按阈值推进

### 1.2 不变约束（必须满足）

1. 不允许兼容性分叉：仿真与真机使用同一个 `dywa_policy_server.py` 与 `stacking_pipeline.py`。  
2. 不允许补丁式绕过：不能直接跳过 LINEMOD/点云或伪造 `pos_err`。  
3. 不允许扩散业务逻辑到 server 入口：`dywa_policy_server.py` 仍仅做协议收发。  
4. 坐标定义不变：请求中的 `T_base_cam` 语义仍为 `T_world_cam`。

---

## 2. 第一性原理拆解

服务端真正需要的只有一份“观测快照”：

- 几何输入：`rgb_jpeg_b64`、`depth_zlib_b64`、内参、`depth_scale`、`T_base_cam`
- 机器人状态：`ee_state`
- 任务索引：`scene_id`

因此，真机与仿真的本质差异仅在**观测快照如何产生**。  
结论：仿真部署只需要新增一个“仿真侧请求节点”，复用原协议与原 server。

---

## 3. 仿真端最小实现方案

## 3.1 组件划分

1. **Policy Server（复用）**  
   - 直接使用 `start_dywa_policy_server.sh` 启动。
2. **仿真客户端节点（新增）**  
   - 新文件建议：`sim_dywa_inference_node.py`  
   - 职责：从仿真环境取 `RGB-D + intrinsics + T_base_cam + ee_state`，按现有 WebSocket 协议请求 server，回写动作到仿真控制接口。

## 3.2 仿真客户端必须实现的函数（对齐真机节点）

1. `_get_sim_camera_frame()`  
   - 输出：`bgr(uint8)`、`depth_u16`（或可转换到 uint16）、相机内参。
2. `_get_sim_ee_state()`  
   - 输出：7 维末端状态，语义与真机 `ee_state` 一致。
3. `_get_T_base_cam()`  
   - 输出：4x4 行主序 16 float，语义为 `T_world_cam`。
4. `_build_request()`  
   - 字段与现有协议完全一致（不增减字段）。
5. `_request_server()`  
   - WebSocket 请求/响应逻辑与 `franka_dywa_inference_node.py` 一致。
6. `_apply_actions_to_sim(actions)`  
   - 逐条执行 7 维动作；`action[6]` 抓取语义保持一致（`>=0.5` 闭合）。

## 3.3 数据编码约束（必须一致）

1. RGB：JPEG 后 base64，字段名 `rgb_jpeg_b64`。  
2. Depth：`uint16` 原始字节 zlib 压缩后 base64，字段名 `depth_zlib_b64`。  
3. `image_width/height`、`fx/fy/cx/cy`、`depth_scale` 与当前帧严格对应。  
4. 深度零值/无效值处理与真机保持同等语义（无效深度不应被当作真实几何）。

---

## 4. 部署步骤（执行顺序）

### 步骤 1：启动 server（不改代码）

```bash
./start_dywa_policy_server.sh
```

### 步骤 2：启动仿真环境

要求仿真侧已经能稳定提供：

- 彩色图、深度图（与彩色像素对齐）
- 相机内参
- `T_world_cam`（作为 `T_base_cam` 发送）
- 机器人末端 7 维状态

### 步骤 3：启动仿真客户端节点

```bash
python sim_dywa_inference_node.py \
  --policy-server-host 127.0.0.1 \
  --policy-server-port 33060 \
  --scene-id 0
```

### 步骤 4：闭环执行

1. 客户端按频率请求 server。  
2. server 返回 `actions/phase/block_index/pos_err_block_m`。  
3. 客户端执行动作并推进仿真。  
4. 循环至 `scene_done=True`。

---

## 5. 全链路正确性验证（必须通过）

## 5.1 单次请求验证

目标：确认协议与几何链路正确。

- 输入固定一帧仿真 RGB-D + EE + `T_base_cam`
- 期望：`error=""` 且 `actions` 非空（`ADJUST` 阶段）

## 5.2 状态机验证

目标：确认流程推进逻辑正确。

1. 初始为 `phase=ADJUST`。  
2. 当 `pos_err_block_m < grasp_tol_m`，返回抓取序列并切到 `GRASPED`。  
3. `GRASPED` 下当 `pos_err_block_m < place_tol_m`，`block_index += 1` 且 phase 回到 `ADJUST`。  
4. 全部目标块完成后 `scene_done=True`。

## 5.3 坐标系验证

目标：确认 `T_base_cam` 方向正确。

- 将仿真中已知物体中心点投影-反投影并变换到 world，误差应在可接受范围内。  
- 若误差系统性偏大，先检查是否误发了 `T_cam_world`（方向反了）。

---

## 6. 交付物与完成判定

## 6.1 交付物

1. `sim_dywa_inference_node.py`（仿真侧请求与执行节点）  
2. 启动命令文档（可写入 README 或脚本注释）  
3. 一段完整运行日志（至少覆盖一次 `ADJUST -> GRASPED -> next block`）

## 6.2 完成判定（DoD）

同时满足以下条件才算完成：

1. 不修改 server 核心业务模块仍可在仿真闭环运行。  
2. 协议字段 100% 对齐真机链路。  
3. 阶段推进与阈值语义完全一致。  
4. 至少一个 `scene_id` 跑到 `scene_done=True`。

---

## 7. 本计划的唯一实现路径说明

本计划只保留一条实现路径：  
**复用现有 server + 新增仿真客户端节点替换真机数据采集端**。  
不引入兜底路径、不引入双实现、不引入旁路逻辑。

