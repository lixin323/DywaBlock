# DyWA 真机部署计划（2026-03-19 更新）

## 1. 当前目标（已落地）

保持最短链路、无占位、点云全链路闭环：

1. 读取 GT 目标块序列  
2. 真机上传 RGB-D + EE 状态 + `T_base_cam`  
3. Server 侧 LINEMOD + 点云得到 `current_T_world_block` / `partial_cloud_world`  
4. DyWA Student（对齐评估语义）输出调整目标  
5. 几何抓取模块生成 `pre/grasp/lift` 动作序列  
6. Server 状态机按阈值推进 `ADJUST -> GRASPED -> next block`

---

## 2. 代码结构（当前有效）

### 2.1 流程编排
- `dywa/src/control/stacking_pipeline.py`
- 入口：`BlockStackingPipeline.process_request(req)`
- 负责：状态机、模块调用顺序、返回 `actions/phase/block_index`

### 2.2 WebSocket 服务入口
- `dywa/src/control/dywa_policy_server.py`
- 只做协议收发 + 异常封装
- 配置项：`dywa_export_dir`（已替换旧 `dywa_ckpt`）

### 2.3 感知与任务
- `dywa/src/control/gt_task_reader.py`：读取 `SCENEs_400_Goal_Jsons/{scene_id:03d}.json`
- `dywa/src/control/pose_recognition_module.py`：解码 RGB-D + LINEMOD + 点云

### 2.4 调整与抓取
- `dywa/src/control/dywa_adjust_module.py`：DyWA 调整 chunk（夹爪保持打开）
- `dywa/src/control/grasp_point_module.py`：几何抓取点生成
- `dywa/src/control/grasp_action_module.py`：抓取动作序列（含 `gripper=1` 触发）

### 2.5 Student 部署适配（重点）
- `dywa/src/control/dywa_model_policy.py`
- 已切换为“导出资产驱动”：
  - 读取 `export_dir/student.yaml` + `student.ckpt`
  - 读取 `export_dir/normalizer.yaml` + `normalize.ckpt`
  - 观测 key 对齐 `eval_student_unseen_obj.sh` 语义：`abs_goal/hand_state/robot_state/previous_action/partial_cloud(+goal_cloud)`
  - 覆盖关键语义：`student.norm="ln"`、`student.decoder.film_mlp=1`

---

## 3. 协议（不变）

### 请求（客户端 -> server）
- `scene_id`
- `ee_state` (7)
- `image_width`, `image_height`
- `fx`, `fy`, `cx`, `cy`
- `depth_scale`
- `T_base_cam` (4x4 行主序, 16 floats)
- `rgb_jpeg_b64`
- `depth_zlib_b64`

### 响应（server -> 客户端）
- `error`
- `actions`（7维动作序列）
- `scene_done`
- `block_index`
- `phase`
- `pos_err_block_m`
- `object_name`

---

## 4. 阈值语义（server 侧）

- `grasp_tol_m`：  
  `ADJUST` 阶段当 `translation error < grasp_tol_m` 时生成抓取序列并切换 `GRASPED`
- `place_tol_m`：  
  `GRASPED` 阶段当 `translation error < place_tol_m` 时 `block_index += 1` 并 reset DyWA episode

---

## 5. 已解决问题记录

1. `ConfigKeyError: use_partial_cloud`  
   - 处理：合并配置时允许非结构化 key（`hydra_cli.py`）

2. `env-last.ckpt` 路径找不到  
   - 处理：`test_rma.py` 支持 `load_student` 为目录或文件两种形式

3. `replace() should be called on dataclass instances`  
   - 处理：`dywa_model_policy.py` 中先将 `student.yaml` merge 到 `StudentAgentRMAConfig` structured base，再 `to_object` 生成 dataclass

4. `history_tokenizer.history` shape mismatch (4096 vs 1)  
   - 处理：加载 student.ckpt 时排除  
   `history_tokenizer.history`、`aggregator.aggregator.memory`

---

## 6. 当前启动方式

### 6.1 启动 server
- 脚本：`start_dywa_policy_server.sh`
- 默认从 `exported_abs_goal_1view` 读取部署模型资产

### 6.2 真机侧 inference node
- 文件：`franka_dywa_inference_node.py`
- 必填：`--t-base-cam-csv`（16 个 float，行主序）

---

## 7. 下一步（立即执行）

1. 在容器内重新启动 server，确认不再报 student 初始化错误  
2. 用单次客户端请求验证返回非空 `actions`  
3. 真机联调观察 `phase/block_index/pos_err_block_m` 是否按阈值推进

---

## 8. 维护约束

1. 禁止回退到占位策略或“兼容补丁”路径。  
2. 新逻辑只进对应模块，不在 `dywa_policy_server.py` 扩散业务实现。  
3. 坐标系统一：`T_world_cam == T_base_cam`（由客户端提供并校验）。

你现在部署里就固定用这三路：

RGB: /camera2/camera2/color/image_raw
Depth: /camera2/camera2/aligned_depth_to_color/image_raw
CameraInfo: /camera2/camera2/color/camera_info
内参用你刚回显的：

fx=606.2757568359375
fy=605.8397216796875
cx=325.67181396484375
cy=248.4153594970703
标定矩阵是
Translation
	x: 0.327485
	y: -0.526087
	z: 0.455808
Rotation
	x: -0.248910
	y: 0.257862
	z: 0.635468
	w: 0.683909
相机内参是侧边相机

camera matrix
611.023473 0.000000 330.929700
0.000000 612.591246 247.915796
0.000000 0.000000 1.000000

distortion
0.179206 -0.422968 -0.006947 0.007867 0.000000

rectification
1.000000 0.000000 0.000000
0.000000 1.000000 0.000000
0.000000 0.000000 1.000000

projection
620.669434 0.000000 334.781793 0.000000
0.000000 625.729980 245.162042 0.000000
0.000000 0.000000 1.000000 0.000000