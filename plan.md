# plan.md: DywaBlock 真机部署与代码实现计划

## 1. 项目概览
本项目旨在利用 DyWA（Dynamic World Alignment）策略，在 Franka Panda 机器人上实现自动化积木堆叠。
- **核心逻辑**：基于点云对齐的非合拢操控（Push）+ 几何抓取（Pick & Place）。
- **关键挑战**：Sim-to-Real 的坐标对齐、多物体场景下的点云遮罩（Masking）、复杂形状的抓取规划。

---

## 2. 模块化开发路线

### M1: 数据预处理（Template Generation）
**目标**：为 LINEMOD 识别生成 6D 位姿匹配模板。
- **任务**：编写 `generate_templates.py`。
- **逻辑**：利用 `block_assets` 中的 OBJ 文件，通过球形采样渲染 RGBD 图像，提取 LINEMOD 特征并记录相对位姿。

### M2: 视觉感知模块（Masked Perception）
**目标**：在多物体桌面识别目标积木，并提取纯净的点云。
- **任务**：完善 `linemod_opencv.py`。
- **核心点**：
    - 返回目标积木的 `SE3` 位姿和像素级 `Mask`。
    - **点云提取**：结合深度图与 Mask，提取目标积木的当前点云 $PC_{curr}$。
    - **位姿精炼**：使用 Open3D 的 Point-to-Plane ICP 对识别出的位姿进行毫米级微调。

### M3: DyWA 位姿调整器（Point Cloud Adjuster）
**目标**：调用点云 RL 模型，将积木调整至理想抓取姿态。
- **任务**：编写 `pose_adjuster.py`。
- **输入对齐**：
    - `Input A`: 实时 Mask 提取的点云 $PC_{curr}$。
    - `Input B`: 将 OBJ 模型变换至 `goal_pose` 得到的虚拟点云 $PC_{goal}$。
- **控制循环**：模型输出 Push 动作 -> 机械臂执行 -> 视觉反馈 -> 循环直到 Chamfer Distance 达标。

### M4: 几何解析抓取规划（Grasp Planner）
**目标**：针对不同形状（正方体、半圆柱、三角柱）自动计算抓取点。
- **任务**：编写 `grasp_planner.py`。
- **形状策略**：
    - **正方体**：平行侧面中心抓取。
    - **半圆柱**：端面优先，或底面垂直夹持。
    - **三角柱**：端面夹持，或侧面二等分线对角夹持。

### M5: 系统集成与堆叠逻辑（Master Executor）
**目标**：读取 GT 任务，驱动 Franka 完成流水线。
- **任务**：编写 `main_stacking_executor.py`。
- **逻辑**：解析顺序 -> 视觉定位 -> DyWA 调整 -> 抓取 -> 堆叠（根据层数动态计算 Z 轴 Offset）。

---

## 3. 实现步骤与 Cursor Prompt 序列

### 第一阶段：视觉与基础控制
1. **Prompt 1 (Templates)**：
   > “阅读 `block_assets/` 中的 OBJ。编写 `generate_templates.py`，使用 Open3D 渲染各视角并为每种积木生成 OpenCV LINEMOD 模板存入 `linemod_templates/`。”
2. **Prompt 2 (Perception)**：
   > “完善 `linemod_opencv.py`。实现 `detect_and_mask` 函数，要求在识别位姿的同时返回像素 Mask，并利用 Mask 提取 1024 个点的目标物点云。”

### 第二阶段：DyWA 策略对齐
3. **Prompt 3 (Adjustment)**：
   > “编写 `pose_adjuster.py`。它应加载 DyWA 点云策略模型。输入是 Mask 提取的实时点云和目标位姿下的模型点云。输出是 Push 动作，通过 `Ros2RobotController` 执行。”

### 第三阶段：抓取与堆叠执行
4. **Prompt 4 (Grasping)**：
   > “实现 `grasp_planner.py`。针对 Cube, Semi-cylinder, Triangular Prism 实现几何法抓取点计算，返回 [pre_grasp, grasp, lift_up] 序列。”
5. **Prompt 5 (Main)**：
   > “集成所有模块至 `main_stacking_executor.py`。读取 `SCENEs_400_Goal_Jsons/`，按照顺序执行堆叠任务。加入手眼标定矩阵变换逻辑，将相机系坐标转换为机器人基座系。”

---

## 4. 真机部署关键检查清单
- [ ] **手眼标定**：确保 $T_{base}^{cam}$ 精度在 2mm 以内。
- [ ] **单位对齐**：检查点云单位（米 vs 毫米）是否与模型训练时一致。
- [ ] **Mask 质量**：光照是否会导致 Mask 破碎？若有点云离群点，需加入离群点滤波。
- [ ] **安全高度**：在 `lift_up` 和 `pre_place` 时，Z 轴高度需高于已有积木塔 10cm 以上。

---

### 给 Cursor 的建议：
在开始写代码前，请先运行：
1. `se3.py`：熟悉位姿变换工具类。
2. `config.py`：确定参数读取方式。
3. `franka_control_node.py`：熟悉真机 ROS2 接口。
