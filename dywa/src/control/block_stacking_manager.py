#!/usr/bin/env python3

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional, Protocol, Tuple

import numpy as np

try:
    # 可选依赖：仅在几何解析阶段使用
    import open3d as o3d  # type: ignore
except Exception:  # pragma: no cover - 运行环境可按需安装
    o3d = None  # type: ignore

try:
    from pytransform3d.transformations import (  # type: ignore
        euler_matrix,
        transform_from,
        transform,
    )
except Exception:  # pragma: no cover
    euler_matrix = None  # type: ignore
    transform_from = None  # type: ignore
    transform = None  # type: ignore


SE3 = np.ndarray  # 4x4 同质变换矩阵


@dataclass
class BlockSpec:
    """来自 GT JSON 的单块积木目标描述."""

    order: int
    type: str
    color: str
    layer: int
    depend: List[int]
    target_T_world_block: SE3  # 目标位姿（世界到当前积木）


class LinemodDetector(Protocol):
    """LINEMOD 位姿识别接口.

    实际实现可以封装 OpenCV LINEMOD + PnP / ICP 等.
    """

    def estimate_pose(self, block_type: str, color: str) -> SE3:
        """返回当前帧中指定 CAD 模型对应实物的 T_world_block."""
        ...


class DywaPolicyInterface(Protocol):
    """DyWA 推理接口.

    这里不约束具体实现，只约定输入输出的语义。
    """

    def reset_episode(self) -> None:
        """在开始处理新积木前重置内部状态（例如 RNN 隐状态）."""
        ...

    def compute_action(
        self,
        block_type: str,
        current_T_world_block: SE3,
        target_T_world_block: SE3,
    ) -> Dict:
        """根据当前 / 目标位姿输出单步控制动作.

        返回的 Dict 由下游机器人控制器解析，例如
        {
            "ee_target_T_world": <4x4>,
            "gripper_width": 0.03,
        }
        """
        ...


class RobotController(Protocol):
    """机器人执行接口，用于解耦具体 ROS2 / 模拟实现."""

    def move_ee_to_pose(self, T_world_ee: SE3) -> None:
        """笛卡尔空间移动末端到指定位姿（阻塞直到到达或失败）."""
        ...

    def set_gripper(self, width: float) -> None:
        """设置夹爪开合宽度."""
        ...

    def attach_object(self, block_type: str) -> None:
        """在仿真 / TF 树中将当前对象与末端刚性绑定."""
        ...

    def detach_object(self, block_type: str) -> None:
        """解除绑定."""
        ...


class GraspPlanner(Protocol):
    """几何抓取点规划接口."""

    def compute_grasp_poses(
        self,
        block_type: str,
        T_world_block: SE3,
    ) -> List[Tuple[SE3, float]]:
        """返回一组候选抓取位姿及对应夹爪宽度."""
        ...


def _pose_from_json(position: List[float], euler_xyz: List[float]) -> SE3:
    """将 JSON 中的平移 + 欧拉角转换为 4x4 同质矩阵.

    约定欧拉角单位为弧度，顺序为 (x, y, z)，与 json 中字段保持一致。
    """
    if euler_matrix is None:
        raise RuntimeError(
            "pytransform3d 未安装，无法从欧拉角构造位姿矩阵。"
        )
    if len(position) != 3 or len(euler_xyz) != 3:
        raise ValueError("position / euler 维度错误，应为长度 3 的列表。")
    tx, ty, tz = position
    rx, ry, rz = euler_xyz
    T = euler_matrix(rx, ry, rz, axes="sxyz")  # type: ignore[arg-type]
    T = np.asarray(T, dtype=np.float32)
    T[:3, 3] = np.asarray([tx, ty, tz], dtype=np.float32)
    return T


class SimpleBoxGraspPlanner:
    """基于 AABB 的简单几何抓取策略，依赖 CAD 网格或包围盒."""

    def __init__(self, cad_root: Path):
        """
        参数
        ----
        cad_root:
            存放各类积木 CAD 模型的目录，文件名约定为
            ``{block_type}.ply`` / ``{block_type}.obj`` 等。
        """
        self.cad_root = Path(cad_root)

    def _load_cad(self, block_type: str) -> Optional["o3d.geometry.TriangleMesh"]:
        if o3d is None:
            return None
        for ext in (".ply", ".obj", ".stl"):
            path = self.cad_root / f"{block_type}{ext}"
            if path.is_file():
                mesh = o3d.io.read_triangle_mesh(str(path))
                if not mesh.has_vertices():
                    continue
                return mesh
        return None

    def compute_grasp_poses(
        self,
        block_type: str,
        T_world_block: SE3,
    ) -> List[Tuple[SE3, float]]:
        if o3d is None:
            raise RuntimeError("open3d 未安装，无法进行几何抓取规划。")

        mesh = self._load_cad(block_type)
        if mesh is None:
            # 回退策略：使用单位立方体近似
            bbox_local = np.array(
                [
                    [-0.5, -0.5, -0.5],
                    [0.5, 0.5, 0.5],
                ],
                dtype=np.float32,
            )
        else:
            bbox: "o3d.geometry.AxisAlignedBoundingBox" = mesh.get_axis_aligned_bounding_box()  # type: ignore[assignment]
            min_bound = np.asarray(bbox.get_min_bound(), dtype=np.float32)
            max_bound = np.asarray(bbox.get_max_bound(), dtype=np.float32)
            bbox_local = np.stack([min_bound, max_bound], axis=0)

        # 仅使用 X/Y 方向尺寸估计抓取宽度
        size = bbox_local[1] - bbox_local[0]
        width_xy = float(max(size[0], size[1]))
        grasp_width = width_xy * 1.1  # 略大于物体宽度

        # 规划两类抓取：从 +X 和 -X 方向靠近物体质心
        center_local = (bbox_local[0] + bbox_local[1]) * 0.5
        offset = size[0] * 0.6  # 在物体外部一定距离

        def _make_grasp(offset_dir: np.ndarray) -> SE3:
            if transform_from is None:
                raise RuntimeError("pytransform3d 未安装，无法构造抓取位姿。")
            # 抓取点在物体坐标系下的位置
            p_grasp_local = center_local + offset_dir
            # 末端坐标系的 z 轴朝向物体中心
            z_axis = -offset_dir / (np.linalg.norm(offset_dir) + 1e-8)
            x_axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            y_axis = np.cross(z_axis, x_axis)
            if np.linalg.norm(y_axis) < 1e-4:
                x_axis = np.array([0.0, 1.0, 0.0], dtype=np.float32)
                y_axis = np.cross(z_axis, x_axis)
            x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-8)
            y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-8)
            R_be = np.stack([x_axis, y_axis, z_axis], axis=1)
            T_block_ee = transform_from(R_be, p_grasp_local)  # type: ignore[arg-type]
            T_world_ee = transform(T_world_block, T_block_ee)  # type: ignore[arg-type]
            return np.asarray(T_world_ee, dtype=np.float32)

        candidates: List[Tuple[SE3, float]] = []
        for sign in (-1.0, 1.0):
            offset_dir = np.asarray([sign * offset, 0.0, 0.0], dtype=np.float32)
            candidates.append((_make_grasp(offset_dir), grasp_width))
        return candidates


class BlockStackingManager:
    """PhyBlock 任务的高层控制器，实现从 GT 到实际搭建的全流程."""

    def __init__(
        self,
        scene_root: Path,
        linemod: LinemodDetector,
        dywa_policy: DywaPolicyInterface,
        robot: RobotController,
        grasp_planner: Optional[GraspPlanner] = None,
        *,
        scene_extension: str = ".json",
    ):
        """
        参数
        ----
        scene_root:
            GT json 根目录，例如
            ``/home/lixin/DyWA/data/PhyBlock/data/SCENEs_400_Goal_Jsons``。
        linemod:
            已经训练好的 LINEMOD 位姿识别器封装。
        dywa_policy:
            已经加载好权重的 DyWA policy 接口。
        robot:
            机器人控制接口（可以是 ROS2 实机，也可以是仿真环境封装）。
        grasp_planner:
            几何抓取规划器，若为 None，则仅使用 DyWA 输出的末端目标。
        scene_extension:
            场景文件后缀，默认 ``.json``。
        """
        self.scene_root = Path(scene_root)
        self.scene_extension = scene_extension
        self.linemod = linemod
        self.dywa = dywa_policy
        self.robot = robot
        self.grasp_planner = grasp_planner

    # --------------------------------------------------------------------- #
    # 公共 API
    # --------------------------------------------------------------------- #
    def run_scene(self, shape_name: str) -> None:
        """执行一次完整的积木搭建任务."""
        blocks = self._load_scene_blocks(shape_name)
        # 按 order 升序排序，保证搭建顺序一致
        blocks = sorted(blocks, key=lambda b: b.order)

        for block in blocks:
            self._process_single_block(block)

    # --------------------------------------------------------------------- #
    # 内部步骤
    # --------------------------------------------------------------------- #
    def _load_scene_blocks(self, shape_name: str) -> List[BlockSpec]:
        """从 GT json 解析当前场景所有积木."""
        filename = self.scene_root / f"{shape_name}{self.scene_extension}"
        if not filename.is_file():
            raise FileNotFoundError(f"未找到场景文件: {filename}")

        with open(filename, "r") as f:
            data = json.load(f)

        blocks_data = data.get("blocks", [])
        result: List[BlockSpec] = []
        for b in blocks_data:
            target_T = _pose_from_json(
                position=b["position"],
                euler_xyz=b["euler"],
            )
            result.append(
                BlockSpec(
                    order=int(b["order"]),
                    type=str(b["type"]),
                    color=str(b["color"]),
                    layer=int(b["layer"]),
                    depend=list(b.get("depend", [])),
                    target_T_world_block=target_T,
                )
            )
        return result

    def _process_single_block(self, block: BlockSpec) -> None:
        """单块积木从识别到放置的完整流程."""
        # 1) CAD 位姿对齐：通过 LINEMOD 识别当前位姿
        current_T_world_block = self.linemod.estimate_pose(
            block_type=block.type,
            color=block.color,
        )

        # 2) 使用 DyWA 在“工作区域”内将积木调整到目标姿态
        self.dywa.reset_episode()
        current_T = current_T_world_block
        for _ in range(128):
            if self._is_pose_close(current_T, block.target_T_world_block):
                break

            action = self.dywa.compute_action(
                block_type=block.type,
                current_T_world_block=current_T,
                target_T_world_block=block.target_T_world_block,
            )
            self._execute_dywa_action(action)

            # 再次调用 LINEMOD 更新当前位姿（闭环）
            current_T = self.linemod.estimate_pose(
                block_type=block.type,
                color=block.color,
            )

        # 3) 到达目标工作区域后，利用几何解析法生成抓取点并完成抓取
        if self.grasp_planner is not None:
            grasp_candidates = self.grasp_planner.compute_grasp_poses(
                block_type=block.type,
                T_world_block=block.target_T_world_block,
            )
            if len(grasp_candidates) == 0:
                raise RuntimeError(f"未能为积木 {block.type} 生成有效抓取位姿。")
            # 简单策略：取第一个候选
            T_world_ee, grip_width = grasp_candidates[0]
            self._execute_grasp(T_world_ee, grip_width, block)
        else:
            # 若未提供 grasp_planner，则假定 DyWA 的动作中已经包含抓取控制
            pass

    def _execute_dywa_action(self, action: Dict) -> None:
        """根据 DyWA 输出的动作字典调用机器人控制器."""
        T_world_ee: Optional[SE3] = action.get("ee_target_T_world")
        if T_world_ee is not None:
            self.robot.move_ee_to_pose(np.asarray(T_world_ee, dtype=np.float32))

        grip_width = action.get("gripper_width", None)
        if grip_width is not None:
            self.robot.set_gripper(float(grip_width))

    def _execute_grasp(
        self,
        T_world_ee: SE3,
        gripper_width: float,
        block: BlockSpec,
    ) -> None:
        """执行实际抓取动作."""
        # 先张开夹爪
        self.robot.set_gripper(gripper_width)
        # 移动到抓取位姿
        self.robot.move_ee_to_pose(T_world_ee)
        # 闭合夹爪绑定积木
        self.robot.attach_object(block.type)

    @staticmethod
    def _is_pose_close(
        T_a: SE3,
        T_b: SE3,
        pos_tol: float = 5e-3,
        ang_tol: float = 5.0 / 180.0 * np.pi,
    ) -> bool:
        """简单位姿误差度量，用于判断是否已经到达目标."""
        da = np.asarray(T_a, dtype=np.float32)
        db = np.asarray(T_b, dtype=np.float32)
        dp = np.linalg.norm(da[:3, 3] - db[:3, 3])
        R_rel = db[:3, :3].T @ da[:3, :3]
        trace = np.clip((np.trace(R_rel) - 1.0) / 2.0, -1.0, 1.0)
        dtheta = float(np.arccos(trace))
        return (dp <= pos_tol) and (dtheta <= ang_tol)


__all__ = [
    "BlockStackingManager",
    "BlockSpec",
    "LinemodDetector",
    "DywaPolicyInterface",
    "RobotController",
    "GraspPlanner",
    "SimpleBoxGraspPlanner",
]

