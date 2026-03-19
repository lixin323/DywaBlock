#!/usr/bin/env python3

from __future__ import annotations

from typing import Literal, Tuple

import numpy as np

from .se3 import SE3, assert_se3


ShapeType = Literal["cube", "semi_cylinder", "triangular_prism"]


GRIPPER_MAX_WIDTH = 0.085  # Franka Panda 默认最大张开 85mm


def _make_se3(R: np.ndarray, t: np.ndarray) -> SE3:
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = np.asarray(R, dtype=np.float32)
    T[:3, 3] = np.asarray(t, dtype=np.float32).reshape(3)
    assert_se3(T)
    return T


def _select_axis_for_parallel_faces(
    R_world_obj: np.ndarray,
    size_obj: np.ndarray,
    table_normal_world: np.ndarray,
    gripper_max_width: float,
) -> Tuple[int, float]:
    """在 OBB 中选择一对“垂直桌面”的平行侧面所在轴.

    返回:
        axis_idx: 选中的轴索引 (0/1/2)，对应物体系主轴
        width:    该方向上两平行面的间距
    """
    tn = table_normal_world / (np.linalg.norm(table_normal_world) + 1e-8)

    best_axis = -1
    best_width = float("inf")

    for i in range(3):
        n_world = R_world_obj[:, i]
        # 垂直桌面: 法向量与桌面法线近似正交
        if abs(float(np.dot(n_world, tn))) > 0.3:
            continue
        width = float(abs(size_obj[i]))
        # 优先选择宽度在夹爪范围内的轴; 若都不满足, 选最小的
        if width <= gripper_max_width:
            if width < best_width:
                best_axis = i
                best_width = width
        elif best_axis == -1 and width < best_width:
            best_axis = i
            best_width = width

    if best_axis == -1:
        # 回退: 选尺寸最小的轴
        best_axis = int(np.argmin(np.abs(size_obj)))
        best_width = float(abs(size_obj[best_axis]))

    return best_axis, best_width


def _grasp_from_side_faces_cube(
    T_world_obj: SE3,
    size_obj: np.ndarray,
    table_normal_world: np.ndarray,
    gripper_max_width: float,
) -> Tuple[SE3, float]:
    """基于 OBB 的立方体侧面夹持."""
    assert_se3(T_world_obj)
    R_world_obj = T_world_obj[:3, :3]
    p_world_obj = T_world_obj[:3, 3]

    axis_idx, width = _select_axis_for_parallel_faces(
        R_world_obj, size_obj, table_normal_world, gripper_max_width
    )
    if width > gripper_max_width:
        # 所有方向均超出夹爪张开, 此时仍返回最小宽度方向, 由上层决定是否放弃
        pass

    # 物体系中, 选中轴的正向法线
    n_obj = np.zeros(3, dtype=np.float32)
    n_obj[axis_idx] = 1.0
    # 转到世界系
    n_world = R_world_obj @ n_obj
    n_world = n_world / (np.linalg.norm(n_world) + 1e-8)

    # 质心在世界系
    center_world = p_world_obj

    # 抓取接触平面中心 (在世界系): 从质心沿法线移动 size/2
    contact_center_world = center_world + 0.5 * width * n_world

    # 末端靠近方向: 指尖法线朝向物体, 即 -n_world
    z_ee = -n_world
    z_ee = z_ee / (np.linalg.norm(z_ee) + 1e-8)

    # 夹爪开合方向: 要与物体“宽度方向”一致, 且与桌面法线正交
    # 先令 y_ee 与桌面法线正交, 再做正交化
    tn = table_normal_world / (np.linalg.norm(table_normal_world) + 1e-8)
    y_ee = np.cross(tn, z_ee)
    if np.linalg.norm(y_ee) < 1e-4:
        # 若共线, 改用另一辅助向量
        aux = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        y_ee = np.cross(aux, z_ee)
    y_ee = y_ee / (np.linalg.norm(y_ee) + 1e-8)

    x_ee = np.cross(y_ee, z_ee)
    x_ee = x_ee / (np.linalg.norm(x_ee) + 1e-8)

    R_world_ee = np.stack([x_ee, y_ee, z_ee], axis=1)

    # 末端参考点: 在接触平面外侧留一点安全距离
    approach_clearance = 0.02  # 2cm
    p_world_ee = contact_center_world + approach_clearance * z_ee

    T_world_ee = _make_se3(R_world_ee, p_world_ee)
    grasp_width = min(width * 1.05, gripper_max_width)
    return T_world_ee, grasp_width


def _lift_and_pre_grasp(
    T_world_grasp: SE3,
    table_normal_world: np.ndarray,
    pre_grasp_height: float = 0.10,
    lift_height: float = 0.15,
) -> Tuple[SE3, SE3]:
    """根据抓取位姿生成 pre_grasp / lift_up."""
    assert_se3(T_world_grasp)
    tn = table_normal_world / (np.linalg.norm(table_normal_world) + 1e-8)

    T_pre = T_world_grasp.copy()
    T_pre[:3, 3] = T_pre[:3, 3] + pre_grasp_height * tn
    assert_se3(T_pre)

    T_lift = T_world_grasp.copy()
    T_lift[:3, 3] = T_lift[:3, 3] + lift_height * tn
    assert_se3(T_lift)

    return T_pre, T_lift


def generate_grasp_pose(
    object_id: str,
    current_se3: SE3,
    *,
    gripper_max_width: float = GRIPPER_MAX_WIDTH,
    table_normal_world: np.ndarray | None = None,
) -> Tuple[SE3, SE3, SE3, float]:
    """针对不同形状生成 ``(pre_grasp, grasp, lift_up, width)``.

    参数
    ----
    object_id:
        物体 ID, 内部通过前缀简单判断形状:
        - 以 ``cube`` 开头: 正方体/长方体
        - 以 ``semi_cylinder`` 开头: 半圆柱
        - 以 ``tri_prism`` 或 ``triangular_prism`` 开头: 三角柱
    current_se3:
        当前世界到物体的位姿 ``T_world_obj`` (4x4).
    gripper_max_width:
        夹爪最大开口宽度 (米), 默认 0.085 (85mm).
    table_normal_world:
        桌面法线在世界系下的方向, 默认 ``[0, 0, 1]``.

    返回
    ----
    (pre_grasp, grasp, lift_up, width)
    """
    assert_se3(current_se3)
    if table_normal_world is None:
        table_normal_world = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    table_normal_world = np.asarray(table_normal_world, dtype=np.float32).reshape(3)

    oid = object_id.lower()

    # --------------------------------------------------------------- #
    # 1) 近似 OBB 尺寸: 这里假设通过 ID 事先约定了尺寸 (米).
    #    若无精确 CAD, 可以在此处根据具体项目填入真实尺寸.
    # --------------------------------------------------------------- #
    if oid.startswith("cube"):
        # 示例: 5cm 立方体
        size_obj = np.array([0.05, 0.05, 0.05], dtype=np.float32)
        T_world_grasp, grip_width = _grasp_from_side_faces_cube(
            current_se3, size_obj, table_normal_world, gripper_max_width
        )
    elif oid.startswith("semi_cylinder"):
        # 半圆柱: 轴向长度 L, 半径 R
        L = 0.06  # 6cm
        R = 0.025  # 2.5cm
        # 近似 OBB 尺寸 (以局部 x 为轴向, y 宽度, z 高度)
        size_obj = np.array([L, 2.0 * R, R], dtype=np.float32)

        # 优先用两端平整端面 (±x) 抓取
        T_world_obj = current_se3
        R_world_obj = T_world_obj[:3, :3]
        p_world_obj = T_world_obj[:3, 3]

        axis_x_world = R_world_obj[:, 0]
        width_end = L
        use_end_faces = width_end <= gripper_max_width

        if use_end_faces:
            # 末端 z 轴对准物体轴向, 从 ±x 方向之一靠近
            z_ee = -axis_x_world
            z_ee = z_ee / (np.linalg.norm(z_ee) + 1e-8)
            tn = table_normal_world
            y_ee = np.cross(tn, z_ee)
            if np.linalg.norm(y_ee) < 1e-4:
                aux = np.array([0.0, 1.0, 0.0], dtype=np.float32)
                y_ee = np.cross(aux, z_ee)
            y_ee = y_ee / (np.linalg.norm(y_ee) + 1e-8)
            x_ee = np.cross(y_ee, z_ee)
            x_ee = x_ee / (np.linalg.norm(x_ee) + 1e-8)
            R_world_ee = np.stack([x_ee, y_ee, z_ee], axis=1)

            contact_center_world = p_world_obj + 0.5 * width_end * axis_x_world
            approach_clearance = 0.02
            p_world_ee = contact_center_world + approach_clearance * z_ee

            T_world_grasp = _make_se3(R_world_ee, p_world_ee)
            grip_width = min(width_end * 1.05, gripper_max_width)
        else:
            # 端面不可达: 使用“底面中心 - 圆弧顶线中心”的连线,
            # 这里假设 z 轴为“上/下”方向, y 轴为圆弧展开方向
            T_world_obj = current_se3
            R_world_obj = T_world_obj[:3, :3]
            p_world_obj = T_world_obj[:3, 3]

            z_obj = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            y_obj = np.array([0.0, 1.0, 0.0], dtype=np.float32)

            z_world = R_world_obj @ z_obj
            y_world = R_world_obj @ y_obj

            bottom_center_world = p_world_obj - 0.5 * size_obj[2] * z_world
            arc_top_center_world = p_world_obj + R * z_world

            # 连线方向 (指向圆弧顶线)
            line_dir = arc_top_center_world - bottom_center_world
            line_dir = line_dir / (np.linalg.norm(line_dir) + 1e-8)

            # 末端 z 轴: 垂直于底面 (≈ 桌面法线), 选择水平的最佳方向
            # 这里令 z_ee 与 y_world 正交, 指向物体内部
            z_ee = -y_world
            z_ee = z_ee / (np.linalg.norm(z_ee) + 1e-8)

            tn = table_normal_world
            y_ee = np.cross(tn, z_ee)
            if np.linalg.norm(y_ee) < 1e-4:
                aux = np.array([1.0, 0.0, 0.0], dtype=np.float32)
                y_ee = np.cross(aux, z_ee)
            y_ee = y_ee / (np.linalg.norm(y_ee) + 1e-8)
            x_ee = np.cross(y_ee, z_ee)
            x_ee = x_ee / (np.linalg.norm(x_ee) + 1e-8)
            R_world_ee = np.stack([x_ee, y_ee, z_ee], axis=1)

            contact_center_world = bottom_center_world + 0.5 * line_dir * np.linalg.norm(
                line_dir
            )
            approach_clearance = 0.02
            p_world_ee = contact_center_world + approach_clearance * z_ee

            T_world_grasp = _make_se3(R_world_ee, p_world_ee)
            grip_width = min(size_obj[1] * 1.05, gripper_max_width)
    elif oid.startswith("tri_prism") or oid.startswith("triangular_prism"):
        # 三角柱: 假设局部 x 为柱体轴向, 底面为某一三角形面
        T_world_obj = current_se3
        R_world_obj = T_world_obj[:3, :3]
        p_world_obj = T_world_obj[:3, 3]

        # 近似 OBB: 轴向长度 L, 高度 H, 宽度 W
        L = 0.06
        H = 0.04
        W = 0.04
        size_obj = np.array([L, W, H], dtype=np.float32)

        axis_x_world = R_world_obj[:, 0]
        width_end = L

        use_end_faces = width_end <= gripper_max_width

        if use_end_faces:
            # 直接抓取两个三角形端面 (±x)
            z_ee = -axis_x_world
            z_ee = z_ee / (np.linalg.norm(z_ee) + 1e-8)

            tn = table_normal_world
            y_ee = np.cross(tn, z_ee)
            if np.linalg.norm(y_ee) < 1e-4:
                aux = np.array([0.0, 1.0, 0.0], dtype=np.float32)
                y_ee = np.cross(aux, z_ee)
            y_ee = y_ee / (np.linalg.norm(y_ee) + 1e-8)
            x_ee = np.cross(y_ee, z_ee)
            x_ee = x_ee / (np.linalg.norm(x_ee) + 1e-8)
            R_world_ee = np.stack([x_ee, y_ee, z_ee], axis=1)

            contact_center_world = p_world_obj + 0.5 * width_end * axis_x_world
            approach_clearance = 0.02
            p_world_ee = contact_center_world + approach_clearance * z_ee

            T_world_grasp = _make_se3(R_world_ee, p_world_ee)
            grip_width = min(width_end * 1.05, gripper_max_width)
        else:
            # 物体横放: 假设其中一个三角形面贴地
            # 找到法向最接近桌面法线的物体系轴, 其余两个为倾斜侧面
            tn = table_normal_world / (np.linalg.norm(table_normal_world) + 1e-8)
            best_align = -1.0
            base_axis_idx = 2
            for i in range(3):
                n_world = R_world_obj[:, i]
                align = abs(float(np.dot(n_world, tn)))
                if align > best_align:
                    best_align = align
                    base_axis_idx = i

            side_axes = [i for i in range(3) if i != base_axis_idx]
            n1_world = R_world_obj[:, side_axes[0]]
            n2_world = R_world_obj[:, side_axes[1]]
            n1_world = n1_world / (np.linalg.norm(n1_world) + 1e-8)
            n2_world = n2_world / (np.linalg.norm(n2_world) + 1e-8)

            # 夹角二等分线方向
            bisector = n1_world + n2_world
            if np.linalg.norm(bisector) < 1e-4:
                bisector = n1_world
            bisector = bisector / (np.linalg.norm(bisector) + 1e-8)

            z_ee = -bisector
            z_ee = z_ee / (np.linalg.norm(z_ee) + 1e-8)

            y_ee = np.cross(tn, z_ee)
            if np.linalg.norm(y_ee) < 1e-4:
                aux = np.array([1.0, 0.0, 0.0], dtype=np.float32)
                y_ee = np.cross(aux, z_ee)
            y_ee = y_ee / (np.linalg.norm(y_ee) + 1e-8)
            x_ee = np.cross(y_ee, z_ee)
            x_ee = x_ee / (np.linalg.norm(x_ee) + 1e-8)
            R_world_ee = np.stack([x_ee, y_ee, z_ee], axis=1)

            # 对角夹持: 接触点沿二等分线向内一定深度 (依据夹爪开口)
            depth = min(0.5 * min(W, H), gripper_max_width * 0.5)
            contact_center_world = p_world_obj + depth * bisector
            approach_clearance = 0.02
            p_world_ee = contact_center_world + approach_clearance * z_ee

            T_world_grasp = _make_se3(R_world_ee, p_world_ee)
            grip_width = min(min(W, H) * 1.05, gripper_max_width)
    else:
        # 若未识别形状, 退化为简单盒子侧面抓取, 假设 5cm 立方体
        size_obj = np.array([0.05, 0.05, 0.05], dtype=np.float32)
        T_world_grasp, grip_width = _grasp_from_side_faces_cube(
            current_se3, size_obj, table_normal_world, gripper_max_width
        )

    pre_grasp, lift_up = _lift_and_pre_grasp(
        T_world_grasp,
        table_normal_world=table_normal_world,
    )
    return pre_grasp, T_world_grasp, lift_up, grip_width


__all__ = [
    "generate_grasp_pose",
]

