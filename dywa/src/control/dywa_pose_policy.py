#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from .se3 import SE3, assert_se3, step_towards


@dataclass
class WorkspaceBounds:
    """简单工作空间约束（世界坐标系下 AABB）."""

    xyz_min: Tuple[float, float, float] = (-0.6, -0.6, 0.0)
    xyz_max: Tuple[float, float, float] = (0.6, 0.6, 0.8)

    def clamp_pose(self, T_world: SE3) -> SE3:
        assert_se3(T_world)
        T = np.asarray(T_world, dtype=np.float32).copy()
        p = T[:3, 3]
        p = np.maximum(p, np.asarray(self.xyz_min, dtype=np.float32))
        p = np.minimum(p, np.asarray(self.xyz_max, dtype=np.float32))
        T[:3, 3] = p
        return T


class DywaPoseServoPolicy:
    """一个“可直接跑”的 DyWA API 替代实现：SE(3) 伺服策略.

    作用：
    - 输入当前/目标的物体位姿
    - 输出一个末端目标位姿（或增量）用于逐步逼近目标

    说明：
    - 这不是训练好的 DyWA 网络，而是一个可运行的默认实现，
      用于把 BlockStackingManager 的“接口链路”跑通。
    - 你后续可以用真实 DyWA 推理结果替换本类，实现同样的接口即可。
    """

    def __init__(
        self,
        *,
        T_block_ee_nominal: Optional[SE3] = None,
        bounds: WorkspaceBounds = WorkspaceBounds(),
        max_translation_per_step: float = 0.01,
        max_rotation_rad_per_step: float = 10.0 / 180.0 * np.pi,
        default_gripper_width: float = 0.05,
    ):
        self.bounds = bounds
        self.max_translation_per_step = float(max_translation_per_step)
        self.max_rotation_rad_per_step = float(max_rotation_rad_per_step)
        self.default_gripper_width = float(default_gripper_width)

        if T_block_ee_nominal is None:
            # 缺省：末端在物体上方 10cm，朝下
            T_block_ee_nominal = np.eye(4, dtype=np.float32)
            T_block_ee_nominal[:3, 3] = np.array([0.0, 0.0, 0.10], dtype=np.float32)
        self.T_block_ee_nominal = np.asarray(T_block_ee_nominal, dtype=np.float32)
        assert_se3(self.T_block_ee_nominal)

    def reset_episode(self) -> None:
        return

    def compute_action(
        self,
        block_type: str,
        current_T_world_block: SE3,
        target_T_world_block: SE3,
    ) -> Dict:
        _ = block_type
        assert_se3(current_T_world_block)
        assert_se3(target_T_world_block)

        # 末端跟随“目标物体位姿”+固定偏置
        T_world_ee_tgt = np.asarray(target_T_world_block, dtype=np.float32) @ self.T_block_ee_nominal

        # 为了产生平滑动作，我们对末端目标做一步步的限幅逼近；
        # 外部 BlockStackingManager 仍然用 LINEMOD 做物体位姿闭环。
        # 这里假设当前末端位姿未知，则直接给出目标（也支持上层替换为真正机器人反馈）。
        T_world_ee_cmd = self.bounds.clamp_pose(T_world_ee_tgt)

        return {
            "ee_target_T_world": T_world_ee_cmd,
            "gripper_width": self.default_gripper_width,
        }


__all__ = [
    "DywaPoseServoPolicy",
    "WorkspaceBounds",
]

