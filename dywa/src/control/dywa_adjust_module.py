#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np

from .dywa_model_policy import DywaStudentPolicyConfig, DywaStudentPolicyInterface
from .se3 import SE3, assert_se3


def _rpy_to_R(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = float(np.cos(roll)), float(np.sin(roll))
    cp, sp = float(np.cos(pitch)), float(np.sin(pitch))
    cy, sy = float(np.cos(yaw)), float(np.sin(yaw))
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float32)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float32)
    return (rz @ ry @ rx).astype(np.float32)


def _rpy_from_R(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    pitch = float(np.arctan2(-R[2, 0], np.sqrt(R[2, 1] ** 2 + R[2, 2] ** 2)))
    roll = float(np.arctan2(R[2, 1], R[2, 2]))
    yaw = float(np.arctan2(R[1, 0], R[0, 0]))
    return np.array([roll, pitch, yaw], dtype=np.float32)


def _ee7_to_hand9(ee7: np.ndarray) -> np.ndarray:
    x, y, z, r, p, yw, _g = [float(v) for v in np.asarray(ee7, dtype=np.float64).reshape(-1)[:7].tolist()]
    rot6d = _rpy_to_R(r, p, yw)[:, :2].reshape(6).astype(np.float32)
    return np.asarray([x, y, z, *rot6d.tolist()], dtype=np.float32)


def _ee7_to_robot14(ee7: np.ndarray) -> np.ndarray:
    v7 = np.asarray(ee7, dtype=np.float32).reshape(-1)[:7]
    return np.concatenate([v7, np.zeros((7,), dtype=np.float32)], axis=0).astype(np.float32)


def interpolate_ee_chunk(ee_start: List[float], T_goal: SE3, gripper: float, n: int) -> List[List[float]]:
    x0, y0, z0, r0, p0, yw0, g0 = [float(v) for v in ee_start[:7]]
    rg, pg, ywg = _rpy_from_R(np.asarray(T_goal[:3, :3], dtype=np.float64))
    xg, yg, zg = [float(v) for v in np.asarray(T_goal[:3, 3], dtype=np.float64).tolist()]
    n = max(1, int(n))
    out: List[List[float]] = []
    for i in range(1, n + 1):
        a = i / float(n)
        out.append(
            [
                (1 - a) * x0 + a * xg,
                (1 - a) * y0 + a * yg,
                (1 - a) * z0 + a * zg,
                (1 - a) * r0 + a * float(rg),
                (1 - a) * p0 + a * float(pg),
                (1 - a) * yw0 + a * float(ywg),
                (1 - a) * g0 + a * float(gripper),
            ]
        )
    return out


@dataclass
class DywaAdjustModuleConfig:
    export_dir: Path
    device: str
    block_assets_dir: Path
    chunk_size: int = 20


class DywaPoseAdjustModule:
    """DyWA 姿态调整模块：接收当前位姿/目标位姿，输出调整动作 chunk。"""

    def __init__(self, cfg: DywaAdjustModuleConfig):
        self.cfg = cfg
        self._pc_world = np.zeros((1, 3), dtype=np.float32)
        self._ee7 = np.zeros((7,), dtype=np.float32)
        self._policy = DywaStudentPolicyInterface(
            DywaStudentPolicyConfig(
                export_dir=cfg.export_dir,
                device=cfg.device,
                block_assets_dir=cfg.block_assets_dir,
                get_partial_cloud_world=lambda: self._pc_world,
                get_robot_state=lambda: _ee7_to_robot14(self._ee7),
                get_hand_state=lambda: _ee7_to_hand9(self._ee7),
            )
        )

    def reset_episode(self) -> None:
        self._policy.reset_episode()

    def plan_adjust_chunk(
        self,
        *,
        block_type: str,
        current_T_world_block: SE3,
        target_T_world_block: SE3,
        partial_cloud_world: np.ndarray,
        ee_state: List[float],
    ) -> List[List[float]]:
        assert_se3(current_T_world_block)
        assert_se3(target_T_world_block)
        self._pc_world = np.asarray(partial_cloud_world, dtype=np.float32).reshape(-1, 3)
        self._ee7 = np.asarray(ee_state[:7], dtype=np.float32)
        out = self._policy.compute_action(
            block_type=block_type,
            current_T_world_block=np.asarray(current_T_world_block, dtype=np.float32),
            target_T_world_block=np.asarray(target_T_world_block, dtype=np.float32),
        )
        T_goal = np.asarray(out["ee_target_T_world"], dtype=np.float32)
        assert_se3(T_goal)
        return interpolate_ee_chunk(
            ee_start=ee_state,
            T_goal=T_goal,
            gripper=0.0,  # 调整阶段夹爪保持打开
            n=self.cfg.chunk_size,
        )

