#!/usr/bin/env python3
from __future__ import annotations

from typing import List

import numpy as np

from .se3 import SE3, assert_se3


def _rpy_from_R(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    pitch = float(np.arctan2(-R[2, 0], np.sqrt(R[2, 1] ** 2 + R[2, 2] ** 2)))
    roll = float(np.arctan2(R[2, 1], R[2, 2]))
    yaw = float(np.arctan2(R[1, 0], R[0, 0]))
    return np.array([roll, pitch, yaw], dtype=np.float32)


def _ee7_from_T(T: SE3, gripper: float) -> List[float]:
    assert_se3(T)
    rpy = _rpy_from_R(np.asarray(T[:3, :3], dtype=np.float64))
    pos = np.asarray(T[:3, 3], dtype=np.float64)
    return [float(pos[0]), float(pos[1]), float(pos[2]), float(rpy[0]), float(rpy[1]), float(rpy[2]), float(gripper)]


def _interp(start: List[float], end: List[float], n: int) -> List[List[float]]:
    n = max(1, int(n))
    out: List[List[float]] = []
    for i in range(1, n + 1):
        a = i / float(n)
        out.append([(1 - a) * float(s) + a * float(e) for s, e in zip(start[:7], end[:7])])
    return out


class GraspExecutionModule:
    """抓取模块：将 pre_grasp/grasp/lift 转换为 action chunk。"""

    def __init__(self, steps_per_segment: int = 6):
        self.steps_per_segment = int(max(1, steps_per_segment))

    def build_grasp_actions(
        self,
        *,
        ee_state: List[float],
        pre_grasp: SE3,
        grasp: SE3,
        lift_up: SE3,
    ) -> List[List[float]]:
        start = [float(x) for x in ee_state[:7]]
        pre = _ee7_from_T(pre_grasp, gripper=0.0)
        g_open = _ee7_from_T(grasp, gripper=0.0)
        g_close = _ee7_from_T(grasp, gripper=1.0)  # 真机控制节点会触发 grasp()
        lift = _ee7_from_T(lift_up, gripper=1.0)
        actions: List[List[float]] = []
        actions.extend(_interp(start, pre, self.steps_per_segment))
        actions.extend(_interp(pre, g_open, self.steps_per_segment))
        actions.append(g_close)
        actions.extend(_interp(g_close, lift, self.steps_per_segment))
        return actions

