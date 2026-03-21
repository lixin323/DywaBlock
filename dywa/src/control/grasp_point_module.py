#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .grasp_planner import generate_grasp_pose
from .se3 import SE3, assert_se3


@dataclass
class GraspPoseResult:
    pre_grasp: SE3
    grasp: SE3
    lift_up: SE3
    width: float


class GeometricGraspPointModule:
    """几何解析抓取点模块。"""

    def __init__(self, table_normal_world: np.ndarray | None = None):
        if table_normal_world is None:
            table_normal_world = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        self.table_normal_world = np.asarray(table_normal_world, dtype=np.float32).reshape(3)

    def generate(self, *, object_name: str, current_T_world_block: SE3) -> GraspPoseResult:
        assert_se3(current_T_world_block)
        pre, grasp, lift, width = generate_grasp_pose(
            object_id=object_name,
            current_se3=np.asarray(current_T_world_block, dtype=np.float32),
            table_normal_world=self.table_normal_world,
        )
        return GraspPoseResult(
            pre_grasp=np.asarray(pre, dtype=np.float32),
            grasp=np.asarray(grasp, dtype=np.float32),
            lift_up=np.asarray(lift, dtype=np.float32),
            width=float(width),
        )

