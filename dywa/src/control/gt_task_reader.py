#!/usr/bin/env python3
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from .se3 import SE3


def _pose_from_json(position: List[float], euler_xyz: List[float]) -> SE3:
    if len(position) != 3 or len(euler_xyz) != 3:
        raise ValueError("position/euler 必须是长度3")
    tx, ty, tz = [float(v) for v in position]
    rx, ry, rz = [float(v) for v in euler_xyz]
    cx, sx = float(np.cos(rx)), float(np.sin(rx))
    cy, sy = float(np.cos(ry)), float(np.sin(ry))
    cz, sz = float(np.cos(rz)), float(np.sin(rz))
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float32)
    Ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float32)
    Rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = Rz @ Ry @ Rx
    T[:3, 3] = np.array([tx, ty, tz], dtype=np.float32)
    return T


@dataclass
class BlockTarget:
    order: int
    block_type: str
    color: str
    layer: int
    depend: List[int]
    target_T_world_block: SE3

    @property
    def object_name(self) -> str:
        return f"{self.block_type}_{self.color}"


class GtTaskReader:
    """读取 GT 场景，输出按顺序执行的积木目标列表。"""

    def __init__(self, scene_root: Path):
        self.scene_root = Path(scene_root)

    def load_scene(self, scene_id: int) -> List[BlockTarget]:
        path = self.scene_root / f"{int(scene_id):03d}.json"
        if not path.is_file():
            raise FileNotFoundError(f"scene not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data: Dict[str, Any] = json.load(f)
        blocks: List[BlockTarget] = []
        for row in data.get("blocks", []):
            blocks.append(
                BlockTarget(
                    order=int(row["order"]),
                    block_type=str(row["type"]),
                    color=str(row["color"]),
                    layer=int(row["layer"]),
                    depend=list(row.get("depend", [])),
                    target_T_world_block=np.asarray(
                        _pose_from_json(position=row["position"], euler_xyz=row["euler"]),
                        dtype=np.float32,
                    ),
                )
            )
        return sorted(blocks, key=lambda x: x.order)

