#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .linemod_opencv import CameraIntrinsics, detect_object_by_id
from .se3 import SE3, assert_se3


@dataclass
class LinemodPoseEstimatorConfig:
    template_db: Path
    block_assets_dir: Path
    intrinsics: CameraIntrinsics
    depth_scale: float = 0.001
    match_threshold: float = 80.0
    refine: bool = True

    # 外参：T_base_cam（base/world -> camera）
    T_base_cam: Optional[SE3] = None


class LinemodPoseEstimator:
    """将 OpenCV LINEMOD 检测封装成 BlockStackingManager 需要的接口."""

    def __init__(self, cfg: LinemodPoseEstimatorConfig):
        self.cfg = cfg

        if self.cfg.T_base_cam is not None:
            assert_se3(self.cfg.T_base_cam)

    def estimate_pose_from_rgbd(self, *, rgb: np.ndarray, depth: np.ndarray, object_id: str) -> SE3:
        T_cam_obj = detect_object_by_id(
            image=rgb,
            depth=depth,
            object_id=object_id,
            intrinsics=self.cfg.intrinsics,
            template_db=self.cfg.template_db,
            block_assets_dir=self.cfg.block_assets_dir,
            depth_scale=self.cfg.depth_scale,
            match_threshold=self.cfg.match_threshold,
            refine=self.cfg.refine,
        )
        if T_cam_obj is None:
            raise RuntimeError(f"LINEMOD 匹配失败或低于阈值: {object_id}")
        assert_se3(T_cam_obj)

        if self.cfg.T_base_cam is None:
            # 上层若未提供外参，则默认返回相机系位姿（用于离线调试）。
            return np.asarray(T_cam_obj, dtype=np.float32)

        T_base_obj = np.asarray(self.cfg.T_base_cam, dtype=np.float32) @ np.asarray(T_cam_obj, dtype=np.float32)
        assert_se3(T_base_obj)
        return T_base_obj


__all__ = [
    "LinemodPoseEstimator",
    "LinemodPoseEstimatorConfig",
]

