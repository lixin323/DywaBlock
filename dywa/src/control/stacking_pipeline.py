#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from .dywa_adjust_module import DywaAdjustModuleConfig, DywaPoseAdjustModule
from .grasp_action_module import GraspExecutionModule
from .grasp_point_module import GeometricGraspPointModule
from .gt_task_reader import BlockTarget, GtTaskReader
from .linemod_opencv import CameraIntrinsics
from .pose_recognition_module import LinemodPoseRecognitionModule, decode_rgb_depth_payload
from .se3 import SE3, assert_se3


@dataclass
class PipelineConfig:
    scene_root: Path
    template_db: Path
    block_assets_dir: Path
    dywa_export_dir: Path
    dywa_device: str = "cuda:0"
    adjust_chunk_size: int = 20
    # 以厘米为单位的目标位置偏置（相对于 GT）
    # DyWA 输入目标位置偏置
    dywa_goal_offset_cm_x: float = 30.0
    dywa_goal_offset_cm_y: float = 0.0
    dywa_goal_offset_cm_z: float = 9.5
    # 抓取后平移/放置目标位置偏置
    grasp_goal_offset_cm_x: float = 30.0
    grasp_goal_offset_cm_y: float = -30.0
    grasp_goal_offset_cm_z: float = 9.5
    # 姿态误差阈值（度）：小于该阈值认为姿态已对齐
    attitude_tol_deg: float = 8.0
    # 位置工作区半宽（米）：只要在该区域内即可触发抓取
    adjust_region_x_m: float = 0.05
    adjust_region_y_m: float = 0.05
    adjust_region_z_m: float = 0.05
    place_tol_m: float = 0.02
    grasp_steps_per_segment: int = 6


@dataclass
class SceneState:
    block_index: int = 0
    phase: str = "ADJUST"  # ADJUST / GRASPED
    place_delta_xyz: np.ndarray | None = None


class BlockStackingPipeline:
    """主流程编排器（server 调用）：GT -> 识别 -> DyWA -> 几何抓取 -> 下一块。"""

    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self._gt = GtTaskReader(cfg.scene_root)
        self._pose = LinemodPoseRecognitionModule(cfg.template_db)
        self._adjust = DywaPoseAdjustModule(
            DywaAdjustModuleConfig(
                export_dir=cfg.dywa_export_dir,
                device=cfg.dywa_device,
                block_assets_dir=cfg.block_assets_dir,
                chunk_size=cfg.adjust_chunk_size,
            )
        )
        self._grasp_point = GeometricGraspPointModule()
        self._grasp_exec = GraspExecutionModule(steps_per_segment=cfg.grasp_steps_per_segment)
        self._scene_states: Dict[int, SceneState] = {}
        self._scene_targets: Dict[int, List[BlockTarget]] = {}

    def _targets_for_scene(self, scene_id: int) -> List[BlockTarget]:
        if scene_id not in self._scene_targets:
            self._scene_targets[scene_id] = self._gt.load_scene(scene_id)
        return self._scene_targets[scene_id]

    @staticmethod
    def _translation_err(T_a: SE3, T_b: SE3) -> float:
        assert_se3(T_a)
        assert_se3(T_b)
        pa = np.asarray(T_a[:3, 3], dtype=np.float32)
        pb = np.asarray(T_b[:3, 3], dtype=np.float32)
        return float(np.linalg.norm(pa - pb))

    @staticmethod
    def _rotation_err_deg(T_a: SE3, T_b: SE3) -> float:
        assert_se3(T_a)
        assert_se3(T_b)
        R_a = np.asarray(T_a[:3, :3], dtype=np.float64)
        R_b = np.asarray(T_b[:3, :3], dtype=np.float64)
        R_rel = R_a.T @ R_b
        cos_theta = (np.trace(R_rel) - 1.0) * 0.5
        cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
        return float(np.degrees(np.arccos(cos_theta)))

    def _is_within_adjust_region(self, T_current: SE3, T_target: SE3) -> bool:
        assert_se3(T_current)
        assert_se3(T_target)
        delta = np.abs(
            np.asarray(T_current[:3, 3], dtype=np.float32)
            - np.asarray(T_target[:3, 3], dtype=np.float32)
        )
        region = np.asarray(
            [
                float(self.cfg.adjust_region_x_m),
                float(self.cfg.adjust_region_y_m),
                float(self.cfg.adjust_region_z_m),
            ],
            dtype=np.float32,
        )
        return bool(np.all(delta <= region))

    @staticmethod
    def _build_place_actions(ee_state: List[float], delta_xyz: np.ndarray, n_steps: int) -> List[List[float]]:
        """抓取后仅做平移放置：保持姿态不变，末端平移到目标位置后张开夹爪。"""
        x0, y0, z0, r0, p0, yw0, _g0 = [float(v) for v in ee_state[:7]]
        dx, dy, dz = [float(v) for v in np.asarray(delta_xyz, dtype=np.float32).reshape(3).tolist()]
        n_steps = max(1, int(n_steps))
        actions: List[List[float]] = []
        for i in range(1, n_steps + 1):
            a = i / float(n_steps)
            actions.append(
                [
                    x0 + a * dx,
                    y0 + a * dy,
                    z0 + a * dz,
                    r0,
                    p0,
                    yw0,
                    1.0,  # 抓取后搬运阶段保持夹爪闭合
                ]
            )
        # 到达目标后张开夹爪释放
        last = actions[-1][:]
        last[6] = 0.0
        actions.append(last)
        return actions

    def _target_with_offset_m(self, T_gt: SE3, *, use_dywa_offset: bool) -> SE3:
        T = np.asarray(T_gt, dtype=np.float32).copy()
        assert_se3(T)
        if use_dywa_offset:
            off_cm = np.asarray(
                [
                    float(self.cfg.dywa_goal_offset_cm_x),
                    float(self.cfg.dywa_goal_offset_cm_y),
                    float(self.cfg.dywa_goal_offset_cm_z),
                ],
                dtype=np.float32,
            )
        else:
            off_cm = np.asarray(
                [
                    float(self.cfg.grasp_goal_offset_cm_x),
                    float(self.cfg.grasp_goal_offset_cm_y),
                    float(self.cfg.grasp_goal_offset_cm_z),
                ],
                dtype=np.float32,
            )
        T[:3, 3] = T[:3, 3] + off_cm / 100.0  # cm -> m
        return T

    def process_request(self, req: Dict[str, Any]) -> Dict[str, Any]:
        scene_id = int(req["scene_id"])
        ee_state = [float(x) for x in req["ee_state"]]
        if len(ee_state) < 7:
            return {"error": "ee_state 需要7维", "actions": [], "scene_done": False}

        state = self._scene_states.setdefault(scene_id, SceneState())
        targets = self._targets_for_scene(scene_id)
        if state.block_index >= len(targets):
            return {"error": "", "actions": [], "scene_done": True, "block_index": state.block_index}

        target = targets[state.block_index]
        target_T_world_block_dywa = self._target_with_offset_m(
            target.target_T_world_block,
            use_dywa_offset=True,
        )
        target_T_world_block_grasp = self._target_with_offset_m(
            target.target_T_world_block,
            use_dywa_offset=False,
        )
        intr = CameraIntrinsics(
            fx=float(req["fx"]),
            fy=float(req["fy"]),
            cx=float(req["cx"]),
            cy=float(req["cy"]),
        )
        T_world_cam = np.asarray(req["T_base_cam"], dtype=np.float32).reshape(4, 4)
        assert_se3(T_world_cam)
        rgb, depth = decode_rgb_depth_payload(
            rgb_jpeg_b64=req["rgb_jpeg_b64"],
            depth_zlib_b64=req["depth_zlib_b64"],
            image_height=int(req["image_height"]),
            image_width=int(req["image_width"]),
        )
        obs = self._pose.recognize(
            object_name=target.object_name,
            rgb=rgb,
            depth=depth,
            intrinsics=intr,
            depth_scale=float(req["depth_scale"]),
            T_world_cam=T_world_cam,
            target_T_world_block=target_T_world_block_dywa,
            n_points=1024,
        )
        pos_err = self._translation_err(obs.T_world_block, target_T_world_block_dywa)
        rot_err_deg = self._rotation_err_deg(obs.T_world_block, target_T_world_block_dywa)
        in_region = self._is_within_adjust_region(obs.T_world_block, target_T_world_block_dywa)

        # 1) ADJUST 阶段：姿态优先；位置只需进入工作区
        if state.phase == "ADJUST":
            att_ok = rot_err_deg <= float(self.cfg.attitude_tol_deg)
            if not (att_ok and in_region):
                actions = self._adjust.plan_adjust_chunk(
                    block_type=target.block_type,
                    current_T_world_block=obs.T_world_block,
                    target_T_world_block=target_T_world_block_dywa,
                    partial_cloud_world=obs.partial_cloud_world,
                    ee_state=ee_state,
                )
            else:
                # 2) 抓取点生成 + 抓取模块执行
                state.place_delta_xyz = (
                    np.asarray(target_T_world_block_grasp[:3, 3], dtype=np.float32)
                    - np.asarray(obs.T_world_block[:3, 3], dtype=np.float32)
                )
                grasp = self._grasp_point.generate(
                    object_name=target.object_name,
                    current_T_world_block=obs.T_world_block,
                )
                actions = self._grasp_exec.build_grasp_actions(
                    ee_state=ee_state,
                    pre_grasp=grasp.pre_grasp,
                    grasp=grasp.grasp,
                    lift_up=grasp.lift_up,
                )
                state.phase = "GRASPED"
            return {
                "error": "",
                "actions": actions,
                "scene_done": False,
                "block_index": state.block_index,
                "phase": state.phase,
                "pos_err_block_m": pos_err,
                "rot_err_deg": rot_err_deg,
                "in_adjust_region": in_region,
                "target_pos_dywa_m": np.asarray(target_T_world_block_dywa[:3, 3], dtype=np.float32).tolist(),
                "target_pos_grasp_m": np.asarray(target_T_world_block_grasp[:3, 3], dtype=np.float32).tolist(),
                "object_name": target.object_name,
            }

        # 3) GRASPED 阶段：抓取后平移到 GT 目标位置，再切下一块
        if state.phase == "GRASPED":
            delta = (
                np.asarray(state.place_delta_xyz, dtype=np.float32).reshape(3)
                if state.place_delta_xyz is not None
                else np.zeros((3,), dtype=np.float32)
            )
            if float(np.linalg.norm(delta)) <= float(self.cfg.place_tol_m):
                actions = [[float(ee_state[0]), float(ee_state[1]), float(ee_state[2]), float(ee_state[3]), float(ee_state[4]), float(ee_state[5]), 0.0]]
            else:
                actions = self._build_place_actions(
                    ee_state=ee_state,
                    delta_xyz=delta,
                    n_steps=self.cfg.grasp_steps_per_segment,
                )
            state.block_index += 1
            state.phase = "ADJUST"
            state.place_delta_xyz = None
            self._adjust.reset_episode()
            return {
                "error": "",
                "actions": actions,
                "scene_done": state.block_index >= len(targets),
                "block_index": state.block_index,
                "phase": state.phase,
                "pos_err_block_m": pos_err,
                "rot_err_deg": rot_err_deg,
                "in_adjust_region": in_region,
                "target_pos_dywa_m": np.asarray(target_T_world_block_dywa[:3, 3], dtype=np.float32).tolist(),
                "target_pos_grasp_m": np.asarray(target_T_world_block_grasp[:3, 3], dtype=np.float32).tolist(),
                "object_name": target.object_name,
            }

        return {"error": f"unknown phase={state.phase}", "actions": [], "scene_done": False}

