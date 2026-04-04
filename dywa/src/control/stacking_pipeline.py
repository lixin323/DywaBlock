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
from .pose_recognition_module import (
    CameraIntrinsics,
    FoundationPoseInferClient,
    LocalPointCloudRecognitionModule,
    decode_rgb_depth_payload,
)
from .se3 import SE3, assert_se3


@dataclass
class PipelineConfig:
    scene_root: Path
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
    # server 侧动作安全限幅（单步变化）
    max_action_step_pos_m: float = 0.03
    max_action_step_rot_rad: float = 0.35
    max_action_step_gripper: float = 0.6
    pose_debug: bool = False
    # 完成门控：icp=与 DyWA 相同局部位姿；foundationpose=ADJUST 阶段用 FP 位姿判 att_ok/工作区
    completion_method: str = "icp"
    foundationpose_pose_url: str = "http://127.0.0.1:18080/infer_pose"
    foundationpose_http_timeout_s: float = 8.0


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
        self._pose = LocalPointCloudRecognitionModule()
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
        cm = str(cfg.completion_method).lower()
        self._fp_client: FoundationPoseInferClient | None = None
        if cm in ("foundationpose", "fp"):
            self._fp_client = FoundationPoseInferClient(
                service_url=cfg.foundationpose_pose_url,
                timeout_s=cfg.foundationpose_http_timeout_s,
            )

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

    @staticmethod
    def _clamp_norm(delta: np.ndarray, max_norm: float) -> np.ndarray:
        n = float(np.linalg.norm(delta))
        if n <= max_norm or n < 1e-12:
            return delta
        return delta * (max_norm / n)

    def _apply_action_safety(self, actions: List[List[float]], ee_state: List[float]) -> tuple[List[List[float]], Dict[str, int]]:
        """对 server 输出动作做安全限幅，避免相邻动作变化过大。"""
        if not actions:
            return [], {"clamped_steps": 0}

        max_dp = float(self.cfg.max_action_step_pos_m)
        max_dr = float(self.cfg.max_action_step_rot_rad)
        max_dg = float(self.cfg.max_action_step_gripper)

        prev = np.asarray(ee_state[:7], dtype=np.float32).copy()
        safe_actions: List[List[float]] = []
        clamped_steps = 0

        for a in actions:
            cur = np.asarray(a[:7], dtype=np.float32).copy()
            raw = cur.copy()

            # 位置单步限幅（按向量范数）
            dpos = cur[:3] - prev[:3]
            cur[:3] = prev[:3] + self._clamp_norm(dpos, max_dp)

            # 姿态单步限幅（rpy 增量按向量范数）
            drpy = cur[3:6] - prev[3:6]
            cur[3:6] = prev[3:6] + self._clamp_norm(drpy, max_dr)

            # 夹爪变化限幅 + 范围约束
            dg = float(cur[6] - prev[6])
            if dg > max_dg:
                cur[6] = prev[6] + max_dg
            elif dg < -max_dg:
                cur[6] = prev[6] - max_dg
            cur[6] = float(np.clip(cur[6], 0.0, 1.0))

            if not np.allclose(raw, cur, atol=1e-8):
                clamped_steps += 1

            safe_actions.append([float(x) for x in cur.tolist()])
            prev = cur

        return safe_actions, {"clamped_steps": clamped_steps}

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
            debug=bool(self.cfg.pose_debug),
        )
        if self.cfg.pose_debug and obs.debug is not None:
            print(
                "[POSE DEBUG] "
                f"obj={obs.debug.get('object_name')} "
                f"score={obs.debug.get('score')} "
                f"mask_pixels={obs.debug.get('mask_pixels')}"
            )

        cm = str(self.cfg.completion_method).lower()
        gate_source = "icp"
        fp_gate_error: str | None = None
        T_for_gate = np.asarray(obs.T_world_block, dtype=np.float32)
        if (
            cm in ("foundationpose", "fp")
            and state.phase == "ADJUST"
            and self._fp_client is not None
        ):
            try:
                T_for_gate = self._fp_client.infer_T_world_block(
                    object_name=target.object_name,
                    rgb_bgr=rgb,
                    depth_u16=np.asarray(depth, dtype=np.uint16),
                    intrinsics=intr,
                    depth_scale=float(req["depth_scale"]),
                    T_world_cam=T_world_cam,
                )
                gate_source = "foundationpose"
            except Exception as e:
                fp_gate_error = str(e)
                gate_source = "foundationpose_fallback_icp"
                T_for_gate = np.asarray(obs.T_world_block, dtype=np.float32)

        pos_err = self._translation_err(T_for_gate, target_T_world_block_dywa)
        rot_err_deg = self._rotation_err_deg(T_for_gate, target_T_world_block_dywa)
        in_region = self._is_within_adjust_region(T_for_gate, target_T_world_block_dywa)

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
            actions, safety = self._apply_action_safety(actions, ee_state)
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
                "safety": safety,
                "pose_debug": obs.debug if self.cfg.pose_debug else None,
                "completion_gate_source": gate_source,
                "completion_gate_fp_error": fp_gate_error,
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
            actions, safety = self._apply_action_safety(actions, ee_state)
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
                "safety": safety,
                "pose_debug": obs.debug if self.cfg.pose_debug else None,
                "completion_gate_source": gate_source,
                "completion_gate_fp_error": fp_gate_error,
            }

        return {"error": f"unknown phase={state.phase}", "actions": [], "scene_done": False}

