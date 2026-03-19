#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np

from .se3 import SE3, assert_se3, step_towards


def _rot6d_from_R(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=np.float32).reshape(3, 3)
    return R[:, :2].reshape(6).astype(np.float32)


def _quat_xyzw_from_R(R: np.ndarray) -> np.ndarray:
    # Reuse pos_quat_xyzw_from_pose by forming SE3
    from .se3 import pos_quat_xyzw_from_pose

    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = np.asarray(R, dtype=np.float32)
    pos, quat = pos_quat_xyzw_from_pose(T)
    return quat.astype(np.float32)


def _pose9d_from_se3(T: SE3) -> np.ndarray:
    assert_se3(T)
    pos = np.asarray(T[:3, 3], dtype=np.float32).reshape(3)
    rot6d = _rot6d_from_R(T[:3, :3])
    return np.concatenate([pos, rot6d], axis=0).astype(np.float32)


def _pose7d_from_se3(T: SE3) -> np.ndarray:
    assert_se3(T)
    pos = np.asarray(T[:3, 3], dtype=np.float32).reshape(3)
    quat = _quat_xyzw_from_R(T[:3, :3])
    return np.concatenate([pos, quat], axis=0).astype(np.float32)


@dataclass
class DywaStudentPolicyConfig:
    """基于训练脚本 `test_rma.py` 的 StudentAgentRMA 推理适配器配置。

    重要说明：
    - 你当前只给了 `last.ckpt`（训练期 ckpt），它通常只包含 `student` 权重，
      并不包含 PPO actor 的 `policy` 权重与 normalizer 统计。
    - 因此这里实现的是“能把 StudentAgentRMA 跑起来并产生一个用于末端目标的话题控制输出”的最小版，
      输出会偏保守，主要用于把真实模型接入管线，后续可进一步对齐训练时 action 语义。
    """

    ckpt_path: Path
    device: str = "cuda:0"

    # 观测点云点数（与训练时可能不同；此处会做采样/补齐）
    cloud_n: int = 512

    # 官方 abs_goal_1view 通常不启用 goal_cloud；只有训练启用 gpcd 时才需要 True
    use_goal_cloud: bool = False

    # StudentAgentRMA 输出向量的前 6 维作为 (dx,dy,dz,droll,dpitch,dyaw) 小增量（米/弧度）
    delta_scale_pos: float = 0.02
    delta_scale_rot: float = 10.0 / 180.0 * np.pi

    # 末端目标生成：以“目标物体位姿 + 固定偏置”作为 nominal，再叠加模型增量
    T_block_ee_nominal: Optional[SE3] = None

    # 真机观测回调（world/base 坐标系，单位米）
    get_partial_cloud_world: Optional[Callable[[], np.ndarray]] = None  # (N,3)
    get_robot_state: Optional[Callable[[], np.ndarray]] = None  # (14,)
    get_hand_state: Optional[Callable[[], np.ndarray]] = None  # (9,)

    # goal_cloud 生成：CAD 点云采样 + target pose 变换到 world
    block_assets_dir: Optional[Path] = None


class DywaStudentPolicyInterface:
    """实现 `DywaPolicyInterface`：用 StudentAgentRMA 生成末端目标位姿（PoseStamped 链路）。

    输入：
    - block_type/current_T_world_block/target_T_world_block（来自 BlockStackingManager）
    输出：
    - dict: { "ee_target_T_world": (4,4), "gripper_width": float(可选) }
    """

    def __init__(self, cfg: DywaStudentPolicyConfig):
        self.cfg = cfg

        import torch as th  # type: ignore

        self.th = th
        self.device = th.device(cfg.device)

        # 延迟导入：训练侧模型代码在 dywa/exp/train 下
        from dywa.exp.train.distill import StudentAgentRMA  # type: ignore

        # 构造一个尽量通用的 config：只使用训练脚本常见的输入 key
        # 注：action_size 在不同实验里可能不是 20；此处先用 20，若 ckpt 不匹配会在 load 时提示。
        student_cfg = StudentAgentRMA.StudentAgentRMAConfig(
            shapes={
                "goal": 9,  # abs_goal (pose9d)
                "hand_state": 9,
                "robot_state": 14,
                "previous_action": 20,
            },
            state_keys=["goal", "hand_state", "robot_state", "previous_action"],
            batch_size=1,
            max_delay_steps=0,
            without_teacher=True,
            horizon=1,
            action_size=20,
            estimate_level="state",
            use_gpcd=bool(cfg.use_goal_cloud),
        )
        self.student = StudentAgentRMA(student_cfg, writer=None, device=self.device).to(self.device)

        # 加载 ckpt（只加载 student 权重；strict=False 以适配你修改过的结构）
        from dywa.src.train.ckpt import last_ckpt  # type: ignore

        ckpt_file = last_ckpt(str(cfg.ckpt_path))
        if ckpt_file is None:
            raise FileNotFoundError(f"找不到 ckpt: {cfg.ckpt_path}")
        self.student.load(str(ckpt_file), strict=False)
        self.student.eval()

        self._prev_action = np.zeros((20,), dtype=np.float32)

        if cfg.T_block_ee_nominal is None:
            T = np.eye(4, dtype=np.float32)
            T[:3, 3] = np.array([0.0, 0.0, 0.10], dtype=np.float32)
            self.T_block_ee_nominal = T
        else:
            self.T_block_ee_nominal = np.asarray(cfg.T_block_ee_nominal, dtype=np.float32)
            assert_se3(self.T_block_ee_nominal)

    def reset_episode(self) -> None:
        self._prev_action.fill(0.0)

    def _sample_cloud(self, pc: np.ndarray) -> np.ndarray:
        pc = np.asarray(pc, dtype=np.float32).reshape(-1, 3)
        n = int(max(1, self.cfg.cloud_n))
        if pc.shape[0] == 0:
            return np.zeros((n, 3), dtype=np.float32)
        if pc.shape[0] >= n:
            idx = np.random.choice(pc.shape[0], size=n, replace=False)
        else:
            idx = np.random.choice(pc.shape[0], size=n, replace=True)
        return pc[idx].astype(np.float32, copy=False)

    def _load_goal_cloud_world(self, object_id: str, T_world_obj: SE3) -> np.ndarray:
        if self.cfg.block_assets_dir is None:
            return np.zeros((int(self.cfg.cloud_n), 3), dtype=np.float32)
        try:
            import open3d as o3d  # type: ignore
        except Exception:
            return np.zeros((int(self.cfg.cloud_n), 3), dtype=np.float32)

        obj_path = Path(self.cfg.block_assets_dir) / f"{object_id}.obj"
        if not obj_path.exists():
            return np.zeros((int(self.cfg.cloud_n), 3), dtype=np.float32)

        mesh = o3d.io.read_triangle_mesh(str(obj_path))
        if not mesh.has_vertices():
            # 回退 trimesh
            try:
                import trimesh  # type: ignore
            except Exception:
                return np.zeros((int(self.cfg.cloud_n), 3), dtype=np.float32)
            tm = trimesh.load(obj_path, force="mesh")
            if isinstance(tm, trimesh.Scene):
                tm = trimesh.util.concatenate(list(tm.geometry.values()))
            if not isinstance(tm, trimesh.Trimesh) or tm.vertices.size == 0:
                return np.zeros((int(self.cfg.cloud_n), 3), dtype=np.float32)
            mesh = o3d.geometry.TriangleMesh(
                o3d.utility.Vector3dVector(np.asarray(tm.vertices, dtype=np.float64)),
                o3d.utility.Vector3iVector(np.asarray(tm.faces, dtype=np.int32)),
            )

        verts = np.asarray(mesh.vertices, dtype=np.float32)
        mesh.vertices = o3d.utility.Vector3dVector((verts - verts.mean(axis=0)).astype(np.float64))
        pcd = mesh.sample_points_uniformly(number_of_points=int(self.cfg.cloud_n))
        pts_obj = np.asarray(pcd.points, dtype=np.float32)

        assert_se3(T_world_obj)
        R = np.asarray(T_world_obj[:3, :3], dtype=np.float32)
        t = np.asarray(T_world_obj[:3, 3], dtype=np.float32)
        pts_world = (R @ pts_obj.T).T + t
        return pts_world.astype(np.float32, copy=False)

    def compute_action(
        self,
        block_type: str,
        current_T_world_block: SE3,
        target_T_world_block: SE3,
    ) -> Dict:
        object_id = str(block_type)
        assert_se3(current_T_world_block)
        assert_se3(target_T_world_block)

        # 1) 生成 nominal 末端目标（跟随目标物体位姿 + 固定偏置）
        T_nom = np.asarray(target_T_world_block, dtype=np.float32) @ self.T_block_ee_nominal
        assert_se3(T_nom)

        # 2) 构造 student 输入 obs
        goal = _pose9d_from_se3(np.asarray(target_T_world_block, dtype=np.float32))[None, :]
        hand_state = (
            np.asarray(self.cfg.get_hand_state(), dtype=np.float32).reshape(9)[None, :]
            if self.cfg.get_hand_state is not None
            else np.zeros((1, 9), dtype=np.float32)
        )
        robot_state = (
            np.asarray(self.cfg.get_robot_state(), dtype=np.float32).reshape(14)[None, :]
            if self.cfg.get_robot_state is not None
            else np.zeros((1, 14), dtype=np.float32)
        )
        prev_action = self._prev_action[None, :]

        if self.cfg.get_partial_cloud_world is not None:
            pc_world = self._sample_cloud(self.cfg.get_partial_cloud_world())
        else:
            pc_world = np.zeros((int(self.cfg.cloud_n), 3), dtype=np.float32)
        partial_cloud = pc_world[None, :, :]

        goal_cloud = None
        if self.cfg.use_goal_cloud:
            goal_pc_world = self._load_goal_cloud_world(
                object_id=object_id,
                T_world_obj=np.asarray(target_T_world_block, dtype=np.float32),
            )
            goal_cloud = self._sample_cloud(goal_pc_world)[None, :, :]

        th = self.th
        obs = {
            "goal": th.from_numpy(goal).to(self.device),
            "hand_state": th.from_numpy(hand_state).to(self.device),
            "robot_state": th.from_numpy(robot_state).to(self.device),
            "previous_action": th.from_numpy(prev_action).to(self.device),
            "partial_cloud": th.from_numpy(partial_cloud).to(self.device),
        }
        if goal_cloud is not None:
            obs["goal_cloud"] = th.from_numpy(goal_cloud).to(self.device)

        with th.no_grad():
            out = self.student.reset(obs)
            vec = out[0, 0].detach().float().cpu().numpy().reshape(-1)

        # 3) 用输出向量的前 6 维做小增量叠加到 nominal 上
        d = np.zeros((6,), dtype=np.float32)
        if vec.size >= 6:
            d[:] = vec[:6]

        dp = d[:3] * float(self.cfg.delta_scale_pos)
        dr = d[3:6] * float(self.cfg.delta_scale_rot)

        # 构造一个小旋转（用欧拉近似的轴角：直接把 dr 当作 axis-angle 向量）
        ang = float(np.linalg.norm(dr))
        if ang < 1e-9:
            R_delta = np.eye(3, dtype=np.float32)
        else:
            axis = dr / (ang + 1e-12)
            K = np.array(
                [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]],
                dtype=np.float32,
            )
            R_delta = np.eye(3, dtype=np.float32) + np.sin(ang) * K + (1.0 - np.cos(ang)) * (K @ K)

        T_cmd = np.asarray(T_nom, dtype=np.float32).copy()
        T_cmd[:3, :3] = (R_delta @ T_cmd[:3, :3]).astype(np.float32)
        T_cmd[:3, 3] = (T_cmd[:3, 3] + dp).astype(np.float32)
        assert_se3(T_cmd)

        # 4) 限幅平滑（避免抖动）
        T_smooth = step_towards(T_nom, T_cmd, max_translation=0.01, max_rotation_rad=10.0 / 180.0 * np.pi)

        # 更新 previous_action（保持训练结构一致：20维；这里把前6维写回）
        self._prev_action[: min(6, self._prev_action.size)] = d[: min(6, self._prev_action.size)]

        return {
            "ee_target_T_world": T_smooth,
            "gripper_width": 0.05,
        }


__all__ = [
    "DywaStudentPolicyConfig",
    "DywaStudentPolicyInterface",
]

