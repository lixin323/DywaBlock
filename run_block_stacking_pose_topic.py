#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from dywa.src.control.block_stacking_manager import BlockStackingManager
from dywa.src.control.dywa_pose_policy import DywaPoseServoPolicy
from dywa.src.control.block_stacking_manager import SimpleBoxGraspPlanner
from dywa.src.control.dywa_model_policy import DywaStudentPolicyConfig, DywaStudentPolicyInterface
from dywa.src.control.linemod_opencv import CameraIntrinsics, OpenCvLinemodDetector
from dywa.src.control.ros2_rgbd_subscriber import Ros2RgbdSubscriber, Ros2RgbdSubscriberConfig
from dywa.src.control.ros2_robot_controller import Ros2RobotController, Ros2RobotControllerConfig
from dywa.src.control.se3 import SE3, assert_se3
from dywa.src.control.ros2_franka_ee_state_subscriber import (
    Ros2FrankaEeStateSubscriber,
    Ros2FrankaEeStateSubscriberConfig,
)


class PoseTopicLinemodAdapter(OpenCvLinemodDetector):
    """把 OpenCvLinemodDetector 适配为 LinemodDetector(estimate_pose) 接口.

    - 返回 `T_base_obj`（若提供外参）；否则返回 `T_cam_obj`。
    """

    def __init__(
        self,
        *,
        get_rgbd,
        intrinsics: CameraIntrinsics,
        template_db: Optional[Path],
        block_assets_dir: Optional[Path],
        depth_scale: float,
        T_base_cam: Optional[SE3],
    ):
        super().__init__(get_rgbd=get_rgbd, intrinsics=intrinsics, template_db=template_db, depth_scale=depth_scale)
        self.T_base_cam = None if T_base_cam is None else np.asarray(T_base_cam, dtype=np.float32)
        self.block_assets_dir = block_assets_dir
        self._last_partial_cloud_world = None
        if self.T_base_cam is not None:
            assert_se3(self.T_base_cam)

    def estimate_pose(self, block_type: str, color: str) -> SE3:
        object_id = f"{block_type}_{color}"
        # 同时缓存 partial_cloud（world/base 系）
        rgb, depth = self.get_rgbd()
        out = None
        try:
            out = self.detect_and_mask(block_type, color, n_points=1024)
        except Exception:
            out = None
        if out is not None:
            T_cam_obj, _mask, pc_cam = out
            assert_se3(T_cam_obj)
            if self.T_base_cam is not None:
                R = self.T_base_cam[:3, :3]
                t = self.T_base_cam[:3, 3]
                pc_world = (R @ pc_cam.T).T + t
                self._last_partial_cloud_world = pc_world.astype(np.float32, copy=False)
            else:
                self._last_partial_cloud_world = pc_cam.astype(np.float32, copy=False)
        else:
            from dywa.src.control.linemod_opencv import detect_object_by_id

            T_cam_obj = detect_object_by_id(
                image=rgb,
                depth=depth,
                object_id=object_id,
                intrinsics=self.intr,
                template_db=self.template_db,
                block_assets_dir=self.block_assets_dir,
                depth_scale=self.depth_scale,
                match_threshold=80.0,
                refine=True,
            )
            if T_cam_obj is None:
                raise RuntimeError(f"未检测到目标: {object_id}")
            assert_se3(T_cam_obj)

        if self.T_base_cam is None:
            return np.asarray(T_cam_obj, dtype=np.float32)
        T_base_obj = self.T_base_cam @ np.asarray(T_cam_obj, dtype=np.float32)
        assert_se3(T_base_obj)
        return T_base_obj

    def get_last_partial_cloud_world(self) -> np.ndarray:
        if self._last_partial_cloud_world is None:
            return np.zeros((0, 3), dtype=np.float32)
        return np.asarray(self._last_partial_cloud_world, dtype=np.float32)


def _parse_se3_arg(s: str) -> np.ndarray:
    vals = [float(x) for x in s.replace(",", " ").split()]
    if len(vals) != 16:
        raise ValueError("T_base_cam 需要 16 个浮点数（按行展开 4x4）")
    T = np.asarray(vals, dtype=np.float32).reshape(4, 4)
    assert_se3(T)
    return T


def main() -> None:
    p = argparse.ArgumentParser(description="DyWA Block stacking (PoseStamped topic control)")
    p.add_argument("--scene-root", type=Path, default=Path("block_data/SCENEs_400_Goal_Jsons"))
    p.add_argument("--scene-name", type=str, required=True, help="例如 0 / 396（对应 scene-root 下的 JSON 文件名）")
    p.add_argument("--template-db", type=Path, default=Path("block_data/linemod_templates"))
    p.add_argument("--block-assets-dir", type=Path, default=Path("block_data/block_assets"))
    p.add_argument("--cad-root", type=Path, default=Path("block_data/block_assets"), help="抓取规划 CAD 根目录")
    p.add_argument("--dywa-ckpt", type=Path, default=None, help="接入 DyWA 学生 ckpt（last.ckpt 或目录）")
    p.add_argument("--dywa-device", type=str, default="cuda:0")
    p.add_argument("--dywa-use-goal-cloud", action="store_true", help="启用 goal_cloud（需要 CAD，训练时若 use_gpcd 才应打开）")

    p.add_argument("--rgb-topic", type=str, default="/camera/color/image_raw")
    p.add_argument("--depth-topic", type=str, default="/camera/depth/image_raw")
    p.add_argument("--depth-scale", type=float, default=0.001)

    p.add_argument("--fx", type=float, default=500.0)
    p.add_argument("--fy", type=float, default=500.0)
    p.add_argument("--cx", type=float, default=320.0)
    p.add_argument("--cy", type=float, default=240.0)

    p.add_argument("--base-frame", type=str, default="world")
    p.add_argument("--ee-target-topic", type=str, default="/ee_target_pose")
    p.add_argument("--gripper-width-topic", type=str, default="/gripper_width")
    p.add_argument("--settle-time", type=float, default=0.10)
    p.add_argument("--franka-ee-states-topic", type=str, default="/franka/ee_states")
    p.add_argument("--pos-tol-m", type=float, default=0.005)
    p.add_argument("--ang-tol-deg", type=float, default=5.0)
    p.add_argument("--max-adjust-steps", type=int, default=128)

    p.add_argument(
        "--T-base-cam",
        type=_parse_se3_arg,
        default=None,
        help="手眼外参 T_base_cam（4x4 按行展开 16 个数）。不提供则返回相机系位姿用于调试。",
    )

    args = p.parse_args()

    # 1) ROS2 RGBD 订阅器
    rgbd = Ros2RgbdSubscriber(
        Ros2RgbdSubscriberConfig(
            rgb_topic=args.rgb_topic,
            depth_topic=args.depth_topic,
            depth_scale_to_m=float(args.depth_scale),
        )
    )
    rgbd.wait_for_first_frame()

    # 1.5) Franka EE 状态订阅（用于填充 DyWA 输入的 hand_state/robot_state）
    ee_state = Ros2FrankaEeStateSubscriber(
        Ros2FrankaEeStateSubscriberConfig(
            ee_states_topic=str(args.franka_ee_states_topic),
        )
    )
    ee_state.wait_for_first()

    # 2) LINEMOD 识别器（适配到 BlockStackingManager）
    intr = CameraIntrinsics(fx=float(args.fx), fy=float(args.fy), cx=float(args.cx), cy=float(args.cy))
    linemod = PoseTopicLinemodAdapter(
        get_rgbd=rgbd.get_rgbd,
        intrinsics=intr,
        template_db=args.template_db,
        block_assets_dir=args.block_assets_dir,
        depth_scale=float(args.depth_scale),
        T_base_cam=args.T_base_cam,
    )

    # 3) DyWA 策略：若提供 ckpt 则尝试接入 StudentAgentRMA；否则使用伺服策略占位
    if args.dywa_ckpt is not None:
        dywa = DywaStudentPolicyInterface(
            DywaStudentPolicyConfig(
                ckpt_path=args.dywa_ckpt,
                device=str(args.dywa_device),
                use_goal_cloud=bool(args.dywa_use_goal_cloud),
                block_assets_dir=args.block_assets_dir,
                get_partial_cloud_world=linemod.get_last_partial_cloud_world,
                get_hand_state=ee_state.get_hand_state9,
                get_robot_state=ee_state.get_robot_state14,
            )
        )
    else:
        dywa = DywaPoseServoPolicy()

    # 4) 机器人控制（PoseStamped 目标）
    robot = Ros2RobotController(
        Ros2RobotControllerConfig(
            base_frame=str(args.base_frame),
            ee_target_topic=str(args.ee_target_topic),
            gripper_width_topic=str(args.gripper_width_topic),
            settle_time_sec=float(args.settle_time),
        )
    )

    # 5) 运行单场景
    grasp_planner = SimpleBoxGraspPlanner(cad_root=args.cad_root)

    mgr = BlockStackingManager(
        scene_root=args.scene_root,
        linemod=linemod,
        dywa_policy=dywa,
        robot=robot,
        grasp_planner=grasp_planner,
        pos_tol_m=float(args.pos_tol_m),
        ang_tol_rad=float(args.ang_tol_deg) / 180.0 * np.pi,
        max_adjust_steps=int(args.max_adjust_steps),
    )
    mgr.run_scene(args.scene_name)


if __name__ == "__main__":
    main()

