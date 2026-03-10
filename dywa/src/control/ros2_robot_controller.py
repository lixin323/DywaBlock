#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .se3 import SE3, assert_se3, pos_quat_xyzw_from_pose


@dataclass
class Ros2RobotControllerConfig:
    node_name: str = "dywa_block_stacking_controller"
    base_frame: str = "world"

    # EE 目标位姿：默认用 PoseStamped 发布给下游笛卡尔控制器
    ee_target_topic: str = "/ee_target_pose"

    # 夹爪控制：两种方式二选一（优先 action）
    gripper_action_name: Optional[str] = None  # 例如 "/gripper_controller/gripper_cmd"
    gripper_width_topic: Optional[str] = "/gripper_width"

    # 简单阻塞等待
    settle_time_sec: float = 0.10


class Ros2RobotController:
    """ROS2 机器人控制接口的默认实现.

    特点：
    - `move_ee_to_pose`: 发布 `geometry_msgs/PoseStamped` 到 `ee_target_topic`
    - `set_gripper`: 若存在 `control_msgs/GripperCommand` action 则调用；否则发布 Float64 到 topic
    - `attach_object` / `detach_object`: 默认 no-op（实机/仿真差异太大），你可按需继承实现

    依赖：
    - 运行时需要 `rclpy`, `geometry_msgs`, `std_msgs`；
      如启用 action 夹爪，则需要 `control_msgs`。
    """

    def __init__(self, cfg: Ros2RobotControllerConfig):
        self.cfg = cfg

        try:
            import rclpy  # type: ignore
            from rclpy.node import Node  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError("未安装/未配置 ROS2(rclpy)，无法使用 Ros2RobotController") from e

        self._rclpy = rclpy
        self.node: "Node" = Node(cfg.node_name)

        from geometry_msgs.msg import PoseStamped  # type: ignore
        from std_msgs.msg import Float64  # type: ignore

        self._PoseStamped = PoseStamped
        self._Float64 = Float64

        self._ee_pub = self.node.create_publisher(PoseStamped, cfg.ee_target_topic, 10)
        self._grip_pub = None
        if cfg.gripper_width_topic is not None:
            self._grip_pub = self.node.create_publisher(Float64, cfg.gripper_width_topic, 10)

        self._gripper_action = None
        self._gripper_goal_cls = None
        if cfg.gripper_action_name is not None:
            try:
                from rclpy.action import ActionClient  # type: ignore
                from control_msgs.action import GripperCommand  # type: ignore

                self._gripper_goal_cls = GripperCommand.Goal
                self._gripper_action = ActionClient(self.node, GripperCommand, cfg.gripper_action_name)
            except Exception:
                # 若 control_msgs 不存在则回退 topic
                self._gripper_action = None
                self._gripper_goal_cls = None

    # ------------------------------------------------------------------ #
    # RobotController 接口
    # ------------------------------------------------------------------ #
    def move_ee_to_pose(self, T_world_ee: SE3) -> None:
        assert_se3(T_world_ee)
        pos, quat = pos_quat_xyzw_from_pose(T_world_ee)

        msg = self._PoseStamped()
        msg.header.frame_id = self.cfg.base_frame
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.pose.position.x = float(pos[0])
        msg.pose.position.y = float(pos[1])
        msg.pose.position.z = float(pos[2])
        msg.pose.orientation.x = float(quat[0])
        msg.pose.orientation.y = float(quat[1])
        msg.pose.orientation.z = float(quat[2])
        msg.pose.orientation.w = float(quat[3])

        self._ee_pub.publish(msg)
        self._spin_sleep(self.cfg.settle_time_sec)

    def set_gripper(self, width: float) -> None:
        width = float(width)
        if self._gripper_action is not None and self._gripper_goal_cls is not None:
            goal = self._gripper_goal_cls()
            goal.command.position = width
            goal.command.max_effort = 0.0

            if not self._gripper_action.wait_for_server(timeout_sec=1.0):
                # 回退 topic
                self._publish_gripper_width(width)
                return

            fut = self._gripper_action.send_goal_async(goal)
            self._spin_until_future_complete(fut, timeout_sec=2.0)
            return

        self._publish_gripper_width(width)

    def attach_object(self, block_type: str) -> None:
        _ = block_type
        # 默认 no-op：仿真可以用“吸附/约束”，实机可用“认为抓取成功”的状态机
        return

    def detach_object(self, block_type: str) -> None:
        _ = block_type
        return

    # ------------------------------------------------------------------ #
    # ROS2 helpers
    # ------------------------------------------------------------------ #
    def _publish_gripper_width(self, width: float) -> None:
        if self._grip_pub is None:
            return
        msg = self._Float64()
        msg.data = float(width)
        self._grip_pub.publish(msg)
        self._spin_sleep(self.cfg.settle_time_sec)

    def _spin_sleep(self, seconds: float) -> None:
        # 允许外部没有 rclpy.spin 的情况下也能工作
        end = self.node.get_clock().now().nanoseconds + int(seconds * 1e9)
        while self.node.get_clock().now().nanoseconds < end:
            self._rclpy.spin_once(self.node, timeout_sec=0.01)

    def _spin_until_future_complete(self, future, timeout_sec: float = 2.0) -> None:
        # 兼容 rclpy Future
        start = self.node.get_clock().now().nanoseconds
        timeout_ns = int(timeout_sec * 1e9)
        while not future.done():
            self._rclpy.spin_once(self.node, timeout_sec=0.01)
            if self.node.get_clock().now().nanoseconds - start > timeout_ns:
                break


__all__ = [
    "Ros2RobotController",
    "Ros2RobotControllerConfig",
]

