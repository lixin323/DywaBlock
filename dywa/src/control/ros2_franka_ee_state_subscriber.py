#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class Ros2FrankaEeStateSubscriberConfig:
    node_name: str = "dywa_franka_ee_state_subscriber"
    ee_states_topic: str = "/franka/ee_states"

    wait_timeout_sec: float = 5.0


def _rpy_to_R(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = float(np.cos(roll)), float(np.sin(roll))
    cp, sp = float(np.cos(pitch)), float(np.sin(pitch))
    cy, sy = float(np.cos(yaw)), float(np.sin(yaw))
    # R = Rz(yaw) * Ry(pitch) * Rx(roll)
    Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    Ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float32)
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float32)
    return (Rz @ Ry @ Rx).astype(np.float32)


class Ros2FrankaEeStateSubscriber:
    """订阅 `franka_control_node.py` 发布的 `/franka/ee_states`。

    该话题复用 `sensor_msgs/JointState`，但 `position` 字段携带：
    [x, y, z, roll, pitch, yaw, gripper_width]
    （单位：米 / 弧度 / 米）
    """

    def __init__(self, cfg: Ros2FrankaEeStateSubscriberConfig):
        self.cfg = cfg
        self._last: Optional[np.ndarray] = None

        try:
            import rclpy  # type: ignore
            from rclpy.node import Node  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError("未安装/未配置 ROS2(rclpy)，无法订阅 Franka EE 状态") from e

        self._rclpy = rclpy
        self.node: "Node" = Node(cfg.node_name)

        from sensor_msgs.msg import JointState  # type: ignore

        self._sub = self.node.create_subscription(JointState, cfg.ee_states_topic, self._cb, 10)

    def _cb(self, msg) -> None:
        try:
            arr = np.asarray(msg.position, dtype=np.float32).reshape(-1)
            if arr.size >= 7:
                self._last = arr[:7].copy()
        except Exception:
            return

    def wait_for_first(self) -> None:
        end_ns = self.node.get_clock().now().nanoseconds + int(self.cfg.wait_timeout_sec * 1e9)
        while self.node.get_clock().now().nanoseconds < end_ns:
            self._rclpy.spin_once(self.node, timeout_sec=0.05)
            if self._last is not None:
                return
        raise TimeoutError(f"等待 EE 状态超时: {self.cfg.ee_states_topic}")

    def get_hand_state9(self) -> np.ndarray:
        """返回 hand_state (9,)：pos(3)+rot6d(6)。"""
        if self._last is None:
            self.wait_for_first()
        assert self._last is not None
        x, y, z, r, p, yw, _g = [float(v) for v in self._last.tolist()]
        R = _rpy_to_R(r, p, yw)
        rot6d = R[:, :2].reshape(6).astype(np.float32)
        return np.asarray([x, y, z, *rot6d.tolist()], dtype=np.float32)

    def get_robot_state14(self) -> np.ndarray:
        """返回 robot_state (14,)。

        训练里常见 robot_state=pos_vel7/14；真机这里暂时用：
        - 前 7 维：x,y,z,roll,pitch,yaw,gripper_width
        - 后 7 维：先置 0（没有速度/关节信息时的兜底）
        """
        if self._last is None:
            self.wait_for_first()
        assert self._last is not None
        v7 = self._last.astype(np.float32)
        tail = np.zeros((7,), dtype=np.float32)
        return np.concatenate([v7, tail], axis=0).astype(np.float32)


__all__ = [
    "Ros2FrankaEeStateSubscriber",
    "Ros2FrankaEeStateSubscriberConfig",
]

