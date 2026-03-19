#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class Ros2RgbdSubscriberConfig:
    node_name: str = "dywa_rgbd_subscriber"
    rgb_topic: str = "/camera/color/image_raw"
    depth_topic: str = "/camera/depth/image_raw"

    # 若相机发布的是 depth in mm(uint16) 则 scale=0.001；若是 float32(m) 则 scale=1.0
    depth_scale_to_m: float = 0.001

    # 等待首帧超时
    wait_timeout_sec: float = 5.0


class Ros2RgbdSubscriber:
    """订阅 ROS2 Image 话题并缓存最新 RGB/Depth 帧.

    说明：
    - 依赖 `rclpy`, `sensor_msgs`，以及将 ROS Image 转 numpy 的工具。
    - 优先使用 `cv_bridge`；若不可用则仅支持常见 encoding 的简化路径（可能不完整）。
    """

    def __init__(self, cfg: Ros2RgbdSubscriberConfig):
        self.cfg = cfg
        self._rgb: Optional[np.ndarray] = None
        self._depth: Optional[np.ndarray] = None

        try:
            import rclpy  # type: ignore
            from rclpy.node import Node  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError("未安装/未配置 ROS2(rclpy)，无法使用 Ros2RgbdSubscriber") from e

        self._rclpy = rclpy
        self.node: "Node" = Node(cfg.node_name)

        from sensor_msgs.msg import Image  # type: ignore

        self._Image = Image

        # cv_bridge 优先
        self._cv_bridge = None
        try:
            from cv_bridge import CvBridge  # type: ignore

            self._cv_bridge = CvBridge()
        except Exception:
            self._cv_bridge = None

        self._rgb_sub = self.node.create_subscription(Image, cfg.rgb_topic, self._on_rgb, 10)
        self._depth_sub = self.node.create_subscription(Image, cfg.depth_topic, self._on_depth, 10)

    def wait_for_first_frame(self) -> None:
        end_ns = self.node.get_clock().now().nanoseconds + int(self.cfg.wait_timeout_sec * 1e9)
        while self.node.get_clock().now().nanoseconds < end_ns:
            self._rclpy.spin_once(self.node, timeout_sec=0.05)
            if self._rgb is not None and self._depth is not None:
                return
        raise TimeoutError(
            f"等待 RGBD 首帧超时：rgb={self.cfg.rgb_topic}, depth={self.cfg.depth_topic}"
        )

    def get_rgbd(self) -> Tuple[np.ndarray, np.ndarray]:
        """返回 (rgb, depth).

        - rgb: uint8, 形状 (H,W,3)，BGR 或 RGB 取决于相机驱动；上层需保持与模板一致。
        - depth: float32, 形状 (H,W)，单位为米。
        """
        if self._rgb is None or self._depth is None:
            self.wait_for_first_frame()
        assert self._rgb is not None and self._depth is not None
        return self._rgb, self._depth

    # ------------------------------------------------------------------ #
    # callbacks
    # ------------------------------------------------------------------ #
    def _on_rgb(self, msg: "object") -> None:
        arr = self._img_to_numpy(msg, want_depth=False)
        if arr is None:
            return
        self._rgb = arr

    def _on_depth(self, msg: "object") -> None:
        arr = self._img_to_numpy(msg, want_depth=True)
        if arr is None:
            return
        # depth 统一为米
        depth = arr.astype(np.float32)
        if depth.max() > 10.0:  # 多数 uint16(mm)
            depth = depth * float(self.cfg.depth_scale_to_m)
        self._depth = depth

    def _img_to_numpy(self, msg: "object", *, want_depth: bool) -> Optional[np.ndarray]:
        if self._cv_bridge is not None:
            try:
                if want_depth:
                    cv = self._cv_bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
                    return np.asarray(cv)
                cv = self._cv_bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
                return np.asarray(cv)
            except Exception:
                return None

        # 无 cv_bridge 的兜底：只处理最常见格式（不保证全覆盖）
        try:
            data = np.frombuffer(msg.data, dtype=np.uint8)
            h, w = int(msg.height), int(msg.width)
            enc = str(msg.encoding).lower()
            if want_depth:
                if "16uc1" in enc or "mono16" in enc:
                    return data.view(np.uint16).reshape(h, w)
                if "32fc1" in enc:
                    return data.view(np.float32).reshape(h, w)
                return None
            # color
            if "rgb8" in enc or "bgr8" in enc:
                return data.reshape(h, w, 3)
            if "rgba8" in enc or "bgra8" in enc:
                return data.reshape(h, w, 4)[:, :, :3]
            return None
        except Exception:
            return None


__all__ = [
    "Ros2RgbdSubscriber",
    "Ros2RgbdSubscriberConfig",
]

