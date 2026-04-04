#!/usr/bin/env python3
from __future__ import annotations

import base64
import json
import urllib.error
import urllib.request
import zlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .se3 import SE3, assert_se3


@dataclass
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass
class PoseRecognitionResult:
    object_name: str
    T_world_block: SE3
    partial_cloud_world: np.ndarray
    mask: np.ndarray
    debug: Optional[Dict[str, Any]] = None


class LocalPointCloudRecognitionModule:
    """本地点云识别：颜色分割 + 深度反投影（不调用 FoundationPose 位姿服务）。"""

    @staticmethod
    def _color_ranges_hsv(color_name: str) -> List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]]:
        c = str(color_name).lower()
        table: Dict[str, List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]]] = {
            "red": [((0, 70, 50), (10, 255, 255)), ((170, 70, 50), (179, 255, 255))],
            "orange": [((10, 80, 60), (25, 255, 255))],
            "yellow": [((20, 80, 60), (38, 255, 255))],
            "green": [((40, 60, 40), (90, 255, 255))],
            "blue": [((95, 60, 40), (135, 255, 255))],
        }
        return table.get(c, [])

    @staticmethod
    def _largest_component_mask(mask: np.ndarray) -> np.ndarray:
        n_lbl, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if n_lbl <= 1:
            return mask
        keep = int(np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1)
        return np.where(labels == keep, np.uint8(255), np.uint8(0))

    def _segment_by_color(self, object_name: str, bgr: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        color_name = str(object_name).lower().split("_")[-1]
        ranges = self._color_ranges_hsv(color_name)
        out = np.zeros(bgr.shape[:2], dtype=np.uint8)
        for lo, hi in ranges:
            out = np.maximum(out, cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8)))
        if np.any(out > 0):
            kernel = np.ones((5, 5), dtype=np.uint8)
            out = cv2.morphologyEx(out, cv2.MORPH_OPEN, kernel)
            out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel)
            out = self._largest_component_mask(out)
        return out

    @staticmethod
    def _mask_from_depth(depth_u16: np.ndarray) -> np.ndarray:
        valid = np.where(np.asarray(depth_u16) > 0, np.uint8(255), np.uint8(0))
        if int((valid > 0).sum()) == 0:
            return valid
        n_lbl, labels, stats, _ = cv2.connectedComponentsWithStats(valid, connectivity=8)
        if n_lbl <= 1:
            return valid
        keep = int(np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1)
        return np.where(labels == keep, np.uint8(255), np.uint8(0))

    @staticmethod
    def _partial_cloud_from_mask(
        *,
        depth_u16: np.ndarray,
        mask: np.ndarray,
        intrinsics: CameraIntrinsics,
        depth_scale: float,
        T_world_cam: SE3,
        n_points: int,
    ) -> np.ndarray:
        depth_m = np.asarray(depth_u16, dtype=np.float32) * float(depth_scale)
        ys, xs = np.where(np.asarray(mask) > 0)
        if ys.size == 0:
            return np.zeros((0, 3), dtype=np.float32)

        z = depth_m[ys, xs]
        ok = np.isfinite(z) & (z > 0.0)
        ys = ys[ok]
        xs = xs[ok]
        z = z[ok]
        if z.size == 0:
            return np.zeros((0, 3), dtype=np.float32)

        x = (xs.astype(np.float32) - float(intrinsics.cx)) / float(intrinsics.fx) * z
        y = (ys.astype(np.float32) - float(intrinsics.cy)) / float(intrinsics.fy) * z
        pc_cam = np.stack([x, y, z], axis=1).astype(np.float32)

        n = int(max(1, n_points))
        if pc_cam.shape[0] >= n:
            idx = np.random.choice(pc_cam.shape[0], size=n, replace=False)
        else:
            idx = np.random.choice(pc_cam.shape[0], size=n, replace=True)
        pc_cam = pc_cam[idx]

        Rwc = np.asarray(T_world_cam[:3, :3], dtype=np.float32)
        twc = np.asarray(T_world_cam[:3, 3], dtype=np.float32)
        pc_world = (Rwc @ pc_cam.T).T + twc
        return np.asarray(pc_world, dtype=np.float32)

    def recognize(
        self,
        *,
        object_name: str,
        rgb: np.ndarray,
        depth: np.ndarray,
        intrinsics: CameraIntrinsics,
        depth_scale: float,
        T_world_cam: SE3,
        target_T_world_block: Optional[SE3] = None,
        n_points: int = 1024,
        debug: bool = False,
    ) -> PoseRecognitionResult:
        depth_u16 = np.asarray(depth, dtype=np.uint16)
        bgr = np.asarray(rgb, dtype=np.uint8)
        mask = self._segment_by_color(object_name=object_name, bgr=bgr)
        mask_source = "color"
        if int((mask > 0).sum()) < 32:
            mask = self._mask_from_depth(depth_u16)
            mask_source = "depth_fallback"
        if int((mask > 0).sum()) < 32:
            raise RuntimeError(f"segmentation too small: {object_name}")

        pc_world = self._partial_cloud_from_mask(
            depth_u16=depth_u16,
            mask=mask,
            intrinsics=intrinsics,
            depth_scale=float(depth_scale),
            T_world_cam=np.asarray(T_world_cam, dtype=np.float32),
            n_points=int(n_points),
        )
        if pc_world.shape[0] == 0:
            raise RuntimeError("segmentation success but mask has no valid depth points")

        # 无位姿估计时，使用点云质心 + 目标姿态（若提供）构造 current_T_world_block 占位。
        T_world_block = np.eye(4, dtype=np.float32)
        T_world_block[:3, 3] = np.mean(pc_world, axis=0).astype(np.float32)
        if target_T_world_block is not None:
            T_world_block[:3, :3] = np.asarray(target_T_world_block, dtype=np.float32)[:3, :3]
        assert_se3(T_world_block)

        debug_info = {
            "object_name": str(object_name),
            "score": None,
            "mask_pixels": int((mask > 0).sum()),
            "mask_source": mask_source,
        }
        return PoseRecognitionResult(
            object_name=object_name,
            T_world_block=np.asarray(T_world_block, dtype=np.float32),
            partial_cloud_world=np.asarray(pc_world, dtype=np.float32),
            mask=np.asarray(mask, dtype=np.uint8),
            debug=debug_info if debug else None,
        )


def decode_rgb_depth_payload(
    *,
    rgb_jpeg_b64: str,
    depth_zlib_b64: str,
    image_height: int,
    image_width: int,
) -> Tuple[np.ndarray, np.ndarray]:
    rgb = cv2.imdecode(
        np.frombuffer(base64.b64decode(rgb_jpeg_b64), np.uint8),
        cv2.IMREAD_COLOR,
    )
    if rgb is None or rgb.shape[0] != int(image_height) or rgb.shape[1] != int(image_width):
        raise ValueError("rgb_jpeg 解码失败或尺寸与 image_height/image_width 不一致")
    depth = np.frombuffer(
        zlib.decompress(base64.b64decode(depth_zlib_b64)),
        dtype=np.uint16,
    ).reshape(int(image_height), int(image_width))
        return rgb, depth


class FoundationPoseInferClient:
    """HTTP 调用 FoundationPose `/infer_pose`，返回物体在世界系下的 4x4 位姿。

    与训练侧 ``ReplaceObjectStateWithFoundationPose`` 使用相同 JSON 负载约定。
    ``T_world_block = T_world_cam @ T_cam_obj``。
    """

    def __init__(self, service_url: str = "http://127.0.0.1:18080/infer_pose", timeout_s: float = 8.0):
        self._service_url = str(service_url)
        self._timeout_s = float(timeout_s)

    def infer_T_world_block(
        self,
        *,
        object_name: str,
        rgb_bgr: np.ndarray,
        depth_u16: np.ndarray,
        intrinsics: CameraIntrinsics,
        depth_scale: float,
        T_world_cam: SE3,
    ) -> np.ndarray:
        bgr = np.asarray(rgb_bgr, dtype=np.uint8)
        if bgr.ndim != 3 or bgr.shape[2] != 3:
            raise ValueError("rgb_bgr must be HxWx3 uint8 BGR")
        depth_u16 = np.asarray(depth_u16, dtype=np.uint16)
        if depth_u16.shape[:2] != bgr.shape[:2]:
            raise ValueError("depth shape must match rgb")

        ok, enc = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            raise RuntimeError("cv2.imencode failed for rgb")

        payload = {
            "object_name": str(object_name),
            "image_width": int(bgr.shape[1]),
            "image_height": int(bgr.shape[0]),
            "fx": float(intrinsics.fx),
            "fy": float(intrinsics.fy),
            "cx": float(intrinsics.cx),
            "cy": float(intrinsics.cy),
            "depth_scale": float(depth_scale),
            "rgb_jpeg_b64": base64.b64encode(enc.tobytes()).decode("ascii"),
            "depth_zlib_b64": base64.b64encode(zlib.compress(depth_u16.tobytes())).decode("ascii"),
        }
        req = urllib.request.Request(
            self._service_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout_s) as resp:
                out = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError) as e:
            raise RuntimeError(f"FoundationPose infer_pose request failed: {e}") from e

        if not bool(out.get("ok", False)):
            raise RuntimeError(f"FoundationPose infer_pose ok=false: {out!r}")

        T_cam_obj = np.asarray(out.get("T_cam_obj", []), dtype=np.float32).reshape(4, 4)
        Twc = np.asarray(T_world_cam, dtype=np.float32).reshape(4, 4)
        T_world_obj = Twc @ T_cam_obj
        assert_se3(T_world_obj)
        return T_world_obj.astype(np.float32)

