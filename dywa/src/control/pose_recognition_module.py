#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from .linemod_opencv import CameraIntrinsics, detect_object_mask_pointcloud
from .se3 import SE3, assert_se3


@dataclass
class PoseRecognitionResult:
    object_name: str
    T_world_block: SE3
    partial_cloud_world: np.ndarray
    mask: np.ndarray


class LinemodPoseRecognitionModule:
    """位姿识别模块：LINEMOD + 深度反投影点云。"""

    def __init__(self, template_db: Path):
        self.template_db = Path(template_db)

    @staticmethod
    def _rot_z(rad: float) -> np.ndarray:
        c = float(np.cos(rad))
        s = float(np.sin(rad))
        return np.asarray(
            [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )

    def _symmetry_local_transforms(self, object_name: str) -> List[SE3]:
        """
        Return local-frame symmetry transforms T_obj_sym.
        Candidate in world frame is T_world_obj @ T_obj_sym.
        """
        name = str(object_name).lower()
        out: List[SE3] = []

        def _mk(R: np.ndarray) -> SE3:
            T = np.eye(4, dtype=np.float32)
            T[:3, :3] = np.asarray(R, dtype=np.float32)
            return T

        # Cube: C4 symmetry around local Z.
        if name.startswith("cube"):
            mats = []
            eye = np.eye(3, dtype=np.float32)
            # Proper rotation group of cube (24 elements).
            for perm in ((0, 1, 2), (0, 2, 1), (1, 0, 2),
                         (1, 2, 0), (2, 0, 1), (2, 1, 0)):
                p = eye[:, perm]
                for sx in (-1.0, 1.0):
                    for sy in (-1.0, 1.0):
                        for sz in (-1.0, 1.0):
                            s = np.diag([sx, sy, sz]).astype(np.float32)
                            r = p @ s
                            if np.linalg.det(r) > 0.0:
                                mats.append(r)
            for r in mats:
                out.append(_mk(r))
            return out

        # Cuboid / arch: 180-degree rotational symmetry around local Z.
        if name.startswith("cuboid") or name.startswith("arch"):
            out.append(_mk(np.eye(3, dtype=np.float32)))
            out.append(_mk(self._rot_z(np.pi)))
            return out

        # Triangle prism: 180-degree rotational symmetry around local Z.
        if name.startswith("triangle"):
            out.append(_mk(np.eye(3, dtype=np.float32)))
            out.append(_mk(self._rot_z(np.pi)))
            return out

        # Semi-cylinder: treat 180-degree yaw as equivalent.
        if name.startswith("semi_cylinder"):
            out.append(_mk(np.eye(3, dtype=np.float32)))
            out.append(_mk(self._rot_z(np.pi)))
            return out

        # Cylinder: approximate continuous yaw symmetry using dense discretization.
        if name.startswith("cylinder"):
            n = 36
            for k in range(n):
                out.append(_mk(self._rot_z(2.0 * np.pi * (k / n))))
            return out

        # Default: no symmetry.
        out.append(_mk(np.eye(3, dtype=np.float32)))
        return out

    @staticmethod
    def _rotation_distance(R_a: np.ndarray, R_b: np.ndarray) -> float:
        R_rel = np.asarray(R_a, dtype=np.float32).T @ np.asarray(R_b, dtype=np.float32)
        tr = float(np.trace(R_rel))
        cos_theta = max(-1.0, min(1.0, (tr - 1.0) * 0.5))
        return float(np.arccos(cos_theta))

    def _closest_symmetric_pose_to_target(
        self,
        *,
        object_name: str,
        T_world_block: SE3,
        target_T_world_block: SE3,
    ) -> SE3:
        T_cur = np.asarray(T_world_block, dtype=np.float32)
        T_tgt = np.asarray(target_T_world_block, dtype=np.float32)
        assert_se3(T_cur)
        assert_se3(T_tgt)

        syms = self._symmetry_local_transforms(object_name)
        best_T = T_cur
        best_score = float("inf")
        for T_obj_sym in syms:
            T_cand = T_cur @ np.asarray(T_obj_sym, dtype=np.float32)
            pos_err = float(np.linalg.norm(T_cand[:3, 3] - T_tgt[:3, 3]))
            rot_err = self._rotation_distance(T_cand[:3, :3], T_tgt[:3, :3])
            score = pos_err + 0.05 * rot_err
            if score < best_score:
                best_score = score
                best_T = T_cand
        return np.asarray(best_T, dtype=np.float32)

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
        match_threshold: float = 80.0,
    ) -> PoseRecognitionResult:
        out = detect_object_mask_pointcloud(
            rgb=rgb,
            depth=depth,
            object_name=object_name,
            intrinsics=intrinsics,
            template_db=self.template_db,
            depth_scale=float(depth_scale),
            match_threshold=float(match_threshold),
            n_points=int(n_points),
            roi_radius_px=80,
            depth_band_m=0.03,
        )
        if out is None:
            raise RuntimeError(f"LINEMOD/点云检测失败: {object_name}")
        T_cam_block, mask, pc_cam = out
        assert_se3(T_cam_block)
        assert_se3(T_world_cam)
        T_world_block = np.asarray(T_world_cam, dtype=np.float32) @ np.asarray(T_cam_block, dtype=np.float32)
        if target_T_world_block is not None:
            T_world_block = self._closest_symmetric_pose_to_target(
                object_name=object_name,
                T_world_block=T_world_block,
                target_T_world_block=np.asarray(target_T_world_block, dtype=np.float32),
            )

        pc_cam = np.asarray(pc_cam, dtype=np.float32).reshape(-1, 3)
        Rwc = np.asarray(T_world_cam[:3, :3], dtype=np.float32)
        twc = np.asarray(T_world_cam[:3, 3], dtype=np.float32)
        pc_world = (Rwc @ pc_cam.T).T + twc

        return PoseRecognitionResult(
            object_name=object_name,
            T_world_block=np.asarray(T_world_block, dtype=np.float32),
            partial_cloud_world=np.asarray(pc_world, dtype=np.float32),
            mask=np.asarray(mask),
        )


def decode_rgb_depth_payload(
    *,
    rgb_jpeg_b64: str,
    depth_zlib_b64: str,
    image_height: int,
    image_width: int,
) -> Tuple[np.ndarray, np.ndarray]:
    import base64
    import zlib
    import cv2  # type: ignore

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

