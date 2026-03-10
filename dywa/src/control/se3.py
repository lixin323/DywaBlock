#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

SE3 = np.ndarray


def assert_se3(T: SE3) -> None:
    T = np.asarray(T)
    if T.shape != (4, 4):
        raise ValueError(f"SE3 必须是 (4,4)，但得到 {T.shape}")
    if not np.isfinite(T).all():
        raise ValueError("SE3 含有 NaN/Inf")
    if not np.allclose(T[3, :], np.array([0, 0, 0, 1], dtype=T.dtype), atol=1e-5):
        raise ValueError("SE3 最后一行必须为 [0,0,0,1]")


def invert_se3(T: SE3) -> SE3:
    assert_se3(T)
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4, dtype=np.float32)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -(R.T @ t)
    return Ti


def pose_from_pos_quat_xyzw(pos: np.ndarray, quat_xyzw: np.ndarray) -> SE3:
    pos = np.asarray(pos, dtype=np.float32).reshape(3)
    q = np.asarray(quat_xyzw, dtype=np.float32).reshape(4)
    x, y, z, w = [float(e) for e in q]
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        raise ValueError("四元数范数过小")
    s = 2.0 / n
    xx, yy, zz = x * x * s, y * y * s, z * z * s
    xy, xz, yz = x * y * s, x * z * s, y * z * s
    wx, wy, wz = w * x * s, w * y * s, w * z * s

    R = np.array(
        [
            [1.0 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1.0 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1.0 - (xx + yy)],
        ],
        dtype=np.float32,
    )
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    T[:3, 3] = pos
    return T


def pos_quat_xyzw_from_pose(T: SE3) -> Tuple[np.ndarray, np.ndarray]:
    assert_se3(T)
    R = np.asarray(T[:3, :3], dtype=np.float64)
    t = np.asarray(T[:3, 3], dtype=np.float32)

    tr = float(np.trace(R))
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s

    q = np.array([x, y, z, w], dtype=np.float32)
    q = q / (np.linalg.norm(q) + 1e-12)
    return t, q


def rotation_angle(R_rel: np.ndarray) -> float:
    R_rel = np.asarray(R_rel, dtype=np.float64).reshape(3, 3)
    trace = np.clip((np.trace(R_rel) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.arccos(trace))


def step_towards(
    T_curr: SE3,
    T_tgt: SE3,
    max_translation: float = 0.01,
    max_rotation_rad: float = 10.0 / 180.0 * np.pi,
) -> SE3:
    """对 SE3 做一个“限幅”步进，用于伺服/闭环控制."""
    assert_se3(T_curr)
    assert_se3(T_tgt)
    Tc = np.asarray(T_curr, dtype=np.float64)
    Tt = np.asarray(T_tgt, dtype=np.float64)

    # translation
    dt = (Tt[:3, 3] - Tc[:3, 3])
    dtn = np.linalg.norm(dt)
    if dtn > max_translation:
        dt = dt / (dtn + 1e-12) * max_translation

    # rotation (use axis-angle from relative R)
    R_rel = Tt[:3, :3] @ Tc[:3, :3].T
    ang = rotation_angle(R_rel)
    if ang < 1e-9:
        R_step = np.eye(3)
    else:
        wx = np.array(
            [
                R_rel[2, 1] - R_rel[1, 2],
                R_rel[0, 2] - R_rel[2, 0],
                R_rel[1, 0] - R_rel[0, 1],
            ],
            dtype=np.float64,
        )
        axis = wx / (np.linalg.norm(wx) + 1e-12)
        ang_step = min(ang, max_rotation_rad)
        K = np.array(
            [
                [0.0, -axis[2], axis[1]],
                [axis[2], 0.0, -axis[0]],
                [-axis[1], axis[0], 0.0],
            ],
            dtype=np.float64,
        )
        R_step = np.eye(3) + np.sin(ang_step) * K + (1.0 - np.cos(ang_step)) * (K @ K)

    Tn = np.eye(4, dtype=np.float32)
    Tn[:3, :3] = (R_step @ Tc[:3, :3]).astype(np.float32)
    Tn[:3, 3] = (Tc[:3, 3] + dt).astype(np.float32)
    return Tn


__all__ = [
    "SE3",
    "assert_se3",
    "invert_se3",
    "pose_from_pos_quat_xyzw",
    "pos_quat_xyzw_from_pose",
    "rotation_angle",
    "step_towards",
]

