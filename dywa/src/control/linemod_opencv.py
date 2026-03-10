#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np

from .se3 import SE3


@dataclass
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float

    def K(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )


class OpenCvLinemodDetector:
    """OpenCV LINEMOD 的一个可运行封装（需要 opencv-contrib）.

    设计目标：
    - 对接 BlockStackingManager 的 `estimate_pose(block_type, color) -> T_world_block`
    - 内部通过 `get_rgbd()` 拉取最新彩色/深度帧
    - 用 LINEMOD 得到最佳模板匹配的像素位置，然后用深度反投影得到平移
    - 姿态默认返回单位旋转（你可以在模板数据库里存姿态，或叠加 PnP/ICP）

    注意：
    - 这份实现重点是“接口链路打通 + 可跑”，不是完整的高精度 6D pose pipeline。
    - 如果你已有成熟的 LINEMOD+PnP/ICP，建议直接实现同名方法替换即可。
    """

    def __init__(
        self,
        *,
        get_rgbd: Callable[[], Tuple[np.ndarray, np.ndarray]],
        intrinsics: CameraIntrinsics,
        template_db: Optional[Path] = None,
        depth_scale: float = 0.001,
    ):
        self.get_rgbd = get_rgbd
        self.intr = intrinsics
        self.template_db = Path(template_db) if template_db is not None else None
        self.depth_scale = float(depth_scale)

        self._detector = None
        self._loaded = False

    def _lazy_init(self) -> None:
        if self._detector is not None:
            return
        try:
            import cv2  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError("未安装 opencv-python / opencv-contrib-python") from e

        if not hasattr(cv2, "linemod"):  # pragma: no cover
            raise RuntimeError(
                "当前 OpenCV 不包含 linemod（通常需要安装 opencv-contrib-python）。"
            )

        # 默认 LINEMOD detector
        self._detector = cv2.linemod.getDefaultLINEMOD()

    def _load_templates_if_needed(self) -> None:
        if self._loaded:
            return
        if self.template_db is None:
            # 允许无模板模式：只用深度中心点估计
            self._loaded = True
            return

        self._lazy_init()
        import cv2  # type: ignore

        db = self.template_db
        if not db.exists():
            raise FileNotFoundError(f"LINEMOD 模板库路径不存在: {db}")

        # 约定模板以 OpenCV FileStorage 的 yaml/xml 保存：
        # 每个物体一个文件：{block_type}_{color}.yaml
        # 文件内部结构由用户生成；这里仅做 Detector.read 的尝试。
        loaded_any = False
        for p in sorted(db.glob("*.y*ml")):
            fs = cv2.FileStorage(str(p), cv2.FILE_STORAGE_READ)
            node = fs.getNode("linemod_detector")
            if node.empty():
                fs.release()
                continue
            self._detector.read(node)  # type: ignore[union-attr]
            fs.release()
            loaded_any = True

        if not loaded_any:
            raise RuntimeError(
                f"模板库 {db} 中未发现可读取的 'linemod_detector' 节点。"
            )
        self._loaded = True

    def estimate_pose(self, block_type: str, color: str) -> SE3:
        self._load_templates_if_needed()
        rgb, depth = self.get_rgbd()
        if rgb is None or depth is None:
            raise RuntimeError("get_rgbd() 返回了空帧。")

        # depth 允许 uint16(mm) 或 float32(m)
        depth_m = depth.astype(np.float32)
        if depth_m.max() > 10.0:  # 典型 uint16(mm)
            depth_m = depth_m * self.depth_scale

        # 1) 如果模板可用，尝试 LINEMOD 匹配得到像素位置
        u, v = self._match_pixel(rgb, depth_m, block_type, color)

        # 2) 深度反投影得到平移
        z = float(depth_m[int(v), int(u)])
        if not np.isfinite(z) or z <= 0.0:
            # 回退：找邻域最小非零深度
            z = self._fallback_depth(depth_m, int(u), int(v))

        x = (float(u) - self.intr.cx) / self.intr.fx * z
        y = (float(v) - self.intr.cy) / self.intr.fy * z

        T = np.eye(4, dtype=np.float32)
        T[:3, 3] = np.array([x, y, z], dtype=np.float32)
        return T

    def _fallback_depth(self, depth_m: np.ndarray, u: int, v: int) -> float:
        h, w = depth_m.shape[:2]
        r = 10
        u0, u1 = max(0, u - r), min(w, u + r + 1)
        v0, v1 = max(0, v - r), min(h, v + r + 1)
        patch = depth_m[v0:v1, u0:u1]
        patch = patch[np.isfinite(patch) & (patch > 0)]
        if patch.size == 0:
            raise RuntimeError("深度图中找不到有效深度，无法估计平移。")
        return float(np.median(patch))

    def _match_pixel(
        self,
        rgb: np.ndarray,
        depth_m: np.ndarray,
        block_type: str,
        color: str,
    ) -> Tuple[float, float]:
        # 无模板库：直接用深度最小点/或中心
        if self.template_db is None:
            h, w = depth_m.shape[:2]
            return float(w * 0.5), float(h * 0.5)

        self._lazy_init()
        import cv2  # type: ignore

        # LINEMOD 需要多模态输入，这里用单 RGB 作为简化输入
        # 如果你用 RGB+Depth 的 modality，请在模板生成阶段保持一致。
        sources = [rgb]
        matches = self._detector.match(sources, 80)  # type: ignore[union-attr]
        if matches is None or len(matches) == 0:
            # 回退：中心点
            h, w = depth_m.shape[:2]
            return float(w * 0.5), float(h * 0.5)

        want_id = f"{block_type}_{color}"
        best = None
        for m in matches:
            # OpenCV Python 的 match item 通常含：class_id, template_id, similarity, x, y
            if hasattr(m, "class_id") and str(m.class_id) != want_id:
                continue
            if best is None or float(m.similarity) > float(best.similarity):
                best = m

        if best is None:
            best = max(matches, key=lambda mm: float(mm.similarity))
        return float(best.x), float(best.y)


__all__ = [
    "CameraIntrinsics",
    "OpenCvLinemodDetector",
]

