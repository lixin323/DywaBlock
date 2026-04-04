#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

from .se3 import SE3, assert_se3


# 默认模板库路径：与 block_data/linemod_templates 对应（dywa 包上一级为项目根时有效）
def _default_template_dir() -> Path:
    return Path(__file__).resolve().parents[2].parent / "block_data" / "linemod_templates"


@dataclass
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float


def _match_numeric_attr(match_item: Any, name: str) -> Optional[float]:
    """安全读取 match 条目的数值字段，兼容属性/可调用属性。"""
    try:
        if not hasattr(match_item, name):
            return None
        value = getattr(match_item, name)
        if callable(value):
            value = value()
        return float(value)
    except Exception:
        return None


def _match_similarity_score(match_item: Any) -> Tuple[float, str]:
    """
    提取匹配分数并归一化到越大越好、常见 0~100 的范围。
    兼容不同 OpenCV 绑定字段：
    - similarity / score: 越大越好
    - distance: 越小越好（映射为 100-distance）
    """
    for key in ("similarity", "score"):
        val = _match_numeric_attr(match_item, key)
        if val is None:
            continue
        # 某些实现返回 0~1，统一放大到 0~100
        if 0.0 <= val <= 1.0:
            val = val * 100.0
        return float(val), key

    dist = _match_numeric_attr(match_item, "distance")
    if dist is not None:
        return float(max(0.0, 100.0 - dist)), "distance"

    return 0.0, "missing"

    def K(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )


def _read_detector_from_fs(detector, fs) -> bool:
    """兼容读取不同 OpenCV 版本导出的 LINEMOD detector。"""
    node = fs.getNode("linemod_detector")
    if not node.empty():
        detector.read(node)  # type: ignore[union-attr]
        return True

    for key in ("opencv_linemod_Detector", "linemod_Detector"):
        node = fs.getNode(key)
        if not node.empty():
            detector.read(node)  # type: ignore[union-attr]
            return True

    try:
        root = fs.root()
        if root is not None and not root.empty():
            detector.read(root)  # type: ignore[union-attr]
            return True
    except Exception:
        pass
    return False


def _read_detector_from_file(detector, *, class_id: str, path: Path) -> bool:
    """优先使用 readClasses(file) 读取；失败再回退到 FileStorage 节点读取。"""
    try:
        if hasattr(detector, "readClasses"):
            detector.readClasses([str(class_id)], str(path))
            return True
    except Exception:
        pass

    import cv2  # type: ignore

    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        return False
    ok = _read_detector_from_fs(detector, fs)
    fs.release()
    return ok


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
            class_id = p.stem
            if not _read_detector_from_file(self._detector, class_id=class_id, path=p):
                continue
            loaded_any = True

        if not loaded_any:
            raise RuntimeError(
                f"模板库 {db} 中未发现可读取的 LINEMOD detector。"
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

    def detect_and_mask(
        self,
        block_type: str,
        color: str,
        *,
        n_points: int = 1024,
        roi_radius_px: int = 80,
        depth_band_m: float = 0.03,
    ) -> Tuple[SE3, np.ndarray, np.ndarray]:
        """返回 (T_cam_obj, mask, pc_cam)."""
        self._load_templates_if_needed()
        rgb, depth = self.get_rgbd()
        object_id = f"{block_type}_{color}"
        out = detect_object_mask_pointcloud(
            rgb=rgb,
            depth=depth,
            object_name=object_id,
            intrinsics=self.intr,
            template_db=self.template_db,
            depth_scale=self.depth_scale,
            match_threshold=80.0,
            n_points=int(n_points),
            roi_radius_px=int(roi_radius_px),
            depth_band_m=float(depth_band_m),
        )
        if out is None:
            raise RuntimeError(f"detect_and_mask 失败: {object_id}")
        T_cam_obj, mask, pc_cam, _ = out
        return T_cam_obj, mask, pc_cam

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
            m_score, _ = _match_similarity_score(m)
            b_score, _ = _match_similarity_score(best) if best is not None else (float("-inf"), "none")
            if best is None or m_score > b_score:
                best = m

        if best is None:
            best = max(matches, key=lambda mm: _match_similarity_score(mm)[0])
        return float(best.x), float(best.y)


def detect_object_pose(
    rgb: np.ndarray,
    depth: np.ndarray,
    object_name: str,
    intrinsics: CameraIntrinsics,
    template_db: Optional[Path] = None,
    depth_scale: float = 0.001,
    match_threshold: float = 80.0,
    use_icp: bool = False,
    block_assets_dir: Optional[Path] = None,
) -> Optional[SE3]:
    """基于 LINEMOD 模板匹配 + 可选 ICP 的 6D 位姿估计。

    输入：
        rgb: 彩色图像 (H, W, 3) BGR，与模板训练时一致。
        depth: 深度图，uint16 毫米或 float 米均可，会按 depth_scale 统一为米。
        object_name: 积木名称，与模板文件名一致，如 "cube_red"、"arch_red"。
        intrinsics: 相机内参，用于深度反投影与 ICP。
        template_db: 模板目录，默认 block_data/linemod_templates。
        depth_scale: 若 depth 为 uint16 毫米，则 depth_scale=0.001。
        match_threshold: LINEMOD 相似度阈值，低于则返回 None。
        use_icp: 是否用深度点云做 ICP 精化位姿。
        block_assets_dir: 可选，OBJ 模型目录，仅 use_icp=True 时用于采样模型点云。

    输出：
        物体相对于相机的 4x4 变换 T_cam_obj（OpenCV 系：X 右、Y 下、Z 前），
        封装为 SE3；匹配失败或分值不足时返回 None。
    """
    import cv2  # type: ignore

    if template_db is None:
        template_db = _default_template_dir()
    template_db = Path(template_db)
    yml_path = template_db / f"{object_name}.yml"
    if not yml_path.exists():
        return None

    # 深度统一为 float 米，供反投影与 ICP 使用
    depth_m = np.asarray(depth, dtype=np.float32)
    if depth_m.size == 0:
        return None
    if depth_m.max() > 10.0:
        depth_m = depth_m * depth_scale

    # 1) 加载该积木的 LINEMOD 检测器与位姿表
    detector = _get_linemod_detector()
    if not _read_detector_from_file(detector, class_id=object_name, path=yml_path):
        return None
    fs = cv2.FileStorage(str(yml_path), cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        return None
    nt_node = fs.getNode("num_templates")
    if nt_node.empty():
        fs.release()
        return None
    try:
        num_templates = int(nt_node.real())
    except Exception:
        num_templates = int(nt_node.mat().item())
    poses_R = []
    poses_t = []
    for i in range(num_templates):
        R_node = fs.getNode(f"R_{i}")
        t_node = fs.getNode(f"t_{i}")
        if R_node.empty() or t_node.empty():
            break
        poses_R.append(R_node.mat())
        t_mat = t_node.mat()
        poses_t.append(t_mat.ravel() if t_mat.size > 0 else np.zeros(3, dtype=np.float32))
    fs.release()
    if len(poses_R) == 0:
        return None

    # 2) LINEMOD 匹配：获取最佳模板与像素位置（OpenCV 惯例：u=x, v=y）
    depth_uint16 = _to_linemod_depth_uint16(depth, depth_scale)
    sources_rgbd = [rgb, depth_uint16]
    sources_rgb = [rgb]
    matches = None
    try:
        matches = detector.match(sources_rgbd, match_threshold)  # type: ignore[union-attr]
    except Exception:
        pass
    if matches is None or len(matches) == 0:
        try:
            matches = detector.match(sources_rgb, match_threshold)  # type: ignore[union-attr]
        except Exception:
            pass
    if matches is None or len(matches) == 0:
        return None

    # 只保留当前 object_name 的匹配，并取相似度最高
    want_id = object_name
    best = None
    for m in matches:
        cid = getattr(m, "class_id", None)
        if cid is not None and str(cid) != want_id:
            continue
        sim, _ = _match_similarity_score(m)
        best_sim, _ = _match_similarity_score(best) if best is not None else (float("-inf"), "none")
        if best is None or sim > best_sim:
            best = m
    if best is None:
        best = max(matches, key=lambda mm: _match_similarity_score(mm)[0])
    similarity, _ = _match_similarity_score(best)
    if similarity < match_threshold:
        return None

    template_id = int(getattr(best, "template_id", -1))
    u = float(getattr(best, "x", 0))
    v = float(getattr(best, "y", 0))
    if template_id < 0 or template_id >= len(poses_R):
        template_id = 0

    # 3) 初始位姿：使用模板库中该 template_id 对应的 R、t（物体在相机系下）
    R = np.asarray(poses_R[template_id], dtype=np.float32)
    t = np.asarray(poses_t[template_id], dtype=np.float32).reshape(3)
    # 可选：用当前帧匹配点处的深度反投影修正平移，提升鲁棒性
    h, w = depth_m.shape[:2]
    ui, vi = int(round(u)), int(round(v))
    if 0 <= ui < w and 0 <= vi < h:
        z = float(depth_m[vi, ui])
        if np.isfinite(z) and z > 0:
            t = np.array(
                [
                    (u - intrinsics.cx) / intrinsics.fx * z,
                    (v - intrinsics.cy) / intrinsics.fy * z,
                    z,
                ],
                dtype=np.float32,
            )
        else:
            # 邻域内取有效深度中值再反投影
            z = _fallback_depth_neighborhood(depth_m, ui, vi)
            if z > 0:
                t = np.array(
                    [
                        (u - intrinsics.cx) / intrinsics.fx * z,
                        (v - intrinsics.cy) / intrinsics.fy * z,
                        z,
                    ],
                    dtype=np.float32,
                )

    T_cam_obj = np.eye(4, dtype=np.float32)
    T_cam_obj[:3, :3] = R
    T_cam_obj[:3, 3] = t

    # 4) 可选：使用深度图 + 模型点云做 ICP 精化 6D 位姿
    if use_icp and block_assets_dir is not None:
        T_refined = _refine_pose_icp(
            rgb=rgb,
            depth_m=depth_m,
            T_init=T_cam_obj,
            intrinsics=intrinsics,
            object_name=object_name,
            block_assets_dir=Path(block_assets_dir),
        )
        if T_refined is not None:
            T_cam_obj = T_refined

    assert_se3(T_cam_obj)
    return T_cam_obj


def detect_object_mask_pointcloud(
    rgb: np.ndarray,
    depth: np.ndarray,
    object_name: str,
    intrinsics: CameraIntrinsics,
    *,
    template_db: Optional[Path] = None,
    depth_scale: float = 0.001,
    match_threshold: float = 80.0,
    n_points: int = 1024,
    roi_radius_px: int = 80,
    depth_band_m: float = 0.03,
    debug: bool = False,
) -> Optional[Tuple[SE3, np.ndarray, np.ndarray, Optional[Dict[str, Any]]]]:
    """检测位姿并输出 mask 与目标点云（v1：ROI+深度阈值近似）.

    输出：
      - `T_cam_obj`: (4,4) SE3（OpenCV 相机系：X右Y下Z前，平移单位米）
      - `mask`: (H,W) uint8，目标为 255，背景 0
      - `pc_cam`: (n_points,3) float32，相机系点云，单位米
      - `debug_info`: 调试信息（可选）
    """
    T_cam_obj = detect_object_pose(
        rgb=rgb,
        depth=depth,
        object_name=object_name,
        intrinsics=intrinsics,
        template_db=template_db,
        depth_scale=depth_scale,
        match_threshold=match_threshold,
        use_icp=False,
        block_assets_dir=None,
    )
    if T_cam_obj is None:
        return None

    # 深度统一为米
    depth_m = np.asarray(depth, dtype=np.float32)
    if depth_m.size == 0:
        return None
    if depth_m.max() > 10.0:
        depth_m = depth_m * float(depth_scale)

    import cv2  # type: ignore

    if template_db is None:
        template_db = _default_template_dir()
    template_db = Path(template_db)
    yml_path = template_db / f"{object_name}.yml"
    if not yml_path.exists():
        return None

    # 重新匹配一次以获取匹配点像素 (u,v)
    detector = _get_linemod_detector()
    if not _read_detector_from_file(detector, class_id=object_name, path=yml_path):
        return None
    fs = cv2.FileStorage(str(yml_path), cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        return None
    fs.release()

    depth_uint16 = _to_linemod_depth_uint16(depth, depth_scale)
    sources_rgbd = [rgb, depth_uint16]
    sources_rgb = [rgb]
    matches = None
    try:
        matches = detector.match(sources_rgbd, match_threshold)  # type: ignore[union-attr]
    except Exception:
        matches = None
    if matches is None or len(matches) == 0:
        try:
            matches = detector.match(sources_rgb, match_threshold)  # type: ignore[union-attr]
        except Exception:
            matches = None
    if matches is None or len(matches) == 0:
        return None

    best = None
    for m in matches:
        cid = getattr(m, "class_id", None)
        if cid is not None and str(cid) != object_name:
            continue
        sim, _ = _match_similarity_score(m)
        best_sim, _ = _match_similarity_score(best) if best is not None else (float("-inf"), "none")
        if best is None or sim > best_sim:
            best = m
    if best is None:
        best = max(matches, key=lambda mm: _match_similarity_score(mm)[0])

    h, w = depth_m.shape[:2]
    u = int(round(float(getattr(best, "x", w * 0.5))))
    v = int(round(float(getattr(best, "y", h * 0.5))))
    u = int(np.clip(u, 0, w - 1))
    v = int(np.clip(v, 0, h - 1))
    similarity, similarity_key = _match_similarity_score(best)
    template_id = int(getattr(best, "template_id", -1))

    # ROI + 深度带宽：近似 mask
    r = int(max(1, roi_radius_px))
    u0, u1 = max(0, u - r), min(w, u + r + 1)
    v0, v1 = max(0, v - r), min(h, v + r + 1)
    roi = depth_m[v0:v1, u0:u1]
    valid = roi[np.isfinite(roi) & (roi > 0)]
    if valid.size == 0:
        return None
    z0 = float(np.median(valid))
    band = float(max(1e-4, depth_band_m))

    mask = np.zeros((h, w), dtype=np.uint8)
    roi_mask = (roi > 0) & np.isfinite(roi) & (np.abs(roi - z0) <= band)
    mask[v0:v1, u0:u1] = np.where(roi_mask, np.uint8(255), np.uint8(0))

    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        return None
    z = depth_m[ys, xs].astype(np.float32)
    ok = (z > 0) & np.isfinite(z)
    ys = ys[ok]
    xs = xs[ok]
    z = z[ok]
    if z.size == 0:
        return None

    x = (xs.astype(np.float32) - float(intrinsics.cx)) / float(intrinsics.fx) * z
    y = (ys.astype(np.float32) - float(intrinsics.cy)) / float(intrinsics.fy) * z
    pts = np.stack([x, y, z], axis=1)

    n = int(max(1, n_points))
    if pts.shape[0] >= n:
        idx = np.random.choice(pts.shape[0], size=n, replace=False)
    else:
        idx = np.random.choice(pts.shape[0], size=n, replace=True)
    pts = pts[idx].astype(np.float32, copy=False)

    assert_se3(T_cam_obj)
    debug_info: Optional[Dict[str, Any]] = None
    if debug:
        import base64

        overlay = np.ascontiguousarray(rgb.copy())
        try:
            cv2.rectangle(overlay, (u0, v0), (u1 - 1, v1 - 1), (0, 255, 0), 2)
            cv2.circle(overlay, (u, v), 3, (0, 0, 255), -1)
            cv2.putText(
                overlay,
                f"{object_name} sim={similarity:.1f}",
                (max(0, u0), max(16, v0 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )
            ok_dbg, enc_dbg = cv2.imencode(".jpg", overlay, [cv2.IMWRITE_JPEG_QUALITY, 85])
            overlay_b64 = base64.b64encode(enc_dbg.tobytes()).decode("ascii") if ok_dbg else None
        except Exception:
            overlay_b64 = None

        debug_info = {
            "object_name": str(object_name),
            "similarity": similarity,
            "similarity_key": similarity_key,
            "template_id": template_id,
            "bbox_xyxy": [int(u0), int(v0), int(u1 - 1), int(v1 - 1)],
            "center_xy": [int(u), int(v)],
            "overlay_jpeg_b64": overlay_b64,
        }

    return T_cam_obj, mask, pts, debug_info


def diagnose_linemod_failure(
    *,
    rgb: np.ndarray,
    depth: np.ndarray,
    object_name: str,
    template_db: Optional[Path] = None,
    depth_scale: float = 0.001,
    match_threshold: float = 80.0,
) -> Dict[str, Any]:
    """诊断 LINEMOD 失败原因，返回结构化调试信息。"""
    info: Dict[str, Any] = {
        "object_name": str(object_name),
        "match_threshold": float(match_threshold),
        "reason": "unknown",
    }

    depth_arr = np.asarray(depth)
    finite = np.isfinite(depth_arr) if depth_arr.size > 0 else np.zeros((0,), dtype=bool)
    positive = (depth_arr > 0) if depth_arr.size > 0 else np.zeros((0,), dtype=bool)
    valid = finite & positive if depth_arr.size > 0 else np.zeros((0,), dtype=bool)
    info["depth_dtype"] = str(depth_arr.dtype)
    info["depth_shape"] = list(depth_arr.shape) if depth_arr.ndim >= 2 else []
    info["depth_valid_ratio"] = float(valid.mean()) if valid.size > 0 else 0.0

    if template_db is None:
        template_db = _default_template_dir()
    template_db = Path(template_db)
    yml_path = template_db / f"{object_name}.yml"
    info["template_path"] = str(yml_path)
    info["template_exists"] = bool(yml_path.exists())
    if not yml_path.exists():
        info["reason"] = "template_yml_missing"
        return info

    try:
        detector = _get_linemod_detector()
    except Exception as e:
        info["reason"] = "linemod_detector_unavailable"
        info["detector_error"] = str(e)
        return info

    loaded = False
    try:
        loaded = bool(_read_detector_from_file(detector, class_id=object_name, path=yml_path))
    except Exception as e:
        info["reason"] = "template_read_exception"
        info["template_read_error"] = str(e)
        return info
    info["template_loaded"] = loaded
    if not loaded:
        info["reason"] = "template_read_failed"
        return info

    depth_u16 = _to_linemod_depth_uint16(depth, float(depth_scale))
    info["depth_u16_nonzero_ratio"] = float((depth_u16 > 0).mean()) if depth_u16.size > 0 else 0.0

    def _run_match(sources: list[np.ndarray], name: str) -> Tuple[int, float, str]:
        try:
            ms = detector.match(sources, float(match_threshold))  # type: ignore[union-attr]
        except Exception as e:
            info[f"{name}_error"] = str(e)
            return 0, 0.0, "missing"
        if ms is None:
            return 0, 0.0, "missing"
        best_sim = 0.0
        best_key = "missing"
        count = 0
        for m in ms:
            cid = getattr(m, "class_id", None)
            if cid is not None and str(cid) != object_name:
                continue
            count += 1
            sim, key = _match_similarity_score(m)
            if sim >= best_sim:
                best_sim = sim
                best_key = key
        return count, best_sim, best_key

    rgbd_count, rgbd_best, rgbd_key = _run_match([rgb, depth_u16], "rgbd_match")
    rgb_count, rgb_best, rgb_key = _run_match([rgb], "rgb_match")
    info["rgbd_match_count"] = int(rgbd_count)
    info["rgb_match_count"] = int(rgb_count)
    info["best_similarity"] = float(max(rgbd_best, rgb_best))
    info["similarity_key"] = rgbd_key if rgbd_best >= rgb_best else rgb_key

    if rgbd_count == 0 and rgb_count == 0:
        info["reason"] = "no_matches"
    elif float(info["best_similarity"]) < float(match_threshold):
        info["reason"] = "similarity_below_threshold"
    else:
        info["reason"] = "pose_or_mask_stage_failed"
    return info


def _get_linemod_detector():
    """获取 LINEMOD 检测器（优先 rgbd.linemod，否则 linemod）。"""
    import cv2  # type: ignore

    if not hasattr(cv2, "linemod"):
        raise RuntimeError("当前 OpenCV 不包含 linemod（需 opencv-contrib-python）。")
    if hasattr(cv2, "rgbd") and hasattr(cv2.rgbd, "linemod"):
        return cv2.rgbd.linemod.getDefaultLINEMOD()
    return cv2.linemod.getDefaultLINEMOD()


def _depth_m_to_uint16_mm(depth_m: np.ndarray) -> np.ndarray:
    """将深度从米转为 uint16 毫米，无效处为 0。"""
    depth_mm = np.clip(depth_m * 1000.0, 0, 65535)
    return np.where(depth_m > 0, depth_mm.astype(np.uint16), np.uint16(0))


def _to_linemod_depth_uint16(depth: np.ndarray, depth_scale: float) -> np.ndarray:
    """强制转换为 LINEMOD 需要的 uint16(mm) 深度图。"""
    d = np.asarray(depth)
    if d.size == 0:
        return np.zeros((0, 0), dtype=np.uint16)

    if d.dtype == np.uint16:
        out = d.copy()
        return np.ascontiguousarray(out)

    d = d.astype(np.float32, copy=False)
    d[~np.isfinite(d)] = 0.0
    # 输入可能是米(float)或毫米(float)；这里统一转换到米再转毫米
    if float(np.max(d)) <= 10.0:
        depth_m = d
    else:
        depth_m = d * float(depth_scale)
    return np.ascontiguousarray(_depth_m_to_uint16_mm(depth_m))


def _fallback_depth_neighborhood(depth_m: np.ndarray, u: int, v: int, radius: int = 10) -> float:
    """在 (u,v) 邻域内取有效深度的中值，用于反投影。"""
    h, w = depth_m.shape[:2]
    u0, u1 = max(0, u - radius), min(w, u + radius + 1)
    v0, v1 = max(0, v - radius), min(h, v + radius + 1)
    patch = depth_m[v0:v1, u0:u1]
    valid = patch[np.isfinite(patch) & (patch > 0)]
    if valid.size == 0:
        return 0.0
    return float(np.median(valid))


def _refine_pose_icp(
    rgb: np.ndarray,
    depth_m: np.ndarray,
    T_init: np.ndarray,
    intrinsics: CameraIntrinsics,
    object_name: str,
    block_assets_dir: Path,
) -> Optional[np.ndarray]:
    """使用深度图与 CAD 模型点云做 ICP 对齐，精化物体相对相机的位姿。

    - 从 depth_m 中在物体大致区域（根据 T_init 投影到图像）采样得到观测点云（相机系）。
    - 从 OBJ 模型采样得到物体模型点云，经 T_init 变换到相机系作为源。
    - 运行 point-to-plane ICP，得到精化后的 T_cam_obj。
    """
    try:
        import open3d as o3d  # type: ignore
    except ImportError:
        return None

    obj_path = block_assets_dir / f"{object_name}.obj"
    if not obj_path.exists():
        return None

    # 加载模型并中心化，与 generate_templates 一致
    try:
        mesh = o3d.io.read_triangle_mesh(str(obj_path))
        if not mesh.has_vertices():
            import trimesh  # type: ignore

            tm = trimesh.load(obj_path, force="mesh")
            if isinstance(tm, trimesh.Scene):
                tm = trimesh.util.concatenate(list(tm.geometry.values()))
            if isinstance(tm, trimesh.Trimesh) and tm.vertices.size > 0:
                mesh = o3d.geometry.TriangleMesh(
                    o3d.utility.Vector3dVector(np.asarray(tm.vertices, dtype=np.float64)),
                    o3d.utility.Vector3iVector(np.asarray(tm.faces, dtype=np.int32)),
                )
        if not mesh.has_vertices():
            return None
        verts = np.asarray(mesh.vertices)
        mesh.vertices = o3d.utility.Vector3dVector(verts - verts.mean(axis=0))
        mesh.compute_vertex_normals()
    except Exception:
        return None

    # 模型点云：均匀采样表面
    pcd_model = mesh.sample_points_uniformly(number_of_points=1024)
    model_pts = np.asarray(pcd_model.points, dtype=np.float32)
    # 用 T_init 将物体点从物体系变到相机系（当前估计的观测）
    R_init = T_init[:3, :3]
    t_init = T_init[:3, 3]
    source_pts = (R_init @ model_pts.T).T + t_init

    # 观测点云：从深度图反投影，仅保留物体附近（用 T_init 投影中心到图像，取邻域）
    K = intrinsics.K()
    h, w = depth_m.shape[:2]
    cx, cy = intrinsics.cx, intrinsics.cy
    fx, fy = intrinsics.fx, intrinsics.fy
    # 物体中心在图像上的投影
    t_proj = R_init @ np.array([0, 0, 0]) + t_init
    if t_proj[2] <= 0:
        return None
    uc = fx * t_proj[0] / t_proj[2] + cx
    vc = fy * t_proj[1] / t_proj[2] + cy
    radius_px = 80
    u0 = max(0, int(uc) - radius_px)
    u1 = min(w, int(uc) + radius_px + 1)
    v0 = max(0, int(vc) - radius_px)
    v1 = min(h, int(vc) + radius_px + 1)
    obs_pts = []
    for v in range(v0, v1, 2):
        for u in range(u0, u1, 2):
            z = float(depth_m[v, u])
            if not (np.isfinite(z) and z > 0):
                continue
            x = (u - cx) / fx * z
            y = (v - cy) / fy * z
            obs_pts.append([x, y, z])
    if len(obs_pts) < 20:
        return None
    target_pts = np.asarray(obs_pts, dtype=np.float32)

    # Open3D ICP：source 为模型点（相机系），target 为观测点（相机系），估计从 source 到 target 的变换
    # 即 T_refine 满足 target ≈ T_refine @ source；这里 source 已是相机系，所以 T_refine 是修正量
    source_pcd = o3d.geometry.PointCloud()
    source_pcd.points = o3d.utility.Vector3dVector(source_pts.astype(np.float64))
    target_pcd = o3d.geometry.PointCloud()
    target_pcd.points = o3d.utility.Vector3dVector(target_pts.astype(np.float64))
    source_pcd.estimate_normals()
    target_pcd.estimate_normals()

    result = o3d.pipelines.registration.registration_icp(
        source_pcd,
        target_pcd,
        max_correspondence_distance=0.02,
        init=np.eye(4),
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
    )
    T_refine = np.asarray(result.transformation, dtype=np.float32)
    # 精化后的物体在相机系位姿：T_cam_obj_new = T_refine @ T_cam_obj_old
    T_cam_obj_new = T_refine @ T_init
    assert_se3(T_cam_obj_new)
    return T_cam_obj_new


def _rot_z(rad: float) -> np.ndarray:
    c, s = float(np.cos(rad)), float(np.sin(rad))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)


def _rot_y(rad: float) -> np.ndarray:
    c, s = float(np.cos(rad)), float(np.sin(rad))
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float32)


def _pose_score_by_depth_consistency(
    *,
    model_pts_obj: np.ndarray,
    T_cam_obj: np.ndarray,
    depth_m: np.ndarray,
    intr: CameraIntrinsics,
    step: int = 5,
    max_abs_err_m: float = 0.02,
) -> float:
    """用深度一致性给一个位姿打分（越大越好）。

    做法：将一部分模型点投影到图像，与该像素的深度比较，统计落在阈值内的比例。
    这能用于：
    - 正方体 90° 旋转对称候选的选择
    - 三角柱正反面（180° 翻转）候选的消歧
    """
    assert_se3(T_cam_obj)
    h, w = depth_m.shape[:2]
    R = T_cam_obj[:3, :3].astype(np.float32)
    t = T_cam_obj[:3, 3].astype(np.float32)
    pts_cam = (R @ model_pts_obj.T).T + t

    z = pts_cam[:, 2]
    valid = z > 1e-6
    if not np.any(valid):
        return 0.0
    pts_cam = pts_cam[valid]
    # 下采样
    pts_cam = pts_cam[:: max(1, int(step))]

    u = intr.fx * (pts_cam[:, 0] / pts_cam[:, 2]) + intr.cx
    v = intr.fy * (pts_cam[:, 1] / pts_cam[:, 2]) + intr.cy
    ui = np.round(u).astype(np.int32)
    vi = np.round(v).astype(np.int32)

    inside = (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
    if not np.any(inside):
        return 0.0
    ui = ui[inside]
    vi = vi[inside]
    z_pred = pts_cam[inside, 2]
    z_obs = depth_m[vi, ui].astype(np.float32)
    ok = (z_obs > 0) & np.isfinite(z_obs) & (np.abs(z_obs - z_pred) <= max_abs_err_m)
    if ok.size == 0:
        return 0.0
    return float(np.mean(ok.astype(np.float32)))


def _load_centered_model_points(
    object_id: str,
    block_assets_dir: Path,
    n_points: int = 2048,
) -> Optional[np.ndarray]:
    """加载 OBJ 并中心化，采样点云 (N,3)（物体系）。"""
    try:
        import open3d as o3d  # type: ignore
    except ImportError:
        return None

    obj_path = block_assets_dir / f"{object_id}.obj"
    if not obj_path.exists():
        return None

    try:
        mesh = o3d.io.read_triangle_mesh(str(obj_path))
        if not mesh.has_vertices():
            import trimesh  # type: ignore

            tm = trimesh.load(obj_path, force="mesh")
            if isinstance(tm, trimesh.Scene):
                tm = trimesh.util.concatenate(list(tm.geometry.values()))
            if isinstance(tm, trimesh.Trimesh) and tm.vertices.size > 0:
                mesh = o3d.geometry.TriangleMesh(
                    o3d.utility.Vector3dVector(np.asarray(tm.vertices, dtype=np.float64)),
                    o3d.utility.Vector3iVector(np.asarray(tm.faces, dtype=np.int32)),
                )
        if not mesh.has_vertices():
            return None
        verts = np.asarray(mesh.vertices, dtype=np.float32)
        mesh.vertices = o3d.utility.Vector3dVector((verts - verts.mean(axis=0)).astype(np.float64))
        mesh.compute_vertex_normals()
        pcd = mesh.sample_points_uniformly(number_of_points=int(n_points))
        return np.asarray(pcd.points, dtype=np.float32)
    except Exception:
        return None


def detect_object_by_id(
    image: np.ndarray,
    depth: np.ndarray,
    object_id: str,
    *,
    intrinsics: Optional[CameraIntrinsics] = None,
    template_db: Optional[Path] = None,
    block_assets_dir: Optional[Path] = None,
    depth_scale: float = 0.001,
    match_threshold: float = 80.0,
    refine: bool = True,
) -> Optional[SE3]:
    """多形状识别入口：按 object_id 动态加载模板并输出 6D 位姿（SE3）。

    约定：
    - object_id 与模板文件名/OBJ 文件名一致（如 \"cube_red\"、\"triangle_orange\"）。 
    - 坐标系遵循 OpenCV：X 右、Y 下、Z 前。

    特殊处理：
    - 正方体：处理 90° 旋转对称性。我们对 4 个候选旋转打分（深度一致性/ICP 效果），取最优。
      （如果你在上层有“目标搭建位姿”，建议把它转到相机系并在此处加入“更接近目标”的额外项。）
    - 三角柱：通过深度一致性消除正反面（180° 翻转）误判。
    """
    if intrinsics is None:
        # 默认值只作为兜底；推荐由真实标定值传入
        intrinsics = CameraIntrinsics(fx=500.0, fy=500.0, cx=image.shape[1] * 0.5, cy=image.shape[0] * 0.5)

    if template_db is None:
        template_db = _default_template_dir()

    # 初始位姿（来自 LINEMOD + 平移反投影 + 可选 ICP）
    T0 = detect_object_pose(
        rgb=image,
        depth=depth,
        object_name=object_id,
        intrinsics=intrinsics,
        template_db=template_db,
        depth_scale=depth_scale,
        match_threshold=match_threshold,
        use_icp=False,  # 我们在下面做“带对称性消歧”的 ICP
        block_assets_dir=block_assets_dir,
    )
    if T0 is None:
        return None

    # 深度统一为米，供打分与 ICP 使用
    depth_m = np.asarray(depth, dtype=np.float32)
    if depth_m.max() > 10.0:
        depth_m = depth_m * float(depth_scale)

    # 如果要做对称性/正反面消歧，需要模型点云
    model_pts = None
    if block_assets_dir is not None:
        model_pts = _load_centered_model_points(object_id, Path(block_assets_dir))

    # 1) 正方体：90° 对称候选
    obj_lower = str(object_id).lower()
    is_cube = obj_lower.startswith("cube") or obj_lower.startswith("cuboid") is False and "cube" in obj_lower
    # 注意：本仓库还有 cuboid*，它不是完全的 90° 对称体，这里仅对 cube* 做处理
    is_cube = obj_lower.startswith("cube_") or obj_lower == "cube" or obj_lower.startswith("cube")

    # 2) 三角柱：正反面消歧（此处按文件命名包含 triangle 判断）
    is_tri_prism = "triangle" in obj_lower

    candidates: list[np.ndarray] = [np.asarray(T0, dtype=np.float32)]
    if model_pts is not None:
        R0 = T0[:3, :3].astype(np.float32)
        t0 = T0[:3, 3].astype(np.float32)

        if is_cube:
            # 假设正方体绕“物体局部 Z 轴”90°对称（模板训练时物体中心化，局部轴由模板姿态决定）
            candidates = []
            for k in range(4):
                Rk = R0 @ _rot_z(k * (np.pi / 2.0))
                Tk = np.eye(4, dtype=np.float32)
                Tk[:3, :3] = Rk
                Tk[:3, 3] = t0
                candidates.append(Tk)

        if is_tri_prism:
            # 三角柱常见误判是正反面（等价于绕局部 Y 或 Z 轴 180°翻转）。
            # 这里生成 2 个候选：原始与绕局部 Y 轴旋转 180°。
            c2 = []
            for Tk in candidates:
                Rk = Tk[:3, :3].astype(np.float32)
                t = Tk[:3, 3].astype(np.float32)
                Ta = np.asarray(Tk, dtype=np.float32)
                Tb = np.eye(4, dtype=np.float32)
                Tb[:3, :3] = Rk @ _rot_y(np.pi)
                Tb[:3, 3] = t
                c2.extend([Ta, Tb])
            candidates = c2

    # 候选打分：优先使用“深度一致性”；若 refine=True 再以 ICP 结果进一步筛
    if model_pts is not None and len(candidates) > 1:
        scored = []
        for Tk in candidates:
            s = _pose_score_by_depth_consistency(
                model_pts_obj=model_pts,
                T_cam_obj=Tk,
                depth_m=depth_m,
                intr=intrinsics,
                step=5,
                max_abs_err_m=0.02,
            )
            scored.append((s, Tk))
        scored.sort(key=lambda x: x[0], reverse=True)
        # 取前 2 个进 ICP（节省时间）
        candidates = [t for _, t in scored[:2]]

    # ICP 精炼（毫米级精调的关键步骤）
    if refine and block_assets_dir is not None:
        best_T = None
        best_score = -1.0
        for Tk in candidates:
            Tref = _refine_pose_icp(
                rgb=image,
                depth_m=depth_m,
                T_init=Tk,
                intrinsics=intrinsics,
                object_name=object_id,
                block_assets_dir=Path(block_assets_dir),
            )
            if Tref is None:
                continue
            if model_pts is not None:
                s = _pose_score_by_depth_consistency(
                    model_pts_obj=model_pts,
                    T_cam_obj=Tref,
                    depth_m=depth_m,
                    intr=intrinsics,
                    step=5,
                    max_abs_err_m=0.01,
                )
            else:
                s = 0.0
            if s > best_score:
                best_score = s
                best_T = Tref
        if best_T is not None:
            assert_se3(best_T)
            return best_T

    # 不精炼时返回最优候选（若没有模型点云则直接返回 T0）
    assert_se3(candidates[0])
    return candidates[0]


__all__ = [
    "CameraIntrinsics",
    "OpenCvLinemodDetector",
    "detect_object_pose",
    "detect_object_mask_pointcloud",
    "detect_object_by_id",
]

