#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="使用 LINEMOD 模板(yml)在图像中匹配并输出物体位姿。"
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=ROOT_DIR / "test.png",
        help="输入 RGB 图像路径。",
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=ROOT_DIR / "block_data" / "linemod_templates-real" / "arch_red.yml",
        help="LINEMOD 模板 yml 路径（如 arch_red.yml）。",
    )
    parser.add_argument(
        "--depth",
        type=Path,
        default=None,
        help="保留参数（当前安全匹配路径不依赖深度）。",
    )
    parser.add_argument("--fx", type=float, default=500.0, help="相机内参 fx。")
    parser.add_argument("--fy", type=float, default=500.0, help="相机内参 fy。")
    parser.add_argument("--cx", type=float, default=None, help="相机内参 cx，默认取图像中心。")
    parser.add_argument("--cy", type=float, default=None, help="相机内参 cy，默认取图像中心。")
    parser.add_argument(
        "--depth-scale",
        type=float,
        default=0.001,
        help="保留参数（当前安全匹配路径不依赖深度）。",
    )
    parser.add_argument(
        "--match-threshold",
        type=float,
        default=0.3,
        help="匹配分数阈值（0~1，越高越严格）。",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=5000,
        help="每个模板最多评估多少候选平移位置（越大越稳但更慢）。",
    )
    return parser


def _quantize_gradient_labels(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    import cv2  # type: ignore

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    ang = (np.arctan2(gy, gx) + 2.0 * np.pi) % (2.0 * np.pi)
    bins = np.floor(ang / (2.0 * np.pi / 8.0)).astype(np.int32) % 8
    return bins.astype(np.uint8), mag


def _load_template_doc(template_path: Path) -> dict:
    try:
        import yaml
    except Exception as exc:
        raise RuntimeError("缺少 PyYAML，请先安装：pip install pyyaml") from exc

    raw = template_path.read_text(encoding="utf-8", errors="ignore")
    lines = [ln for ln in raw.splitlines() if not ln.startswith("%YAML:")]
    text = "\n".join(lines)
    first_doc = text.split("\n---\n")[0]
    data = yaml.safe_load(first_doc)
    if not isinstance(data, dict):
        raise RuntimeError("模板 yml 第一段内容解析失败。")
    return data


def _load_pose_table_cv2(template_path: Path) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    import cv2  # type: ignore

    fs = cv2.FileStorage(str(template_path), cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise RuntimeError(f"无法打开模板文件: {template_path}")

    node = fs.getNode("num_templates")
    num_templates = int(node.real()) if not node.empty() else 0
    max_scan = max(1000, num_templates + 50)
    pose_map: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for tid in range(max_scan):
        r_node = fs.getNode(f"R_{tid}")
        t_node = fs.getNode(f"t_{tid}")
        if r_node.empty() or t_node.empty():
            continue
        R = np.asarray(r_node.mat(), dtype=np.float32)
        t = np.asarray(t_node.mat(), dtype=np.float32).reshape(-1)
        if R.shape == (3, 3) and t.size >= 3:
            pose_map[tid] = (R, t[:3])
    fs.release()
    return pose_map


def _template_entries(template_doc: dict) -> list[tuple[int, int, int, np.ndarray]]:
    out: list[tuple[int, int, int, np.ndarray]] = []
    tps = template_doc.get("template_pyramids", [])
    if not isinstance(tps, list):
        return out

    for item in tps:
        if not isinstance(item, dict):
            continue
        tid = int(item.get("template_id", -1))
        arr = item.get("templates", [])
        if tid < 0 or not isinstance(arr, list):
            continue
        # 取 pyramid_level=0 的第一组（通常是 ColorGradient）
        chosen = None
        for one in arr:
            if isinstance(one, dict) and int(one.get("pyramid_level", 0)) == 0:
                chosen = one
                break
        if chosen is None:
            continue
        w = int(chosen.get("width", 0))
        h = int(chosen.get("height", 0))
        feats = chosen.get("features", [])
        if w <= 0 or h <= 0 or not isinstance(feats, list):
            continue

        parsed = []
        for f in feats:
            if isinstance(f, (list, tuple)) and len(f) >= 3:
                x, y, lb = int(f[0]), int(f[1]), int(f[2]) % 8
                parsed.append((x, y, lb))
        if len(parsed) < 10:
            continue
        out.append((tid, w, h, np.asarray(parsed, dtype=np.int32)))
    return out


def _uniform_subsample(arr: np.ndarray, max_n: int) -> np.ndarray:
    n = arr.shape[0]
    if n <= max_n:
        return arr
    idx = np.linspace(0, n - 1, num=max_n, dtype=np.int32)
    return arr[idx]


def _match_best_template(
    bins: np.ndarray,
    mag: np.ndarray,
    templates: list[tuple[int, int, int, np.ndarray]],
    max_candidates: int,
) -> tuple[int, int, int, float]:
    h_img, w_img = bins.shape[:2]
    mag_valid = mag > 10.0
    ys_by_label = []
    xs_by_label = []
    for lb in range(8):
        ys, xs = np.where((bins == lb) & mag_valid)
        ys_by_label.append(ys.astype(np.int32))
        xs_by_label.append(xs.astype(np.int32))

    best_tid = -1
    best_x = 0
    best_y = 0
    best_score = -1.0

    for tid, w_t, h_t, feats in templates:
        anchor = feats[0]
        ax, ay, al = int(anchor[0]), int(anchor[1]), int(anchor[2])
        ys = ys_by_label[al]
        xs = xs_by_label[al]
        if ys.size == 0:
            continue
        cand = np.stack([xs - ax, ys - ay], axis=1)
        valid = (
            (cand[:, 0] >= 0)
            & (cand[:, 1] >= 0)
            & (cand[:, 0] + w_t < w_img)
            & (cand[:, 1] + h_t < h_img)
        )
        cand = cand[valid]
        if cand.size == 0:
            continue
        cand = _uniform_subsample(cand, max_candidates)
        ox = cand[:, 0:1]
        oy = cand[:, 1:2]

        fx = feats[:, 0].reshape(1, -1)
        fy = feats[:, 1].reshape(1, -1)
        fl = feats[:, 2].reshape(1, -1)

        px = ox + fx
        py = oy + fy
        labels = bins[py, px]
        valid_mag = mag_valid[py, px]
        hit = (labels == fl) & valid_mag
        scores = hit.mean(axis=1)
        i = int(np.argmax(scores))
        s = float(scores[i])
        if s > best_score:
            best_score = s
            best_tid = int(tid)
            best_x = int(cand[i, 0])
            best_y = int(cand[i, 1])

    return best_tid, best_x, best_y, best_score


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if not args.image.exists():
        print(f"[错误] 图像不存在: {args.image}", file=sys.stderr)
        return 2
    if not args.template.exists():
        print(f"[错误] 模板不存在: {args.template}", file=sys.stderr)
        return 2

    import cv2  # type: ignore

    rgb = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if rgb is None:
        print(f"[错误] 无法读取图像: {args.image}", file=sys.stderr)
        return 2

    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    bins, mag = _quantize_gradient_labels(gray)

    template_doc = _load_template_doc(args.template)
    templates = _template_entries(template_doc)
    if not templates:
        print("[错误] 未从 yml 中解析到有效模板特征。", file=sys.stderr)
        return 2

    tid, x, y, score = _match_best_template(
        bins=bins,
        mag=mag,
        templates=templates,
        max_candidates=max(100, int(args.max_candidates)),
    )
    if tid < 0 or score < float(args.match_threshold):
        print(f"[结果] 未匹配到目标，best_score={score:.4f} < threshold={args.match_threshold:.4f}")
        return 1

    pose_map = _load_pose_table_cv2(args.template)
    if tid not in pose_map:
        if not pose_map:
            print("[错误] 模板中未找到任何 R_i/t_i 位姿数据。", file=sys.stderr)
            return 2
        # 极端情况下兜底：若缺该 id，则取最小 id 位姿。
        tid = sorted(pose_map.keys())[0]
    R, t = pose_map[tid]

    T_cam_obj = np.eye(4, dtype=np.float32)
    T_cam_obj[:3, :3] = R
    T_cam_obj[:3, 3] = t

    # 若用户提供了深度和内参，可用匹配点深度粗修正平移 z（不会调用 linemod.match，不会崩溃）
    if args.depth is not None:
        if not args.depth.exists():
            print(f"[错误] 深度图不存在: {args.depth}", file=sys.stderr)
            return 2
        depth = cv2.imread(str(args.depth), cv2.IMREAD_UNCHANGED)
        if depth is None:
            print(f"[错误] 无法读取深度图: {args.depth}", file=sys.stderr)
            return 2
        d = np.asarray(depth, dtype=np.float32)
        if d.size > 0 and np.max(d) > 10.0:
            d = d * float(args.depth_scale)
        h, w = d.shape[:2]
        u = int(np.clip(x, 0, w - 1))
        v = int(np.clip(y, 0, h - 1))
        z = float(d[v, u])
        if np.isfinite(z) and z > 0:
            cx = float(args.cx) if args.cx is not None else (w * 0.5)
            cy = float(args.cy) if args.cy is not None else (h * 0.5)
            fx = float(args.fx)
            fy = float(args.fy)
            t = np.array([(u - cx) / fx * z, (v - cy) / fy * z, z], dtype=np.float32)
            T_cam_obj[:3, 3] = t

    object_name = args.template.stem
    np.set_printoptions(precision=6, suppress=True)
    print(f"[输入] image={args.image}")
    print(f"[输入] template={args.template} (object_name={object_name})")
    print(f"[匹配] template_id={tid}, x={x}, y={y}, score={score:.4f}")
    print("[输出] T_cam_obj (4x4):")
    print(T_cam_obj)
    print("[输出] R (3x3):")
    print(T_cam_obj[:3, :3])
    print("[输出] t (x,y,z) [m]:")
    print(T_cam_obj[:3, 3])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
