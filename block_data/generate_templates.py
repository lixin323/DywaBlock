#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于 OpenCV LINEMOD + Open3D 的积木模板生成脚本。

- 从 block_assets/ 加载 OBJ 模型，在球面上采样视角并渲染 RGB + 深度。
- 使用 cv2.linemod（或 cv2.rgbd）Detector 提取模板并记录每位姿的 R、t。
- 结果保存到 linemod_templates/{block_name}.yml。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# 项目根目录
SCRIPT_DIR = Path(__file__).resolve().parent
BLOCK_ASSETS_DIR = SCRIPT_DIR / "block_assets"
LINEMOD_OUTPUT_DIR = SCRIPT_DIR / "linemod_templates-real"

# 默认相机内参（来自标定 camera matrix）

DEFAULT_FX = 611.023473
DEFAULT_FY = 612.591246
DEFAULT_CX = 330.929700
DEFAULT_CY = 247.915796
DEFAULT_WIDTH = 640
DEFAULT_HEIGHT = 480

#仿真内参
'''
DEFAULT_FX = 160.0      # fx = width / (2 * tan(FOV/2))  
DEFAULT_FY = 160.0      # fy = fx (方形像素)  
DEFAULT_CX = 64.0       # cx = width/2  
DEFAULT_CY = 64.0       # cy = height/2  
DEFAULT_WIDTH = 128  
DEFAULT_HEIGHT = 128
'''
# 球形采样半径（米），物体中心在原点
SPHERE_RADIUS = 0.4
# 视角数量
NUM_VIEWS = 300
# 深度图单位：LINEMOD 常用 uint16 毫米
DEPTH_SCALE_M_TO_MM = 1000.0


def fibonacci_sphere(n: int) -> np.ndarray:
    """在单位球面上均匀采样 n 个点（Fibonacci 球面采样），返回 (n, 3) 的相机位置。"""
    indices = np.arange(n, dtype=np.float64) + 0.5
    phi = np.arccos(1.0 - 2.0 * indices / n)
    theta = np.pi * (1.0 + 5.0**0.5) * indices
    x = np.cos(theta) * np.sin(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(phi)
    return np.stack([x, y, z], axis=1)


def view_matrix_look_at(eye: np.ndarray, center: np.ndarray, up: np.ndarray) -> np.ndarray:
    """构建 OpenGL 风格 look-at 视图矩阵 4x4（相机坐标系：右 x，上 y，-z 朝向场景）。"""
    f = (center - eye).astype(np.float64)
    f = f / (np.linalg.norm(f) + 1e-9)
    s = np.cross(f, up)
    s = s / (np.linalg.norm(s) + 1e-9)
    u = np.cross(s, f)
    R = np.array([[s[0], s[1], s[2]], [u[0], u[1], u[2]], [-f[0], -f[1], -f[2]]], dtype=np.float64)
    t = -R @ eye
    V = np.eye(4)
    V[:3, :3] = R
    V[:3, 3] = t
    return V


def get_linemod_detector():
    """获取 LINEMOD 检测器（优先 rgbd，否则 linemod）。"""
    try:
        import cv2
    except ImportError as e:
        raise RuntimeError("请安装 opencv-python 与 opencv-contrib-python") from e

    if hasattr(cv2, "rgbd") and hasattr(cv2.rgbd, "linemod"):
        return cv2.rgbd.linemod.getDefaultLINEMOD()
    if hasattr(cv2, "linemod"):
        return cv2.linemod.getDefaultLINEMOD()
    raise RuntimeError("当前 OpenCV 未包含 linemod（请安装 opencv-contrib-python）。")


def _load_mtl_base_color(obj_path: Path) -> Optional[np.ndarray]:
    """
    从 MTL 文件读取首个 Kd 颜色。
    优先级:
      1) OBJ 同目录 `<name>.mtl`
      2) `<obj_dir>/MTLs/<name>.mtl`
    """
    candidates = [
        obj_path.with_suffix(".mtl"),
        obj_path.parent / "MTLs" / f"{obj_path.stem}.mtl",
    ]
    for mtl in candidates:
        if not mtl.is_file():
            continue
        try:
            with open(mtl, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    s = line.strip()
                    if s.startswith("Kd "):
                        vals = s.split()[1:4]
                        if len(vals) == 3:
                            rgb = np.asarray([float(vals[0]), float(vals[1]), float(vals[2])], dtype=np.float32)
                            rgb = np.clip(rgb, 0.0, 1.0)
                            return np.asarray([rgb[0], rgb[1], rgb[2], 1.0], dtype=np.float32)
        except Exception:
            continue
    return None


def load_mesh_centered(obj_path: Path):
    """使用 Open3D 或 trimesh 加载 OBJ，平移使几何中心在原点，返回 (mesh, rgba_base_color)。"""
    import open3d as o3d
    base_color = _load_mtl_base_color(obj_path)

    mesh = o3d.io.read_triangle_mesh(str(obj_path))
    if not mesh.has_vertices():
        # Open3D/ASSIMP 常因 MTL 解析失败（如 "niso"）返回空网格，改用 trimesh 再转 Open3D
        try:
            import trimesh
        except ImportError as e:
            raise ValueError(f"模型无顶点且未安装 trimesh，无法回退加载: {obj_path}") from e
        tm = trimesh.load(obj_path, force="mesh")
        if isinstance(tm, trimesh.Scene):
            tm = trimesh.util.concatenate(list(tm.geometry.values()))
        if not isinstance(tm, trimesh.Trimesh) or tm.vertices.size == 0:
            raise ValueError(f"模型无顶点: {obj_path}")
        mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(np.asarray(tm.vertices, dtype=np.float64)),
            o3d.utility.Vector3iVector(np.asarray(tm.faces, dtype=np.int32)),
        )

    # 中心化
    verts = np.asarray(mesh.vertices)
    center = verts.mean(axis=0)
    mesh.vertices = o3d.utility.Vector3dVector(verts - center)
    mesh.compute_vertex_normals()
    return mesh, base_color


def _create_offscreen_renderer(width: int, height: int):
    """创建离屏渲染器，兼容不同 Open3D 版本。"""
    import open3d as o3d

    try:
        return o3d.visualization.rendering.OffscreenRenderer(width, height)
    except Exception:
        pass
    try:
        return o3d.visualization.rendering.OffscreenRenderer(width, height, resource_path="")
    except Exception:
        pass
    raise RuntimeError("当前 Open3D 版本不支持 OffscreenRenderer，请升级 Open3D（>=0.13）。")


def _set_renderer_background(renderer, r: float, g: float, b: float, a: float = 1.0):
    """兼容不同 Open3D 版本的背景设置。"""
    color = np.array([r, g, b, a], dtype=np.float64)
    for target in (getattr(renderer.scene, "scene", None), renderer.scene):
        if target is None:
            continue
        for name in ("set_background", "set_background_color"):
            fn = getattr(target, name, None)
            if callable(fn):
                try:
                    fn(color)
                    return
                except Exception:
                    pass
    # 无可用 API 时忽略，使用默认背景


def render_rgb_depth(
    mesh,
    base_color: Optional[np.ndarray],
    eye: np.ndarray,
    width: int,
    height: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    near: float = 0.01,
    far: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    在指定视角渲染 RGB 和深度图。
    - eye: 相机在世界系下的位置（物体在原点）。
    - 返回 rgb (H,W,3) uint8 BGR，depth_m (H,W) float 米。
    """
    import open3d as o3d

    renderer = _create_offscreen_renderer(width, height)
    try:
        mat = o3d.visualization.rendering.MaterialRecord()
    except Exception:
        mat = o3d.visualization.rendering.Material()
    mat.shader = "defaultLit"
    if hasattr(mat, "base_color"):
        if base_color is None:
            mat.base_color = [0.7, 0.7, 0.7, 1.0]
        else:
            mat.base_color = [
                float(base_color[0]),
                float(base_color[1]),
                float(base_color[2]),
                float(base_color[3]),
            ]
    renderer.scene.add_geometry("mesh", mesh, mat)
    _set_renderer_background(renderer, 1.0, 1.0, 1.0, 1.0)

    center = np.array([0.0, 0.0, 0.0])
    up = np.array([0.0, 1.0, 0.0])
    renderer.scene.camera.look_at(center, eye, up)

    vfov_deg = 2.0 * np.degrees(np.arctan2(height / 2.0, fy))
    try:
        fov_type = o3d.visualization.rendering.Camera.Projection.Perspective
    except Exception:
        fov_type = 0
    try:
        renderer.scene.camera.set_projection(
            fov_type,
            vfov_deg,
            width / height,
            near,
            far,
        )
    except Exception:
        try:
            renderer.setup_camera(
                vfov_deg, center, eye, up,
            )
        except Exception:
            pass

    # 深度：视空间 Z（实际距离）
    try:
        depth_img = renderer.render_to_depth_image(z_in_view_space=True)
    except TypeError:
        depth_img = renderer.render_to_depth_image()
    depth_np = np.asarray(depth_img)
    depth_m = depth_np.astype(np.float64)
    invalid = (depth_m <= 0) | (depth_m >= far)
    depth_m[invalid] = 0.0

    color_img = renderer.render_to_image()
    rgb = np.asarray(color_img)
    if rgb.ndim == 3 and rgb.shape[2] == 4:
        rgb = rgb[:, :, :3]
    rgb = np.ascontiguousarray(rgb[:, :, ::-1])

    del renderer
    return rgb, depth_m


def depth_meters_to_uint16_mm(depth_m: np.ndarray) -> np.ndarray:
    """将深度从米转为 uint16 毫米，无效处为 0。"""
    depth_mm = depth_m * DEPTH_SCALE_M_TO_MM
    depth_mm = np.clip(depth_mm, 0, 65535)
    return np.where(depth_m > 0, depth_mm.astype(np.uint16), np.uint16(0))


def process_one_obj(
    obj_path: Path,
    output_dir: Path,
    num_views: int,
    width: int,
    height: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    sphere_radius: float,
) -> bool:
    """对单个 OBJ 生成 LINEMOD 模板并保存到 output_dir/{block_name}.yml。"""
    import cv2

    block_name = obj_path.stem
    mesh, base_color = load_mesh_centered(obj_path)
    object_id = block_name

    detector = get_linemod_detector()
    poses_R: List[np.ndarray] = []
    poses_t: List[np.ndarray] = []

    eyes = fibonacci_sphere(num_views) * sphere_radius
    up = np.array([0.0, 1.0, 0.0])
    center = np.array([0.0, 0.0, 0.0])

    for i, eye in enumerate(eyes):
        V = view_matrix_look_at(eye, center, up)
        R_cam = V[:3, :3]
        t_cam = V[:3, 3]
        # 物体在相机系下的位姿：物体中心在相机系为 -R_cam^T @ eye = -R_cam.T @ eye
        # 更常用的是“物体到相机”：p_cam = R @ p_obj + t，物体原点 p_obj=0 -> t = 物体中心在相机系
        # 相机看原点，所以物体中心在相机系 = R_cam @ 0 + t_cam = t_cam（因为 view 矩阵是 world to camera，原点在 world 是 0，所以 camera 里是 t_cam）
        # 实际上 view 矩阵是 p_cam = R_cam @ p_world + t_cam，其中 t_cam = -R_cam @ eye，所以原点在相机系为 t_cam。
        t_obj_in_cam = R_cam @ np.array([0.0, 0.0, 0.0]) + t_cam
        R_obj_to_cam = R_cam

        rgb, depth_m = render_rgb_depth(
            mesh, base_color, eye, width, height, fx, fy, cx, cy
        )
        depth_uint16 = depth_meters_to_uint16_mm(depth_m)
        # 物体掩码：深度有效处为物体（255），背景为 0；LINEMOD 部分版本强制要求 object_mask
        object_mask = np.where(depth_uint16 > 0, np.uint8(255), np.uint8(0))

        # LINEMOD addTemplate: (sources, class_id, object_mask) 或 (sources, class_id)
        sources = [rgb, depth_uint16]
        try:
            ret = detector.addTemplate(sources, object_id, object_mask)
        except TypeError:
            try:
                ret = detector.addTemplate(sources, object_id)
            except Exception:
                try:
                    ret = detector.addTemplate([rgb], object_id, object_mask)
                except Exception as e2:
                    print(f"  [警告] 视角 {i} addTemplate 失败: {e2}", file=sys.stderr)
                    continue
        except Exception as e2:
            print(f"  [警告] 视角 {i} addTemplate 失败: {e2}", file=sys.stderr)
            continue
        template_id = ret if isinstance(ret, int) else getattr(ret, "template_id", ret)
        if isinstance(template_id, (tuple, list)):
            template_id = template_id[0]
        if template_id < 0:
            continue
        poses_R.append(R_obj_to_cam.astype(np.float32))
        poses_t.append(t_obj_in_cam.astype(np.float32))

    if len(poses_R) == 0:
        print(f"  [警告] {block_name}: 未成功添加任何模板，跳过保存。", file=sys.stderr)
        return False

    output_dir.mkdir(parents=True, exist_ok=True)
    out_yml = output_dir / f"{block_name}.yml"

    detector_saved = False
    if hasattr(detector, "writeClasses"):
        try:
            detector.writeClasses(str(out_yml))
            detector_saved = True
        except Exception:
            pass
    elif hasattr(detector, "write"):
        fs = cv2.FileStorage(str(out_yml), cv2.FILE_STORAGE_WRITE)
        if fs.isOpened():
            try:
                detector.write(fs)
                detector_saved = True
            except Exception:
                pass
            fs.release()

    if not detector_saved:
        print(
            "  [错误] 当前 OpenCV 的 linemod.Detector 不支持 writeClasses/write，无法持久化模板。",
            file=sys.stderr,
        )
        return False

    fs = cv2.FileStorage(str(out_yml), cv2.FILE_STORAGE_APPEND)
    if not fs.isOpened():
        print(f"  [错误] detector 已保存，但无法追加位姿到 {out_yml}", file=sys.stderr)
        return False
    # 保存位姿映射：template_id -> R(3x3), t(3,) 供 6D 位姿恢复
    fs.write("object_id", object_id)
    fs.write("num_templates", len(poses_R))
    for i in range(len(poses_R)):
        fs.write(f"R_{i}", poses_R[i])
        fs.write(f"t_{i}", poses_t[i])
    fs.release()
    print(f"  已保存: {out_yml} (模板数 {len(poses_R)})")
    return True


def find_obj_files(root: Path, exclude_dirs: Tuple[str, ...] = ("dywa_processed",)) -> List[Path]:
    """遍历 root 下所有 .obj，可排除指定子目录名。"""
    out = []
    for p in root.rglob("*.obj"):
        if any(ex in p.parts for ex in exclude_dirs):
            continue
        out.append(p)
    return sorted(out)


def main():
    parser = argparse.ArgumentParser(description="LINEMOD 积木模板生成（Open3D 渲染 + OpenCV LINEMOD）")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=BLOCK_ASSETS_DIR,
        help="OBJ 模型根目录",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=LINEMOD_OUTPUT_DIR,
        help="模板输出目录",
    )
    parser.add_argument(
        "--num-views",
        type=int,
        default=NUM_VIEWS,
        help="球面采样视角数量",
    )
    parser.add_argument("--width", type=int, default=None, help="渲染宽度")
    parser.add_argument("--height", type=int, default=None, help="渲染高度")
    parser.add_argument("--fx", type=float, default=DEFAULT_FX, help="相机 fx")
    parser.add_argument("--fy", type=float, default=DEFAULT_FY, help="相机 fy")
    parser.add_argument("--cx", type=float, default=DEFAULT_CX, help="相机 cx")
    parser.add_argument("--cy", type=float, default=DEFAULT_CY, help="相机 cy")
    parser.add_argument("--sphere-radius", type=float, default=SPHERE_RADIUS, help="采样球半径（米）")
    parser.add_argument("--no-exclude-processed", action="store_true", help="不排除 dywa_processed 下的 OBJ")
    parser.add_argument("--obj", type=Path, default=None, help="仅处理该 OBJ 文件（相对或绝对路径）")
    args = parser.parse_args()

    width = args.width if args.width is not None else DEFAULT_WIDTH
    height = args.height if args.height is not None else DEFAULT_HEIGHT
    exclude = () if args.no_exclude_processed else ("dywa_processed",)

    if args.obj is not None:
        obj_path = args.obj if args.obj.is_absolute() else args.input_dir / args.obj
        if not obj_path.exists():
            print(f"文件不存在: {obj_path}", file=sys.stderr)
            sys.exit(1)
        obj_files = [obj_path]
    else:
        if not args.input_dir.exists():
            print(f"输入目录不存在: {args.input_dir}", file=sys.stderr)
            sys.exit(1)
        obj_files = find_obj_files(args.input_dir, exclude_dirs=exclude)
        if not obj_files:
            print(f"在 {args.input_dir} 下未找到 OBJ 文件。", file=sys.stderr)
            sys.exit(1)

    print(f"将处理 {len(obj_files)} 个 OBJ，输出目录: {args.output_dir}")
    ok = 0
    for p in obj_files:
        print(f"处理: {p.name}")
        if process_one_obj(
            p,
            args.output_dir,
            num_views=args.num_views,
            width=width,
            height=height,
            fx=args.fx,
            fy=args.fy,
            cx=args.cx,
            cy=args.cy,
            sphere_radius=args.sphere_radius,
        ):
            ok += 1
    print(f"完成: 成功 {ok}/{len(obj_files)}")


if __name__ == "__main__":
    main()
