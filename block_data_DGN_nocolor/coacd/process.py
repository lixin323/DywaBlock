#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import traceback
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from tqdm import tqdm
import trimesh

try:
    import coacd
except ImportError as e:
    print("导入 coacd 失败，请先执行 `pip install coacd`。")
    raise e

# 配置：根据当前脚本位置自动推断路径
THIS_FILE = Path(__file__).resolve()

# process.py 位于: <project_root>/data/PhyBlock/data/block_assets/process.py
# 所以 block_assets 目录就是输入根目录
INPUT_ROOT = THIS_FILE.parent

# 输出根目录设在: <project_root>/data/PhyBlock/dywa_processed
# 即从 block_assets 向上两级到 data，再向上一级到 PhyBlock
OUTPUT_ROOT = THIS_FILE.parents[2] / "dywa_processed"

POINT_CLOUD_ROOT = OUTPUT_ROOT / "point_clouds"
COLLISION_ROOT = OUTPUT_ROOT / "collision_meshes"

NUM_POINTS = 2048


def load_trimesh_mesh(obj_path: Path) -> trimesh.Trimesh:
    """
    尝试将任意 .obj（包括场景、多子网格）转为单个 Trimesh。
    出错时抛异常，由上层捕获。
    """
    mesh = trimesh.load(obj_path, force="mesh")

    if isinstance(mesh, trimesh.Scene):
        # 将 scene 中所有几何体合并成一个 mesh
        if len(mesh.geometry) == 0:
            raise ValueError("Scene 中不包含几何体。")
        mesh_list = [g for g in mesh.geometry.values()]
        mesh = trimesh.util.concatenate(mesh_list)

    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"无法将 {obj_path} 转为 Trimesh。类型：{type(mesh)}")

    if mesh.vertices.shape[0] == 0 or mesh.faces.shape[0] == 0:
        raise ValueError("Mesh 顶点或面数量为 0。")

    return mesh


def process_single_obj(obj_path_str: str) -> str:
    """
    单个 .obj 文件的处理逻辑：
    - 采样点云并保存 .npy
    - 使用 coacd 凸分解并保存 *_coacd.obj

    返回值：处理成功或失败的简要字符串，仅用于日志。
    """
    obj_path = Path(obj_path_str)

    try:
        # 计算相对路径，用于输出结构映射
        rel_dir = obj_path.parent.relative_to(INPUT_ROOT)
        base_name = obj_path.stem

        # 点云输出路径
        pc_dir = POINT_CLOUD_ROOT / rel_dir
        pc_dir.mkdir(parents=True, exist_ok=True)
        pc_path = pc_dir / f"{base_name}.npy"

        # 碰撞体输出路径
        col_dir = COLLISION_ROOT / rel_dir
        col_dir.mkdir(parents=True, exist_ok=True)
        col_path = col_dir / f"{base_name}_coacd.obj"

        # 1. 加载 mesh
        mesh = load_trimesh_mesh(obj_path)

        # 2. 采样表面点云
        points, _ = trimesh.sample.sample_surface(mesh, NUM_POINTS)
        points = points.astype(np.float32)
        np.save(pc_path, points)

        # 3. 使用 coacd 做凸分解
        mesh_coacd = coacd.Mesh(mesh.vertices, mesh.faces)

        # 可以根据需要调整 coacd 参数（这里使用默认较通用配置）
        parts = coacd.run_coacd(
            mesh_coacd,
            threshold=0.05,         # 凹度阈值
            max_convex_hull=-1,     # -1 表示不限制凸块数量
            resolution=2000,
            mcts_iterations=150,
        )

        if len(parts) == 0:
            raise RuntimeError("coacd 未返回任何凸分解结果。")

        convex_meshes = []
        for verts, faces in parts:
            if verts.size == 0 or faces.size == 0:
                continue
            convex_meshes.append(
                trimesh.Trimesh(vertices=verts, faces=faces, process=False)
            )

        if not convex_meshes:
            raise RuntimeError("所有 coacd 返回的凸块均为空。")

        # 将所有凸块组成一个 Scene 输出为单个 obj 文件
        scene = trimesh.Scene(convex_meshes)
        scene.export(col_path)

        return f"OK: {obj_path}"

    except Exception as e:
        # 打印详细错误，但不要终止整个程序
        print(f"[警告] 处理失败，文件已跳过: {obj_path}")
        print(f"  原因: {e}")
        traceback.print_exc(file=sys.stdout)
        return f"FAIL: {obj_path} ({e})"


def find_all_obj_files(root: Path):
    """
    递归查找 root 下所有 .obj 文件。
    """
    return sorted(root.rglob("*.obj"))


def main(use_multiprocessing: bool = True, max_workers =None):
    if not INPUT_ROOT.exists():
        print(f"[错误] 输入目录不存在: {INPUT_ROOT}")
        return

    obj_files = find_all_obj_files(INPUT_ROOT)
    if not obj_files:
        print(f"[提示] 在 {INPUT_ROOT} 中未找到任何 .obj 文件。")
        return

    print(f"待处理 .obj 文件数量: {len(obj_files)}")
    print(f"输入根目录: {INPUT_ROOT}")
    print(f"点云输出根目录: {POINT_CLOUD_ROOT}")
    print(f"碰撞体输出根目录: {COLLISION_ROOT}")
    print(f"使用多进程: {use_multiprocessing}")

    # 确保输出根目录存在
    POINT_CLOUD_ROOT.mkdir(parents=True, exist_ok=True)
    COLLISION_ROOT.mkdir(parents=True, exist_ok=True)

    results = []

    if use_multiprocessing:
        # 多进程模式
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(process_single_obj, str(p)): p for p in obj_files
            }

            for future in tqdm(as_completed(futures), total=len(futures), desc="处理 OBJ 文件", ncols=80):
                res = future.result()
                results.append(res)
    else:
        # 单进程（调试或机器核数较少时可选）
        for p in tqdm(obj_files, desc="处理 OBJ 文件", ncols=80):
            res = process_single_obj(str(p))
            results.append(res)

    # 简单统计
    ok_count = sum(1 for r in results if r.startswith("OK:"))
    fail_count = len(results) - ok_count

    print(f"\n处理完成：成功 {ok_count} 个，失败 {fail_count} 个。")
    if fail_count > 0:
        print("失败文件列表（简要）：")
        for r in results:
            if r.startswith("FAIL:"):
                print("  ", r)


if __name__ == "__main__":
    # 如需关闭多进程，将 use_multiprocessing 设为 False
    main(use_multiprocessing=True, max_workers=None)