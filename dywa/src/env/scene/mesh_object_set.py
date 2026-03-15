#!/usr/bin/env python3

import os
import glob
from tempfile import mkdtemp
from dataclasses import dataclass
from typing import Tuple, Optional
from pathlib import Path
import shutil

import numpy as np
import trimesh
from tqdm.auto import tqdm

from icecream import ic

from util.path import ensure_directory

from util.config import ConfigBase
from env.scene.util import sample_stable_poses
from env.scene.object_set import ObjectSet
from env.scene.apply_coacd import (Config as COACDConfig,
                                       apply_coacd)

DATA_ROOT = os.getenv('PKM_DATA_ROOT', '/input')

URDF_TEMPLATE: str = '''<robot name="robot">
    <link name="base_link">
        <inertial>
            <mass value="{mass}"/>
            <origin xyz="0 0 0"/>
            <inertia ixx="{ixx}" ixy="{ixy}" ixz="{ixz}"
            iyy="{iyy}" iyz="{iyz}" izz="{izz}"/>
        </inertial>
        <visual>
            <origin xyz="0 0 0" rpy="0 0 0"/>
            <geometry>
                <mesh filename="{vis_mesh}" scale="1.0 1.0 1.0"/>
            </geometry>
        </visual>
        <collision>
            <origin xyz="0 0 0" rpy="0 0 0"/>
            <geometry>
                <mesh filename="{col_mesh}" scale="1.0 1.0 1.0"/>
            </geometry>
        </collision>
    </link>
</robot>
'''


class MeshObjectSet(ObjectSet):
    @dataclass
    class Config(ConfigBase):
        # Is this necessary? I guess we'll just
        # configure the mass according to this density parameter
        # and the volume ... unless we also apply mass DR
        density: float = 300.0
        table_dims: Tuple[float, float, float] = (0.4, 0.5, 0.4)
        # Might be necessary for on-line generation of
        # stable poses
        num_poses: int = 256
        # = CloudSize
        num_points: int = 512
        cache_dir: str = '/tmp/poses'

        # Remember: we also need to run CoACD
        # (probably)... to generate collision mesh
        filename: Optional[str] = None

        acd: bool = True
        coacd: COACDConfig = COACDConfig(
            mcts_max_depth=3,
            mcts_iterations=32,
            mcts_nodes=16,
            verbose=True
        )

        # --- PhyBlock 预处理数据支持 ---
        # 是否使用离线生成的 coacd.obj 和点云 .npy
        use_phyblock_preprocess: bool = False
        # 预处理后的点云与 collision mesh 根目录（容器内路径）
        phyblock_pc_root: str = '/input/PhyBlock/dywa_processed/point_clouds'
        phyblock_coacd_root: str = '/input/PhyBlock/dywa_processed/collision_meshes'

    def __init__(self, cfg: Config):
        self.cfg = cfg

        if cfg.filename is None:
            raise ValueError('cfg.filename should not be None!')

        files = glob.glob(cfg.filename,
                          recursive=True)

        # 原始可视 mesh 文件
        self.__files = {str(Path(m).stem): m
                        for m in files}

        # 加载网格
        self.__mesh = {k: trimesh.load(v, force='mesh')
                       for k, v in self.__files.items()}
        self.__keys = sorted(list(self.__mesh.keys()))
        print(self.__keys)
        self.__metadata = {k: {} for k in self.__keys}

        self.__radius = {}
        self.__volume = {}
        self.__masses = {}
        for k, m in self.__mesh.items():
            self.__radius[k] = float(
                0.5 *
                np.linalg.norm(m.vertices,
                               axis=-1).max())
            self.__volume[k] = m.volume
            # TODO: consider randomization
            self.__masses[k] = cfg.density * m.volume

        table_dims = np.asarray(cfg.table_dims,
                                dtype=np.float32)
        self.__poses = {}
        for k, v in tqdm(self.__mesh.items(), desc='pose'):
            # FIXME: height=table_dims[2] assumes
            # table is on the ground.
            # TODO: should _probably_ use a different function;
            # mostly since trimesh implementation is super slow.
            path = ensure_directory(cfg.cache_dir)
            if (path/f'{k}.npy').is_file():
                poses = np.load(path/f'{k}.npy')
                print(f"load cached poses from {path}/{k}.npy")
            else:
                poses = sample_stable_poses(v.convex_hull,
                                            table_dims[2],
                                            cfg.num_poses)
                np.save(path/f'{k}.npy', poses)
            self.__poses[k] = poses.astype(np.float32)

        # NOTE: unnecessarily computationally costly maybe
        # 为了兼容原有逻辑，仍然维护在线采样的 cloud / normal / bbox 信息；
        # 如果启用了 PhyBlock 预处理，则优先使用离线点云 `.npy` 作为 canonical cloud。
        self.__cloud = {}
        self.__normal = {}
        self.__bbox = {}
        self.__aabb = {}
        self.__obb = {}

        # 离线 canonical 点云（仅在 use_phyblock_preprocess=True 时有效）
        self.canonical_clouds = {}

        for k, v in self.__mesh.items():
            # 默认：从 mesh 表面在线采样 cfg.num_points 个点
            samples, face_index = trimesh.sample.sample_surface(
                v, cfg.num_points)

            # 如果使用 PhyBlock 预处理，尝试从 .npy 加载离线点云作为 canonical cloud
            if cfg.use_phyblock_preprocess:
                mesh_path = Path(self.__files[k])
                try:
                    # /input/PhyBlock/data/block_assets/... -> /input/PhyBlock/dywa_processed/point_clouds/...
                    rel = mesh_path.relative_to('/input/PhyBlock/data/block_assets')
                    pc_path = Path(cfg.phyblock_pc_root) / rel.parent / (rel.stem + '.npy')
                except ValueError:
                    # 路径结构不符合预期时，直接拼接文件名
                    pc_path = Path(cfg.phyblock_pc_root) / (mesh_path.stem + '.npy')

                if pc_path.is_file():
                    try:
                        pc = np.load(pc_path).astype(np.float32)  # 期望形状约为 [2048, 3]
                        # 如有需要，下采样到 cfg.num_points（通常为 512），
                        # 以兼容已有 cloud_size / point_tokenizer 配置。
                        if pc.shape[0] > cfg.num_points:
                            idx = np.random.choice(pc.shape[0],
                                                   cfg.num_points,
                                                   replace=False)
                            pc_sub = pc[idx]
                        else:
                            pc_sub = pc
                        samples = pc_sub  # 用离线点云替换在线采样
                        # 记录完整 canonical cloud（保留原始点数，如 2048），供后续 initial_cloud 使用
                        self.canonical_clouds[k] = pc
                    except Exception:
                        # 如果 .npy 损坏或读取失败，则退回在线采样结果
                        self.canonical_clouds[k] = samples.astype(np.float32)
                else:
                    # 找不到离线文件时，仍然提供在线采样 cloud，避免中断
                    self.canonical_clouds[k] = samples.astype(np.float32)

            self.__cloud[k] = samples
            self.__normal[k] = v.face_normals[face_index]
            self.__aabb[k] = v.bounds
            self.__bbox[k] = trimesh.bounds.corners(v.bounds)
            obb = v.bounding_box_oriented
            self.__obb[k] = (
                np.asarray(obb.transform, dtype=np.float32),
                np.asarray(obb.extents, dtype=np.float32))

        # Unfortunately, no guarantee of deletion
        self.__tmpdir = mkdtemp()
        self.__write_urdf()

    def __write_urdf(self):
        cfg = self.cfg
        self.__urdf = {}
        for k in self.__keys:
            m = self.__masses[k]
            I = self.__mesh[k].moment_inertia

            # 可视 mesh 一直用原始 obj
            vis_mesh_file = self.__files[k]

            aux = {}
            col_mesh_file = F'{self.__tmpdir}/{k}.obj'

            if cfg.use_phyblock_preprocess:
                # 使用预处理生成的 coacd.obj 作为碰撞体
                # 从原始路径推断出相对路径，例如：
                # /input/PhyBlock/data/block_assets/arch_red.obj
                # -> arch_red_coacd.obj 位于
                # /input/PhyBlock/dywa_processed/collision_meshes/...
                src_path = Path(vis_mesh_file)
                try:
                    rel = src_path.relative_to('/input/PhyBlock/data/block_assets')
                except ValueError:
                    # 若无法 relative_to，直接退化为旧逻辑
                    rel = src_path.name
                rel = Path(rel)
                coacd_path = Path(cfg.phyblock_coacd_root) / rel.parent / f'{rel.stem}_coacd.obj'
                if coacd_path.is_file():
                    shutil.copy(str(coacd_path), col_mesh_file)
                    aux['num_part'] = 1  # 多凸块已在 coacd.obj 中合并
                else:
                    # 找不到预处理文件时，回退到旧行为：直接使用可视 mesh
                    shutil.copy(vis_mesh_file, col_mesh_file)
                    aux['num_part'] = 1
            else:
                # 旧行为：直接使用可视 mesh 或在线调用 CoACD
                if True:
                    shutil.copy(vis_mesh_file, col_mesh_file)
                    aux['num_part'] = 1
                else:
                    apply_coacd(cfg.coacd,
                                vis_mesh_file,
                                col_mesh_file, aux=aux)

            ic(vis_mesh_file, col_mesh_file)
            self.__metadata[k]['num_chulls'] = (
                aux['num_part']
            )

            params = dict(
                mass=m,
                ixy=I[0, 1], ixz=I[0, 2], iyz=I[1, 2],
                ixx=m * I[0, 0], iyy=m * I[1, 1], izz=m * I[2, 2],
                vis_mesh=vis_mesh_file,
                col_mesh=col_mesh_file
            )
            filename = F'{self.__tmpdir}/{k}.urdf'
            with open(filename, 'w') as fp:
                fp.write(URDF_TEMPLATE.format(**params))
            self.__urdf[k] = filename

    def keys(self):
        return self.__keys

    def label(self, key: str) -> str:
        return key

    def urdf(self, key: str):
        return self.__urdf[key]

    def pose(self, key: str):
        return self.__poses[key]

    def code(self, key: str):
        # return self.codes[key]
        return None

    def cloud(self, key: str):
        return self.__cloud[key]

    def normal(self, key: str):
        return self.__normal[key]

    def bbox(self, key: str):
        return self.__bbox[key]

    def aabb(self, key: str):
        return self.__aabb[key]

    def obb(self, key: str) -> Tuple[np.ndarray, np.ndarray]:
        return self.__obb[key]

    def hull(self, key: str) -> trimesh.Trimesh:
        return self.__mesh[key]

    def radius(self, key: str) -> float:
        return self.__radius[key]

    def volume(self, key: str) -> float:
        return self.__volume[key]

    def num_verts(self, key: str) -> float:
        return len(self.__mesh[key].vertices)

    def num_faces(self, key: str) -> float:
        return len(self.__mesh[key].faces)

    def num_hulls(self, key: str) -> float:
        return self.__metadata[key]['num_chulls']


def main():
    # _convert_from_previous_version()
    dataset = MeshObjectSet(
        MeshObjectSet.Config(
            num_poses=1,
            filename='/tmp/docker/bd_debug5/textured_mesh.obj'
        ))
    for attr in dir(dataset):
        print(attr)
        if hasattr(dataset, attr):
            (getattr(dataset, attr))
    # print(len(dataset.codes))


if __name__ == '__main__':
    main()
