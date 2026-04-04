#!/usr/bin/env python3

from typing import Tuple, Optional, Callable, List

from isaacgym import gymtorch
from isaacgym import gymapi
from pathlib import Path

from dataclasses import dataclass
from util.config import ConfigBase

from einops import rearrange
import numpy as np
import torch as th


from env.env.iface import EnvIface
from env.env.wrap.base import WrapperEnv

from env.env.help.apply_camera_transform import ApplyCameraTransform
from env.env.wrap.nvdr_camera_wrapper import NvdrCameraWrapper
from env.robot.cube_poker import CubePoker
from env.robot.ur5_fe import UR5FE
from models.common import merge_shapes

from util.path import ensure_directory
from util.torch_util import dcn
from util.vis.img import tile_images, to_hwc

import cv2


class NvdrRecordEpisode(WrapperEnv):
    """
    Wrapper to record images from gym viewer.
    """

    @dataclass
    class Config(ConfigBase):
        record_dir: str = '/tmp/docker/record'
        episode_type: str = 'succ'
        img_size: Tuple[int, int] = (256, 256)
        cam_eye: Tuple[float, float, float] = (0.54, 0.0, 0.9)
        cam_at: Tuple[float, float, float] = (-0.2, 0.0, 0.4)
        # False: URDF visual + 材质/贴图，RGB 更彩；True: 碰撞网格，RGB 常发灰、像黑白
        use_col: bool = False

        video_fps: int = 30

    def __init__(self, cfg: Config, env: EnvIface, **kwds):
        super().__init__(env)
        self.cfg = cfg

        self.img_env = NvdrCameraWrapper(
            env,
            NvdrCameraWrapper.Config(
                img_size=cfg.img_size,

                eye=cfg.cam_eye,
                at=cfg.cam_at,

                use_flow=False,
                use_label=False,
                use_color=True,

                use_col=cfg.use_col,
                **kwds
            )
        )

        self._record_dir: Path = ensure_directory(
            cfg.record_dir)

        # self.__eps_count: int = 0
        self.eps_count_batch: list = [0] * env.num_env
        # Create a giant buffer for storing images.
        self.__images = np.zeros(
            merge_shapes(env.timeout, env.num_env, cfg.img_size, 3),
            dtype=np.uint8)
        self.__index = np.zeros((env.num_env,),
                                dtype=np.int32)
        self._roll_chunks: List[List[np.ndarray]] = [
            [] for _ in range(env.num_env)]
        self.project = ApplyCameraTransform()
        self.project.reset(self)

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):
        return self.env.action_space

    def __clear_record_buffer(self, env_id: int) -> None:
        self._roll_chunks[env_id].clear()
        self.__index[env_id] = 0

    def __flush_full_chunk(self, env_id: int) -> None:
        n_cap = int(self.__images.shape[0])
        self._roll_chunks[env_id].append(
            self.__images[:n_cap, env_id].copy())
        self.__index[env_id] = 0

    def __export_episode(self, images_hwc: np.ndarray, env_id: int = 0):
        count = int(images_hwc.shape[0])
        if count <= 0:
            return
        out_dir = ensure_directory(
            self._record_dir /
            F'env_{env_id:04d}')
        video_path = out_dir / f'episode_{self.eps_count_batch[env_id]:04d}.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        height, width = images_hwc.shape[1:3]
        self.video_writer = cv2.VideoWriter(
            str(video_path), fourcc, self.cfg.video_fps, (width, height))
        for frame in images_hwc:
            self.video_writer.write(frame)
        self.video_writer.release()

        self.eps_count_batch[env_id] += 1

    def __export_episode_merged(self, env_id: int) -> None:
        idx = int(self.__index[env_id])
        parts = list(self._roll_chunks[env_id])
        if idx > 0:
            parts.append(self.__images[:idx, env_id].copy())
        self._roll_chunks[env_id].clear()
        self.__index[env_id] = 0
        if not parts:
            return
        full = np.concatenate(parts, axis=0)
        self.__export_episode(full, env_id)

    def __get_lines(self, lines) -> Tuple[np.ndarray, np.ndarray]:
        vss = []
        css = []
        nss = []
        for env_id in sorted(lines.keys()):
            # self.lines[env_id].append((num_lines, vertices.copy(), colors.copy()))
            vs = []
            cs = []
            ns = []
            for line in lines[env_id]:
                n, v, c = line
                vs.append(v)
                cs.append(c)
                ns.append(n)
            vss.append(np.concatenate(vs, axis=0).reshape(-1, 6))
            css.append(np.concatenate(cs, axis=0).reshape(-1, 3))
            nss.append(ns)

        if len(vss) <= 0:
            vss = np.empty((self.env.num_env, 0, 6))
            css = np.empty((self.env.num_env, 0, 3))
            nss = np.empty((self.env.num_env, 0))
        else:
            vss = np.stack(vss, axis=0)
            css = np.stack(css, axis=0)
            nss = np.stack(nss, axis=0)
        return (vss, css, nss)

    def __record(self, obs, rew, done, info):
        # Render & Query.
        alt = self.img_env._wrap_obs(obs)
        rgb_imgs = alt['color']  # NCHW

        # Convert.
        rgb_imgs_np = dcn(rgb_imgs)
        if rgb_imgs_np.dtype != np.uint8:
            rgb_imgs_np = (255 * rgb_imgs_np.clip(0.0, 1.0)).astype(np.uint8)
        rgb_imgs_np = to_hwc(rgb_imgs_np)
        done_np = dcn(done)

        # Draw all debug lines on our image as well
        if True:
            vertices, colors, counts = self.__get_lines(self.env.gym.lines)
            vertices = vertices.reshape(self.env.num_env, -1, 3)
            vertices = dcn(self.project(vertices, self.cfg.img_size))
            lines = vertices.reshape(self.env.num_env, -1, 2, 2)
            lines_np = dcn(lines)

            def _to_cvpt(x):
                return tuple(int(e) for e in x)

            for i in range(self.env.num_env):
                for l in range(lines_np.shape[1]):
                    color = (255 *
                             (colors[i, l]).clip(0.0, 1.0)).astype(np.uint8)
                    cv2.line(
                        rgb_imgs_np[i],
                        _to_cvpt(lines_np[i, l, 0]),
                        _to_cvpt(lines_np[i, l, 1]),
                        _to_cvpt(color),
                        thickness=1
                    )

        # 缓冲区长度为 env.timeout；满则把整段推到 _roll_chunks 并复位写入指针，避免越界；回合结束再拼接导出一条 mp4
        n_cap = int(self.__images.shape[0])
        env_ids = np.arange(self.env.num_env, dtype=np.int32)
        valid = self.__index < n_cap
        if np.any(valid):
            self.__images[self.__index[valid], env_ids[valid]] = rgb_imgs_np[valid]
        self.__index[valid] += 1
        overflow = self.__index >= n_cap
        if np.any(overflow):
            for e in np.where(overflow)[0]:
                self.__flush_full_chunk(int(e))

        # Process episodes.
        sel = None
        if self.cfg.episode_type == 'succ':
            assert (('success' in info))
            sel = info['success']
        elif self.cfg.episode_type == 'fail':
            assert (('success' in info))
            sel = ~info['success']

        for env_id in np.argwhere(done_np).ravel():
            e = int(env_id)
            if self.env.buffers['step'][e] < 30:
                self.__clear_record_buffer(e)
                continue
            if (sel is not None) and (not sel[e]):
                self.__clear_record_buffer(e)
                continue
            self.__export_episode_merged(e)

    def step(self, *args, **kwds):
        # Step original env.
        obs, rew, done, info = self.env.step(*args, **kwds)
        # Record.
        self.__record(obs, rew, done, info)
        # Return original outputs.
        return (obs, rew, done, info)
