#!/usr/bin/env python3
"""DyWA 策略 WebSocket 服务：每帧用客户端 RGB-D 做 LINEMOD+点云，再 DyWA 前向，输出末端动作 chunk。

请求 JSON（必填）：
  scene_id, block_index, ee_state[7], image_width, image_height,
  fx, fy, cx, cy, depth_scale,
  T_base_cam: 16 个 float（行主序 4x4，OpenCV 相机系 → 机器人 base/world），
  rgb_jpeg_b64: JPEG 的 base64（BGR，与 OpenCV imencode 一致）,
  depth_zlib_b64: zlib.compress(depth_uint16.tobytes()) 的 base64

响应：
  actions: [[x,y,z,r,p,y,g], ...]（在上一帧 EE 与 DyWA 目标之间插值，长度 chunk_size）,
  error: 可选字符串,
  scene_done: block_index 超出场景积木数量时为 true
"""
from __future__ import annotations

import asyncio
import base64
import json
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import websockets
from websockets.server import WebSocketServerProtocol

from dywa.src.control.dywa_model_policy import DywaStudentPolicyInterface, DywaStudentPolicyConfig
from dywa.src.control.linemod_opencv import CameraIntrinsics, detect_object_mask_pointcloud
from dywa.src.control.se3 import assert_se3


def _rpy_to_R(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = float(np.cos(roll)), float(np.sin(roll))
    cp, sp = float(np.cos(pitch)), float(np.sin(pitch))
    cy, sy = float(np.cos(yaw)), float(np.sin(yaw))
    Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    Ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float32)
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float32)
    return (Rz @ Ry @ Rx).astype(np.float32)


def _ee7_to_hand9(ee7: np.ndarray) -> np.ndarray:
    x, y, z, r, p, yw, _g = [float(v) for v in np.asarray(ee7, dtype=np.float64).reshape(-1)[:7].tolist()]
    R = _rpy_to_R(r, p, yw)
    rot6d = R[:, :2].reshape(6).astype(np.float32)
    return np.asarray([x, y, z, *rot6d.tolist()], dtype=np.float32)


def _ee7_to_robot14(ee7: np.ndarray) -> np.ndarray:
    v7 = np.asarray(ee7, dtype=np.float32).reshape(-1)[:7]
    tail = np.zeros((7,), dtype=np.float32)
    return np.concatenate([v7, tail], axis=0).astype(np.float32)


def _T_from_ee6(x: float, y: float, z: float, r: float, p: float, yw: float) -> np.ndarray:
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = _rpy_to_R(r, p, yw)
    T[:3, 3] = np.array([x, y, z], dtype=np.float32)
    return T


def _rpy_from_R(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    pitch = float(np.arctan2(-R[2, 0], np.sqrt(R[2, 1] ** 2 + R[2, 2] ** 2)))
    roll = float(np.arctan2(R[2, 1], R[2, 2]))
    yaw = float(np.arctan2(R[1, 0], R[0, 0]))
    return np.array([roll, pitch, yaw], dtype=np.float32)


def _interp_chunk_7d(
    ee_start: List[float],
    T_goal: np.ndarray,
    grip_goal: float,
    n: int,
) -> List[List[float]]:
    """从当前 EE(7) 到目标 SE3 + 夹爪宽度，线性插值位置与 RPY（小步足够）。"""
    x0, y0, z0, r0, p0, yw0, g0 = [float(x) for x in ee_start[:7]]
    Rg = T_goal[:3, :3].astype(np.float64)
    rg, pg, ywg = _rpy_from_R(Rg)
    xg, yg, zg = float(T_goal[0, 3]), float(T_goal[1, 3]), float(T_goal[2, 3])
    n = max(1, int(n))
    out: List[List[float]] = []
    for i in range(1, n + 1):
        a = i / float(n)
        x = (1 - a) * x0 + a * xg
        y = (1 - a) * y0 + a * yg
        z = (1 - a) * z0 + a * zg
        r = (1 - a) * r0 + a * float(rg)
        p = (1 - a) * p0 + a * float(pg)
        yw = (1 - a) * yw0 + a * float(ywg)
        g = (1 - a) * g0 + a * float(grip_goal)
        out.append([x, y, z, r, p, yw, g])
    return out


def _load_sorted_blocks(scene_root: Path, scene_id: int) -> List[Dict[str, Any]]:
    shape = f"{int(scene_id):03d}"
    path = scene_root / f"{shape}.json"
    if not path.is_file():
        raise FileNotFoundError(str(path))
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    blocks = data.get("blocks", [])
    blocks = sorted(blocks, key=lambda b: int(b["order"]))
    return blocks


def _target_T_from_block_row(b: Dict[str, Any]) -> np.ndarray:
    from dywa.src.control.block_stacking_manager import _pose_from_json

    return np.asarray(
        _pose_from_json(position=b["position"], euler_xyz=b["euler"]),
        dtype=np.float32,
    )


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 33060
    scene_root: Path = Path("block_data/SCENEs_400_Goal_Jsons")
    template_db: Path = Path("block_data/linemod_templates")
    block_assets_dir: Path = Path("block_data/block_assets")
    dywa_ckpt: Path = Path("Dywa_abs_1view/ckpt")
    dywa_device: str = "cuda:0"
    chunk_size: int = 20
    grasp_tol_m: float = 0.02
    place_tol_m: float = 0.02


class _ObsBucket:
    pc_base: np.ndarray
    ee7: np.ndarray

    def __init__(self) -> None:
        self.pc_base = np.zeros((1, 3), dtype=np.float32)
        self.ee7 = np.zeros((7,), dtype=np.float32)


class DywaPolicyServer:
    def __init__(self, cfg: ServerConfig):
        self.cfg = cfg
        self._bucket = _ObsBucket()
        self._dywa = DywaStudentPolicyInterface(
            DywaStudentPolicyConfig(
                ckpt_path=self.cfg.dywa_ckpt,
                device=self.cfg.dywa_device,
                use_goal_cloud=False,
                block_assets_dir=self.cfg.block_assets_dir,
                get_partial_cloud_world=lambda: self._bucket.pc_base,
                get_robot_state=lambda: _ee7_to_robot14(self._bucket.ee7),
                get_hand_state=lambda: _ee7_to_hand9(self._bucket.ee7),
            )
        )
        self._lock: Optional[asyncio.Lock] = None

    def _decode_rgb_depth(self, req: Dict[str, Any]) -> tuple:
        import cv2  # type: ignore

        h = int(req["image_height"])
        w = int(req["image_width"])
        rgb_buf = base64.b64decode(req["rgb_jpeg_b64"])
        rgb = cv2.imdecode(np.frombuffer(rgb_buf, np.uint8), cv2.IMREAD_COLOR)
        if rgb is None or rgb.shape[0] != h or rgb.shape[1] != w:
            raise ValueError("rgb_jpeg 解码尺寸与 image_height/width 不一致")
        dz = zlib.decompress(base64.b64decode(req["depth_zlib_b64"]))
        depth = np.frombuffer(dz, dtype=np.uint16).reshape(h, w)
        return rgb, depth

    def _handle_one(self, req: Dict[str, Any]) -> Dict[str, Any]:
        scene_id = int(req["scene_id"])
        block_index = int(req["block_index"])
        ee_state = [float(x) for x in req["ee_state"]]
        if len(ee_state) < 7:
            return {"error": "ee_state 需要至少 7 维", "actions": [], "scene_done": False}

        T_bc = np.asarray(req["T_base_cam"], dtype=np.float64).reshape(4, 4).astype(np.float32)
        assert_se3(T_bc)

        blocks = _load_sorted_blocks(self.cfg.scene_root, scene_id)
        if block_index < 0 or block_index >= len(blocks):
            return {"actions": [], "scene_done": True, "block_index": block_index}

        b = blocks[block_index]
        block_type = str(b["type"])
        color = str(b["color"])
        target_T = _target_T_from_block_row(b)

        rgb, depth = self._decode_rgb_depth(req)
        intr = CameraIntrinsics(
            fx=float(req["fx"]),
            fy=float(req["fy"]),
            cx=float(req["cx"]),
            cy=float(req["cy"]),
        )
        depth_scale = float(req["depth_scale"])
        object_name = f"{block_type}_{color}"

        out = detect_object_mask_pointcloud(
            rgb=rgb,
            depth=depth,
            object_name=object_name,
            intrinsics=intr,
            template_db=Path(self.cfg.template_db),
            depth_scale=depth_scale,
            match_threshold=80.0,
            n_points=1024,
            roi_radius_px=80,
            depth_band_m=0.03,
        )
        if out is None:
            return {
                "error": f"LINEMOD/点云检测失败: {object_name}（检查模板与标定）",
                "actions": [],
                "scene_done": False,
                "block_index": block_index,
            }

        T_cam_obj, _mask, pc_cam = out
        assert_se3(T_cam_obj)
        pc_cam = np.asarray(pc_cam, dtype=np.float32).reshape(-1, 3)
        Rbc = T_bc[:3, :3]
        tbc = T_bc[:3, 3]
        pc_base = (Rbc @ pc_cam.T).T + tbc
        self._bucket.pc_base = np.ascontiguousarray(pc_base, dtype=np.float32)
        self._bucket.ee7 = np.asarray(ee_state[:7], dtype=np.float32)

        T_cam_obj_f = np.asarray(T_cam_obj, dtype=np.float32)
        T_base_block = T_bc @ T_cam_obj_f

        act = self._dywa.compute_action(
            block_type=block_type,
            current_T_world_block=T_base_block,
            target_T_world_block=target_T,
        )
        T_ee = np.asarray(act["ee_target_T_world"], dtype=np.float32)
        assert_se3(T_ee)
        grip = float(act.get("gripper_width", ee_state[6]))

        actions = _interp_chunk_7d(ee_state, T_ee, grip, self.cfg.chunk_size)

        pos_err = float(
            np.linalg.norm(T_base_block[:3, 3].astype(np.float64) - target_T[:3, 3].astype(np.float64))
        )
        return {
            "actions": actions,
            "scene_done": False,
            "block_index": block_index,
            "pos_err_block_m": pos_err,
            "object_name": object_name,
        }

    async def handle_connection(self, ws: WebSocketServerProtocol) -> None:
        assert self._lock is not None
        # 服务器侧状态机：服务器决定何时抓取/换块
        block_index = 0
        phase = "ADJUST"  # ADJUST -> GRASPED (抓取已发出) -> ADJUST(next block)
        async for message in ws:
            try:
                req = json.loads(message)
            except Exception:
                await ws.send(json.dumps({"error": "invalid json", "actions": [], "scene_done": False}))
                continue
            async with self._lock:
                # 忽略客户端传入 block_index，以服务器为准
                req["block_index"] = int(block_index)
                try:
                    resp = self._handle_one(req)
                except Exception as e:
                    resp = {"error": str(e), "actions": [], "scene_done": False}
                else:
                    if resp.get("scene_done"):
                        # 场景结束，保持 block_index 不变
                        pass
                    else:
                        pos_err = float(resp.get("pos_err_block_m", 1e9))
                        # 阈值逻辑：translation error
                        if phase == "ADJUST" and pos_err < float(self.cfg.grasp_tol_m):
                            # 调整完成：夹爪保持打开，然后触发抓取（最后一步 gripper=1）
                            actions = resp.get("actions", [])
                            if isinstance(actions, list) and len(actions) > 0:
                                # 强制前面的动作 gripper=0
                                for a in actions:
                                    if isinstance(a, list) and len(a) >= 7:
                                        a[6] = 0.0
                                # 追加一步：保持末端不动，仅触发抓取函数
                                last = actions[-1]
                                if isinstance(last, list) and len(last) >= 7:
                                    grasp = [float(x) for x in last[:7]]
                                    grasp[6] = 1.0
                                    actions.append(grasp)
                                    resp["actions"] = actions
                            phase = "GRASPED"
                        elif phase == "GRASPED" and pos_err < float(self.cfg.place_tol_m):
                            block_index += 1
                            phase = "ADJUST"
                            self._dywa.reset_episode()
                            resp["block_index"] = int(block_index)
            await ws.send(json.dumps(resp))

    async def run(self) -> None:
        self._lock = asyncio.Lock()
        async with websockets.serve(self.handle_connection, self.cfg.host, self.cfg.port):
            print(f"[DywaPolicyServer] listening on {self.cfg.host}:{self.cfg.port}")
            await asyncio.Future()


def main() -> None:
    cfg = ServerConfig()
    asyncio.run(DywaPolicyServer(cfg).run())


if __name__ == "__main__":
    main()
