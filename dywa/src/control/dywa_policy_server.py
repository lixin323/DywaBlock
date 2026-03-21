#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import tyro
import websockets
from websockets.server import WebSocketServerProtocol

from .stacking_pipeline import BlockStackingPipeline, PipelineConfig


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 33060
    scene_root: Path = Path("block_data/SCENEs_400_Goal_Jsons")
    template_db: Path = Path("block_data/linemod_templates")
    block_assets_dir: Path = Path("block_data/block_assets")
    dywa_export_dir: Path = Path("exported_abs_goal_1view")
    dywa_device: str = "cuda:0"
    chunk_size: int = 20
    dywa_goal_offset_cm_x: float = 30.0
    dywa_goal_offset_cm_y: float = 0.0
    dywa_goal_offset_cm_z: float = 9.5
    grasp_goal_offset_cm_x: float = 30.0
    grasp_goal_offset_cm_y: float = -30.0
    grasp_goal_offset_cm_z: float = 9.5
    attitude_tol_deg: float = 8.0
    adjust_region_x_m: float = 0.05
    adjust_region_y_m: float = 0.05
    adjust_region_z_m: float = 0.05
    place_tol_m: float = 0.02
    grasp_steps_per_segment: int = 6


class DywaPolicyServer:
    """WebSocket Server：协议收发 + 流水线调用。"""

    def __init__(self, cfg: ServerConfig):
        self.cfg = cfg
        self.pipeline = BlockStackingPipeline(
            PipelineConfig(
                scene_root=cfg.scene_root,
                template_db=cfg.template_db,
                block_assets_dir=cfg.block_assets_dir,
                dywa_export_dir=cfg.dywa_export_dir,
                dywa_device=cfg.dywa_device,
                adjust_chunk_size=cfg.chunk_size,
                dywa_goal_offset_cm_x=cfg.dywa_goal_offset_cm_x,
                dywa_goal_offset_cm_y=cfg.dywa_goal_offset_cm_y,
                dywa_goal_offset_cm_z=cfg.dywa_goal_offset_cm_z,
                grasp_goal_offset_cm_x=cfg.grasp_goal_offset_cm_x,
                grasp_goal_offset_cm_y=cfg.grasp_goal_offset_cm_y,
                grasp_goal_offset_cm_z=cfg.grasp_goal_offset_cm_z,
                attitude_tol_deg=cfg.attitude_tol_deg,
                adjust_region_x_m=cfg.adjust_region_x_m,
                adjust_region_y_m=cfg.adjust_region_y_m,
                adjust_region_z_m=cfg.adjust_region_z_m,
                place_tol_m=cfg.place_tol_m,
                grasp_steps_per_segment=cfg.grasp_steps_per_segment,
            )
        )
        self._lock = asyncio.Lock()

    async def handle_connection(self, ws: WebSocketServerProtocol) -> None:
        async for message in ws:
            try:
                req: Dict[str, Any] = json.loads(message)
            except Exception:
                await ws.send(json.dumps({"error": "invalid json", "actions": [], "scene_done": False}))
                continue

            async with self._lock:
                try:
                    resp = self.pipeline.process_request(req)
                except Exception as e:
                    resp = {"error": str(e), "actions": [], "scene_done": False}
            await ws.send(json.dumps(resp))

    async def run(self) -> None:
        async with websockets.serve(self.handle_connection, self.cfg.host, self.cfg.port):
            print(f"[DywaPolicyServer] listening on {self.cfg.host}:{self.cfg.port}")
            await asyncio.Future()


def main() -> None:
    cfg = tyro.cli(ServerConfig)
    asyncio.run(DywaPolicyServer(cfg).run())


if __name__ == "__main__":
    main()

