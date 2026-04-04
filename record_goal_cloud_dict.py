#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import zlib
from pathlib import Path
from typing import List
import urllib.request

import cv2
import numpy as np
import pyrealsense2 as rs


def _load_object_names(scene_json: Path) -> List[str]:
    data = json.loads(scene_json.read_text(encoding="utf-8"))
    names: List[str] = []
    for b in data.get("blocks", []):
        n = f"{str(b['type'])}_{str(b['color'])}"
        if n not in names:
            names.append(n)
    if len(names) == 0:
        raise RuntimeError(f"no blocks in scene json: {scene_json}")
    return names


def _capture_one_frame(serial: str, width: int, height: int, fps: int):
    pipe = rs.pipeline()
    cfg = rs.config()
    if serial:
        cfg.enable_device(serial)
    cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    prof = pipe.start(cfg)
    try:
        # warmup
        for _ in range(20):
            pipe.wait_for_frames(timeout_ms=2000)
        frames = pipe.wait_for_frames(timeout_ms=3000)
        color = frames.get_color_frame()
        depth = frames.get_depth_frame()
        if not color or not depth:
            raise RuntimeError("failed to read color/depth frame")
        color_bgr = np.asanyarray(color.get_data(), dtype=np.uint8)
        depth_u16 = np.asanyarray(depth.get_data(), dtype=np.uint16)
        intr = prof.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        return color_bgr, depth_u16, intr
    finally:
        pipe.stop()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene-json", type=str, required=True)
    ap.add_argument("--save-path", type=str, required=True)
    ap.add_argument("--service-url", type=str, default="http://127.0.0.1:7780/extract_goal_clouds")
    ap.add_argument("--cam-serial", type=str, default="")
    ap.add_argument("--image-width", type=int, default=640)
    ap.add_argument("--image-height", type=int, default=480)
    ap.add_argument("--camera-fps", type=int, default=30)
    ap.add_argument("--cloud-size", type=int, default=512)
    args = ap.parse_args()

    scene_json = Path(args.scene_json)
    save_path = Path(args.save_path)
    object_names = _load_object_names(scene_json)
    print(f"[goal_cloud_record] scene={scene_json.name} objects={object_names}", flush=True)

    bgr, depth_u16, intr = _capture_one_frame(
        serial=str(args.cam_serial),
        width=int(args.image_width),
        height=int(args.image_height),
        fps=int(args.camera_fps),
    )
    ok, enc = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise RuntimeError("rgb encode failed")

    payload = {
        "object_names": object_names,
        "image_width": int(bgr.shape[1]),
        "image_height": int(bgr.shape[0]),
        "fx": float(intr.fx),
        "fy": float(intr.fy),
        "cx": float(intr.ppx),
        "cy": float(intr.ppy),
        "depth_scale": 0.001,
        "cloud_size": int(args.cloud_size),
        "save_path": str(save_path),
        "include_clouds": False,
        "rgb_jpeg_b64": base64.b64encode(enc.tobytes()).decode("ascii"),
        "depth_zlib_b64": base64.b64encode(zlib.compress(depth_u16.tobytes())).decode("ascii"),
    }
    req = urllib.request.Request(
        str(args.service_url),
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=20.0) as resp:
        out = json.loads(resp.read().decode("utf-8"))
    if not bool(out.get("ok", False)):
        raise RuntimeError(f"extract_goal_clouds failed: {out}")
    if not save_path.exists():
        raise RuntimeError(f"goal cloud pkl not saved: {save_path}")
    print(f"[goal_cloud_record] saved={save_path}", flush=True)


if __name__ == "__main__":
    main()

