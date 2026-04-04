#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import base64
import dataclasses
import json
import threading
import time
import zlib
from typing import Optional

import cv2
import numpy as np
import pyrealsense2 as rs
import rclpy
import tyro
import websockets
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float32MultiArray


@dataclasses.dataclass
class Args:
    # server
    policy_server_host: str = "127.0.0.1"
    policy_server_port: int = 33060
    scene_id: int = 0
    t_base_cam_csv: str = ""

    # camera
    cam_side_serial: str = "405622072640"
    image_width: int = 640
    image_height: int = 480
    camera_fps: int = 30
    jpeg_quality: int = 88

    # loop
    inference_frequency: float = 1.0
    max_actions_to_publish: int = 20
    action_publish_interval: float = 0.01
    max_state_age: float = 0.2

    # topics
    ee_states_topic: str = "/franka/ee_states"
    action_topic: str = "/franka/action_command"
    queue_status_topic: str = "/franka/queue_status"
    allow_inference_topic: str = "/franka/allow_inference"


class FrankaDywaInferenceNode(Node):
    """真机侧最小节点：采集 RGB-D + EE，向 server 请求动作，再发布 action_command。"""

    def __init__(self, args: Args):
        super().__init__("franka_dywa_inference_node")
        self.args = args
        self.ws_uri = f"ws://{args.policy_server_host}:{args.policy_server_port}"

        self._T_base_cam_row = self._parse_t_base_cam(args.t_base_cam_csv)
        self.get_logger().info(f"Policy Server: {self.ws_uri}")

        self._state_lock = threading.Lock()
        self._policy_lock = threading.Lock()
        self._queue_lock = threading.Lock()

        self.latest_ee: Optional[list[float]] = None
        self.last_ee_time: float = 0.0
        self.queue_empty: bool = False
        self.allow_inference: bool = True
        self.can_infer: bool = False

        self._rgbd_payload: Optional[dict] = None
        self._cam_fx = self._cam_fy = self._cam_cx = self._cam_cy = 0.0
        self._depth_scale = 0.001

        self.current_block_index = 0
        self.current_phase = "ADJUST"
        self.infer_count = 0

        self._init_camera()
        self._start_camera_thread()

        self.ee_sub = self.create_subscription(JointState, args.ee_states_topic, self.ee_cb, 10)
        self.queue_sub = self.create_subscription(Bool, args.queue_status_topic, self.queue_cb, 10)
        self.allow_sub = self.create_subscription(Bool, args.allow_inference_topic, self.allow_cb, 10)
        self.action_pub = self.create_publisher(Float32MultiArray, args.action_topic, 10)
        self.timer = self.create_timer(1.0 / max(0.1, args.inference_frequency), self.inference_cb)

    @staticmethod
    def _parse_t_base_cam(csv: str) -> list[float]:
        s = (csv or "").strip()
        if not s:
            raise ValueError("必须提供 --t-base-cam-csv（16个浮点数，行主序）")
        arr = [float(x.strip()) for x in s.split(",")]
        if len(arr) != 16:
            raise ValueError(f"--t-base-cam-csv 需要16个数，收到 {len(arr)}")
        return arr

    def _init_camera(self) -> None:
        W, H, fps = self.args.image_width, self.args.image_height, self.args.camera_fps
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_device(self.args.cam_side_serial)
        cfg.enable_stream(rs.stream.depth, W, H, rs.format.z16, fps)
        cfg.enable_stream(rs.stream.color, W, H, rs.format.yuyv, fps)
        self.frame_queue = rs.frame_queue(50)
        self.pipeline.start(cfg, self.frame_queue)
        profile = self.pipeline.get_active_profile()
        self._depth_scale = float(profile.get_device().first_depth_sensor().get_depth_scale())
        self._align = rs.align(rs.stream.color)
        self.get_logger().info(f"Camera started, depth_scale={self._depth_scale}")

    def _start_camera_thread(self) -> None:
        t = threading.Thread(target=self._camera_loop, daemon=True, name="camera_rgbd_loop")
        t.start()

    def _camera_loop(self) -> None:
        H, W = self.args.image_height, self.args.image_width
        while rclpy.ok():
            try:
                frame = self.frame_queue.wait_for_frame(timeout_ms=2000)
                fs = frame.as_frameset()
                fs = self._align.process(fs)
                color_frame = fs.get_color_frame()
                depth_frame = fs.get_depth_frame()
                if not color_frame or not depth_frame:
                    continue

                if self._cam_fx == 0.0:
                    intr = color_frame.profile.as_video_stream_profile().intrinsics
                    self._cam_fx = float(intr.fx)
                    self._cam_fy = float(intr.fy)
                    self._cam_cx = float(intr.ppx)
                    self._cam_cy = float(intr.ppy)
                    self.get_logger().info(
                        f"Camera intrinsics fx={self._cam_fx:.3f} fy={self._cam_fy:.3f} "
                        f"cx={self._cam_cx:.3f} cy={self._cam_cy:.3f}"
                    )

                yuyv = np.asanyarray(color_frame.get_data()).view(np.uint8).reshape(H, W, 2)
                bgr = cv2.cvtColor(yuyv, cv2.COLOR_YUV2BGR_YUYV)
                depth_u16 = np.asanyarray(depth_frame.get_data(), dtype=np.uint16).copy()
                ok, enc = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, int(self.args.jpeg_quality)])
                if not ok:
                    continue
                payload = {
                    "rgb_jpeg_b64": base64.b64encode(enc.tobytes()).decode("ascii"),
                    "depth_zlib_b64": base64.b64encode(zlib.compress(depth_u16.tobytes())).decode("ascii"),
                }
                with self._policy_lock:
                    self._rgbd_payload = payload
            except Exception as e:
                self.get_logger().warn(f"Camera loop error: {e}")
                time.sleep(0.1)

    def ee_cb(self, msg: JointState) -> None:
        if len(msg.position) < 7:
            return
        with self._state_lock:
            self.latest_ee = [float(x) for x in msg.position[:7]]
            self.last_ee_time = time.time()

    def queue_cb(self, msg: Bool) -> None:
        with self._queue_lock:
            self.queue_empty = bool(msg.data)
            self.can_infer = self.allow_inference and self.queue_empty

    def allow_cb(self, msg: Bool) -> None:
        with self._queue_lock:
            self.allow_inference = bool(msg.data)
            self.can_infer = self.allow_inference and self.queue_empty

    def _build_request(self) -> Optional[dict]:
        with self._queue_lock:
            if not self.can_infer:
                return None
        with self._state_lock:
            ee = self.latest_ee
            age = time.time() - self.last_ee_time if self.last_ee_time > 0 else 1e9
        if ee is None or age > float(self.args.max_state_age):
            return None
        with self._policy_lock:
            rgbd = self._rgbd_payload
        if rgbd is None or self._cam_fx <= 0:
            return None
        req = {
            "scene_id": int(self.args.scene_id),
            "block_index": int(self.current_block_index),  # 服务器会覆盖，但这里保留可观测性
            "ee_state": ee,
            "image_width": int(self.args.image_width),
            "image_height": int(self.args.image_height),
            "fx": float(self._cam_fx),
            "fy": float(self._cam_fy),
            "cx": float(self._cam_cx),
            "cy": float(self._cam_cy),
            "depth_scale": float(self._depth_scale),
            "T_base_cam": self._T_base_cam_row,
            "rgb_jpeg_b64": rgbd["rgb_jpeg_b64"],
            "depth_zlib_b64": rgbd["depth_zlib_b64"],
        }
        return req

    async def _request_server(self, req: dict) -> dict:
        async with websockets.connect(self.ws_uri) as ws:
            await ws.send(json.dumps(req))
            raw = await ws.recv()
            return json.loads(raw)

    def _publish_actions(self, actions: list) -> None:
        with self._queue_lock:
            self.can_infer = False
        n = min(len(actions), int(self.args.max_actions_to_publish))
        for i in range(n):
            msg = Float32MultiArray()
            msg.data = [float(x) for x in actions[i]]
            self.action_pub.publish(msg)
            if i < n - 1:
                time.sleep(float(self.args.action_publish_interval))

    def inference_cb(self) -> None:
        req = self._build_request()
        if req is None:
            return
        try:
            t0 = time.time()
            resp = asyncio.run(self._request_server(req))
            dt = (time.time() - t0) * 1000.0
            if resp.get("error"):
                self.get_logger().error(f"Server error: {resp['error']}")
                return
            if resp.get("scene_done"):
                self.get_logger().warn("scene_done=True")
                return

            self.current_block_index = int(resp.get("block_index", self.current_block_index))
            self.current_phase = str(resp.get("phase", self.current_phase))
            actions = resp.get("actions", [])
            self._publish_actions(actions)
            self.infer_count += 1
            self.get_logger().info(
                f"[{self.infer_count}] block={self.current_block_index} phase={self.current_phase} "
                f"actions={len(actions)} pos_err={resp.get('pos_err_block_m', 'n/a')} infer_ms={dt:.1f}"
            )
        except Exception as e:
            self.get_logger().error(f"Inference failed: {e}")

    def destroy_node(self) -> bool:
        try:
            self.pipeline.stop()
        except Exception:
            pass
        return super().destroy_node()


def main(args: Args) -> None:
    rclpy.init()
    node = FrankaDywaInferenceNode(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


## NOTE:
## 此文件历史上被拼接了两份实现，这里禁用前半段入口，
## 统一使用文件末尾的唯一入口（完整版本）。
#!/usr/bin/env python3
"""
Franka DyWA 推理节点（运行在真机侧机器）

功能：
- 侧边 RealSense：彩色(YUYV) + 深度(Z16)，对齐到彩色后编码上传策略服务器
- 订阅 EE 状态与队列/推理控制；WebSocket 发送 RGB-D + 内参 + T_base_cam + block_index
- 接收 7 维动作 chunk，发布到 /franka/action_command

使用示例：
---------
python franka_dywa_inference_node.py \\
    --policy-server-host 192.168.x.x \\
    --policy-server-port 33060 \\
    --scene-id 0 --block-index 0 \\
    --t-base-cam-csv "r11,r12,...,r44"   # 4x4 行主序，OpenCV 相机系 → base

"""
import asyncio
import base64
import websockets
import json
import dataclasses
import zlib
import time
import logging
import threading
import copy
from datetime import datetime
from typing import Optional
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32MultiArray, Bool
import tyro

try:
    from websockets.sync.client import connect as ws_connect_sync
except Exception:
    ws_connect_sync = None

@dataclasses.dataclass
class Args:
    """命令行参数"""
    
    # ========== 可调参数 ==========
    max_actions_to_publish: int = 20
    """每次推理最多发布多少个动作。
    越小=观测越频繁=任务适应性越好但chunk间停顿概率越高；
    越大=执行越连续但场景变化响应慢。建议10-25。"""
    
    gripper_threshold: float = 0.07
    """夹爪归一化阈值(米) - <=阈值表示闭合(1.0)，>阈值表示张开(0.0)"""
    
    max_state_age: float = 0.1
    """最大允许状态年龄(秒) - 超过此值跳过推理，避免使用过时状态"""
    
    action_publish_interval: float = 0.01
    """动作发布间隔(秒) - 快速填满队列；过大会导致队列始终为空、机器人停顿"""

    inference_frequency: float = 1.0
    """推理频率(Hz) - 每秒推理几次"""

    action_chunk_size: int = 50
    """每次推理返回的动作数量"""
    
    # 图像尺寸
    image_width: int = 640
    image_height: int = 480
    
    # ========== 服务器配置 ==========
    policy_server_host: str = "localhost"
    """DyWA Policy Server 地址"""

    policy_server_port: int = 33060
    """DyWA Policy Server 端口"""

    scene_id: int = 0
    """当前任务使用的 GT 场景 ID（0 -> 000.json）"""

    block_index: int = 0
    """当前搭建 GT 中第几块（与 blocks 排序后下标一致，换块时改此参数或上层逻辑）"""

    t_base_cam_csv: str = ""
    """必填：T_base_cam 4x4 行主序共16个浮点数（逗号分隔）。相机(OpenCV)坐标系到机器人 base。"""

    # ========== 相机配置 ==========
    cam_side_serial: str = "405622072640"
    camera_fps: int = 30

    # ========== ROS2话题配置 ==========
    # 机器人状态topic
    ee_states_topic: str = "/franka/ee_states"

    # 动作发布topic
    action_topic: str = "/franka/action_command"
    
    # 队列状态topic
    queue_status_topic: str = "/franka/queue_status"
    """订阅队列完成状态的topic"""
    
    # 推理控制topic
    allow_inference_topic: str = "/franka/allow_inference"
    """订阅是否允许推理的topic"""


class FrankaInferenceNode(Node):
    """Franka推理节点 - 只负责推理，不控制机器人"""

    def __init__(self, args: Args):
        super().__init__('franka_inference_node')
        self.args = args

        # 初始化文件日志
        self._setup_file_logging()

        # 连接 DyWA Policy Server 的基本信息
        self.ws_uri = f"ws://{args.policy_server_host}:{args.policy_server_port}"
        self.get_logger().info(f"DyWA Policy Server: {self.ws_uri}")
        self._ws = None
        self._ws_ctx = None
        self._ws_lock = threading.Lock()

        csv = (args.t_base_cam_csv or "").strip()
        if not csv:
            self.get_logger().error("必须提供 --t-base-cam-csv（16 个逗号分隔浮点数，camera→base 4x4 行主序）")
            raise ValueError("t_base_cam_csv 为空")
        parts = [float(x.strip()) for x in csv.split(",")]
        if len(parts) != 16:
            raise ValueError(f"t_base_cam_csv 需要 16 个数，当前 {len(parts)}")
        self._T_base_cam_row = parts

        self.latest_images = {"cam_side": None}
        self._policy_snap_lock = threading.Lock()
        self._policy_snap: Optional[dict] = None
        self._cam_fx = self._cam_fy = self._cam_cx = self._cam_cy = 0.0
        self._depth_scale = 0.001
        self.latest_ee_states = None
        self.last_ee_state_time = None  # 记录最后接收EE状态的时间
        
        # 状态锁（确保推理时使用一致的状态快照）
        self.state_lock = threading.Lock()
        
        # 队列状态跟踪
        self.queue_is_empty = False  # 队列是否为空
        self.queue_status_lock = threading.Lock()  # 保护队列状态标志
        self.can_infer = False  # 是否可以推理（收到队列完成信号后为True，推理后为False）
        
        # 推理控制标志（从推理控制节点接收）
        self.allow_inference = True  # 默认允许推理
        self.allow_inference_lock = threading.Lock()  # 保护推理控制标志

        # 初始化侧边 pyrealsense2 相机（直接采图，无需 ROS2 相机节点）
        self.camera_pipelines = {}
        self.camera_frame_queues = {}
        # demo 视频录制（每个摄像头一个 VideoWriter，在采集线程中写帧）
        self.video_writers = {}
        self._init_cameras()
        self._start_camera_threads()

        # 订阅机器人EE状态
        self.ee_states_sub = self.create_subscription(
            JointState,
            args.ee_states_topic,
            self.ee_states_callback,
            10
        )

        # 订阅队列状态
        self.queue_status_sub = self.create_subscription(
            Bool,
            args.queue_status_topic,
            self.queue_status_callback,
            10
        )

        # 订阅推理控制信号
        self.allow_inference_sub = self.create_subscription(
            Bool,
            args.allow_inference_topic,
            self.allow_inference_callback,
            10
        )

        # 发布动作指令
        self.action_pub = self.create_publisher(
            Float32MultiArray,
            args.action_topic,
            10
        )

        # 推理定时器
        inference_period = 1.0 / args.inference_frequency
        self.inference_timer = self.create_timer(
            inference_period,
            self.inference_callback
        )

        # 统计
        self.inference_count = 0
        self.last_inference_time = 0.0
        self.all_chunks: list = []

        # session 目录（用于保存图像 + 生成 HTML 报告）
        self._session_start = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._session_dir = Path(__file__).resolve().parent / "logs" / f"session_{self._session_start}"
        (self._session_dir / "images").mkdir(parents=True, exist_ok=True)
        self._init_video_writers()

        self.get_logger().info("=" * 60)
        self.get_logger().info(f"[推理节点] 已启动 | scene_id={args.scene_id} block_index={args.block_index}")
        self.get_logger().info(f"[推理节点] 频率: {args.inference_frequency}Hz | Server: {args.policy_server_host}:{args.policy_server_port}")
        self.get_logger().info("=" * 60)

    def _setup_file_logging(self):
        """设置文件日志记录"""
        # 创建logs目录
        log_dir = Path("./logs")
        log_dir.mkdir(exist_ok=True)
        
        # 创建日志文件名（带时间戳）
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"inference_{timestamp}.log"
        
        # 配置文件日志记录器
        self.file_logger = logging.getLogger("inference_file")
        self.file_logger.setLevel(logging.INFO)
        
        # 避免重复添加handler
        if not self.file_logger.handlers:
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(logging.INFO)
            formatter = logging.Formatter(
                '%(asctime)s | %(levelname)s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(formatter)
            self.file_logger.addHandler(file_handler)
        
        self.get_logger().info(f"日志文件: {log_file}")

    def _init_cameras(self):
        """侧边相机：深度 Z16 + 彩色 YUYV，深度对齐到彩色（供策略服务器 LINEMOD+点云）。"""
        W, H, fps = self.args.image_width, self.args.image_height, self.args.camera_fps
        camera_serials = {"cam_side": self.args.cam_side_serial}
        for cam_name, serial in camera_serials.items():
            try:
                pipeline = rs.pipeline()
                cfg = rs.config()
                cfg.enable_device(serial)
                cfg.enable_stream(rs.stream.depth, W, H, rs.format.z16, fps)
                cfg.enable_stream(rs.stream.color, W, H, rs.format.yuyv, fps)
                fq = rs.frame_queue(50)
                pipeline.start(cfg, fq)
                profile = pipeline.get_active_profile()
                depth_sensor = profile.get_device().first_depth_sensor()
                self._depth_scale = float(depth_sensor.get_depth_scale())
                self._rs_align = rs.align(rs.stream.color)
                self.camera_pipelines[cam_name] = pipeline
                self.camera_frame_queues[cam_name] = fq
                self.get_logger().info(
                    f"[相机] {cam_name} depth+color 已启动 depth_scale={self._depth_scale}"
                )
                time.sleep(1)
            except Exception as e:
                self.get_logger().error(f"[相机] {cam_name} (SN:{serial}) 启动失败: {e}")

    def _init_video_writers(self):
        """为每个摄像头创建 VideoWriter，保存到 session_dir/videos/"""
        videos_dir = self._session_dir / "videos"
        videos_dir.mkdir(exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        for cam_name in ("cam_side",):
            out_path = videos_dir / f"{cam_name}.mp4"
            writer = cv2.VideoWriter(
                str(out_path), fourcc, 30.0,
                (self.args.image_width, self.args.image_height),
            )
            self.video_writers[cam_name] = writer
            self.get_logger().info(f"[视频] {cam_name} 录制文件: {out_path}")

    def _restart_camera(self, cam_name: str) -> bool:
        """停止并重新启动指定相机的 pipeline，返回是否成功"""
        camera_serials = {
            'cam_side':  self.args.cam_side_serial,
        }
        serial = camera_serials.get(cam_name)
        if serial is None:
            return False
        # 停止旧 pipeline
        try:
            self.camera_pipelines[cam_name].stop()
        except Exception:
            pass
        time.sleep(1)
        # 重新启动
        try:
            W, H, fps = self.args.image_width, self.args.image_height, self.args.camera_fps
            pipeline = rs.pipeline()
            cfg = rs.config()
            cfg.enable_device(serial)
            cfg.enable_stream(rs.stream.depth, W, H, rs.format.z16, fps)
            cfg.enable_stream(rs.stream.color, W, H, rs.format.yuyv, fps)
            fq = rs.frame_queue(50)
            pipeline.start(cfg, fq)
            profile = pipeline.get_active_profile()
            self._depth_scale = float(profile.get_device().first_depth_sensor().get_depth_scale())
            self._rs_align = rs.align(rs.stream.color)
            self.camera_pipelines[cam_name] = pipeline
            self.camera_frame_queues[cam_name] = fq
            self.get_logger().info(f"[相机] {cam_name} 重启成功")
            return True
        except Exception as e:
            self.get_logger().error(f"[相机] {cam_name} 重启失败: {e}")
            return False

    def _camera_capture_thread(self, cam_name: str, fq: rs.frame_queue):
        """对齐 RGB-D，更新策略用快照（JPEG+zlib 深度）与录像帧。"""
        H = self.args.image_height
        W = self.args.image_width
        consecutive_failures = 0
        MAX_FAILURES = 5
        while rclpy.ok():
            writer = self.video_writers.get(cam_name)
            try:
                frame = fq.wait_for_frame(timeout_ms=2000)
                try:
                    fs = frame.as_frameset()
                except Exception:
                    continue
                if not hasattr(self, "_rs_align") or self._rs_align is None:
                    continue
                aligned = self._rs_align.process(fs)
                cf = aligned.get_color_frame()
                df = aligned.get_depth_frame()
                if not cf or not df:
                    continue
                if self._cam_fx == 0.0:
                    intr = cf.profile.as_video_stream_profile().intrinsics
                    self._cam_fx = float(intr.fx)
                    self._cam_fy = float(intr.fy)
                    self._cam_cx = float(intr.ppx)
                    self._cam_cy = float(intr.ppy)
                    self.get_logger().info(
                        f"[相机] 内参 fx={self._cam_fx:.4f} fy={self._cam_fy:.4f} "
                        f"cx={self._cam_cx:.4f} cy={self._cam_cy:.4f}"
                    )
                img_yuyv = np.asanyarray(cf.get_data()).view(np.uint8).reshape(H, W, 2)
                img_bgr = cv2.cvtColor(img_yuyv, cv2.COLOR_YUV2BGR_YUYV)
                depth_u16 = np.asanyarray(df.get_data(), dtype=np.uint16).copy()
                ok, enc = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 88])
                if not ok:
                    raise RuntimeError("JPEG 编码失败")
                with self._policy_snap_lock:
                    self._policy_snap = {
                        "rgb_jpeg_b64": base64.b64encode(enc.tobytes()).decode("ascii"),
                        "depth_zlib_b64": base64.b64encode(
                            zlib.compress(depth_u16.tobytes())
                        ).decode("ascii"),
                    }
                # 仅在有效 RGB-D 成功编码并写入快照后，才重置失败计数
                consecutive_failures = 0
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                with self.state_lock:
                    self.latest_images[cam_name] = img_rgb
                if writer is not None and writer.isOpened():
                    writer.write(img_bgr)
            except Exception as e:
                if not rclpy.ok():
                    break
                consecutive_failures += 1
                self.get_logger().warn(
                    f"[相机] {cam_name} 取帧失败 ({consecutive_failures}/{MAX_FAILURES}): {e}"
                )
                if consecutive_failures >= MAX_FAILURES:
                    self.get_logger().warn(f"[相机] {cam_name} 连续失败，尝试重启...")
                    if self._restart_camera(cam_name):
                        fq = self.camera_frame_queues[cam_name]
                        consecutive_failures = 0
                    else:
                        time.sleep(2)

    def _start_camera_threads(self):
        """为每个相机启动后台采集线程"""
        for cam_name, fq in self.camera_frame_queues.items():
            t = threading.Thread(
                target=self._camera_capture_thread,
                args=(cam_name, fq),
                daemon=True,
                name=f"camera_{cam_name}",
            )
            t.start()
            self.get_logger().info(f"[相机] {cam_name} 采集线程已启动")

    def _save_sent_images(self, obs: dict, inference_num: int):
        """（可选）根据需要保存侧边相机图像，目前 DyWA server 不需要图像，可留空实现。"""
        return

    def _ensure_ws_connected(self):
        if self._ws is not None:
            return
        if ws_connect_sync is None:
            raise RuntimeError("websockets.sync.client 不可用，无法建立长连接")
        # 兼容不同 websockets 版本：
        # - 某些版本 connect() 直接返回连接对象
        # - 某些版本 connect() 返回上下文管理器
        self._ws_ctx = ws_connect_sync(self.ws_uri, open_timeout=3.0, close_timeout=1.0)
        if hasattr(self._ws_ctx, "__enter__") and hasattr(self._ws_ctx, "__exit__"):
            self._ws = self._ws_ctx.__enter__()
        else:
            self._ws = self._ws_ctx
        self.get_logger().info(f"[WS] 已连接 {self.ws_uri}")

    def _close_ws(self):
        if self._ws is None and self._ws_ctx is None:
            return
        try:
            # 优先走上下文退出，保证底层资源完全释放
            if self._ws_ctx is not None and hasattr(self._ws_ctx, "__exit__"):
                self._ws_ctx.__exit__(None, None, None)
            elif self._ws is not None:
                self._ws.close()
        except Exception:
            pass
        finally:
            self._ws = None
            self._ws_ctx = None

    def _request_server_sync(self, req: dict) -> dict:
        with self._ws_lock:
            for attempt in range(2):
                try:
                    self._ensure_ws_connected()
                    self._ws.send(json.dumps(req))
                    resp_raw = self._ws.recv()
                    return json.loads(resp_raw)
                except Exception as e:
                    self._close_ws()
                    if attempt == 0:
                        self.get_logger().warn(f"[WS] 请求失败，重连后重试一次: {e}")
                        continue
                    raise
        raise RuntimeError("WebSocket 请求失败")

    def destroy_node(self):
        """清理：释放视频文件，停止所有相机 pipeline"""
        self._close_ws()
        # 先释放 VideoWriter，确保视频文件写完整
        for cam_name, writer in self.video_writers.items():
            try:
                writer.release()
                self.get_logger().info(f"[视频] {cam_name} 已保存")
            except Exception:
                pass
        for cam_name, pipeline in self.camera_pipelines.items():
            try:
                pipeline.stop()
                self.get_logger().info(f"[相机] {cam_name} 已停止")
            except Exception:
                pass
        super().destroy_node()

    def ee_states_callback(self, msg: JointState):
        """接收机器人EE状态"""
        current_time = time.time()
        with self.state_lock:
            self.latest_ee_states = msg
            self.last_ee_state_time = current_time
    
    def queue_status_callback(self, msg: Bool):
        """接收队列状态"""
        # 先获取两个锁，避免死锁
        with self.queue_status_lock:
            with self.allow_inference_lock:
                old_queue_empty = self.queue_is_empty
                old_can_infer = self.can_infer
                
                if msg.data:  # True表示队列执行完成（队列为空）
                    if not self.queue_is_empty:
                        self.get_logger().info("[队列] 清空 ✓")
                    self.queue_is_empty = True
                    # 队列清空时，如果允许推理，设置can_infer为True
                    self.can_infer = self.allow_inference
                    if self.can_infer and not old_can_infer:
                        self.get_logger().info(f"[队列] 可以推理 ✓ | allow_inference={self.allow_inference}")
                        self.file_logger.info(f"[队列] can_infer变为True | allow_inference={self.allow_inference}")
                else:
                    if self.queue_is_empty:
                        self.get_logger().info("[队列] 有动作")
                        self.file_logger.info("[队列] 收到有动作信号")
                    self.queue_is_empty = False
                    self.can_infer = False
    
    def allow_inference_callback(self, msg: Bool):
        """接收推理控制信号"""
        # 先获取两个锁，避免死锁（与queue_status_callback保持相同顺序）
        with self.queue_status_lock:
            with self.allow_inference_lock:
                old_allow = self.allow_inference
                old_can_infer = self.can_infer
                self.allow_inference = msg.data
                # 每次收到信号都更新can_infer
                # can_infer = 允许推理 AND 队列为空
                self.can_infer = msg.data and self.queue_is_empty
        
        if old_allow != msg.data:
            status = "允许推理 ✓" if msg.data else "禁止推理"
            self.get_logger().info(f"[控制] {status} | queue_empty={self.queue_is_empty} | can_infer={self.can_infer}")
            self.file_logger.info(f"[控制] {status} | queue_empty={self.queue_is_empty} | can_infer={self.can_infer}")
        elif self.can_infer and not old_can_infer:
            self.get_logger().info(f"[控制] can_infer变为True | allow_inference={msg.data} | queue_empty={self.queue_is_empty}")
            self.file_logger.info(f"[控制] can_infer变为True | allow_inference={msg.data} | queue_empty={self.queue_is_empty}")

    def get_observation(self) -> Optional[dict]:
        """构造观察数据：仅使用 EE 7 维状态

        返回:
            dict: {
              \"state\": np.ndarray(7,),  # [x,y,z,roll,pitch,yaw,gripper]
            }
        """
        # 获取状态快照（使用锁确保一致性）
        with self.state_lock:
            # 检查EE状态
            if self.latest_ee_states is None:
                warn_msg = "等待机器人EE状态"
                self.get_logger().warn(warn_msg)
                self.file_logger.warning(warn_msg)
                return None

            # 提取EE状态: 7维 [x, y, z, roll, pitch, yaw, gripper_width]
            if len(self.latest_ee_states.position) < 7:
                warn_msg = f"EE状态维度不足: {len(self.latest_ee_states.position)}, 需要7维"
                self.get_logger().warn(warn_msg)
                self.file_logger.warning(warn_msg)
                return None

            # 深拷贝状态，避免在推理过程中状态被更新
            ee_state_snapshot = copy.deepcopy(self.latest_ee_states.position)

        # 在锁外处理数据（避免长时间持有锁）
        # 提取EE位置和姿态 (前6维)
        ee_position = list(ee_state_snapshot[:3])  # [x, y, z]
        ee_rpy = list(ee_state_snapshot[3:6])  # [roll, pitch, yaw]
        
        # 处理夹爪：归一化到0/1
        gripper_value = 1.0 if ee_state_snapshot[6] <= self.args.gripper_threshold else 0.0

        # 构造7维观察状态: [x, y, z, roll, pitch, yaw, gripper]
        observation_state = np.array(
            ee_position + ee_rpy + [float(gripper_value)],
            dtype=np.float32
        )  # shape: (7,)

        self.file_logger.debug(f"推理使用状态快照: pos={ee_position}, rpy={ee_rpy}, gripper={gripper_value}")

        return {
            "state": observation_state,
        }

    def inference_callback(self):
        """推理回调 - 仅在允许推理时执行推理并发布动作"""
        # 检查推理控制信号和队列状态
        with self.queue_status_lock:
            with self.allow_inference_lock:
                allow_inference = self.allow_inference
                can_infer_now = self.can_infer
                queue_empty = self.queue_is_empty
        
        # 调试：每10次回调记录一次状态（避免日志过多）
        if not hasattr(self, '_inference_callback_count'):
            self._inference_callback_count = 0
        self._inference_callback_count += 1
        
        # 每10次回调记录一次详细状态
        if self._inference_callback_count % 10 == 0:
            self.get_logger().info(
                f"[推理检查#{self._inference_callback_count}] "
                f"allow={allow_inference} | can_infer={can_infer_now} | "
                f"queue_empty={queue_empty} | 已推理={self.inference_count}"
            )
            self.file_logger.info(
                f"[推理检查#{self._inference_callback_count}] "
                f"allow={allow_inference} | can_infer={can_infer_now} | "
                f"queue_empty={queue_empty}"
            )
        
        if not allow_inference:
            if self._inference_callback_count % 20 == 0:  # 每20次输出一次
                self.get_logger().warn("[推理检查] 禁止推理 - 等待允许推理信号")
            return  # 禁止推理，跳过
        
        if not can_infer_now:
            if self._inference_callback_count % 20 == 0:  # 每20次输出一次
                self.get_logger().warn(
                    f"[推理检查] 未满足推理条件 - can_infer={can_infer_now}, queue_empty={queue_empty}"
                )
            return  # 未满足推理条件，跳过
        
        with self.state_lock:
            if self.latest_ee_states is None or len(self.latest_ee_states.position) < 7:
                if self._inference_callback_count % 20 == 0:
                    self.get_logger().warn("[推理检查] 无有效 EE 状态")
                return
            ee_raw = [float(x) for x in self.latest_ee_states.position[:7]]
            state_age = time.time() - self.last_ee_state_time if self.last_ee_state_time else 0

        if state_age > self.args.max_state_age:
            if self._inference_callback_count % 20 == 0:
                self.get_logger().warn(
                    f"[推理检查] 状态过时 - 状态年龄={state_age*1000:.1f}ms > "
                    f"最大允许={self.args.max_state_age*1000:.1f}ms"
                )
            return

        with self._policy_snap_lock:
            snap = self._policy_snap
        if snap is None or self._cam_fx <= 0:
            if self._inference_callback_count % 20 == 0:
                self.get_logger().warn("[推理检查] 等待相机 RGB-D 与内参就绪")
            return

        req = {
            "scene_id": int(self.args.scene_id),
            "block_index": int(self.args.block_index),
            "ee_state": ee_raw,
            "image_width": self.args.image_width,
            "image_height": self.args.image_height,
            "fx": self._cam_fx,
            "fy": self._cam_fy,
            "cx": self._cam_cx,
            "cy": self._cam_cy,
            "depth_scale": self._depth_scale,
            "T_base_cam": self._T_base_cam_row,
            "rgb_jpeg_b64": snap["rgb_jpeg_b64"],
            "depth_zlib_b64": snap["depth_zlib_b64"],
        }

        try:
            start_time = time.time()

            resp = self._request_server_sync(req)
            inference_time = time.time() - start_time
            self.last_inference_time = inference_time

            if resp.get("error"):
                self.get_logger().error(f"[推理] 服务器: {resp['error']}")
                self.file_logger.error(resp["error"])
                return
            if resp.get("scene_done"):
                self.get_logger().warn("[推理] scene_done=True，已无更多积木下标")
                return
            if "block_index" in resp:
                try:
                    new_bi = int(resp["block_index"])
                    if new_bi != int(self.args.block_index):
                        self.get_logger().info(f"[任务] 切换 block_index: {self.args.block_index} -> {new_bi}")
                        self.args.block_index = new_bi
                except Exception:
                    pass

            actions = resp.get("actions", [])
            self.inference_count += 1

            self.get_logger().info(
                f"[推理#{self.inference_count}] 完成 | 耗时: {inference_time*1000:.1f}ms | "
                f"动作数: {len(actions)} | pos_err_block_m={resp.get('pos_err_block_m', 'n/a')}"
            )

            actions_to_publish = actions[: self.args.max_actions_to_publish]

            self.all_chunks.append(actions_to_publish)

            # 发布动作后，重置推理标志，等待下次队列清空信号
            with self.queue_status_lock:
                self.can_infer = False

            # 发布动作
            for i, action in enumerate(actions_to_publish):
                try:
                    msg = Float32MultiArray()
                    msg.data = [float(x) for x in action]
                    self.action_pub.publish(msg)
                except Exception as pub_error:
                    self.get_logger().error(f"[推理#{self.inference_count}] 发布动作[{i}]失败: {pub_error}")
                    self.file_logger.error(f"发布动作失败 [{i}]: {pub_error}", exc_info=True)

                if i < len(actions_to_publish) - 1:
                    time.sleep(self.args.action_publish_interval)

            self.get_logger().info(f"[推理#{self.inference_count}] 已发布 {len(actions_to_publish)} 个动作")

        except Exception as e:
            self.get_logger().error(f"[推理#{self.inference_count + 1}] 失败: {e}")
            self.file_logger.error(f"推理失败: {e}", exc_info=True)


def main(args: Args):
    """主函数"""
    rclpy.init()
    node = FrankaInferenceNode(args)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n收到中断信号，停止...")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main(tyro.cli(Args))
