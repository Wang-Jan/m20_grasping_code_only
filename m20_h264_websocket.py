#!/usr/bin/env python3
"""M20 dual-camera RTSP -> H.264 -> WebSocket bridge.

The robot's original cameras are exposed by MediaMTX as RTSP streams.  This
module does not use ROS2 topics and does not change anything on AOS/NOS/GOS:
ffmpeg reads both RTSP streams, encodes them as low-latency H.264 Annex-B, and
the binary WebSocket messages carry a small camera/sequence header followed by
one complete H.264 access unit.  A publisher connection can send both cameras
to an existing WebSocket server; a server connection can broadcast them to
LAN clients.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import signal
import struct
import sys
import time
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional

from m20_config import H264_WS_SERVER, M20_IP, RTSP_PORT, rtsp_url


MAGIC = b"M20H264\x00"
PROTOCOL_VERSION = 1
# magic, version, camera (0=front, 1=back), flags, sequence, unix timestamp ms
MESSAGE_HEADER = struct.Struct("!8sBBHIQ")
FLAG_KEYFRAME = 0x0001
CAMERA_IDS = {"front": 0, "back": 1}
CAMERA_NAMES = {value: key for key, value in CAMERA_IDS.items()}


def _find_start_code(data: bytearray, offset: int = 0) -> int:
    """Find the earliest 3/4-byte Annex-B start code."""

    start3 = data.find(b"\x00\x00\x01", offset)
    start4 = data.find(b"\x00\x00\x00\x01", offset)
    if start4 >= 0 and (start3 < 0 or start4 <= start3):
        return start4
    return start3


def _nal_type(nal: bytes) -> Optional[int]:
    if nal.startswith(b"\x00\x00\x00\x01"):
        index = 4
    elif nal.startswith(b"\x00\x00\x01"):
        index = 3
    else:
        return None
    if index >= len(nal):
        return None
    return nal[index] & 0x1F


class AnnexBAccessUnitParser:
    """Turn ffmpeg's H.264 Annex-B byte stream into access units.

    The encoder is configured to insert an Access Unit Delimiter (AUD) before
    every frame.  This lets us drop complete old frames when a WebSocket client
    is slow instead of dropping a random NAL unit and corrupting the decoder.
    """

    def __init__(self, max_buffer_bytes: int = 16 * 1024 * 1024):
        self.buffer = bytearray()
        self.current = bytearray()
        self.max_buffer_bytes = max_buffer_bytes
        self.seen_aud = False

    def _complete_nals(self) -> list[bytes]:
        nals: list[bytes] = []
        while True:
            start = _find_start_code(self.buffer)
            if start < 0:
                if len(self.buffer) > 3:
                    del self.buffer[:-3]
                break
            if start:
                del self.buffer[:start]
            next_start = _find_start_code(self.buffer, 3)
            if next_start < 0:
                break
            nals.append(bytes(self.buffer[:next_start]))
            del self.buffer[:next_start]
        if len(self.buffer) > self.max_buffer_bytes:
            raise ValueError("H.264 Annex-B 缓冲区超过上限，流格式可能异常")
        return nals

    def feed(self, data: bytes) -> list[bytes]:
        self.buffer.extend(data)
        access_units: list[bytes] = []
        for nal in self._complete_nals():
            nal_type = _nal_type(nal)
            if nal_type == 9:  # AUD: starts a new access unit
                self.seen_aud = True
                if self.current:
                    access_units.append(bytes(self.current))
                    self.current.clear()
            elif not self.seen_aud and nal_type in (1, 5) and self.current:
                # Fallback for an encoder that ignores aud=1: split on VCL NALs.
                access_units.append(bytes(self.current))
                self.current.clear()
            self.current.extend(nal)
        return access_units

    def flush(self) -> list[bytes]:
        """Flush complete trailing data when ffmpeg exits."""

        access_units = self.feed(b"")
        # ``feed`` deliberately keeps the last NAL until it sees the next
        # start code.  At EOF that final NAL is complete, so append it now.
        if self.buffer:
            start = _find_start_code(self.buffer)
            if start >= 0:
                if start:
                    del self.buffer[:start]
                self.current.extend(self.buffer)
                self.buffer.clear()
        if self.current:
            access_units.append(bytes(self.current))
            self.current.clear()
        return access_units


def access_unit_is_keyframe(access_unit: bytes) -> bool:
    """Return whether an access unit contains an IDR NAL (type 5)."""

    cursor = 0
    data = bytearray(access_unit)
    while True:
        start = _find_start_code(data, cursor)
        if start < 0:
            return False
        next_start = _find_start_code(data, start + 3)
        nal = bytes(data[start:] if next_start < 0 else data[start:next_start])
        if _nal_type(nal) == 5:
            return True
        if next_start < 0:
            return False
        cursor = next_start


def pack_message(camera: str, sequence: int, access_unit: bytes, timestamp_ms: Optional[int] = None) -> bytes:
    """Build one binary WebSocket message for a complete H.264 access unit."""

    if camera not in CAMERA_IDS:
        raise ValueError(f"未知摄像头: {camera}")
    if not access_unit:
        raise ValueError("access_unit 不能为空")
    flags = FLAG_KEYFRAME if access_unit_is_keyframe(access_unit) else 0
    timestamp_ms = int(time.time() * 1000) if timestamp_ms is None else int(timestamp_ms)
    header = MESSAGE_HEADER.pack(
        MAGIC,
        PROTOCOL_VERSION,
        CAMERA_IDS[camera],
        flags,
        int(sequence) & 0xFFFFFFFF,
        timestamp_ms & 0xFFFFFFFFFFFFFFFF,
    )
    return header + access_unit


@dataclass(frozen=True)
class DecodedMessage:
    camera: str
    sequence: int
    timestamp_ms: int
    keyframe: bool
    access_unit: bytes


def unpack_message(message: bytes) -> DecodedMessage:
    """Validate and unpack one protocol message."""

    if len(message) < MESSAGE_HEADER.size:
        raise ValueError("WebSocket H.264 消息过短")
    magic, version, camera_id, flags, sequence, timestamp_ms = MESSAGE_HEADER.unpack_from(message)
    if magic != MAGIC or version != PROTOCOL_VERSION:
        raise ValueError("不支持的 M20H264 WebSocket 消息头")
    if camera_id not in CAMERA_NAMES:
        raise ValueError(f"未知摄像头编号: {camera_id}")
    access_unit = message[MESSAGE_HEADER.size :]
    if not access_unit:
        raise ValueError("WebSocket H.264 消息没有视频载荷")
    return DecodedMessage(
        camera=CAMERA_NAMES[camera_id],
        sequence=sequence,
        timestamp_ms=timestamp_ms,
        keyframe=bool(flags & FLAG_KEYFRAME),
        access_unit=access_unit,
    )


def build_ffmpeg_h264_command(args, url: str, executable: str = "ffmpeg") -> list[str]:
    """Build a CFR 30-FPS low-latency H.264 Annex-B encoder command."""

    filters = []
    if args.width or args.height:
        width = args.width or -2
        height = args.height or -2
        filters.append(f"scale={width}:{height}:flags=fast_bilinear")
    filters.append(f"fps={args.fps:g}")

    command = [
        executable,
        "-hide_banner",
        "-loglevel",
        "warning",
        "-rtsp_transport",
        "tcp",
        "-fflags",
        "nobuffer",
        "-flags",
        "low_delay",
        "-analyzeduration",
        "0",
        "-probesize",
        "32",
        "-i",
        url,
        "-map",
        "0:v:0",
        "-an",
        "-sn",
        "-dn",
        "-vf",
        ",".join(filters),
        "-r",
        f"{args.fps:g}",
        "-c:v",
        args.encoder,
        "-pix_fmt",
        "yuv420p",
        "-g",
        str(max(1, round(args.fps))),
        "-keyint_min",
        str(max(1, round(args.fps))),
        "-bf",
        "0",
    ]
    if args.encoder == "libx264":
        command.extend(
            [
                "-preset",
                args.preset,
                "-tune",
                "zerolatency",
                "-profile:v",
                "baseline",
                "-sc_threshold",
                "0",
                "-x264-params",
                "repeat-headers=1:aud=1",
            ]
        )
    command.extend(["-f", "h264", "pipe:1"])
    return command


async def _terminate_process(process) -> None:
    if process.returncode is not None:
        return
    process.terminate()
    try:
        await asyncio.wait_for(process.wait(), timeout=2.0)
    except asyncio.TimeoutError:
        process.kill()
        await process.wait()


async def _drain_stderr(stream, label: str) -> None:
    data = await stream.read()
    lines = data.decode("utf-8", errors="replace").strip().splitlines()
    if lines:
        print(f"[FFMPEG {label}] " + " | ".join(lines[-5:]), file=sys.stderr)


async def ffmpeg_camera_loop(
    camera: str,
    url: str,
    args,
    on_access_unit: Callable[[str, bytes], Awaitable[None]],
    stop_event: asyncio.Event,
) -> None:
    """Read one RTSP stream, encode it, and reconnect after source failures."""

    import shutil

    executable = shutil.which("ffmpeg")
    if executable is None:
        raise RuntimeError("找不到 ffmpeg；请先安装 ffmpeg")

    sequence = 0
    while not stop_event.is_set():
        command = build_ffmpeg_h264_command(args, url, executable)
        print(f"[VIDEO] {camera} 启动 H.264 RTSP 解码/编码: {' '.join(command)}")
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stderr_task = asyncio.create_task(_drain_stderr(process.stderr, camera))
        parser = AnnexBAccessUnitParser()
        try:
            while not stop_event.is_set():
                chunk = await process.stdout.read(64 * 1024)
                if not chunk:
                    break
                for access_unit in parser.feed(chunk):
                    sequence += 1
                    await on_access_unit(camera, pack_message(camera, sequence, access_unit))
            for access_unit in parser.flush():
                sequence += 1
                await on_access_unit(camera, pack_message(camera, sequence, access_unit))
        finally:
            await _terminate_process(process)
            with contextlib.suppress(asyncio.CancelledError):
                await stderr_task
        if stop_event.is_set():
            return
        print(f"[WARN] {camera} RTSP/ffmpeg 已断开，{args.reconnect_delay:g} 秒后重连")
        await asyncio.sleep(args.reconnect_delay)


class BroadcastHub:
    """Latest-frame broadcaster; slow clients discard old complete frames."""

    def __init__(self, queue_size: int = 3):
        self.queue_size = queue_size
        self.clients: set[tuple[asyncio.Queue, Optional[str]]] = set()

    def add(self, camera_filter: Optional[str]):
        queue: asyncio.Queue = asyncio.Queue(maxsize=self.queue_size)
        client = (queue, camera_filter)
        self.clients.add(client)
        return client

    def remove(self, client) -> None:
        self.clients.discard(client)

    async def publish(self, camera: str, message: bytes) -> None:
        for queue, camera_filter in list(self.clients):
            if camera_filter is not None and camera_filter != camera:
                continue
            if queue.full():
                with contextlib.suppress(asyncio.QueueEmpty):
                    queue.get_nowait()
            with contextlib.suppress(asyncio.QueueFull):
                queue.put_nowait(message)


def _path_camera_filter(path: str) -> Optional[str]:
    path = path.split("?", 1)[0].rstrip("/")
    if path in ("", "/", "/all", "/video"):
        return None
    raise ValueError("WebSocket 视频路径只能是 /all")


async def websocket_server_handler(websocket, path: str, hub: BroadcastHub) -> None:
    try:
        camera_filter = _path_camera_filter(path)
    except ValueError:
        await websocket.close(code=1008, reason="unsupported camera path")
        return
    client = hub.add(camera_filter)
    print(f"[VIDEO] 客户端已连接 path={path or '/all'} filter={camera_filter or 'both'}")
    try:
        while True:
            message = await client[0].get()
            await asyncio.wait_for(websocket.send(message), timeout=1.0)
    except Exception as exc:
        if exc.__class__.__name__ not in {"ConnectionClosed", "ConnectionClosedOK", "ConnectionClosedError"}:
            print(f"[WARN] WebSocket 客户端断开: {exc}")
    finally:
        hub.remove(client)
        print("[VIDEO] WebSocket 客户端已断开")


async def run_server(args) -> None:
    try:
        from websockets.legacy.server import serve
    except ImportError as exc:
        raise RuntimeError("缺少 websockets 依赖，请执行: python3 -m pip install -r requirements-video.txt") from exc

    hub = BroadcastHub()
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(signum, stop_event.set)
    producer_tasks = [
        asyncio.create_task(ffmpeg_camera_loop("front", args.front_url, args, hub.publish, stop_event)),
        asyncio.create_task(ffmpeg_camera_loop("back", args.back_url, args, hub.publish, stop_event)),
    ]
    handler = lambda websocket, path: websocket_server_handler(websocket, path, hub)
    try:
        async with serve(
            handler,
            args.host,
            args.port,
            compression=None,
            max_size=32 * 1024 * 1024,
            ping_interval=20,
            ping_timeout=10,
            close_timeout=2,
        ):
            print(f"[VIDEO] WebSocket 服务: ws://{args.host}:{args.port}/all")
            print("[VIDEO] 客户端路径：/all（前后双路）")
            await stop_event.wait()
    finally:
        stop_event.set()
        for task in producer_tasks:
            task.cancel()
        await asyncio.gather(*producer_tasks, return_exceptions=True)


async def publish_to_server_once(args, websocket_module) -> None:
    async with websocket_module.connect(
        args.server,
        max_size=32 * 1024 * 1024,
        compression=None,
        ping_interval=20,
        ping_timeout=10,
    ) as websocket:
        print(f"[VIDEO] 已连接中心 WebSocket: {args.server}")
        stop_event = asyncio.Event()
        send_lock = asyncio.Lock()

        async def send_message(_camera: str, message: bytes) -> None:
            async with send_lock:
                await websocket.send(message)

        tasks = [
            asyncio.create_task(ffmpeg_camera_loop("front", args.front_url, args, send_message, stop_event)),
            asyncio.create_task(ffmpeg_camera_loop("back", args.back_url, args, send_message, stop_event)),
        ]
        try:
            await asyncio.gather(*tasks)
        finally:
            stop_event.set()
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)


async def run_publisher(args) -> None:
    try:
        import websockets
    except ImportError as exc:
        raise RuntimeError("缺少 websockets 依赖，请执行: python3 -m pip install -r requirements-video.txt") from exc
    while True:
        try:
            await publish_to_server_once(args, websockets)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            print(f"[WARN] 中心 WebSocket/视频链路断开: {exc}；{args.reconnect_delay:g} 秒后重连")
            await asyncio.sleep(args.reconnect_delay)


async def run_receiver(args) -> None:
    try:
        import websockets
    except ImportError as exc:
        raise RuntimeError("缺少 websockets 依赖，请执行: python3 -m pip install -r requirements-video.txt") from exc
    if args.camera not in CAMERA_IDS:
        raise RuntimeError("接收模式必须指定 --camera front 或 --camera back")
    while True:
        try:
            async with websockets.connect(
                args.server,
                max_size=32 * 1024 * 1024,
                compression=None,
                ping_interval=20,
                ping_timeout=10,
            ) as websocket:
                print(f"[VIDEO] 接收 {args.camera} H.264: {args.server}", file=sys.stderr)
                while True:
                    message = await websocket.recv()
                    if isinstance(message, str):
                        continue
                    decoded = unpack_message(message)
                    if decoded.camera != args.camera:
                        continue
                    sys.stdout.buffer.write(decoded.access_unit)
                    sys.stdout.buffer.flush()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            print(f"[WARN] 接收链路断开: {exc}；{args.reconnect_delay:g} 秒后重连", file=sys.stderr)
            await asyncio.sleep(args.reconnect_delay)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="M20 双路 RTSP 转 H.264 WebSocket（不使用 ROS2 话题）")
    parser.add_argument("--mode", choices=("serve", "publish", "receive"), default="serve")
    parser.add_argument("--ip", default=M20_IP, help=f"机器狗 AOS IP，默认 {M20_IP}")
    parser.add_argument("--rtsp-port", type=int, default=RTSP_PORT)
    parser.add_argument("--front-url", help="覆盖前相机 RTSP 地址")
    parser.add_argument("--back-url", help="覆盖后相机 RTSP 地址")
    parser.add_argument(
        "--server",
        default=H264_WS_SERVER or None,
        help="中心 WebSocket 地址，例如 ws://10.21.31.100:9000/m20；也可设 M20_H264_WS_SERVER",
    )
    parser.add_argument("--host", default="0.0.0.0", help="serve 模式监听地址")
    parser.add_argument("--port", type=int, default=8765, help="serve 模式 WebSocket 端口")
    parser.add_argument("--camera", choices=("front", "back"), help="receive 模式要输出的摄像头")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--encoder", default="libx264", help="H.264 编码器，默认 libx264；可选 h264_nvenc")
    parser.add_argument("--preset", default="veryfast", help="libx264 预设，默认 veryfast")
    parser.add_argument("--reconnect-delay", type=float, default=1.0)
    args = parser.parse_args(argv)
    if args.width < 0 or args.height < 0 or (args.width == 0) != (args.height == 0):
        parser.error("--width 和 --height 必须同时为正数，或同时为 0")
    if args.fps <= 0 or args.reconnect_delay <= 0:
        parser.error("--fps 和 --reconnect-delay 必须大于 0")
    if not 1 <= args.port <= 65535:
        parser.error("--port 必须在 1 到 65535 之间")
    args.front_url = args.front_url or rtsp_url("video1", ip=args.ip, port=args.rtsp_port)
    args.back_url = args.back_url or rtsp_url("video2", ip=args.ip, port=args.rtsp_port)
    if args.mode in ("publish", "receive") and not args.server:
        parser.error(f"--mode {args.mode} 必须指定 --server ws://服务器IP:端口/路径")
    if args.mode == "receive" and not args.camera:
        parser.error("--mode receive 必须指定 --camera front 或 --camera back")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        if args.mode == "serve":
            asyncio.run(run_server(args))
        elif args.mode == "publish":
            asyncio.run(run_publisher(args))
        else:
            asyncio.run(run_receiver(args))
    except KeyboardInterrupt:
        pass
    except RuntimeError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
