"""LAN WebSocket control, status, and dual-camera service."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any

if str(Path(__file__).resolve().parents[2]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import m20_h264_websocket as video


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROUTES_FILE = PROJECT_ROOT / "config/m20/routes.json"
ROUTES = ("route1", "route2", "route3", "route4")
STATES = ("idle", "starting", "running", "stopping", "succeeded", "failed", "stopped", "unavailable")


class RouteManager:
    def __init__(self, routes_file: Path = ROUTES_FILE):
        self.command_lock = asyncio.Lock()
        self.routes_file = routes_file
        self.process: asyncio.subprocess.Process | None = None
        self.state = "idle"
        self.route: str | None = None
        self.loops: int | None = None
        self.message = "服务已启动"
        self.failure: str | None = None
        self.last_logs: list[str] = []
        self.subscribers: set[asyncio.Queue] = set()

    def route_available(self, route: str) -> bool:
        try:
            data = json.loads(self.routes_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        return isinstance(data, dict) and isinstance(data.get(route), list) and bool(data[route])

    def snapshot(self) -> dict[str, Any]:
        return {
            "type": "status", "state": self.state, "route": self.route,
            "loops": self.loops, "pid": self.process.pid if self.process else None,
            "message": self.message, "failure": self.failure,
            "logs": self.last_logs[-10:], "updated_at": time.time(),
        }

    async def _publish(self) -> None:
        item = self.snapshot()
        for queue in list(self.subscribers):
            if queue.full():
                with contextlib.suppress(asyncio.QueueEmpty):
                    queue.get_nowait()
            with contextlib.suppress(asyncio.QueueFull):
                queue.put_nowait(item)

    async def _read_output(self, stream) -> None:
        while True:
            line = await stream.readline()
            if not line:
                return
            text = line.decode("utf-8", errors="replace").rstrip()
            self.last_logs.append(text)
            del self.last_logs[:-50]
            if self.state == "starting":
                self.state = "running"
            self.message = text[-300:]
            await self._publish()

    async def start(self, route: str, loops: int) -> dict[str, Any]:
        async with self.command_lock:
            return await self._start(route, loops)

    async def _start(self, route: str, loops: int) -> dict[str, Any]:
        if self.process and self.process.returncode is None:
            return {"ok": False, "state": self.state, "route": self.route, "error": "已有巡检任务运行中"}
        if route not in ROUTES:
            return {"ok": False, "error": f"不支持的路线: {route}"}
        if not self.route_available(route):
            self.state = "unavailable"
            self.route, self.message = route, "该路线尚未完成路线标注"
            await self._publish()
            return {"ok": False, "state": self.state, "route": route, "error": self.message}
        if loops < 1:
            return {"ok": False, "error": "loops 必须至少为 1"}
        self.last_logs.clear()
        self.route, self.loops, self.failure = route, loops, None
        self.state, self.message = "starting", "正在启动巡检"
        await self._publish()
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONPATH"] = str(PROJECT_ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
        try:
            self.process = await asyncio.create_subprocess_exec(
                sys.executable, "-m", "m20_rebot_moving_grasp.route_main",
                "--run-real", "--route-name", route, "--route-loops", str(loops),
                cwd=str(PROJECT_ROOT), env=env,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
            )
        except OSError as exc:
            self.process = None
            self.state, self.failure, self.message = "failed", str(exc), "巡检进程启动失败"
            await self._publish()
            return {"ok": False, "state": self.state, "route": route, "error": self.message}
        asyncio.create_task(self._read_output(self.process.stdout))
        asyncio.create_task(self._watch_process(self.process))
        return {"ok": True, "state": self.state, "route": route, "loops": loops}

    async def _watch_process(self, process) -> None:
        code = await process.wait()
        if self.process is not process:
            return
        if self.state == "stopping" and code == 0:
            self.state, self.message = "stopped", "巡检已安全停止"
        elif self.state == "stopping":
            self.state, self.failure, self.message = "failed", f"停止进程退出码 {code}", "停止未完成确认，请检查日志和设备状态"
        elif code == 0:
            self.state, self.message = "succeeded", "巡检完成"
        else:
            self.state, self.failure, self.message = "failed", f"进程退出码 {code}", "巡检异常退出"
        self.process = None
        await self._publish()

    async def stop(self) -> dict[str, Any]:
        async with self.command_lock:
            return await self._stop()

    async def _stop(self) -> dict[str, Any]:
        process = self.process
        if not process or process.returncode is not None:
            return {"ok": False, "state": self.state, "error": "当前没有运行中的巡检"}
        if self.state == "stopping":
            return {"ok": True, "state": self.state, "message": "停止请求已发送，等待设备清理完成"}
        try:
            process.send_signal(signal.SIGINT)
        except ProcessLookupError:
            return {"ok": False, "state": self.state, "error": "巡检进程已退出，请查询状态"}
        self.state, self.message = "stopping", "正在执行安全停止"
        await self._publish()
        return {"ok": True, "state": self.state, "message": self.message}


async def control_handler(websocket, manager: RouteManager) -> None:
    queue: asyncio.Queue = asyncio.Queue(maxsize=8)
    manager.subscribers.add(queue)
    await websocket.send(json.dumps(manager.snapshot(), ensure_ascii=False))
    sender = asyncio.create_task(_send_status(websocket, queue))
    try:
        async for raw in websocket:
            if not isinstance(raw, str):
                continue
            try:
                request = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send(json.dumps({"ok": False, "error": "请求必须是 JSON"}, ensure_ascii=False))
                continue
            if not isinstance(request, dict):
                await websocket.send(json.dumps({"ok": False, "error": "请求必须是 JSON 对象"}, ensure_ascii=False))
                continue
            action = request.get("action")
            if action == "status":
                result = manager.snapshot()
            elif action == "start":
                loops = request.get("loops", 1)
                if type(loops) is not int:
                    loops = 0
                result = await manager.start(str(request.get("route", "")), loops)
            elif action == "stop":
                result = await manager.stop()
            else:
                result = {"ok": False, "error": "action 只能是 start、status 或 stop"}
            result = {"type": "response", "request_id": request.get("request_id"), **result}
            await websocket.send(json.dumps(result, ensure_ascii=False))
    finally:
        sender.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await sender
        manager.subscribers.discard(queue)


async def _send_status(websocket, queue: asyncio.Queue) -> None:
    while True:
        item = await queue.get()
        await websocket.send(json.dumps(item, ensure_ascii=False))


async def run(args) -> None:
    try:
        from websockets.legacy.server import serve
    except ImportError as exc:
        raise RuntimeError("缺少 websockets 依赖，请安装 requirements-video.txt") from exc
    manager = RouteManager()
    hub = video.BroadcastHub()
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(signum, stop_event.set)
    video_args = video.parse_args(["--mode", "serve", "--host", args.host, "--port", str(args.port)])
    producers = [
        asyncio.create_task(video.ffmpeg_camera_loop("front", video_args.front_url, video_args, hub.publish, stop_event)),
        asyncio.create_task(video.ffmpeg_camera_loop("back", video_args.back_url, video_args, hub.publish, stop_event)),
    ]

    async def handler(websocket, path: str):
        clean = path.split("?", 1)[0].rstrip("/") or "/all"
        if clean == "/control":
            await control_handler(websocket, manager)
            return
        if clean != "/all":
            await websocket.close(code=1008, reason="仅支持 /all 和 /control")
            return
        await video.websocket_server_handler(websocket, "/all", hub)

    try:
        async with serve(handler, args.host, args.port, compression=None, max_size=32 * 1024 * 1024,
                          ping_interval=20, ping_timeout=10, close_timeout=2):
            print(f"[REMOTE] WebSocket 服务: ws://{args.host}:{args.port}/control 和 /all")
            await stop_event.wait()
    finally:
        with contextlib.suppress(Exception):
            await manager.stop()
        stop_event.set()
        for task in producers:
            task.cancel()
        await asyncio.gather(*producers, return_exceptions=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="M20 局域网巡检控制与双路视频服务")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        return 0
    except RuntimeError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
