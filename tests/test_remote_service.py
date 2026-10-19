import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

from m20_rebot_moving_grasp.remote_service import RouteManager
from m20_h264_websocket import _path_camera_filter


class RemoteServiceTest(unittest.IsolatedAsyncioTestCase):
    async def test_route4_is_unavailable_without_route_definition(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "routes.json"
            path.write_text(json.dumps({"route1": ["start"]}), encoding="utf-8")
            manager = RouteManager(path)
            result = await manager.start("route4", 1)
            self.assertFalse(result["ok"])
            self.assertEqual(result["state"], "unavailable")
            self.assertIsNone(manager.process)

    async def test_start_builds_route_command_and_stop_is_safe(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "routes.json"
            path.write_text(json.dumps({"route1": ["start"]}), encoding="utf-8")
            manager = RouteManager(path)
            fake = AsyncMock()
            fake.pid = 123
            fake.returncode = None
            fake.stdout = asyncio.StreamReader()
            fake.send_signal = Mock()
            with patch("m20_rebot_moving_grasp.remote_service.asyncio.create_subprocess_exec", return_value=fake) as create:
                result = await manager.start("route1", 2)
                self.assertTrue(result["ok"])
                self.assertEqual(create.call_args.args[3:7], ("--run-real", "--route-name", "route1", "--route-loops"))
                stopped = await manager.stop()
                self.assertTrue(stopped["ok"])
                fake.send_signal.assert_called_once()


class VideoPathTest(unittest.TestCase):
    def test_only_combined_video_path_is_allowed(self):
        self.assertIsNone(_path_camera_filter("/all"))
        with self.assertRaises(ValueError):
            _path_camera_filter("/front")
        with self.assertRaises(ValueError):
            _path_camera_filter("/back")


class StopLifecycleTest(unittest.IsolatedAsyncioTestCase):
    async def test_failed_cleanup_does_not_become_stopped(self):
        for code, expected in ((0, "stopped"), (1, "failed"), (-2, "failed")):
            manager = RouteManager()
            process = AsyncMock()
            process.wait.return_value = code
            manager.process, manager.state = process, "stopping"
            await manager._watch_process(process)
            self.assertEqual(manager.state, expected)

    async def test_duplicate_stop_sends_one_signal(self):
        manager = RouteManager()
        manager.process = Mock(returncode=None)
        await manager.stop()
        await manager.stop()
        manager.process.send_signal.assert_called_once()

    async def test_unavailable_route_does_not_overwrite_running_state(self):
        manager = RouteManager()
        manager.process = Mock(returncode=None)
        manager.state, manager.route = "running", "route1"
        result = await manager.start("route4", 1)
        self.assertFalse(result["ok"])
        self.assertEqual((manager.state, manager.route), ("running", "route1"))

    async def test_websocket_stop_reaches_real_child_and_reports_exit(self):
        """No robot: exercise the user's stop protocol against a signal-aware child."""
        import sys
        from websockets.legacy.server import serve
        from websockets.legacy.client import connect
        from m20_rebot_moving_grasp.remote_service import control_handler
        manager = RouteManager()
        child = await asyncio.create_subprocess_exec(
            sys.executable, "-u", "-c",
            "import signal,time,sys; signal.signal(signal.SIGINT,lambda *_:sys.exit(0)); print('ready'); time.sleep(30)",
            stdout=asyncio.subprocess.PIPE,
        )
        await child.stdout.readline()
        manager.process, manager.state = child, "running"
        watcher = asyncio.create_task(manager._watch_process(child))
        try:
            async with serve(lambda ws, path: control_handler(ws, manager), "127.0.0.1", 0) as server:
                port = server.sockets[0].getsockname()[1]
                async with connect(f"ws://127.0.0.1:{port}/control") as ws:
                    self.assertEqual(json.loads(await ws.recv())["state"], "running")
                    await ws.send("[]")
                    self.assertFalse(json.loads(await ws.recv())["ok"])
                    await ws.send(json.dumps({"action": "stop", "request_id": "stop-001"}))
                    accepted = stopped = False
                    while not (accepted and stopped):
                        data = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
                        if data.get("request_id") == "stop-001":
                            accepted = data["ok"]
                        if data.get("type") == "status" and data.get("state") == "stopped":
                            stopped = True
                    self.assertEqual(child.returncode, 0)
            await watcher
        finally:
            if child.returncode is None:
                child.kill()
                await child.wait()
            await watcher
