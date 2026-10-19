"""M20 AOS JSON/UDP 协议的最小实现，带 Message ID 关联校验。"""

from __future__ import annotations

import json
import logging
import socket
import threading
import time
from typing import Any


_UNSET_POINT_ID = object()
_AOS_PROTOCOL_TIME = "2023-01-01 00:00:00"
LOGGER = logging.getLogger(__name__)


def build_protocol_header(data_length: int, msg_id: int, asdu_format: int = 0x01) -> bytes:
    if not 0 <= data_length <= 65535 or not 0 <= msg_id <= 65535:
        raise ValueError("协议字段超出范围")
    if asdu_format not in (0, 1):
        raise ValueError("asdu_format 必须为0或1")
    header = bytearray(16)
    header[:4] = bytes((0xEB, 0x91, 0xEB, 0x90))
    header[4:6] = int(data_length).to_bytes(2, "little")
    header[6:8] = int(msg_id).to_bytes(2, "little")
    header[8] = asdu_format
    return bytes(header)


def build_json_message(payload: str, msg_id: int) -> bytes:
    data = payload.encode("utf-8")
    return build_protocol_header(len(data), msg_id) + data


class UdpJsonTransport:
    def __init__(self, server_ip: str, server_port: int = 30000, *, socket_factory=socket.socket):
        self.server_ip = server_ip
        self.server_port = int(server_port)
        self._socket_factory = socket_factory
        self._next_msg_id = 1
        # Route pickup intentionally polls localization and D455 in parallel
        # with a potentially long-running Type=1003 request.  Individual
        # requests use separate UDP sockets, but allocation of the protocol
        # message id itself must remain serialized.
        self._message_id_lock = threading.Lock()

    def send_request(
        self,
        payload: str,
        *,
        timeout: float = 10.0,
        allow_response_id_mismatch: bool = False,
    ) -> dict[str, Any] | None:
        with self._message_id_lock:
            msg_id = self._next_msg_id
            self._next_msg_id = (msg_id + 1) % 65536
        packet = build_json_message(payload, msg_id)
        sock = self._socket_factory(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("", 0))
        sock.settimeout(timeout)
        deadline = time.monotonic() + timeout
        try:
            sock.sendto(packet, (self.server_ip, self.server_port))
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                sock.settimeout(remaining)
                data, _ = sock.recvfrom(8192)
                if len(data) < 16:
                    continue
                response_id = int.from_bytes(data[6:8], "little")
                try:
                    result = json.loads(data[16:].decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    return None
                if response_id != msg_id:
                    if not allow_response_id_mismatch:
                        continue
                    LOGGER.warning(
                        "AOS 响应 Message ID 不匹配；接受单点导航响应: request=%s response=%s",
                        msg_id,
                        response_id,
                    )
                return result if isinstance(result, dict) else None
        except socket.timeout:
            return None
        finally:
            sock.close()


def motion_state_request(motion_param: int) -> str:
    return json.dumps({"PatrolDevice": {"Type": 2, "Command": 22, "Items": {"MotionParam": int(motion_param)}}})


def map_location_request() -> str:
    return json.dumps({"PatrolDevice": {"Type": 1007, "Command": 2, "Items": {}}})


def perception_state_request() -> str:
    """查询导航感知模块的定位和停避障状态。"""

    return json.dumps({"PatrolDevice": {"Type": 2002, "Command": 1, "Items": {}}})


def single_point_nav_request(
    point: dict[str, float | int],
    *,
    point_id: int | None | object = _UNSET_POINT_ID,
    map_id: int = 0,
    point_info: int = 1,
    gait: int = 0x3002,
    speed: int = 1,
    manner: int = 0,
    obs_mode: int = 0,
    nav_mode: int = 1,
) -> str:
    """生成 M20 单点导航请求。

    命名点导航维持原默认参数；D455 动态接近显式传入已验证的直线
    导航参数，避免两种任务彼此影响。
    """

    # AOS uses Value to correlate a task, but coordinate navigation does not
    # require a point saved in the current map.  ``None`` selects the CLI's
    # coordinate-navigation convention Value=0; callers that need a stable
    # task id (D455 approach 901/902) can still provide one explicitly.
    if point_id is _UNSET_POINT_ID:
        value = int(point.get("point_id", 0))
    elif point_id is None:
        value = 0
    else:
        value = int(point_id)
    items = {"Value": value, "MapID": 0, "PosX": float(point["x"]), "PosY": float(point["y"]),
             "PosZ": float(point["z"]), "AngleYaw": float(point["yaw"]), "PointInfo": int(point_info),
             "Gait": int(gait), "Speed": int(speed), "Manner": int(manner), "ObsMode": int(obs_mode),
             "NavMode": int(nav_mode)}
    items["MapID"] = int(map_id)
    # Keep Type=1003 byte-for-byte compatible with the verified m20_cli.py
    # template.  This AOS command path did not respond to compact JSON even
    # though read-only AOS commands do, so do not normalize its whitespace or
    # numeric precision independently.
    return """
        {
            "PatrolDevice": {
                "Type": 1003,
                "Command": 1,
                "Time": "%s",
                "Items": {
                    "Value": %d,
                    "MapID": %d,
                    "PosX": %f,
                    "PosY": %f,
                    "PosZ": %f,
                    "AngleYaw": %f,
                    "PointInfo": %d,
                    "Gait": %d,
                    "Speed": %d,
                    "Manner": %d,
                    "ObsMode": %d,
                    "NavMode": %d
                }
            }
        }
        """ % (
        _AOS_PROTOCOL_TIME,
        value,
        int(map_id),
        float(point["x"]),
        float(point["y"]),
        float(point["z"]),
        float(point["yaw"]),
        int(point_info),
        int(gait),
        int(speed),
        int(manner),
        int(obs_mode),
        int(nav_mode),
    )


def cancel_nav_request() -> str:
    return json.dumps({"PatrolDevice": {"Type": 1004, "Command": 1, "Items": {}}})


def nav_status_request() -> str:
    return json.dumps({"PatrolDevice": {"Type": 1007, "Command": 1, "Items": {}}})


def response_items(response: Any) -> dict[str, Any]:
    if not isinstance(response, dict):
        return {}
    return response.get("PatrolDevice", {}).get("Items", {}) or {}


def response_ok(response: Any) -> bool:
    return bool(response) and response_items(response).get("ErrorCode") in (None, 0)


__all__ = ["UdpJsonTransport", "build_protocol_header", "build_json_message", "motion_state_request",
           "map_location_request", "perception_state_request", "single_point_nav_request", "cancel_nav_request", "nav_status_request",
           "response_items", "response_ok"]
