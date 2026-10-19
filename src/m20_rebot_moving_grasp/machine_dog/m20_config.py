"""M20 Pro 联合任务配置。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class M20Config:
    aos_ip: str = "10.21.31.103"
    udp_port: int = 30000
    scan_point_name: str = "scan_point"
    bin_point_name: str = "bin_point"
    home_point_name: str = "start"
    points_file: str = ""
    # AOS may withhold the Type=1003 response until the coordinate-navigation
    # task reaches a terminal state.  This is deliberately independent from
    # the short timeout used by ordinary command acknowledgements.
    nav_timeout_sec: float = 60.0
    clean_scan_sec: float = 10.0
    # AOS read/state/posture acknowledgements are normally quick, but a loaded
    # controller can exceed the CLI's historical five-second default.
    command_timeout_sec: float = 10.0
    cancel_confirm_timeout_sec: float = 15.0

    @classmethod
    def from_env(cls, *, system_config: dict | None = None) -> "M20Config":
        section = (system_config or {}).get("m20", {})
        def value(name: str, key: str, default):
            env = os.getenv(name)
            return env if env is not None else section.get(key, default)
        points = value("M20_POINTS_FILE", "points_file", "")
        return cls(
            aos_ip=str(value("M20_IP", "aos_ip", cls.aos_ip)),
            udp_port=int(value("M20_UDP_PORT", "udp_port", cls.udp_port)),
            scan_point_name=str(value("M20_SCAN_POINT_NAME", "scan_point_name", cls.scan_point_name)),
            bin_point_name=str(value("M20_BIN_POINT_NAME", "bin_point_name", cls.bin_point_name)),
            home_point_name=str(value("M20_HOME_POINT_NAME", "home_point_name", cls.home_point_name)),
            points_file=str(points),
            nav_timeout_sec=float(value("M20_NAV_TIMEOUT_SEC", "nav_timeout_sec", cls.nav_timeout_sec)),
            clean_scan_sec=float(value("M20_CLEAN_SCAN_SEC", "clean_scan_continuous_no_target_sec", cls.clean_scan_sec)),
            command_timeout_sec=float(value("M20_COMMAND_TIMEOUT_SEC", "command_timeout_sec", cls.command_timeout_sec)),
            cancel_confirm_timeout_sec=float(value("M20_CANCEL_CONFIRM_TIMEOUT_SEC", "cancel_confirm_timeout_sec", cls.cancel_confirm_timeout_sec)),
        )

    def validate(self) -> "M20Config":
        if not self.aos_ip.strip():
            raise ValueError("M20 AOS IP 不能为空")
        if not 1 <= self.udp_port <= 65535:
            raise ValueError("M20 UDP 端口无效")
        if not self.scan_point_name or not self.bin_point_name or not self.home_point_name:
            raise ValueError("扫描点、投放点和 Home 点名称不能为空")
        for name, number in (("nav_timeout_sec", self.nav_timeout_sec), ("clean_scan_sec", self.clean_scan_sec),
                             ("command_timeout_sec", self.command_timeout_sec), ("cancel_confirm_timeout_sec", self.cancel_confirm_timeout_sec)):
            if number <= 0:
                raise ValueError(f"{name} 必须大于0")
        return self

    def resolved_points_file(self, fallback: str | Path | None = None) -> Path:
        if self.points_file:
            path = Path(self.points_file).expanduser()
            return path if path.is_absolute() else PROJECT_ROOT / path
        if fallback is not None:
            path = Path(fallback).expanduser()
            return path if path.is_absolute() else PROJECT_ROOT / path
        return Path.home() / ".config" / "m20" / "points.json"


__all__ = ["M20Config"]
