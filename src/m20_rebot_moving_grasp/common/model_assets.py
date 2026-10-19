"""Local model asset checks used before real YOLOE startup."""

from __future__ import annotations

import os
from pathlib import Path


def mobileclip_weight_candidates() -> tuple[Path, ...]:
    configured = os.getenv("MOBILECLIP_WEIGHT", "").strip()
    candidates: list[Path] = [Path(configured).expanduser()] if configured else []
    try:
        from ultralytics.utils import SETTINGS

        settings_dir = Path(str(SETTINGS["weights_dir"]))
        if not settings_dir.is_absolute():
            settings_dir = Path.cwd() / settings_dir
        candidates.append(settings_dir / "mobileclip2_b.ts")
    except Exception:
        pass
    project_root = Path(__file__).resolve().parents[3]
    candidates.extend((project_root / "weights" / "mobileclip2_b.ts",
                       project_root / "models" / "mobileclip2_b.ts",
                       Path.cwd() / "mobileclip2_b.ts"))
    unique: list[Path] = []
    for path in candidates:
        if path not in unique:
            unique.append(path)
    return tuple(unique)


def require_mobileclip_weight() -> Path:
    """Return the local MobileCLIP asset or fail before hardware initialization."""

    for path in mobileclip_weight_candidates():
        if path.is_file() and path.stat().st_size > 100_000:
            try:
                from ultralytics.utils import downloads

                if not getattr(downloads, "_m20_mobileclip_resolver", False):
                    original = downloads.attempt_download_asset

                    def attempt_download_asset(file, *args, **kwargs):
                        if Path(str(file)).name == "mobileclip2_b.ts":
                            return str(path)
                        return original(file, *args, **kwargs)

                    downloads.attempt_download_asset = attempt_download_asset
                    downloads._m20_mobileclip_resolver = True
            except Exception:
                pass
            return path
    locations = ", ".join(str(path) for path in mobileclip_weight_candidates())
    raise FileNotFoundError(
        "YOLOE 开放词汇需要本地 mobileclip2_b.ts；未找到候选文件。"
        f" 已检查: {locations}。请先运行 scripts/install_jetson_dependencies.sh。"
    )


__all__ = ["mobileclip_weight_candidates", "require_mobileclip_weight"]
