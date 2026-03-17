import os
import sys
from pathlib import Path

from .types import AppPaths


class PathManager:
    """Resolve working, cache, config, and binary paths."""

    def __init__(self) -> None:
        app_dir = Path.home() / ".EAGLE"
        app_dir.mkdir(exist_ok=True)

        if hasattr(sys, "_MEIPASS"):
            working_dir = Path(sys._MEIPASS)
        else:
            working_dir = Path(os.path.abspath("."))

        if sys.platform == "darwin":
            ffmpeg_path = working_dir / "ffmpeg" / "mac" / "ffmpeg"
        elif sys.platform.startswith("win"):
            ffmpeg_path = working_dir / "ffmpeg" / "win" / "ffmpeg.exe"
        else:
            raise RuntimeError("Unsupported OS")

        self.paths = AppPaths(
            working_dir=working_dir,
            app_dir=app_dir,
            yolo_path=app_dir / "yolo26x.pt",
            botsort_template_path=working_dir / "config" / "botsort.yaml",
            botsort_runtime_path=working_dir / "config" / "botsort_temp.yaml",
            ffmpeg_path=ffmpeg_path,
        )

    def get(self) -> AppPaths:
        return self.paths
