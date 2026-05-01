import os
import sys
from pathlib import Path

from .types import AppPaths


class PathManager:
    """Resolve working, cache, config, and binary paths."""

    def __init__(self) -> None:
        app_dir = Path.home() / ".EAGLE"
        app_dir.mkdir(exist_ok=True)
        torch_home = app_dir / "torch"
        torch_hub_dir = torch_home / "hub"
        torch_hub_dir.mkdir(parents=True, exist_ok=True)

        if hasattr(sys, "_MEIPASS"):
            working_dir = Path(sys._MEIPASS)
        else:
            working_dir = Path(__file__).resolve().parent.parent

        if sys.platform == "darwin":
            ffmpeg_path = working_dir / "ffmpeg" / "mac" / "ffmpeg"
        elif sys.platform.startswith("win"):
            ffmpeg_path = working_dir / "ffmpeg" / "win" / "ffmpeg.exe"
        else:
            bundled_linux_ffmpeg = working_dir / "ffmpeg" / "AMD" / "ffmpeg"
            ffmpeg_path = bundled_linux_ffmpeg if bundled_linux_ffmpeg.exists() else Path("ffmpeg")

        self.paths = AppPaths(
            working_dir=working_dir,
            app_dir=app_dir,
            yolo_path=app_dir / "yolo26x.pt",
            yolo_pose_path=app_dir / "yolo26x-pose.pt",
            mediapipe_face_detector_path=app_dir / "blaze_face_short_range.tflite",
            mobile_gaze_path=app_dir / "mobileone_s0.pt",
            torch_home=torch_home,
            torch_hub_dir=torch_hub_dir,
            botsort_template_path=working_dir / "config" / "botsort.yaml",
            botsort_runtime_path=working_dir / "config" / "botsort_temp.yaml",
            ffmpeg_path=ffmpeg_path,
        )

    def get(self) -> AppPaths:
        return self.paths
