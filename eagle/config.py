import shutil
from pathlib import Path
from typing import Any

import cv2
import yaml

from .constants import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS, VISUALIZATION_MODES
from .types import AppPaths, MediaContext, PipelineConfig


class DeviceManager:
    """Select the inference device and expose supported options."""

    def __init__(self) -> None:
        import torch

        self.device_options: list[str] = []
        if torch.cuda.is_available():
            self.default_device = "cuda"
            self.device_options.append("cuda")
        elif torch.backends.mps.is_available():
            self.default_device = "mps"
            self.device_options.append("mps")
        else:
            self.default_device = "cpu"
        self.device_options.append("cpu")

    def resolve(self, requested_device: str | None) -> str:
        return requested_device or self.default_device


class ConfigManager:
    """Normalize config inputs and prepare runtime files."""

    def __init__(self, paths: AppPaths) -> None:
        self.paths = paths

    def build_config(
        self,
        input_path: Path | str,
        output_dir: Path | str,
        object_target_fps: float | None,
        gaze_target_fps: float | None,
        det_thresh: float,
        device: str,
        updates: dict[str, Any] | None,
        visualization_mode: str,
        heatmap_alpha: float,
        object_smoothing_window: int,
        gaze_smoothing_window: int,
        person_only_mode: bool,
        reuse_cached_objects: bool,
    ) -> PipelineConfig:
        if visualization_mode not in VISUALIZATION_MODES:
            raise ValueError(f"visualization_mode must be one of {sorted(VISUALIZATION_MODES)}")
        if not 0.0 <= heatmap_alpha <= 1.0:
            raise ValueError("heatmap_alpha must be between 0.0 and 1.0")
        if object_smoothing_window < 1:
            raise ValueError("object_smoothing_window must be at least 1")
        if gaze_smoothing_window < 1:
            raise ValueError("gaze_smoothing_window must be at least 1")
        if (
            object_target_fps is not None
            and gaze_target_fps is not None
            and float(gaze_target_fps) > float(object_target_fps)
        ):
            raise ValueError("gaze_target_fps must be less than or equal to object_target_fps")

        return PipelineConfig(
            media_path=Path(input_path),
            output_dir=Path(output_dir),
            object_target_fps=0.0 if object_target_fps is None else float(object_target_fps),
            gaze_target_fps=0.0 if gaze_target_fps is None else float(gaze_target_fps),
            det_thresh=float(det_thresh),
            device=device,
            tracker_updates=dict(updates or {}),
            media_type=self.detect_media_type(Path(input_path)),
            visualization_mode=visualization_mode,
            heatmap_alpha=float(heatmap_alpha),
            object_smoothing_window=int(object_smoothing_window),
            gaze_smoothing_window=int(gaze_smoothing_window),
            person_only_mode=bool(person_only_mode),
            reuse_cached_objects=bool(reuse_cached_objects),
        )

    def detect_media_type(self, input_path: Path) -> str:
        suffix = input_path.suffix.lower()
        if suffix in IMAGE_EXTENSIONS:
            return "image"
        if suffix in VIDEO_EXTENSIONS:
            return "video"
        raise ValueError(f"Unsupported input type for {input_path}")

    def prepare_tracker_config(self, updates: dict[str, Any]) -> None:
        shutil.copy(self.paths.botsort_template_path, self.paths.botsort_runtime_path)
        if not updates:
            return

        with self.paths.botsort_runtime_path.open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file) or {}
        config.update(updates)
        with self.paths.botsort_runtime_path.open("w", encoding="utf-8") as file:
            yaml.safe_dump(config, file, sort_keys=False)

    def build_media_context(self, config: PipelineConfig) -> MediaContext:
        config.output_dir.mkdir(parents=True, exist_ok=True)

        if config.media_type == "image":
            image = cv2.imread(str(config.media_path))
            if image is None:
                raise FileNotFoundError(f"Could not open image: {config.media_path}")
            total_frames = 1
            fps = 1.0
            object_target_fps = 1.0
            object_stride = 1
            object_frame_idx = [0]
            gaze_target_fps = 1.0
            gaze_stride = 1
            gaze_frame_idx = [0]
        else:
            capture = cv2.VideoCapture(str(config.media_path))
            if not capture.isOpened():
                raise FileNotFoundError(f"Could not open video: {config.media_path}")
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            capture.release()
            total_frames = self.count_readable_frames(config.media_path)
            if total_frames <= 0 or fps <= 0:
                raise RuntimeError(f"Invalid video metadata for {config.media_path}")

            object_target_fps = config.object_target_fps or fps
            gaze_target_fps = config.gaze_target_fps or object_target_fps
            if object_target_fps <= 0:
                raise ValueError("object_target_fps must be greater than 0")
            if gaze_target_fps <= 0:
                raise ValueError("gaze_target_fps must be greater than 0")

            object_stride = max(1, round(fps / object_target_fps))
            gaze_stride = max(1, round(fps / gaze_target_fps))
            object_frame_idx = list(range(0, total_frames, object_stride))
            gaze_frame_idx = list(range(0, total_frames, gaze_stride))

        temp_dir = config.output_dir / "temp"
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        temp_dir.mkdir(exist_ok=True)

        heatmap_dir = config.output_dir / "heatmaps"
        if heatmap_dir.exists():
            shutil.rmtree(heatmap_dir)
        heatmap_dir.mkdir(exist_ok=True)

        return MediaContext(
            media_path=config.media_path,
            media_type=config.media_type,
            output_dir=config.output_dir,
            temp_dir=temp_dir,
            objects_path=config.output_dir / "objects.csv",
            gaze_path=config.output_dir / "gaze.csv",
            annotation_path=config.output_dir / "annotation.csv",
            annotated_image_path=config.output_dir / "all_points.jpg",
            heatmap_dir=heatmap_dir,
            fps=fps,
            total_frames=total_frames,
            object_target_fps=object_target_fps,
            object_stride=object_stride,
            object_frame_idx=object_frame_idx,
            gaze_target_fps=gaze_target_fps,
            gaze_stride=gaze_stride,
            gaze_frame_idx=gaze_frame_idx,
        )

    def count_readable_frames(self, media_path: Path) -> int:
        """Count frames by reading until failure instead of trusting CAP_PROP_FRAME_COUNT."""

        capture = cv2.VideoCapture(str(media_path))
        if not capture.isOpened():
            raise FileNotFoundError(f"Could not open video: {media_path}")

        total_frames = 0
        try:
            while True:
                ret, _ = capture.read()
                if not ret:
                    break
                total_frames += 1
        finally:
            capture.release()

        return total_frames
