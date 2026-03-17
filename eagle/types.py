from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class AppPaths:
    """Resolved filesystem paths used by the pipeline."""

    working_dir: Path
    app_dir: Path
    yolo_path: Path
    botsort_template_path: Path
    botsort_runtime_path: Path
    ffmpeg_path: Path


@dataclass
class PipelineConfig:
    """Runtime settings for one pipeline run."""

    media_path: Path
    output_dir: Path
    object_target_fps: float
    gaze_target_fps: float
    det_thresh: float
    device: str
    tracker_updates: dict[str, Any]
    media_type: str
    visualization_mode: str
    heatmap_alpha: float
    object_smoothing_window: int
    gaze_smoothing_window: int
    person_only_mode: bool
    reuse_cached_objects: bool


@dataclass
class MediaContext:
    """Derived media metadata and output paths for one run."""

    media_path: Path
    media_type: str
    output_dir: Path
    temp_dir: Path
    objects_path: Path
    gaze_path: Path
    annotation_path: Path
    annotated_image_path: Path
    heatmap_dir: Path
    fps: float
    total_frames: int
    object_target_fps: float
    object_stride: int
    object_frame_idx: list[int]
    gaze_target_fps: float
    gaze_stride: int
    gaze_frame_idx: list[int]


@dataclass
class FaceDetection:
    """Best face assigned to a tracked person."""

    track_id: int
    conf: float
    x1: int
    y1: int
    x2: int
    y2: int


@dataclass
class GazePoint:
    """Estimated gaze result and heatmap for one face."""

    track_id: int
    inout: float
    x_gaze: int
    y_gaze: int
    heatmap: np.ndarray
    frame_width: int
    frame_height: int


@dataclass
class GazeRecord:
    """Row-oriented gaze result used for CSV export."""

    frame_idx: int
    track_id: int
    face_detected: bool
    face_conf: float | None
    face_x1: int | None
    face_y1: int | None
    face_x2: int | None
    face_y2: int | None
    raw_gaze_detected: bool
    raw_inout: float | None
    raw_x_gaze: int | None
    raw_y_gaze: int | None
    gaze_detected: bool
    inout: float | None
    x_gaze: int | None
    y_gaze: int | None
