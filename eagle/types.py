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
    yolo_pose_path: Path
    mediapipe_face_detector_path: Path
    mobile_gaze_path: Path
    torch_home: Path
    torch_hub_dir: Path
    botsort_template_path: Path
    botsort_runtime_path: Path
    ffmpeg_path: Path


@dataclass
class PipelineConfig:
    """Runtime settings for one pipeline run."""

    media_path: Path
    output_dir: Path
    person_target_fps: float
    object_target_fps: float
    face_target_fps: float
    gaze_target_fps: float
    head_pose_target_fps: float
    person_det_thresh: float
    object_det_thresh: float
    face_det_thresh: float
    gaze_det_thresh: float
    device: str
    person_detection_backend: str
    object_detection_backend: str
    gaze_detection_backend: str
    head_pose_detection_backend: str
    tracker_updates: dict[str, Any]
    media_type: str
    visualization_mode: str
    heatmap_alpha: float
    face_detection_backend: str
    gaze_point_method: str
    gaze_target_radius: int
    person_part_distance_scale: float
    person_part_min_conf: float
    person_smoothing_window: int
    object_smoothing_window: int
    face_smoothing_window: int
    gaze_smoothing_window: int
    selected_object_classes: list[str]
    reuse_cached_persons: bool
    reuse_cached_objects: bool
    reuse_cached_faces: bool
    reuse_cached_gaze: bool


@dataclass
class MediaContext:
    """Derived media metadata and output paths for one run."""

    media_path: Path
    media_type: str
    output_dir: Path
    temp_dir: Path
    persons_path: Path
    persons_meta_path: Path
    objects_path: Path
    objects_meta_path: Path
    faces_path: Path
    faces_meta_path: Path
    gaze_path: Path
    gaze_heatmaps_path: Path
    gaze_meta_path: Path
    annotation_path: Path
    annotated_image_path: Path
    heatmap_dir: Path
    fps: float
    total_frames: int
    person_target_fps: float
    person_stride: int
    person_frame_idx: list[int]
    object_target_fps: float
    object_stride: int
    object_frame_idx: list[int]
    face_target_fps: float
    face_stride: int
    face_frame_idx: list[int]
    gaze_target_fps: float
    gaze_stride: int
    gaze_frame_idx: list[int]
    head_pose_target_fps: float
    head_pose_stride: int
    head_pose_frame_idx: list[int]


@dataclass
class FaceDetection:
    """Best face assigned to a tracked person."""

    track_id: str
    conf: float
    x1: int
    y1: int
    x2: int
    y2: int


@dataclass
class GazePoint:
    """Estimated gaze result and heatmap for one face."""

    track_id: str
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
    track_id: str
    gaze_detected: bool
    inout: float | None
    x_gaze: int | None
    y_gaze: int | None
    offscreen_direction: str | None
    offscreen_yaw: float | None
    offscreen_pitch: float | None
