import shutil
from pathlib import Path
from typing import Any

import cv2
import yaml

from .constants import (
    COCO_OBJECT_CLASSES,
    DEFAULT_GAZE_DETECTION_BACKEND,
    DEFAULT_HEAD_POSE_DETECTION_BACKEND,
    DEFAULT_OBJECT_DETECTION_BACKEND,
    DEFAULT_PERSON_DETECTION_BACKEND,
    FACE_DETECTION_BACKENDS,
    GAZE_POINT_METHODS,
    GAZE_DETECTION_BACKENDS,
    HEAD_POSE_DETECTION_BACKENDS,
    IMAGE_EXTENSIONS,
    OBJECT_DETECTION_BACKENDS,
    PERSON_DETECTION_BACKENDS,
    VIDEO_EXTENSIONS,
    VISUALIZATION_MODES,
)
from .types import AppPaths, MediaContext, PipelineConfig


class DeviceManager:
    """Select the inference device and expose supported options."""

    def __init__(self) -> None:
        import torch

        self.device_options: list[str] = []
        self.device_warnings: list[str] = []
        if torch.cuda.is_available():
            cuda_count = int(torch.cuda.device_count())
            usable_cuda_devices = [
                device_idx for device_idx in range(cuda_count) if self._cuda_device_is_usable(torch, device_idx)
            ]
            if len(usable_cuda_devices) > 1:
                self.default_device = f"cuda:{usable_cuda_devices[0]}"
                self.device_options.extend([f"cuda:{idx}" for idx in usable_cuda_devices])
            elif len(usable_cuda_devices) == 1:
                only_device = usable_cuda_devices[0]
                self.default_device = f"cuda:{only_device}"
                self.device_options.append(f"cuda:{only_device}")
            elif torch.backends.mps.is_available():
                self.default_device = "mps"
                self.device_options.append("mps")
            else:
                self.default_device = "cpu"
        elif torch.backends.mps.is_available():
            self.default_device = "mps"
            self.device_options.append("mps")
        else:
            self.default_device = "cpu"
        self.device_options.append("cpu")

    def _cuda_device_is_usable(self, torch, device_idx: int) -> bool:
        device = f"cuda:{device_idx}"
        try:
            with torch.cuda.device(device_idx):
                probe = torch.ones((1,), device=device)
                _ = probe + 1
                torch.cuda.synchronize(device_idx)
            return True
        except Exception as exc:
            try:
                device_name = torch.cuda.get_device_name(device_idx)
            except Exception:
                device_name = f"CUDA device {device_idx}"
            self.device_warnings.append(
                f"{device} ({device_name}) is visible but unusable by the installed PyTorch CUDA build: {exc}"
            )
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            if device_idx == 0:
                # Keep CUDA's error state from leaking into later CPU/MPS startup paths when possible.
                try:
                    torch.cuda.synchronize(device_idx)
                except Exception:
                    pass
            return False

    def resolve(self, requested_device: str | None) -> str:
        device = requested_device or self.default_device
        if device not in self.device_options:
            raise ValueError(f"Unsupported device '{device}'. Available devices: {', '.join(self.device_options)}")
        return device


class ConfigManager:
    """Normalize config inputs and prepare runtime files."""

    def __init__(self, paths: AppPaths) -> None:
        self.paths = paths

    def build_config(
        self,
        input_path: Path | str,
        output_dir: Path | str,
        person_target_fps: float | None,
        object_target_fps: float | None,
        face_target_fps: float | None,
        gaze_target_fps: float | None,
        head_pose_target_fps: float | None,
        person_det_thresh: float,
        object_det_thresh: float,
        face_det_thresh: float,
        gaze_det_thresh: float,
        device: str,
        person_detection_backend: str,
        object_detection_backend: str,
        face_detection_backend: str,
        gaze_detection_backend: str,
        head_pose_detection_backend: str,
        updates: dict[str, Any] | None,
        visualization_mode: str,
        heatmap_alpha: float,
        gaze_point_method: str,
        gaze_target_radius: int,
        person_part_distance_scale: float,
        person_part_min_conf: float,
        person_smoothing_window: int,
        object_smoothing_window: int,
        face_smoothing_window: int,
        gaze_smoothing_window: int,
        selected_object_classes: list[str] | None,
        reuse_cached_persons: bool,
        reuse_cached_objects: bool,
        reuse_cached_faces: bool,
        reuse_cached_gaze: bool,
    ) -> PipelineConfig:
        if visualization_mode not in VISUALIZATION_MODES:
            raise ValueError(f"visualization_mode must be one of {sorted(VISUALIZATION_MODES)}")
        if face_detection_backend not in FACE_DETECTION_BACKENDS:
            raise ValueError(f"face_detection_backend must be one of {sorted(FACE_DETECTION_BACKENDS)}")
        if person_detection_backend not in PERSON_DETECTION_BACKENDS:
            raise ValueError(f"person_detection_backend must be one of {sorted(PERSON_DETECTION_BACKENDS)}")
        if object_detection_backend not in OBJECT_DETECTION_BACKENDS:
            raise ValueError(f"object_detection_backend must be one of {sorted(OBJECT_DETECTION_BACKENDS)}")
        if gaze_detection_backend not in GAZE_DETECTION_BACKENDS:
            raise ValueError(f"gaze_detection_backend must be one of {sorted(GAZE_DETECTION_BACKENDS)}")
        if head_pose_detection_backend not in HEAD_POSE_DETECTION_BACKENDS:
            raise ValueError(f"head_pose_detection_backend must be one of {sorted(HEAD_POSE_DETECTION_BACKENDS)}")
        if not 0.0 <= heatmap_alpha <= 1.0:
            raise ValueError("heatmap_alpha must be between 0.0 and 1.0")
        if gaze_point_method not in GAZE_POINT_METHODS:
            raise ValueError(f"gaze_point_method must be one of {sorted(GAZE_POINT_METHODS)}")
        if gaze_target_radius < 0:
            raise ValueError("gaze_target_radius must be greater than or equal to 0")
        if person_part_distance_scale <= 0:
            raise ValueError("person_part_distance_scale must be greater than 0")
        if not 0.0 <= person_part_min_conf <= 1.0:
            raise ValueError("person_part_min_conf must be between 0.0 and 1.0")
        if person_smoothing_window < 1:
            raise ValueError("person_smoothing_window must be at least 1")
        if object_smoothing_window < 1:
            raise ValueError("object_smoothing_window must be at least 1")
        if face_smoothing_window < 1:
            raise ValueError("face_smoothing_window must be at least 1")
        if gaze_smoothing_window < 1:
            raise ValueError("gaze_smoothing_window must be at least 1")
        normalized_selected_classes = self.normalize_selected_object_classes(selected_object_classes)

        return PipelineConfig(
            media_path=Path(input_path),
            output_dir=Path(output_dir),
            person_target_fps=0.0 if person_target_fps is None else float(person_target_fps),
            object_target_fps=0.0 if object_target_fps is None else float(object_target_fps),
            face_target_fps=0.0 if face_target_fps is None else float(face_target_fps),
            gaze_target_fps=0.0 if gaze_target_fps is None else float(gaze_target_fps),
            head_pose_target_fps=0.0 if head_pose_target_fps is None else float(head_pose_target_fps),
            person_det_thresh=float(person_det_thresh),
            object_det_thresh=float(object_det_thresh),
            face_det_thresh=float(face_det_thresh),
            gaze_det_thresh=float(gaze_det_thresh),
            device=device,
            person_detection_backend=person_detection_backend or DEFAULT_PERSON_DETECTION_BACKEND,
            object_detection_backend=object_detection_backend or DEFAULT_OBJECT_DETECTION_BACKEND,
            gaze_detection_backend=gaze_detection_backend or DEFAULT_GAZE_DETECTION_BACKEND,
            head_pose_detection_backend=head_pose_detection_backend or DEFAULT_HEAD_POSE_DETECTION_BACKEND,
            tracker_updates=dict(updates or {}),
            media_type=self.detect_media_type(Path(input_path)),
            visualization_mode=visualization_mode,
            heatmap_alpha=float(heatmap_alpha),
            face_detection_backend=face_detection_backend,
            gaze_point_method=gaze_point_method,
            gaze_target_radius=int(gaze_target_radius),
            person_part_distance_scale=float(person_part_distance_scale),
            person_part_min_conf=float(person_part_min_conf),
            person_smoothing_window=int(person_smoothing_window),
            object_smoothing_window=int(object_smoothing_window),
            face_smoothing_window=int(face_smoothing_window),
            gaze_smoothing_window=int(gaze_smoothing_window),
            selected_object_classes=normalized_selected_classes,
            reuse_cached_persons=bool(reuse_cached_persons),
            reuse_cached_objects=bool(reuse_cached_objects),
            reuse_cached_faces=bool(reuse_cached_faces),
            reuse_cached_gaze=bool(reuse_cached_gaze),
        )

    def normalize_selected_object_classes(self, selected_object_classes: list[str] | None) -> list[str]:
        if not selected_object_classes:
            return list(COCO_OBJECT_CLASSES)

        seen: set[str] = set()
        normalized: list[str] = []
        invalid = sorted({cls_name for cls_name in selected_object_classes if cls_name not in COCO_OBJECT_CLASSES})
        if invalid:
            raise ValueError(f"Unsupported object classes: {', '.join(invalid)}")

        for cls_name in COCO_OBJECT_CLASSES:
            if cls_name in selected_object_classes and cls_name not in seen:
                normalized.append(cls_name)
                seen.add(cls_name)
        if not normalized:
            raise ValueError("Select at least one object class.")
        return normalized

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
            person_target_fps = 1.0
            person_stride = 1
            person_frame_idx = [0]
            object_target_fps = 1.0
            object_stride = 1
            object_frame_idx = [0]
            face_target_fps = 1.0
            face_stride = 1
            face_frame_idx = [0]
            gaze_target_fps = 1.0
            gaze_stride = 1
            gaze_frame_idx = [0]
            head_pose_target_fps = 1.0
            head_pose_stride = 1
            head_pose_frame_idx = [0]
        else:
            capture = cv2.VideoCapture(str(config.media_path))
            if not capture.isOpened():
                raise FileNotFoundError(f"Could not open video: {config.media_path}")
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            capture.release()
            total_frames = self.count_readable_frames(config.media_path)
            if total_frames <= 0 or fps <= 0:
                raise RuntimeError(f"Invalid video metadata for {config.media_path}")

            person_target_fps = config.person_target_fps or fps
            object_target_fps = config.object_target_fps or fps
            face_target_fps = config.face_target_fps or person_target_fps
            gaze_target_fps = config.gaze_target_fps or face_target_fps
            head_pose_target_fps = config.head_pose_target_fps or gaze_target_fps
            if person_target_fps <= 0:
                raise ValueError("person_target_fps must be greater than 0")
            if object_target_fps <= 0:
                raise ValueError("object_target_fps must be greater than 0")
            if face_target_fps <= 0:
                raise ValueError("face_target_fps must be greater than 0")
            if gaze_target_fps <= 0:
                raise ValueError("gaze_target_fps must be greater than 0")
            if head_pose_target_fps <= 0:
                raise ValueError("head_pose_target_fps must be greater than 0")

            person_stride = max(1, round(fps / person_target_fps))
            object_stride = max(1, round(fps / object_target_fps))
            face_stride = max(1, round(fps / face_target_fps))
            gaze_stride = max(1, round(fps / gaze_target_fps))
            head_pose_stride = max(1, round(fps / head_pose_target_fps))
            person_frame_idx = list(range(0, total_frames, person_stride))
            object_frame_idx = list(range(0, total_frames, object_stride))
            face_frame_idx = list(range(0, total_frames, face_stride))
            gaze_frame_idx = list(range(0, total_frames, gaze_stride))
            head_pose_frame_idx = list(range(0, total_frames, head_pose_stride))

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
            persons_path=config.output_dir / "persons.csv",
            persons_meta_path=config.output_dir / ".persons_meta.json",
            objects_path=config.output_dir / "objects.csv",
            objects_meta_path=config.output_dir / ".objects_meta.json",
            faces_path=config.output_dir / "faces.csv",
            faces_meta_path=config.output_dir / ".faces_meta.json",
            gaze_path=config.output_dir / "gaze.csv",
            gaze_heatmaps_path=config.output_dir / "gaze_heatmaps.npz",
            gaze_meta_path=config.output_dir / ".gaze_meta.json",
            annotation_path=config.output_dir / "annotation.csv",
            annotated_image_path=config.output_dir / "all_points.jpg",
            heatmap_dir=heatmap_dir,
            fps=fps,
            total_frames=total_frames,
            person_target_fps=person_target_fps,
            person_stride=person_stride,
            person_frame_idx=person_frame_idx,
            object_target_fps=object_target_fps,
            object_stride=object_stride,
            object_frame_idx=object_frame_idx,
            face_target_fps=face_target_fps,
            face_stride=face_stride,
            face_frame_idx=face_frame_idx,
            gaze_target_fps=gaze_target_fps,
            gaze_stride=gaze_stride,
            gaze_frame_idx=gaze_frame_idx,
            head_pose_target_fps=head_pose_target_fps,
            head_pose_stride=head_pose_stride,
            head_pose_frame_idx=head_pose_frame_idx,
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
