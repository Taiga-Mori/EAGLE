import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .annotate import FrameAnnotator
from .config import ConfigManager, DeviceManager
from .constants import COCO_OBJECT_CLASSES, DEFAULT_OFFSCREEN_DIRECTION_BACKEND, DEFAULT_YOLO_OBJECT_MODEL
from .exporters import AnnotationExporter
from .gaze import FaceGazeEstimator
from .models import ModelManager
from .paths import PathManager
from .temporal import GazePointResolver, GazeTemporalProcessor, ObjectTrackSmoother
from .tracking import ObjectTracker
from .types import MediaContext, PipelineConfig


class EAGLE:
    """Facade over the gaze annotation pipeline."""

    def __init__(self) -> None:
        self.paths = PathManager().get()
        self.device_manager = DeviceManager()
        self.device_options = self.device_manager.device_options
        self.config_manager = ConfigManager(self.paths)
        self.model_manager = ModelManager(self.paths)
        self.annotator = FrameAnnotator()
        self.object_tracker = ObjectTracker(self.model_manager, self.paths, ObjectTrackSmoother())
        self.face_gaze_estimator = FaceGazeEstimator(
            self.model_manager,
            self.annotator,
            GazeTemporalProcessor(),
            GazePointResolver(),
        )
        self.exporter = AnnotationExporter(self.paths)
        self.config: PipelineConfig | None = None
        self.context: MediaContext | None = None

    @property
    def working_dir(self) -> Path:
        return self.paths.working_dir

    def preprocess(
        self,
        input_path: Path | str,
        output_dir: Path | str,
        object_target_fps: float | None = None,
        gaze_target_fps: float | None = None,
        det_thresh: float = 0.5,
        yolo_object_model: str = DEFAULT_YOLO_OBJECT_MODEL,
        updates: dict[str, Any] | None = None,
        device: str | None = None,
        visualization_mode: str = "both",
        heatmap_alpha: float = 0.35,
        face_detection_backend: str = "mediapipe",
        offscreen_direction_backend: str = DEFAULT_OFFSCREEN_DIRECTION_BACKEND,
        gaze_point_method: str = "peak_region_centroid",
        gaze_target_radius: int = 15,
        person_part_distance_scale: float = 0.10,
        object_smoothing_window: int = 5,
        face_smoothing_window: int = 5,
        gaze_smoothing_window: int = 5,
        selected_object_classes: list[str] | None = None,
        reuse_cached_objects: bool = True,
        reuse_cached_gaze: bool = False,
        force_reuse_cached_objects: bool = False,
        force_reuse_cached_gaze: bool = False,
    ) -> None:
        resolved_device = self.device_manager.resolve(device)
        self.config = self.config_manager.build_config(
            input_path=input_path,
            output_dir=output_dir,
            object_target_fps=object_target_fps,
            gaze_target_fps=gaze_target_fps,
            det_thresh=det_thresh,
            device=resolved_device,
            yolo_object_model=yolo_object_model,
            updates=updates,
            visualization_mode=visualization_mode,
            heatmap_alpha=heatmap_alpha,
            face_detection_backend=face_detection_backend,
            offscreen_direction_backend=offscreen_direction_backend,
            gaze_point_method=gaze_point_method,
            gaze_target_radius=gaze_target_radius,
            person_part_distance_scale=person_part_distance_scale,
            object_smoothing_window=object_smoothing_window,
            face_smoothing_window=face_smoothing_window,
            gaze_smoothing_window=gaze_smoothing_window,
            selected_object_classes=selected_object_classes,
            reuse_cached_objects=reuse_cached_objects,
            reuse_cached_gaze=reuse_cached_gaze,
            force_reuse_cached_objects=force_reuse_cached_objects,
            force_reuse_cached_gaze=force_reuse_cached_gaze,
        )
        self.config_manager.prepare_tracker_config(self.config.tracker_updates)
        self.context = self.config_manager.build_media_context(self.config)
        self.model_manager.load(
            self.config.device,
            self.config.face_detection_backend,
            self.config.offscreen_direction_backend,
            self.config.yolo_object_model,
        )

    def update_botsort_yaml(self, updates: dict[str, Any]) -> None:
        config = self._require_config()
        merged_updates = dict(config.tracker_updates)
        merged_updates.update(updates)
        self.config_manager.prepare_tracker_config(merged_updates)
        config.tracker_updates = merged_updates

    def det_objects(self, progress_bar=None):
        context = self._require_context()
        config = self._require_config()
        if config.reuse_cached_objects:
            cached_objects = self._load_cached_objects(config, force_reuse=config.force_reuse_cached_objects)
            if cached_objects is not None:
                message = "Skipping object detection: reusing cached objects.csv."
                if config.force_reuse_cached_objects:
                    message = "Skipping object detection: force reusing cached objects.csv despite setting differences."
                self._notify_skip(progress_bar, message)
                return cached_objects
        return self.object_tracker.detect(
            context=context,
            device=config.device,
            yolo_object_model=config.yolo_object_model,
            det_thresh=config.det_thresh,
            smoothing_window=config.object_smoothing_window,
            selected_object_classes=config.selected_object_classes,
            progress_bar=progress_bar,
        )

    def det_faces_and_gaze(self, progress_bar=None):
        context = self._require_context()
        config = self._require_config()
        return self.face_gaze_estimator.estimate(
            context=context,
            device=config.device,
            det_thresh=config.det_thresh,
            visualization_mode=config.visualization_mode,
            heatmap_alpha=config.heatmap_alpha,
            face_detection_backend=config.face_detection_backend,
            offscreen_direction_backend=config.offscreen_direction_backend,
            gaze_point_method=config.gaze_point_method,
            gaze_target_radius=config.gaze_target_radius,
            person_part_distance_scale=config.person_part_distance_scale,
            face_smoothing_window=config.face_smoothing_window,
            gaze_smoothing_window=config.gaze_smoothing_window,
            selected_object_classes=config.selected_object_classes,
            reuse_cached_gaze=config.reuse_cached_gaze,
            force_reuse_cached_gaze=config.force_reuse_cached_gaze,
            progress_bar=progress_bar,
        )

    def make_video(self):
        return self.exporter.make_video(self._require_context(), self._require_config().visualization_mode)

    def make_image(self):
        return self.exporter.make_image(self._require_context(), self._require_config().visualization_mode)

    def export_visualization(self):
        if self._require_context().media_type == "video":
            return self.make_video()
        return self.make_image()

    def make_elan_csv(self):
        config = self._require_config()
        return self.exporter.make_elan_csv(
            self._require_context(),
            config.det_thresh,
            config.gaze_target_radius,
            config.person_part_distance_scale,
            config.selected_object_classes,
        )

    def run_all(self, progress_bar=None) -> dict[str, Any]:
        object_df = self.det_objects(progress_bar=progress_bar)
        gaze_df = self.det_faces_and_gaze(progress_bar=progress_bar)
        media_output_paths = self.export_visualization()
        annotation_df = self.make_elan_csv()
        return {
            "objects": object_df,
            "gaze": gaze_df,
            "media_output_paths": media_output_paths,
            "annotation": annotation_df,
        }

    def _require_config(self) -> PipelineConfig:
        if self.config is None:
            raise RuntimeError("Call preprocess() before running this step.")
        return self.config

    def _require_context(self) -> MediaContext:
        if self.context is None:
            raise RuntimeError("Call preprocess() before running this step.")
        return self.context

    def _load_cached_objects(self, config: PipelineConfig, force_reuse: bool = False):
        context = self._require_context()
        if not context.objects_path.exists():
            return None
        if not context.objects_meta_path.exists():
            return pd.read_csv(context.objects_path).reset_index(drop=True) if force_reuse else None
        try:
            cached_df = pd.read_csv(context.objects_path)
            with context.objects_meta_path.open("r", encoding="utf-8") as file:
                meta = json.load(file)
        except Exception:
            return None
        if cached_df.empty:
            return None
        if force_reuse:
            return cached_df.reset_index(drop=True)
        if meta.get("raw_detection_cache") is not True:
            return None
        if meta.get("media_path") != str(context.media_path.resolve()):
            return None
        if int(meta.get("media_mtime_ns", -1)) != context.media_path.stat().st_mtime_ns:
            return None
        if abs(float(meta.get("det_thresh", -1.0)) - float(config.det_thresh)) > 1e-9:
            return None
        if str(meta.get("yolo_object_model", "")) != str(config.yolo_object_model):
            return None
        if int(meta.get("object_stride", -1)) != int(context.object_stride):
            return None
        if int(meta.get("object_smoothing_window", -1)) != int(config.object_smoothing_window):
            return None
        if str(meta.get("person_detection_source", "")) != "pose":
            return None
        requested_non_person_detections = any(cls_name != "person" for cls_name in config.selected_object_classes)
        cached_includes_non_person = bool(meta.get("includes_non_person_detections", True))
        if requested_non_person_detections and not cached_includes_non_person:
            return None
        cached_tracker_config = meta.get("tracker_config")
        current_tracker_config = self._current_tracker_config()
        if cached_tracker_config != current_tracker_config:
            return None
        return cached_df.reset_index(drop=True)

    def _current_tracker_config(self) -> dict[str, Any] | None:
        try:
            with self.paths.botsort_runtime_path.open("r", encoding="utf-8") as file:
                return yaml.safe_load(file) or {}
        except Exception:
            return None

    def _notify_skip(self, progress_bar, message: str) -> None:
        if progress_bar is not None:
            progress_bar.progress(0.0, text=message)
        print(message, flush=True)
