from pathlib import Path
from typing import Any

import pandas as pd

from .annotate import FrameAnnotator
from .config import ConfigManager, DeviceManager
from .exporters import AnnotationExporter
from .gaze import FaceGazeEstimator
from .models import ModelManager
from .paths import PathManager
from .temporal import GazeTemporalProcessor, ObjectTrackSmoother
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
        updates: dict[str, Any] | None = None,
        device: str | None = None,
        visualization_mode: str = "both",
        heatmap_alpha: float = 0.35,
        object_smoothing_window: int = 5,
        gaze_smoothing_window: int = 5,
        person_only_mode: bool = False,
        reuse_cached_objects: bool = True,
    ) -> None:
        resolved_device = self.device_manager.resolve(device)
        self.config = self.config_manager.build_config(
            input_path=input_path,
            output_dir=output_dir,
            object_target_fps=object_target_fps,
            gaze_target_fps=gaze_target_fps,
            det_thresh=det_thresh,
            device=resolved_device,
            updates=updates,
            visualization_mode=visualization_mode,
            heatmap_alpha=heatmap_alpha,
            object_smoothing_window=object_smoothing_window,
            gaze_smoothing_window=gaze_smoothing_window,
            person_only_mode=person_only_mode,
            reuse_cached_objects=reuse_cached_objects,
        )
        self.config_manager.prepare_tracker_config(self.config.tracker_updates)
        self.context = self.config_manager.build_media_context(self.config)
        self.model_manager.load(self.config.device)

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
            cached_objects = self._load_cached_objects(config.person_only_mode)
            if cached_objects is not None:
                return cached_objects
        return self.object_tracker.detect(
            context=context,
            device=config.device,
            det_thresh=config.det_thresh,
            smoothing_window=config.object_smoothing_window,
            person_only_mode=config.person_only_mode,
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
            gaze_smoothing_window=config.gaze_smoothing_window,
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
        return self.exporter.make_elan_csv(self._require_context(), self._require_config().det_thresh)

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

    def _load_cached_objects(self, person_only_mode: bool):
        context = self._require_context()
        if not context.objects_path.exists():
            return None
        try:
            cached_df = pd.read_csv(context.objects_path)
        except Exception:
            return None
        if cached_df.empty:
            return None
        if person_only_mode and "cls" in cached_df.columns:
            if not cached_df["cls"].eq("person").all():
                return None
        return cached_df
