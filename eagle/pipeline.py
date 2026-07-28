import json
import subprocess
import time
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import torch
import yaml

from .annotate import FrameAnnotator
from .config import ConfigManager, DeviceManager
from .constants import (
    COCO_OBJECT_CLASSES,
    DEFAULT_FACE_DETECTION_BACKEND,
    DEFAULT_GAZE_DETECTION_BACKEND,
    DEFAULT_HEAD_POSE_DETECTION_BACKEND,
    DEFAULT_OBJECT_DETECTION_BACKEND,
    DEFAULT_PERSON_DETECTION_BACKEND,
)
from .exporters import AnnotationExporter
from .gaze import FaceGazeEstimator
from .models import ModelManager
from .paths import PathManager
from .progress import finish_progress, format_elapsed, reset_progress_timers, update_progress
from .temporal import GazePointResolver, GazeTemporalProcessor, ObjectTrackSmoother
from .tracking import ObjectTracker
from .types import MediaContext, PipelineConfig


class EAGLE:
    """Facade over the gaze annotation pipeline."""

    def __init__(self) -> None:
        self.paths = PathManager().get()
        self.device_manager = DeviceManager()
        self.device_options = self.device_manager.device_options
        self.device_warnings = self.device_manager.device_warnings
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
        person_target_fps: float | None = None,
        object_target_fps: float | None = None,
        face_target_fps: float | None = None,
        gaze_target_fps: float | None = None,
        head_pose_target_fps: float | None = None,
        person_det_thresh: float = 0.5,
        object_det_thresh: float = 0.5,
        face_det_thresh: float = 0.5,
        gaze_det_thresh: float = 0.5,
        person_detection_backend: str = DEFAULT_PERSON_DETECTION_BACKEND,
        object_detection_backend: str = DEFAULT_OBJECT_DETECTION_BACKEND,
        face_detection_backend: str = DEFAULT_FACE_DETECTION_BACKEND,
        gaze_detection_backend: str = DEFAULT_GAZE_DETECTION_BACKEND,
        head_pose_detection_backend: str = DEFAULT_HEAD_POSE_DETECTION_BACKEND,
        updates: dict[str, Any] | None = None,
        device: str | None = None,
        visualization_mode: str = "both",
        heatmap_alpha: float = 0.35,
        gaze_point_method: str = "peak_region_centroid",
        gaze_target_radius: int = 15,
        person_part_distance_scale: float = 0.10,
        person_part_min_conf: float = 0.0,
        face_fallback_min_size_scale: float = 1.0,
        person_smoothing_window: int = 5,
        person_max_switch_gap: int = 15,
        object_smoothing_window: int = 5,
        face_smoothing_window: int = 5,
        gaze_smoothing_window: int = 5,
        selected_object_classes: list[str] | None = None,
        reuse_cached_persons: bool = True,
        reuse_cached_objects: bool = True,
        reuse_cached_faces: bool = False,
        reuse_cached_gaze: bool = False,
    ) -> None:
        resolved_device = self.device_manager.resolve(device)
        self.config = self.config_manager.build_config(
            input_path=input_path,
            output_dir=output_dir,
            person_target_fps=person_target_fps,
            object_target_fps=object_target_fps,
            face_target_fps=face_target_fps,
            gaze_target_fps=gaze_target_fps,
            head_pose_target_fps=head_pose_target_fps,
            person_det_thresh=person_det_thresh,
            object_det_thresh=object_det_thresh,
            face_det_thresh=face_det_thresh,
            gaze_det_thresh=gaze_det_thresh,
            device=resolved_device,
            person_detection_backend=person_detection_backend,
            object_detection_backend=object_detection_backend,
            gaze_detection_backend=gaze_detection_backend,
            head_pose_detection_backend=head_pose_detection_backend,
            updates=updates,
            visualization_mode=visualization_mode,
            heatmap_alpha=heatmap_alpha,
            face_detection_backend=face_detection_backend,
            gaze_point_method=gaze_point_method,
            gaze_target_radius=gaze_target_radius,
            person_part_distance_scale=person_part_distance_scale,
            person_part_min_conf=person_part_min_conf,
            face_fallback_min_size_scale=face_fallback_min_size_scale,
            person_smoothing_window=person_smoothing_window,
            person_max_switch_gap=person_max_switch_gap,
            object_smoothing_window=object_smoothing_window,
            face_smoothing_window=face_smoothing_window,
            gaze_smoothing_window=gaze_smoothing_window,
            selected_object_classes=selected_object_classes,
            reuse_cached_persons=reuse_cached_persons,
            reuse_cached_objects=reuse_cached_objects,
            reuse_cached_faces=reuse_cached_faces,
            reuse_cached_gaze=reuse_cached_gaze,
        )
        self.config_manager.prepare_tracker_config(self.config.tracker_updates)
        self.context = self.config_manager.build_media_context(self.config)
        self.model_manager.load(
            self.config.device,
            self.config.person_detection_backend,
            self.config.object_detection_backend,
            self.config.face_detection_backend,
            self.config.gaze_detection_backend,
            self.config.head_pose_detection_backend,
        )

    def update_botsort_yaml(self, updates: dict[str, Any]) -> None:
        config = self._require_config()
        merged_updates = dict(config.tracker_updates)
        merged_updates.update(updates)
        self.config_manager.prepare_tracker_config(merged_updates)
        config.tracker_updates = merged_updates

    def det_persons(self, progress_bar=None):
        context = self._require_context()
        config = self._require_config()
        if config.reuse_cached_persons:
            cached_persons = self._load_cached_tracks(
                context.persons_path,
                context.persons_meta_path,
                config.person_det_thresh,
                config.person_smoothing_window,
                expected_stage="persons",
                max_switch_gap=config.person_max_switch_gap,
                backend=config.person_detection_backend,
            )
            if cached_persons is not None:
                self._notify_skip(progress_bar, "Skipping person detection: reusing cached persons.csv.")
                return cached_persons
        return self.object_tracker.detect_persons(
            context=context,
            device=config.device,
            det_thresh=config.person_det_thresh,
            person_detection_backend=config.person_detection_backend,
            smoothing_window=config.person_smoothing_window,
            max_switch_gap=config.person_max_switch_gap,
            progress_bar=progress_bar,
        )

    def det_objects(self, progress_bar=None):
        context = self._require_context()
        config = self._require_config()
        if config.reuse_cached_objects:
            cached_objects = self._load_cached_tracks(
                context.objects_path,
                context.objects_meta_path,
                config.object_det_thresh,
                config.object_smoothing_window,
                expected_stage="objects",
                backend=config.object_detection_backend,
            )
            if cached_objects is not None:
                self._notify_skip(progress_bar, "Skipping object detection: reusing cached objects.csv.")
                return cached_objects
        return self.object_tracker.detect_objects(
            context=context,
            device=config.device,
            object_detection_backend=config.object_detection_backend,
            det_thresh=config.object_det_thresh,
            smoothing_window=config.object_smoothing_window,
            selected_object_classes=config.selected_object_classes,
            progress_bar=progress_bar,
        )

    def det_faces(self, progress_bar=None):
        context = self._require_context()
        config = self._require_config()
        if config.reuse_cached_faces:
            cached_faces = self._load_cached_faces(config)
            if cached_faces is not None:
                self._notify_skip(progress_bar, "Skipping face detection: reusing cached faces.csv.")
                return cached_faces
        return self.face_gaze_estimator.detect_faces(
            context=context,
            det_thresh=config.face_det_thresh,
            face_detection_backend=config.face_detection_backend,
            face_smoothing_window=config.face_smoothing_window,
            face_fallback_min_size_scale=config.face_fallback_min_size_scale,
            progress_bar=progress_bar,
        )

    def det_gaze(self, progress_bar=None):
        context = self._require_context()
        config = self._require_config()
        return self.face_gaze_estimator.estimate_gaze(
            context=context,
            device=config.device,
            det_thresh=config.gaze_det_thresh,
            gaze_detection_backend=config.gaze_detection_backend,
            head_pose_detection_backend=config.head_pose_detection_backend,
            visualization_mode=config.visualization_mode,
            heatmap_alpha=config.heatmap_alpha,
            gaze_point_method=config.gaze_point_method,
            gaze_target_radius=config.gaze_target_radius,
            person_part_distance_scale=config.person_part_distance_scale,
            person_part_min_conf=config.person_part_min_conf,
            gaze_smoothing_window=config.gaze_smoothing_window,
            selected_object_classes=config.selected_object_classes,
            reuse_cached_gaze=config.reuse_cached_gaze,
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
            config.gaze_det_thresh,
            config.gaze_target_radius,
            config.person_part_distance_scale,
            config.person_part_min_conf,
            config.selected_object_classes,
        )

    def run_all(self, progress_bar=None) -> dict[str, Any]:
        reset_progress_timers()
        phase_seconds: dict[str, float] = {}
        device = self._require_config().device
        if device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(device)

        def _timed(label: str, func: Callable[[], Any]) -> Any:
            start = time.monotonic()
            result = func()
            phase_seconds[label] = time.monotonic() - start
            return result

        person_df = _timed("Person detection", lambda: self.det_persons(progress_bar=progress_bar))
        object_df = _timed("Object detection", lambda: self.det_objects(progress_bar=progress_bar))
        if person_df.empty and object_df.empty:
            self._notify_skip(
                progress_bar,
                "No persons or objects were detected in this media; skipping face/gaze detection and export.",
            )
            elapsed_seconds = finish_progress(progress_bar)
            summary = self._build_summary(person_df, pd.DataFrame(), phase_seconds, elapsed_seconds, device)
            return {
                "persons": person_df,
                "objects": object_df,
                "faces": pd.DataFrame(),
                "gaze": pd.DataFrame(),
                "media_output_paths": [],
                "annotation": pd.DataFrame(),
                "elapsed_seconds": elapsed_seconds,
                "summary": summary,
            }
        face_df = _timed("Face detection", lambda: self.det_faces(progress_bar=progress_bar))
        gaze_df = _timed("Gaze estimation", lambda: self.det_gaze(progress_bar=progress_bar))
        update_progress(progress_bar, 0, 1, "Creating annotation.csv...")
        annotation_df = _timed("Annotation export", self.make_elan_csv)
        update_progress(progress_bar, 1, 1, "Creating annotation.csv...")
        update_progress(progress_bar, 0, 1, "Exporting visualization...")
        media_output_paths = _timed("Visualization export", self.export_visualization)
        update_progress(progress_bar, 1, 1, "Exporting visualization...")
        elapsed_seconds = finish_progress(progress_bar)
        summary = self._build_summary(person_df, face_df, phase_seconds, elapsed_seconds, device)
        return {
            "persons": person_df,
            "objects": object_df,
            "faces": face_df,
            "gaze": gaze_df,
            "media_output_paths": media_output_paths,
            "annotation": annotation_df,
            "elapsed_seconds": elapsed_seconds,
            "summary": summary,
        }

    def _build_summary(
        self,
        person_df: pd.DataFrame,
        face_df: pd.DataFrame,
        phase_seconds: dict[str, float],
        total_elapsed_seconds: float,
        device: str,
    ) -> dict[str, Any]:
        person_count = int(person_df["track_id"].nunique()) if not person_df.empty else 0

        total_faces_found = 0
        detected_without_fallback = 0
        fallback_used = 0
        success_rate_without_fallback: float | None = None
        if not face_df.empty and "face_x1" in face_df.columns:
            found_faces = face_df[face_df["face_x1"].notna()]
            total_faces_found = int(len(found_faces))
            if total_faces_found > 0:
                detected_without_fallback = int((found_faces["face_conf"].astype(float) > 0.0).sum())
                fallback_used = total_faces_found - detected_without_fallback
                success_rate_without_fallback = detected_without_fallback / total_faces_found

        summary = {
            "total_elapsed_seconds": round(total_elapsed_seconds, 2),
            "phase_seconds": {label: round(seconds, 2) for label, seconds in phase_seconds.items()},
            "person_count": person_count,
            "face_detection": {
                "total_faces_found": total_faces_found,
                "detected_without_fallback": detected_without_fallback,
                "fallback_used": fallback_used,
                "success_rate_without_fallback": success_rate_without_fallback,
            },
            "gpu": self._gpu_stats(device),
        }
        self._print_summary(summary)
        self._write_summary(summary)
        return summary

    def _write_summary(self, summary: dict[str, Any]) -> None:
        output_dir = self._require_config().output_dir
        summary_path = output_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    def _gpu_stats(self, device: str) -> dict[str, Any] | None:
        if not device.startswith("cuda"):
            return None
        index = int(device.split(":", 1)[1]) if ":" in device else 0
        stats: dict[str, Any] = {
            "device": device,
            "peak_allocated_gb": round(torch.cuda.max_memory_allocated(index) / (1024**3), 2),
            "peak_reserved_gb": round(torch.cuda.max_memory_reserved(index) / (1024**3), 2),
        }
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    f"--id={index}",
                    "--query-gpu=name,utilization.gpu,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
                check=True,
            )
            name, util, mem_used, mem_total = (part.strip() for part in result.stdout.strip().split(","))
            stats.update(
                {
                    "name": name,
                    "utilization_percent": float(util),
                    "memory_used_mib": float(mem_used),
                    "memory_total_mib": float(mem_total),
                }
            )
        except Exception:
            pass
        return stats

    def _print_summary(self, summary: dict[str, Any]) -> None:
        print("\n" + " Pipeline Summary ".center(72, "="), flush=True)
        print(f"  Total elapsed: {format_elapsed(summary['total_elapsed_seconds'])}", flush=True)
        print("  Phase timings:", flush=True)
        for label, seconds in summary["phase_seconds"].items():
            print(f"    - {label}: {format_elapsed(seconds)}", flush=True)
        print(f"  Detected persons: {summary['person_count']}", flush=True)
        face_stats = summary["face_detection"]
        if face_stats["total_faces_found"] > 0:
            rate = face_stats["success_rate_without_fallback"]
            print(
                "  Face detection success rate (excluding pose fallback): "
                f"{rate:.1%} ({face_stats['detected_without_fallback']}/{face_stats['total_faces_found']}, "
                f"fallback used {face_stats['fallback_used']} times)",
                flush=True,
            )
        else:
            print("  Face detection: no faces found", flush=True)
        gpu_stats = summary.get("gpu")
        if gpu_stats is not None:
            gpu_line = (
                f"  GPU ({gpu_stats['device']}"
                + (f", {gpu_stats['name']}" if "name" in gpu_stats else "")
                + f"): peak allocated {gpu_stats['peak_allocated_gb']:.2f} GB, "
                f"peak reserved {gpu_stats['peak_reserved_gb']:.2f} GB"
            )
            if "utilization_percent" in gpu_stats:
                gpu_line += (
                    f", current util {gpu_stats['utilization_percent']:.0f}%, "
                    f"mem {gpu_stats['memory_used_mib']:.0f}/{gpu_stats['memory_total_mib']:.0f} MiB"
                )
            print(gpu_line, flush=True)
        print("=" * 72, flush=True)

    def _require_config(self) -> PipelineConfig:
        if self.config is None:
            raise RuntimeError("Call preprocess() before running this step.")
        return self.config

    def _require_context(self) -> MediaContext:
        if self.context is None:
            raise RuntimeError("Call preprocess() before running this step.")
        return self.context

    def _load_cached_tracks(
        self,
        csv_path: Path,
        meta_path: Path,
        det_thresh: float,
        smoothing_window: int,
        expected_stage: str,
        max_switch_gap: int | None = None,
        backend: str | None = None,
    ):
        context = self._require_context()
        if not csv_path.exists():
            return None
        if not meta_path.exists():
            return None
        try:
            cached_df = pd.read_csv(csv_path)
            with meta_path.open("r", encoding="utf-8") as file:
                meta = json.load(file)
        except Exception:
            return None
        if meta.get("raw_detection_cache") is not True:
            return None
        if str(meta.get("detection_stage", "")) != expected_stage:
            return None
        if meta.get("media_path") != str(context.media_path.resolve()):
            return None
        if int(meta.get("media_mtime_ns", -1)) != context.media_path.stat().st_mtime_ns:
            return None
        if abs(float(meta.get("det_thresh", -1.0)) - float(det_thresh)) > 1e-9:
            return None
        if int(meta.get("stride", -1)) != int(self._stage_stride(context, expected_stage)):
            return None
        if int(meta.get("smoothing_window", -1)) != int(smoothing_window):
            return None
        if max_switch_gap is not None and int(meta.get("max_switch_gap", -1)) != int(max_switch_gap):
            return None
        if backend is not None and str(meta.get("backend", "")) != str(backend):
            return None
        if expected_stage == "persons" and str(meta.get("pose_keypoint_assignment", "")) != "bbox_geometry_v1":
            return None
        cached_tracker_config = meta.get("tracker_config")
        current_tracker_config = self._current_tracker_config()
        if cached_tracker_config != current_tracker_config:
            return None
        return cached_df.reset_index(drop=True)

    def _stage_stride(self, context: MediaContext, stage: str) -> int:
        if stage == "persons":
            return context.person_stride
        if stage == "objects":
            return context.object_stride
        raise ValueError(f"Unsupported cached track stage: {stage}")

    def _load_cached_faces(self, config: PipelineConfig):
        context = self._require_context()
        if not context.faces_path.exists() or not context.faces_meta_path.exists():
            return None
        try:
            cached_df = pd.read_csv(context.faces_path)
            with context.faces_meta_path.open("r", encoding="utf-8") as file:
                meta = json.load(file)
        except Exception:
            return None
        if meta.get("media_path") != str(context.media_path.resolve()):
            return None
        if int(meta.get("media_mtime_ns", -1)) != context.media_path.stat().st_mtime_ns:
            return None
        if abs(float(meta.get("det_thresh", -1.0)) - float(config.face_det_thresh)) > 1e-9:
            return None
        if int(meta.get("face_stride", -1)) != int(context.face_stride):
            return None
        if int(meta.get("face_smoothing_window", -1)) != int(config.face_smoothing_window):
            return None
        if abs(float(meta.get("face_fallback_min_size_scale", -1.0)) - float(config.face_fallback_min_size_scale)) > 1e-9:
            return None
        if str(meta.get("face_detection_backend", "")) != str(config.face_detection_backend):
            return None
        persons_mtime = context.persons_path.stat().st_mtime_ns if context.persons_path.exists() else -1
        objects_mtime = context.objects_path.stat().st_mtime_ns if context.objects_path.exists() else -1
        if int(meta.get("persons_mtime_ns", -1)) != persons_mtime:
            return None
        if int(meta.get("objects_mtime_ns", -1)) != objects_mtime:
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
