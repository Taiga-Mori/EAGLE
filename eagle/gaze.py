import json
import shutil
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image

from .annotate import FrameAnnotator
from .body_parts import build_person_part_shapes, resolve_person_part_label
from .constants import FACE_COLUMNS, GAZE_COLUMNS
from .models import ModelManager
from .progress import update_progress
from .temporal import GazePointResolver, GazeTemporalProcessor
from .types import FaceDetection, GazePoint, GazeRecord, MediaContext


class FaceGazeEstimator:
    """Estimate face and gaze results for sampled media frames."""

    def __init__(
        self,
        models: ModelManager,
        annotator: FrameAnnotator,
        temporal_processor: GazeTemporalProcessor,
        point_resolver: GazePointResolver | None = None,
    ) -> None:
        self.models = models
        self.annotator = annotator
        self.temporal_processor = temporal_processor
        self.point_resolver = point_resolver or GazePointResolver()

    @staticmethod
    def _offscreen_direction_label(estimate: dict[str, float | str] | str | None) -> str | None:
        if estimate is None:
            return None
        if isinstance(estimate, str):
            return estimate
        value = estimate.get("direction")
        return str(value) if value is not None else None

    @staticmethod
    def _offscreen_angle_tuple(estimate: dict[str, float | str] | str | None) -> tuple[float, float] | None:
        if estimate is None:
            return None
        if isinstance(estimate, str):
            return None
        yaw = estimate.get("yaw")
        pitch = estimate.get("pitch")
        if yaw is None or pitch is None:
            return None
        return float(yaw), float(pitch)

    def detect_faces(
        self,
        context: MediaContext,
        det_thresh: float,
        face_detection_backend: str,
        face_smoothing_window: int,
        progress_bar=None,
    ) -> pd.DataFrame:
        self._update_progress(progress_bar, 0, 1, "Loading person/object detections...")
        object_df = self._load_scene_detections(context)
        self._update_progress(progress_bar, 1, 1, "Loading person/object detections...")
        face_records: list[dict] = []
        if object_df.empty:
            face_df = pd.DataFrame(columns=FACE_COLUMNS)
            face_df.to_csv(context.faces_path, index=False)
            self._save_faces_meta(context, object_df, face_detection_backend, face_smoothing_window, det_thresh)
            return face_df

        if context.media_type == "image":
            frame = cv2.imread(str(context.media_path))
            if frame is None:
                raise FileNotFoundError(f"Could not open image: {context.media_path}")
            frame_objects = object_df.to_dict(orient="records")
            face_map = self.detect_faces_for_frame(frame, frame_objects, det_thresh, face_detection_backend)
            face_records.extend(self._face_records_from_maps([0], {0: face_map}, object_df, {0: face_map}))
        else:
            raw_face_maps_by_frame: dict[int, dict[int, FaceDetection]] = {}
            self._collect_raw_faces_video(
                context,
                object_df,
                det_thresh,
                face_detection_backend,
                raw_face_maps_by_frame,
                progress_bar,
            )
            self._update_progress(progress_bar, 0, 1, "Interpolating and smoothing faces...")
            face_maps_by_frame = self.temporal_processor.interpolate_faces(
                frame_indices=list(range(context.total_frames)),
                raw_face_maps_by_frame=raw_face_maps_by_frame,
                object_df=object_df,
                smoothing_window=face_smoothing_window,
            )
            self._update_progress(progress_bar, 1, 1, "Interpolating and smoothing faces...")
            face_records.extend(
                self._face_records_from_maps(
                    list(range(context.total_frames)),
                    face_maps_by_frame,
                    object_df,
                    raw_face_maps_by_frame,
                )
            )

        self._update_progress(progress_bar, 0, 1, "Saving faces.csv...")
        face_df = pd.DataFrame(face_records, columns=FACE_COLUMNS)
        face_df.to_csv(context.faces_path, index=False)
        self._save_faces_meta(context, object_df, face_detection_backend, face_smoothing_window, det_thresh)
        self._update_progress(progress_bar, 1, 1, "Saving faces.csv...")
        return face_df

    def estimate_gaze(
        self,
        context: MediaContext,
        device: str,
        det_thresh: float,
        gaze_detection_backend: str,
        head_pose_detection_backend: str,
        visualization_mode: str,
        heatmap_alpha: float,
        gaze_point_method: str,
        gaze_target_radius: int,
        person_part_distance_scale: float,
        person_part_min_conf: float,
        gaze_smoothing_window: int,
        selected_object_classes: list[str],
        reuse_cached_gaze: bool = False,
        progress_bar=None,
    ) -> pd.DataFrame:
        object_df = self._load_scene_detections(context)
        if not context.faces_path.exists():
            raise FileNotFoundError(f"Run face detection first: {context.faces_path}")
        face_df = self._normalize_face_df(pd.read_csv(context.faces_path))
        if object_df.empty or face_df.empty:
            gaze_df = pd.DataFrame(columns=GAZE_COLUMNS)
            gaze_df.to_csv(context.gaze_path, index=False)
            self._save_heatmap_cache(
                context,
                {},
                gaze_detection_backend,
                head_pose_detection_backend,
                gaze_point_method,
                gaze_smoothing_window,
                det_thresh,
            )
            return gaze_df

        if reuse_cached_gaze:
            cached_result = self._load_cached_gaze(
                context,
                object_df,
                det_thresh,
                visualization_mode,
                heatmap_alpha,
                gaze_detection_backend,
                head_pose_detection_backend,
                gaze_target_radius,
                person_part_distance_scale,
                person_part_min_conf,
                gaze_point_method,
                gaze_smoothing_window,
                selected_object_classes,
                progress_bar,
            )
            if cached_result is not None:
                return cached_result

        records: list[dict] = []
        face_maps_by_frame = self._face_maps_from_face_df(face_df)
        if context.media_type == "image":
            gaze_maps = self._estimate_image_from_faces(
                context,
                object_df,
                face_maps_by_frame.get(0, {}),
                device,
                det_thresh,
                visualization_mode,
                heatmap_alpha,
                head_pose_detection_backend,
                gaze_detection_backend,
                gaze_point_method,
                gaze_target_radius,
                person_part_distance_scale,
                person_part_min_conf,
                records,
                selected_object_classes,
                progress_bar,
            )
        else:
            gaze_maps = self._estimate_video_from_faces(
                context,
                object_df,
                face_maps_by_frame,
                device,
                det_thresh,
                visualization_mode,
                heatmap_alpha,
                head_pose_detection_backend,
                gaze_point_method,
                gaze_target_radius,
                person_part_distance_scale,
                person_part_min_conf,
                gaze_smoothing_window,
                records,
                selected_object_classes,
                progress_bar,
            )

        self._update_progress(progress_bar, 0, 1, "Saving gaze outputs...")
        gaze_df = pd.DataFrame(records, columns=GAZE_COLUMNS)
        gaze_df.to_csv(context.gaze_path, index=False)
        self._save_heatmap_cache(
            context,
            gaze_maps,
            gaze_detection_backend,
            head_pose_detection_backend,
            gaze_point_method,
            gaze_smoothing_window,
            det_thresh,
        )
        self._update_progress(progress_bar, 1, 1, "Saving gaze outputs...")
        return gaze_df

    def _estimate_image_from_faces(
        self,
        context: MediaContext,
        object_df: pd.DataFrame,
        face_map: dict[int, FaceDetection],
        device: str,
        det_thresh: float,
        visualization_mode: str,
        heatmap_alpha: float,
        gaze_detection_backend: str,
        head_pose_detection_backend: str,
        gaze_point_method: str,
        gaze_target_radius: int,
        person_part_distance_scale: float,
        person_part_min_conf: float,
        records: list[dict],
        selected_object_classes: list[str],
        progress_bar=None,
    ) -> dict[int, dict[int, GazePoint]]:
        frame = cv2.imread(str(context.media_path))
        if frame is None:
            raise FileNotFoundError(f"Could not open image: {context.media_path}")
        point_enabled = self._point_enabled(visualization_mode)
        heatmap_enabled = self._heatmap_enabled(visualization_mode)
        frame_objects = object_df.to_dict(orient="records")
        visible_objects = self._filter_visible_objects(frame_objects, selected_object_classes)
        gaze_map = self.detect_gazes(frame, face_map, device, gaze_point_method)
        offscreen_direction_map = self.detect_offscreen_directions(
            frame,
            face_map,
            gaze_map,
            det_thresh,
            device,
            head_pose_detection_backend,
        )
        if point_enabled:
            annotated_frame = frame.copy()
            for detection in visible_objects:
                track_id = str(detection["track_id"])
                face = face_map.get(track_id)
                gaze = gaze_map.get(track_id)
                offscreen_direction = self._offscreen_direction_label(offscreen_direction_map.get(track_id))
                offscreen_angles = self._offscreen_angle_tuple(offscreen_direction_map.get(track_id))
                annotated_frame = self.annotator.draw_object(annotated_frame, detection)
                if detection["cls"] == "person":
                    annotated_frame = self.annotator.draw_person_keypoints(
                        annotated_frame,
                        detection,
                        person_part_distance_scale,
                        person_part_min_conf,
                    )
                    annotated_frame = self.annotator.draw_face_and_gaze_point(
                        annotated_frame,
                        face,
                        gaze,
                        det_thresh,
                        gaze_target_radius,
                        offscreen_angles,
                    )
                    records.append(
                        asdict(
                            self.to_record(
                                0,
                                track_id,
                                face,
                                gaze,
                                offscreen_direction,
                                offscreen_angles,
                            )
                        )
                    )
                    target_label = self._resolve_target_label(
                        track_id,
                        gaze,
                        frame_objects,
                        face_map,
                        det_thresh,
                        gaze_target_radius,
                        person_part_distance_scale,
                        person_part_min_conf,
                        selected_object_classes,
                        offscreen_direction,
                    )
                    annotated_frame = self.annotator.draw_gaze_target_label(annotated_frame, track_id, face, target_label)
            cv2.imwrite(str(context.annotated_image_path), annotated_frame)
            cv2.imwrite(str(context.temp_dir / "0.jpg"), annotated_frame)
        if heatmap_enabled:
            self._ensure_person_heatmap_dirs(context, visible_objects)
            for detection in visible_objects:
                if detection["cls"] != "person":
                    continue
                track_id = str(detection["track_id"])
                face = face_map.get(track_id)
                gaze = gaze_map.get(track_id)
                if not point_enabled:
                    offscreen_direction = self._offscreen_direction_label(offscreen_direction_map.get(track_id))
                    offscreen_angles = self._offscreen_angle_tuple(offscreen_direction_map.get(track_id))
                    records.append(
                        asdict(self.to_record(0, track_id, face, gaze, offscreen_direction, offscreen_angles))
                    )
                person_frame = frame.copy()
                person_frame = self.annotator.draw_object(person_frame, detection)
                person_frame = self.annotator.draw_person_keypoints(
                    person_frame,
                    detection,
                    person_part_distance_scale,
                    person_part_min_conf,
                )
                person_frame = self.annotator.draw_face_and_heatmap(person_frame, face, gaze, det_thresh, heatmap_alpha)
                cv2.imwrite(str(self._person_heatmap_frame_path(context, track_id, 0)), person_frame)
        self._update_progress(progress_bar, 1, 1, "Estimating gaze...")
        return {0: gaze_map}

    def _estimate_video_from_faces(
        self,
        context: MediaContext,
        object_df: pd.DataFrame,
        face_maps_by_frame: dict[int, dict[int, FaceDetection]],
        device: str,
        det_thresh: float,
        visualization_mode: str,
        heatmap_alpha: float,
        head_pose_detection_backend: str,
        gaze_point_method: str,
        gaze_target_radius: int,
        person_part_distance_scale: float,
        person_part_min_conf: float,
        gaze_smoothing_window: int,
        records: list[dict],
        selected_object_classes: list[str],
        progress_bar=None,
    ) -> dict[int, dict[int, GazePoint]]:
        sparse_gaze_by_frame: dict[int, dict[int, GazePoint]] = {}
        self._collect_sparse_gazes(
            context,
            device,
            gaze_point_method,
            face_maps_by_frame,
            sparse_gaze_by_frame,
            progress_bar,
        )
        self._update_progress(progress_bar, 0, 1, "Interpolating and smoothing gaze...")
        dense_gaze_by_frame = self.temporal_processor.interpolate_and_smooth(
            frame_indices=list(range(context.total_frames)),
            face_maps_by_frame=face_maps_by_frame,
            sparse_gaze_by_frame=sparse_gaze_by_frame,
            smoothing_window=gaze_smoothing_window,
            point_method=gaze_point_method,
        )
        self._update_progress(progress_bar, 1, 1, "Interpolating and smoothing gaze...")
        offscreen_estimates_by_frame = self._collect_offscreen_estimates_video(
            context,
            face_maps_by_frame,
            dense_gaze_by_frame,
            det_thresh,
            device,
            head_pose_detection_backend,
            progress_bar,
        )
        required_offscreen_frames = self._required_offscreen_frames_by_track(
            face_maps_by_frame,
            dense_gaze_by_frame,
            det_thresh,
        )
        self._update_progress(progress_bar, 0, 1, "Smoothing off-screen directions...")
        offscreen_directions_by_frame, offscreen_angles_by_frame = self._smooth_offscreen_estimates(
            offscreen_estimates_by_frame,
            gaze_smoothing_window,
            required_offscreen_frames,
        )
        missing_offscreen_frames = self._missing_offscreen_frames(
            required_offscreen_frames,
            offscreen_directions_by_frame,
        )
        if missing_offscreen_frames:
            fallback_estimates = self._collect_missing_offscreen_estimates_video(
                context,
                face_maps_by_frame,
                dense_gaze_by_frame,
                det_thresh,
                device,
                head_pose_detection_backend,
                missing_offscreen_frames,
            )
            self._merge_offscreen_estimates(offscreen_estimates_by_frame, fallback_estimates)
            offscreen_directions_by_frame, offscreen_angles_by_frame = self._smooth_offscreen_estimates(
                offscreen_estimates_by_frame,
                gaze_smoothing_window,
                required_offscreen_frames,
            )
        self._update_progress(progress_bar, 1, 1, "Smoothing off-screen directions...")
        self._render_video_outputs(
            context,
            object_df,
            face_maps_by_frame,
            sparse_gaze_by_frame,
            dense_gaze_by_frame,
            det_thresh,
            visualization_mode,
            heatmap_alpha,
            gaze_target_radius,
            person_part_distance_scale,
            person_part_min_conf,
            records,
            selected_object_classes,
            None,
            head_pose_detection_backend,
            offscreen_directions_by_frame,
            offscreen_angles_by_frame,
            progress_bar,
        )
        return dense_gaze_by_frame

    def _collect_offscreen_estimates_video(
        self,
        context: MediaContext,
        face_maps_by_frame: dict[int, dict[int, FaceDetection]],
        dense_gaze_by_frame: dict[int, dict[int, GazePoint]],
        det_thresh: float,
        device: str,
        head_pose_detection_backend: str,
        progress_bar=None,
    ) -> dict[int, dict[str, dict[str, float | str]]]:
        capture = cv2.VideoCapture(str(context.media_path))
        if not capture.isOpened():
            raise FileNotFoundError(f"Could not open video: {context.media_path}")

        estimates_by_frame: dict[int, dict[str, dict[str, float | str]]] = {}
        offscreen_frame_set = set(context.head_pose_frame_idx)
        expected_steps = max(1, len(context.head_pose_frame_idx))
        update_interval = max(1, expected_steps // 200)
        offscreen_step = 0
        try:
            for frame_idx in range(context.total_frames):
                ret, frame = capture.read()
                if not ret:
                    break
                if frame_idx not in offscreen_frame_set:
                    continue
                estimates = self.detect_offscreen_directions(
                    frame,
                    face_maps_by_frame.get(frame_idx, {}),
                    dense_gaze_by_frame.get(frame_idx, {}),
                    det_thresh,
                    device,
                    head_pose_detection_backend,
                )
                if estimates:
                    estimates_by_frame[frame_idx] = estimates
                offscreen_step += 1
                if offscreen_step == expected_steps or offscreen_step % update_interval == 0:
                    self._update_progress(
                        progress_bar,
                        offscreen_step,
                        expected_steps,
                        "Estimating off-screen directions...",
                    )
        finally:
            capture.release()
        return estimates_by_frame

    def _smooth_offscreen_estimates(
        self,
        estimates_by_frame: dict[int, dict[str, dict[str, float | str]]],
        window: int,
        required_frames_by_track: dict[str, list[int]] | None = None,
    ) -> tuple[dict[int, dict[str, str]], dict[int, dict[str, tuple[float, float]]]]:
        directions_by_frame: dict[int, dict[str, str]] = {}
        angles_by_frame: dict[int, dict[str, tuple[float, float]]] = {}
        if not estimates_by_frame:
            return directions_by_frame, angles_by_frame

        rows: list[dict[str, float | str | int]] = []
        for frame_idx, frame_estimates in estimates_by_frame.items():
            for track_id, estimate in frame_estimates.items():
                yaw = estimate.get("yaw")
                pitch = estimate.get("pitch")
                if yaw is None or pitch is None:
                    continue
                rows.append(
                    {
                        "frame_idx": int(frame_idx),
                        "track_id": str(track_id),
                        "yaw": float(yaw),
                        "pitch": float(pitch),
                    }
                )

        if not rows:
            return directions_by_frame, angles_by_frame

        estimate_df = pd.DataFrame(rows).sort_values(["track_id", "frame_idx"])
        for track_id, group in estimate_df.groupby("track_id", sort=False):
            target_frames = sorted(set(required_frames_by_track.get(str(track_id), []))) if required_frames_by_track else []
            if target_frames:
                full_index = sorted(set(group["frame_idx"].astype(int)).union(target_frames))
                smoothed = (
                    group.set_index("frame_idx")[["yaw", "pitch"]]
                    .reindex(full_index)
                    .interpolate(method="index", limit_direction="both")
                    .ffill()
                    .bfill()
                    .loc[target_frames]
                    .reset_index()
                    .rename(columns={"index": "frame_idx"})
                )
                smoothed["track_id"] = str(track_id)
            else:
                smoothed = group.copy()
            if window > 1:
                smoothed["yaw"] = smoothed["yaw"].rolling(window=window, min_periods=1, center=True).mean()
                smoothed["pitch"] = smoothed["pitch"].rolling(window=window, min_periods=1, center=True).mean()
            for _, row in smoothed.iterrows():
                frame_idx = int(row["frame_idx"])
                yaw = float(row["yaw"])
                pitch = float(row["pitch"])
                directions_by_frame.setdefault(frame_idx, {})[str(track_id)] = self._direction_from_angles(yaw, pitch)
                angles_by_frame.setdefault(frame_idx, {})[str(track_id)] = (yaw, pitch)

        return directions_by_frame, angles_by_frame

    def _required_offscreen_frames_by_track(
        self,
        face_maps_by_frame: dict[int, dict[int, FaceDetection]],
        dense_gaze_by_frame: dict[int, dict[int, GazePoint]],
        det_thresh: float,
    ) -> dict[str, list[int]]:
        required: dict[str, list[int]] = {}
        for frame_idx, gaze_map in dense_gaze_by_frame.items():
            face_map = face_maps_by_frame.get(frame_idx, {})
            for track_id, gaze in gaze_map.items():
                if gaze is None or gaze.inout > det_thresh:
                    continue
                if face_map.get(track_id) is None and face_map.get(str(track_id)) is None:
                    continue
                required.setdefault(str(track_id), []).append(int(frame_idx))
        return {track_id: sorted(set(frame_indices)) for track_id, frame_indices in required.items()}

    def _missing_offscreen_frames(
        self,
        required_frames_by_track: dict[str, list[int]],
        directions_by_frame: dict[int, dict[str, str]],
    ) -> dict[int, set[str]]:
        missing: dict[int, set[str]] = {}
        for track_id, frame_indices in required_frames_by_track.items():
            for frame_idx in frame_indices:
                if track_id not in directions_by_frame.get(frame_idx, {}):
                    missing.setdefault(frame_idx, set()).add(track_id)
        return missing

    def _collect_missing_offscreen_estimates_video(
        self,
        context: MediaContext,
        face_maps_by_frame: dict[int, dict[int, FaceDetection]],
        dense_gaze_by_frame: dict[int, dict[int, GazePoint]],
        det_thresh: float,
        device: str,
        head_pose_detection_backend: str,
        missing_frames: dict[int, set[str]],
    ) -> dict[int, dict[str, dict[str, float | str]]]:
        capture = cv2.VideoCapture(str(context.media_path))
        if not capture.isOpened():
            raise FileNotFoundError(f"Could not open video: {context.media_path}")

        estimates_by_frame: dict[int, dict[str, dict[str, float | str]]] = {}
        missing_frame_set = set(missing_frames)
        try:
            for frame_idx in range(context.total_frames):
                ret, frame = capture.read()
                if not ret:
                    break
                if frame_idx not in missing_frame_set:
                    continue
                wanted_track_ids = missing_frames[frame_idx]
                face_map = {
                    track_id: face
                    for track_id, face in face_maps_by_frame.get(frame_idx, {}).items()
                    if str(track_id) in wanted_track_ids
                }
                gaze_map = {
                    track_id: gaze
                    for track_id, gaze in dense_gaze_by_frame.get(frame_idx, {}).items()
                    if str(track_id) in wanted_track_ids
                }
                estimates = self.detect_offscreen_directions(
                    frame,
                    face_map,
                    gaze_map,
                    det_thresh,
                    device,
                    head_pose_detection_backend,
                )
                if estimates:
                    estimates_by_frame[frame_idx] = estimates
        finally:
            capture.release()
        return estimates_by_frame

    def _merge_offscreen_estimates(
        self,
        base: dict[int, dict[str, dict[str, float | str]]],
        updates: dict[int, dict[str, dict[str, float | str]]],
    ) -> None:
        for frame_idx, frame_updates in updates.items():
            base.setdefault(frame_idx, {}).update(frame_updates)

    def _collect_raw_faces_video(
        self,
        context: MediaContext,
        object_df: pd.DataFrame,
        det_thresh: float,
        face_detection_backend: str,
        face_maps_by_frame: dict[int, dict[int, FaceDetection]],
        progress_bar=None,
    ) -> None:
        capture = cv2.VideoCapture(str(context.media_path))
        if not capture.isOpened():
            raise FileNotFoundError(f"Could not open video: {context.media_path}")

        face_frame_set = set(context.face_frame_idx)
        expected_steps = max(1, len(context.face_frame_idx))
        update_interval = max(1, expected_steps // 200)
        face_step = 0
        try:
            for frame_idx in range(context.total_frames):
                ret, frame = capture.read()
                if not ret:
                    break
                if frame_idx not in face_frame_set:
                    continue

                frame_objects = object_df[object_df["frame_idx"] == frame_idx].to_dict(orient="records")
                face_map = self.detect_faces_for_frame(frame, frame_objects, det_thresh, face_detection_backend)
                face_maps_by_frame[frame_idx] = face_map
                face_step += 1
                if face_step == expected_steps or face_step % update_interval == 0:
                    self._update_progress(progress_bar, face_step, expected_steps, "Detecting faces...")
        finally:
            capture.release()

    def _collect_sparse_gazes(
        self,
        context: MediaContext,
        device: str,
        gaze_point_method: str,
        face_maps_by_frame: dict[int, dict[int, FaceDetection]],
        sparse_gaze_by_frame: dict[int, dict[int, GazePoint]],
        progress_bar=None,
    ) -> None:
        capture = cv2.VideoCapture(str(context.media_path))
        if not capture.isOpened():
            raise FileNotFoundError(f"Could not open video: {context.media_path}")

        gaze_frame_set = set(context.gaze_frame_idx)
        expected_steps = max(1, len(context.gaze_frame_idx))
        gaze_step = 0

        try:
            for frame_idx in range(context.total_frames):
                ret, frame = capture.read()
                if not ret:
                    break
                if frame_idx not in gaze_frame_set:
                    continue

                sparse_gaze_by_frame[frame_idx] = self.detect_gazes(
                    frame,
                    face_maps_by_frame.get(frame_idx, {}),
                    device,
                    gaze_point_method,
                )
                gaze_step += 1
                self._update_progress(progress_bar, gaze_step, expected_steps, "Estimating gaze...")
        finally:
            capture.release()

    def _render_video_outputs(
        self,
        context: MediaContext,
        object_df: pd.DataFrame,
        face_maps_by_frame: dict[int, dict[int, FaceDetection]],
        sparse_gaze_by_frame: dict[int, dict[int, GazePoint]],
        dense_gaze_by_frame: dict[int, dict[int, GazePoint]],
        det_thresh: float,
        visualization_mode: str,
        heatmap_alpha: float,
        gaze_target_radius: int,
        person_part_distance_scale: float,
        person_part_min_conf: float,
        records: list[dict],
        selected_object_classes: list[str],
        device: str | None = None,
        head_pose_detection_backend: str = "mobileone",
        offscreen_directions_by_frame: dict[int, dict[str, str]] | None = None,
        offscreen_angles_by_frame: dict[int, dict[str, tuple[float, float]]] | None = None,
        progress_bar=None,
    ) -> None:
        render_capture = cv2.VideoCapture(str(context.media_path))
        if not render_capture.isOpened():
            raise FileNotFoundError(f"Could not open video: {context.media_path}")

        point_enabled = self._point_enabled(visualization_mode)
        heatmap_enabled = self._heatmap_enabled(visualization_mode)
        expected_steps = max(1, context.total_frames)
        render_step = 0
        frame_name_width = len(str(context.total_frames - 1))
        heatmap_track_ids: list[str] = []
        if heatmap_enabled:
            shutil.rmtree(context.heatmap_dir, ignore_errors=True)
            person_detections = object_df[object_df["cls"].astype(str) == "person"]
            heatmap_track_ids = sorted({str(track_id) for track_id in person_detections["track_id"].dropna()})
            self._ensure_person_heatmap_dirs(
                context,
                person_detections.drop_duplicates("track_id").to_dict(orient="records"),
            )

        try:
            for frame_idx in range(context.total_frames):
                ret, frame = render_capture.read()
                if not ret:
                    break

                frame_objects = object_df[object_df["frame_idx"] == frame_idx].to_dict(orient="records")
                visible_objects = self._filter_visible_objects(frame_objects, selected_object_classes)
                face_map = face_maps_by_frame.get(frame_idx, {})
                gaze_map = dense_gaze_by_frame.get(frame_idx, {})
                if offscreen_directions_by_frame is not None:
                    offscreen_direction_map = offscreen_directions_by_frame.get(frame_idx, {})
                elif device is not None:
                    offscreen_direction_map = self.detect_offscreen_directions(
                        frame,
                        face_map,
                        gaze_map,
                        det_thresh,
                        device,
                        head_pose_detection_backend,
                    )
                else:
                    offscreen_direction_map = {}
                offscreen_angle_map = offscreen_angles_by_frame.get(frame_idx, {}) if offscreen_angles_by_frame else {}
                annotated_frame = frame.copy() if point_enabled else None
                visible_person_detections: dict[str, dict] = {}

                for detection in visible_objects:
                    track_id = str(detection["track_id"])
                    face = face_map.get(track_id)
                    gaze = gaze_map.get(track_id)
                    offscreen_direction = self._offscreen_direction_label(offscreen_direction_map.get(track_id))
                    offscreen_angles = offscreen_angle_map.get(track_id) or self._offscreen_angle_tuple(
                        offscreen_direction_map.get(track_id)
                    )
                    if point_enabled:
                        assert annotated_frame is not None
                        annotated_frame = self.annotator.draw_object(annotated_frame, detection)
                        if detection["cls"] == "person":
                            annotated_frame = self.annotator.draw_person_keypoints(
                                annotated_frame,
                                detection,
                                person_part_distance_scale,
                                person_part_min_conf,
                            )
                            annotated_frame = self.annotator.draw_face_and_gaze_point(
                                annotated_frame,
                                face,
                                gaze,
                                det_thresh,
                                gaze_target_radius,
                                offscreen_angles,
                            )
                        if detection["cls"] == "person":
                            records.append(
                                asdict(
                                    self.to_record(
                                        frame_idx,
                                        track_id,
                                        face,
                                        gaze,
                                        offscreen_direction,
                                        offscreen_angles,
                                    )
                                )
                            )
                            target_label = self._resolve_target_label(
                                track_id,
                                gaze,
                                frame_objects,
                                face_map,
                                det_thresh,
                                gaze_target_radius,
                                person_part_distance_scale,
                                person_part_min_conf,
                                selected_object_classes,
                                offscreen_direction,
                            )
                            annotated_frame = self.annotator.draw_gaze_target_label(
                                annotated_frame,
                                track_id,
                                face,
                                target_label,
                            )
                    if heatmap_enabled and detection["cls"] == "person":
                        visible_person_detections[track_id] = detection
                        if not point_enabled:
                            records.append(
                                asdict(
                                    self.to_record(
                                        frame_idx,
                                        track_id,
                                        face,
                                        gaze,
                                        offscreen_direction,
                                        offscreen_angles,
                                    )
                                )
                            )

                if heatmap_enabled:
                    for track_id in heatmap_track_ids:
                        detection = visible_person_detections.get(track_id)
                        if detection is None:
                            cv2.imwrite(str(self._person_heatmap_frame_path(context, track_id, frame_idx)), frame)
                            continue
                        face = face_map.get(track_id)
                        gaze = gaze_map.get(track_id)
                        person_frame = frame.copy()
                        person_frame = self.annotator.draw_object(person_frame, detection)
                        person_frame = self.annotator.draw_person_keypoints(
                            person_frame,
                            detection,
                            person_part_distance_scale,
                            person_part_min_conf,
                        )
                        person_frame = self.annotator.draw_face_and_heatmap(
                            person_frame, face, gaze, det_thresh, heatmap_alpha
                        )
                        cv2.imwrite(
                            str(self._person_heatmap_frame_path(context, track_id, frame_idx)),
                            person_frame,
                        )

                if point_enabled:
                    assert annotated_frame is not None
                    cv2.imwrite(str(context.temp_dir / f"{frame_idx:0{frame_name_width}d}.jpg"), annotated_frame)

                render_step += 1
                self._update_progress(
                    progress_bar,
                    render_step,
                    expected_steps,
                    "Rendering outputs...",
                )
        finally:
            render_capture.release()

    def _load_cached_gaze(
        self,
        context: MediaContext,
        object_df: pd.DataFrame,
        det_thresh: float,
        visualization_mode: str,
        heatmap_alpha: float,
        gaze_detection_backend: str,
        head_pose_detection_backend: str,
        gaze_target_radius: int,
        person_part_distance_scale: float,
        person_part_min_conf: float,
        gaze_point_method: str,
        gaze_smoothing_window: int,
        selected_object_classes: list[str],
        progress_bar=None,
    ) -> pd.DataFrame | None:
        if not context.gaze_path.exists() or not context.gaze_heatmaps_path.exists() or not context.gaze_meta_path.exists():
            return None
        try:
            gaze_df = pd.read_csv(context.gaze_path)
            face_df = self._load_face_df_for_cached_gaze(context, gaze_df)
            if face_df is None:
                return None
            with context.gaze_meta_path.open("r", encoding="utf-8") as file:
                gaze_meta = json.load(file)
        except Exception:
            return None
        if gaze_df.empty:
            return None
        if gaze_meta.get("media_path") != str(context.media_path.resolve()):
            return None
        if int(gaze_meta.get("media_mtime_ns", -1)) != context.media_path.stat().st_mtime_ns:
            return None
        if int(gaze_meta.get("gaze_smoothing_window", -1)) != int(gaze_smoothing_window):
            return None
        if int(gaze_meta.get("gaze_stride", -1)) != int(context.gaze_stride):
            return None
        if int(gaze_meta.get("head_pose_stride", -1)) != int(context.head_pose_stride):
            return None
        if abs(float(gaze_meta.get("det_thresh", -1.0)) - float(det_thresh)) > 1e-9:
            return None
        if str(gaze_meta.get("gaze_detection_backend", "")) != str(gaze_detection_backend):
            return None
        if str(gaze_meta.get("head_pose_detection_backend", "")) != str(head_pose_detection_backend):
            return None
        if int(gaze_meta.get("faces_mtime_ns", -1)) != context.faces_path.stat().st_mtime_ns:
            return None

        cached_method = str(gaze_meta.get("gaze_point_method", ""))
        if cached_method == gaze_point_method:
            self._notify_skip(
                progress_bar,
                "Skipping gaze estimation: reusing cached gaze.csv and gaze_heatmaps.npz.",
            )
        else:
            self._notify_skip(
                progress_bar,
                (
                    "Skipping gaze estimation: reusing cached gaze_heatmaps.npz "
                    f"and recalculating gaze points with '{gaze_point_method}'."
                ),
            )

        frame_width, frame_height = self._resolve_media_size(context)
        dense_gaze_by_frame = self._load_heatmaps(context, gaze_df, frame_width, frame_height, gaze_point_method)
        gaze_df = self._rebuild_gaze_df_from_dense_heatmaps(gaze_df, dense_gaze_by_frame)
        gaze_df.to_csv(context.gaze_path, index=False)
        if context.media_type == "image":
            self._render_cached_image(
                context,
                object_df,
                face_df,
                gaze_df,
                dense_gaze_by_frame.get(0, {}),
                det_thresh,
                visualization_mode,
                heatmap_alpha,
                gaze_target_radius,
                person_part_distance_scale,
                person_part_min_conf,
                selected_object_classes,
            )
        else:
            self._render_cached_video(
                context,
                object_df,
                face_df,
                gaze_df,
                dense_gaze_by_frame,
                det_thresh,
                visualization_mode,
                heatmap_alpha,
                gaze_target_radius,
                person_part_distance_scale,
                person_part_min_conf,
                selected_object_classes,
                progress_bar,
            )
        return gaze_df

    def _render_cached_image(
        self,
        context: MediaContext,
        object_df: pd.DataFrame,
        face_df: pd.DataFrame,
        gaze_df: pd.DataFrame,
        gaze_map: dict[int, GazePoint],
        det_thresh: float,
        visualization_mode: str,
        heatmap_alpha: float,
        gaze_target_radius: int,
        person_part_distance_scale: float,
        person_part_min_conf: float,
        selected_object_classes: list[str],
    ) -> None:
        frame = cv2.imread(str(context.media_path))
        if frame is None:
            raise FileNotFoundError(f"Could not open image: {context.media_path}")

        point_enabled = self._point_enabled(visualization_mode)
        heatmap_enabled = self._heatmap_enabled(visualization_mode)
        frame_objects = object_df.to_dict(orient="records")
        visible_objects = self._filter_visible_objects(frame_objects, selected_object_classes)
        face_map = self._face_map_from_face_df(face_df, 0)
        offscreen_direction_map = self._offscreen_direction_map_from_gaze_df(gaze_df, 0)
        offscreen_angle_map = self._offscreen_angle_map_from_gaze_df(gaze_df, 0)

        if point_enabled:
            annotated_frame = frame.copy()
            for detection in visible_objects:
                track_id = str(detection["track_id"])
                face = face_map.get(track_id)
                gaze = gaze_map.get(track_id)
                annotated_frame = self.annotator.draw_object(annotated_frame, detection)
                if detection["cls"] == "person":
                    annotated_frame = self.annotator.draw_person_keypoints(
                        annotated_frame,
                        detection,
                        person_part_distance_scale,
                        person_part_min_conf,
                    )
                    annotated_frame = self.annotator.draw_face_and_gaze_point(
                        annotated_frame,
                        face,
                        gaze,
                        det_thresh,
                        gaze_target_radius,
                        offscreen_angle_map.get(track_id),
                    )
                if detection["cls"] == "person":
                    target_label = self._resolve_target_label(
                        track_id,
                        gaze,
                        frame_objects,
                        face_map,
                        det_thresh,
                        gaze_target_radius,
                        person_part_distance_scale,
                        person_part_min_conf,
                        selected_object_classes,
                        offscreen_direction_map.get(track_id),
                    )
                    annotated_frame = self.annotator.draw_gaze_target_label(annotated_frame, track_id, face, target_label)
            cv2.imwrite(str(context.annotated_image_path), annotated_frame)
            cv2.imwrite(str(context.temp_dir / "0.jpg"), annotated_frame)

        if heatmap_enabled:
            self._ensure_person_heatmap_dirs(context, visible_objects)
            for detection in visible_objects:
                if detection["cls"] != "person":
                    continue
                track_id = str(detection["track_id"])
                face = face_map.get(track_id)
                gaze = gaze_map.get(track_id)
                person_frame = frame.copy()
                person_frame = self.annotator.draw_object(person_frame, detection)
                person_frame = self.annotator.draw_person_keypoints(
                    person_frame,
                    detection,
                    person_part_distance_scale,
                    person_part_min_conf,
                )
                person_frame = self.annotator.draw_face_and_heatmap(person_frame, face, gaze, det_thresh, heatmap_alpha)
                cv2.imwrite(str(self._person_heatmap_frame_path(context, track_id, 0)), person_frame)

    def _render_cached_video(
        self,
        context: MediaContext,
        object_df: pd.DataFrame,
        face_df: pd.DataFrame,
        gaze_df: pd.DataFrame,
        dense_gaze_by_frame: dict[int, dict[int, GazePoint]],
        det_thresh: float,
        visualization_mode: str,
        heatmap_alpha: float,
        gaze_target_radius: int,
        person_part_distance_scale: float,
        person_part_min_conf: float,
        selected_object_classes: list[str],
        progress_bar=None,
    ) -> None:
        face_maps_by_frame = self._face_maps_from_face_df(face_df)
        offscreen_directions_by_frame = self._offscreen_directions_from_gaze_df(gaze_df)
        offscreen_angles_by_frame = self._offscreen_angles_from_gaze_df(gaze_df)
        self._render_video_outputs(
            context,
            object_df,
            face_maps_by_frame,
            {},
            dense_gaze_by_frame,
            det_thresh,
            visualization_mode,
            heatmap_alpha,
            gaze_target_radius,
            person_part_distance_scale,
            person_part_min_conf,
            [],
            selected_object_classes,
            None,
            "mobileone",
            offscreen_directions_by_frame,
            offscreen_angles_by_frame,
            progress_bar,
        )

    def _save_heatmap_cache(
        self,
        context: MediaContext,
        dense_gaze_by_frame: dict[int, dict[int, GazePoint]],
        gaze_detection_backend: str,
        head_pose_detection_backend: str,
        gaze_point_method: str,
        gaze_smoothing_window: int,
        det_thresh: float,
    ) -> None:
        arrays: dict[str, np.ndarray] = {}
        for frame_idx, gaze_map in dense_gaze_by_frame.items():
            for track_id, gaze in gaze_map.items():
                arrays[f"frame_{frame_idx}_track_{track_id}"] = gaze.heatmap.astype(np.float32)
        np.savez_compressed(context.gaze_heatmaps_path, **arrays)
        with context.gaze_meta_path.open("w", encoding="utf-8") as file:
            json.dump(
                {
                    "gaze_point_method": gaze_point_method,
                    "gaze_detection_backend": gaze_detection_backend,
                    "gaze_smoothing_window": gaze_smoothing_window,
                    "gaze_stride": int(context.gaze_stride),
                    "head_pose_stride": int(context.head_pose_stride),
                    "det_thresh": float(det_thresh),
                    "head_pose_detection_backend": head_pose_detection_backend,
                    "media_path": str(context.media_path.resolve()),
                    "media_mtime_ns": context.media_path.stat().st_mtime_ns,
                    "faces_mtime_ns": context.faces_path.stat().st_mtime_ns,
                },
                file,
                ensure_ascii=False,
                indent=2,
            )

    def _save_faces_meta(
        self,
        context: MediaContext,
        object_df: pd.DataFrame,
        face_detection_backend: str,
        face_smoothing_window: int,
        det_thresh: float,
    ) -> None:
        with context.faces_meta_path.open("w", encoding="utf-8") as file:
            json.dump(
                {
                    "face_detection_backend": face_detection_backend,
                    "face_smoothing_window": face_smoothing_window,
                    "face_stride": int(context.face_stride),
                    "det_thresh": float(det_thresh),
                    "media_path": str(context.media_path.resolve()),
                    "media_mtime_ns": context.media_path.stat().st_mtime_ns,
                    "persons_mtime_ns": context.persons_path.stat().st_mtime_ns if context.persons_path.exists() else -1,
                    "objects_mtime_ns": context.objects_path.stat().st_mtime_ns if context.objects_path.exists() else -1,
                },
                file,
                ensure_ascii=False,
                indent=2,
            )

    def _load_heatmaps(
        self,
        context: MediaContext,
        gaze_df: pd.DataFrame,
        frame_width: int,
        frame_height: int,
        gaze_point_method: str,
    ) -> dict[int, dict[int, GazePoint]]:
        dense_gaze_by_frame: dict[int, dict[int, GazePoint]] = {}
        with np.load(context.gaze_heatmaps_path) as heatmaps:
            for _, row in gaze_df[gaze_df["gaze_detected"] == True].iterrows():
                frame_idx = int(row["frame_idx"])
                track_id = str(row["track_id"])
                key = f"frame_{frame_idx}_track_{track_id}"
                if key not in heatmaps:
                    continue
                heatmap = heatmaps[key].astype(np.float32)
                x_gaze, y_gaze = self.point_resolver.resolve(heatmap, frame_width, frame_height, gaze_point_method)
                dense_gaze_by_frame.setdefault(frame_idx, {})[track_id] = GazePoint(
                    track_id=track_id,
                    inout=float(row["inout"]),
                    x_gaze=x_gaze,
                    y_gaze=y_gaze,
                    heatmap=heatmap,
                    frame_width=frame_width,
                    frame_height=frame_height,
                )
        return dense_gaze_by_frame

    def _rebuild_gaze_df_from_dense_heatmaps(
        self,
        gaze_df: pd.DataFrame,
        dense_gaze_by_frame: dict[int, dict[int, GazePoint]],
    ) -> pd.DataFrame:
        rebuilt_df = gaze_df.copy()
        if "offscreen_direction" not in rebuilt_df.columns:
            rebuilt_df["offscreen_direction"] = None
        if "offscreen_yaw" not in rebuilt_df.columns:
            rebuilt_df["offscreen_yaw"] = None
        if "offscreen_pitch" not in rebuilt_df.columns:
            rebuilt_df["offscreen_pitch"] = None
        for index, row in rebuilt_df.iterrows():
            frame_idx = int(row["frame_idx"])
            track_id = str(row["track_id"])
            gaze = dense_gaze_by_frame.get(frame_idx, {}).get(track_id)
            if gaze is None:
                rebuilt_df.at[index, "gaze_detected"] = False
                rebuilt_df.at[index, "inout"] = None
                rebuilt_df.at[index, "x_gaze"] = None
                rebuilt_df.at[index, "y_gaze"] = None
                continue
            rebuilt_df.at[index, "gaze_detected"] = True
            rebuilt_df.at[index, "inout"] = gaze.inout
            rebuilt_df.at[index, "x_gaze"] = gaze.x_gaze
            rebuilt_df.at[index, "y_gaze"] = gaze.y_gaze
        for column in GAZE_COLUMNS:
            if column not in rebuilt_df.columns:
                rebuilt_df[column] = None
        return rebuilt_df[GAZE_COLUMNS].copy()

    def _resolve_media_size(self, context: MediaContext) -> tuple[int, int]:
        if context.media_type == "image":
            frame = cv2.imread(str(context.media_path))
            if frame is None:
                raise FileNotFoundError(f"Could not open image: {context.media_path}")
            return int(frame.shape[1]), int(frame.shape[0])

        capture = cv2.VideoCapture(str(context.media_path))
        if not capture.isOpened():
            raise FileNotFoundError(f"Could not open video: {context.media_path}")
        try:
            ret, frame = capture.read()
        finally:
            capture.release()
        if not ret or frame is None:
            raise FileNotFoundError(f"Could not read video frame: {context.media_path}")
        return int(frame.shape[1]), int(frame.shape[0])


    def _load_json_file(self, path: Path) -> dict | None:
        if not path.exists():
            return None
        try:
            with path.open("r", encoding="utf-8") as file:
                data = json.load(file)
            return data if isinstance(data, dict) else None
        except Exception:
            return None

    def _load_scene_detections(self, context: MediaContext) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        if context.persons_path.exists():
            frames.append(pd.read_csv(context.persons_path))
        if context.objects_path.exists():
            frames.append(pd.read_csv(context.objects_path))
        if not frames:
            if context.objects_path.exists():
                return pd.read_csv(context.objects_path)
            return pd.DataFrame()
        combined = pd.concat(frames, ignore_index=True)
        if combined.empty:
            return combined
        return combined.sort_values(["frame_idx", "track_id"]).reset_index(drop=True)

    def _load_face_df_for_cached_gaze(self, context: MediaContext, gaze_df: pd.DataFrame) -> pd.DataFrame | None:
        if context.faces_path.exists():
            face_df = pd.read_csv(context.faces_path)
            return self._normalize_face_df(face_df)
        if all(column in gaze_df.columns for column in FACE_COLUMNS):
            return self._normalize_face_df(gaze_df[FACE_COLUMNS].copy())
        return None

    def _normalize_face_df(self, face_df: pd.DataFrame) -> pd.DataFrame:
        for column in FACE_COLUMNS:
            if column not in face_df.columns:
                face_df[column] = None
        return face_df[FACE_COLUMNS].copy()

    def _face_records_from_maps(
        self,
        frame_indices: list[int],
        face_maps_by_frame: dict[int, dict[int, FaceDetection]],
        object_df: pd.DataFrame,
        raw_face_maps_by_frame: dict[int, dict[int, FaceDetection]] | None = None,
    ) -> list[dict]:
        frame_index_set = set(int(frame_idx) for frame_idx in frame_indices)
        person_df = object_df[object_df["cls"] == "person"].copy()
        records: list[dict] = []
        if person_df.empty:
            return records
        person_df["track_id"] = person_df["track_id"].astype(str)
        person_df = person_df.sort_values(["frame_idx", "track_id"])
        for _, row in person_df.iterrows():
            frame_idx = int(row["frame_idx"])
            if frame_idx not in frame_index_set:
                continue
            track_id = str(row["track_id"])
            face = face_maps_by_frame.get(frame_idx, {}).get(track_id)
            face_detected = track_id in (raw_face_maps_by_frame or {}).get(frame_idx, {})
            if face is None:
                records.append(
                    {
                        "frame_idx": frame_idx,
                        "track_id": track_id,
                        "face_detected": False,
                        "face_conf": None,
                        "face_x1": None,
                        "face_y1": None,
                        "face_x2": None,
                        "face_y2": None,
                    }
                )
                continue
            records.append(
                {
                    "frame_idx": frame_idx,
                    "track_id": track_id,
                    "face_detected": face_detected,
                    "face_conf": face.conf,
                    "face_x1": face.x1,
                    "face_y1": face.y1,
                    "face_x2": face.x2,
                    "face_y2": face.y2,
                }
            )
        return records

    def _face_maps_from_face_df(self, face_df: pd.DataFrame) -> dict[int, dict[int, FaceDetection]]:
        face_maps_by_frame: dict[int, dict[int, FaceDetection]] = {}
        normalized_face_df = self._normalize_face_df(face_df)
        coordinate_face_df = normalized_face_df[normalized_face_df["face_x1"].notna()]
        for _, row in coordinate_face_df.iterrows():
            frame_idx = int(row["frame_idx"])
            track_id = str(row["track_id"])
            face_maps_by_frame.setdefault(frame_idx, {})[track_id] = FaceDetection(
                track_id=track_id,
                conf=float(row["face_conf"]),
                x1=int(row["face_x1"]),
                y1=int(row["face_y1"]),
                x2=int(row["face_x2"]),
                y2=int(row["face_y2"]),
            )
        return face_maps_by_frame

    def _face_map_from_face_df(self, face_df: pd.DataFrame, frame_idx: int) -> dict[int, FaceDetection]:
        return self._face_maps_from_face_df(face_df).get(frame_idx, {})

    def _offscreen_directions_from_gaze_df(self, gaze_df: pd.DataFrame) -> dict[int, dict[str, str]]:
        if "offscreen_direction" not in gaze_df.columns:
            return {}
        directions_by_frame: dict[int, dict[str, str]] = {}
        valid_rows = gaze_df[gaze_df["offscreen_direction"].notna()]
        for _, row in valid_rows.iterrows():
            directions_by_frame.setdefault(int(row["frame_idx"]), {})[str(row["track_id"])] = str(
                row["offscreen_direction"]
            )
        return directions_by_frame

    def _offscreen_direction_map_from_gaze_df(self, gaze_df: pd.DataFrame, frame_idx: int) -> dict[str, str]:
        return self._offscreen_directions_from_gaze_df(gaze_df).get(frame_idx, {})

    def _offscreen_angles_from_gaze_df(self, gaze_df: pd.DataFrame) -> dict[int, dict[str, tuple[float, float]]]:
        if "offscreen_yaw" not in gaze_df.columns or "offscreen_pitch" not in gaze_df.columns:
            return {}
        angles_by_frame: dict[int, dict[str, tuple[float, float]]] = {}
        valid_rows = gaze_df[gaze_df["offscreen_yaw"].notna() & gaze_df["offscreen_pitch"].notna()]
        for _, row in valid_rows.iterrows():
            angles_by_frame.setdefault(int(row["frame_idx"]), {})[str(row["track_id"])] = (
                float(row["offscreen_yaw"]),
                float(row["offscreen_pitch"]),
            )
        return angles_by_frame

    def _offscreen_angle_map_from_gaze_df(
        self,
        gaze_df: pd.DataFrame,
        frame_idx: int,
    ) -> dict[str, tuple[float, float]]:
        return self._offscreen_angles_from_gaze_df(gaze_df).get(frame_idx, {})

    def detect_faces_for_frame(
        self,
        frame: np.ndarray,
        detections: list[dict],
        det_thresh: float,
        face_detection_backend: str,
    ) -> dict[int, FaceDetection]:
        person_detections = [d for d in detections if d["cls"] == "person"]
        if not person_detections:
            return {}

        valid_faces = self._detect_face_candidates(frame, det_thresh, face_detection_backend)
        if not valid_faces:
            return {}

        scored_pairs: list[tuple[float, str, dict]] = []
        for detection in person_detections:
            keypoint_summary = self._face_keypoint_summary(detection, frame)
            if keypoint_summary is None:
                continue
            for face in valid_faces:
                score = self._score_face_person_match(face, keypoint_summary)
                scored_pairs.append((score, str(detection["track_id"]), face))

        face_map: dict[int, FaceDetection] = {}
        assigned_tracks: set[str] = set()
        assigned_face_ids: set[int] = set()
        for _, track_id, face in sorted(scored_pairs, key=lambda item: item[0]):
            face_id = id(face)
            if track_id in assigned_tracks or face_id in assigned_face_ids:
                continue
            fx1, fy1, fx2, fy2 = map(int, face["bbox"])
            face_map[track_id] = FaceDetection(
                track_id=track_id,
                conf=float(face["score"]),
                x1=fx1,
                y1=fy1,
                x2=fx2,
                y2=fy2,
            )
            assigned_tracks.add(track_id)
            assigned_face_ids.add(face_id)
        return face_map

    def _detect_face_candidates(
        self,
        frame: np.ndarray,
        det_thresh: float,
        face_detection_backend: str,
    ) -> list[dict]:
        if face_detection_backend == "retinaface":
            assert self.models.retinaface is not None
            raw_faces = self.models.retinaface.predict_jsons(frame)
            return [face for face in raw_faces if float(face["score"]) >= det_thresh and face.get("bbox")]
        if face_detection_backend == "mediapipe":
            if self.models.mediapipe_face_detector is None:
                raise RuntimeError("MediaPipe face detection is not initialized.")
            height, width = frame.shape[:2]
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if self.models.mediapipe_face_detector_api == "tasks":
                import mediapipe as mp

                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                result = self.models.mediapipe_face_detector.detect(image)
                faces: list[dict] = []
                for detection in result.detections or []:
                    score = max((float(category.score) for category in detection.categories), default=0.0)
                    if score < det_thresh:
                        continue
                    box = detection.bounding_box
                    x1 = int(round(box.origin_x))
                    y1 = int(round(box.origin_y))
                    x2 = int(round(box.origin_x + box.width))
                    y2 = int(round(box.origin_y + box.height))
                    x1 = max(0, min(width - 1, x1))
                    y1 = max(0, min(height - 1, y1))
                    x2 = max(x1 + 1, min(width, x2))
                    y2 = max(y1 + 1, min(height, y2))
                    faces.append({"bbox": [x1, y1, x2, y2], "score": score})
                return faces

            result = self.models.mediapipe_face_detector.process(rgb_frame)
            faces: list[dict] = []
            for detection in result.detections or []:
                score = float(detection.score[0]) if detection.score else 0.0
                if score < det_thresh:
                    continue
                relative_box = detection.location_data.relative_bounding_box
                x1 = int(round(relative_box.xmin * width))
                y1 = int(round(relative_box.ymin * height))
                x2 = int(round((relative_box.xmin + relative_box.width) * width))
                y2 = int(round((relative_box.ymin + relative_box.height) * height))
                x1 = max(0, min(width - 1, x1))
                y1 = max(0, min(height - 1, y1))
                x2 = max(x1 + 1, min(width, x2))
                y2 = max(y1 + 1, min(height, y2))
                faces.append({"bbox": [x1, y1, x2, y2], "score": score})
            return faces
        raise ValueError(f"Unsupported face detection backend: {face_detection_backend}")

    def _face_keypoint_summary(self, detection: dict, frame: np.ndarray) -> dict[str, object] | None:
        keypoints = self._pose_keypoints(detection)
        height, width = frame.shape[:2]
        points: list[tuple[float, float]] = []
        for keypoint_index in [0, 1, 2, 3, 4]:
            if keypoint_index >= len(keypoints):
                continue
            keypoint = keypoints[keypoint_index]
            if not isinstance(keypoint, (list, tuple)) or len(keypoint) < 2:
                continue
            x = float(keypoint[0])
            y = float(keypoint[1])
            if not np.isfinite(x) or not np.isfinite(y):
                continue
            if x < 0 or y < 0 or x >= width or y >= height:
                continue
            points.append((x, y))
        if not points:
            return None
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        return {
            "points": points,
            "center": (float(np.mean(xs)), float(np.mean(ys))),
            "bbox": (float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))),
        }

    def _score_face_person_match(self, face: dict, keypoint_summary: dict[str, object]) -> float:
        fx1, fy1, fx2, fy2 = map(float, face["bbox"])
        face_width = max(fx2 - fx1, 1.0)
        face_height = max(fy2 - fy1, 1.0)
        face_diag = float(np.hypot(face_width, face_height))
        face_area = max(face_width * face_height, 1.0)
        face_center_x = (fx1 + fx2) / 2.0
        face_center_y = (fy1 + fy2) / 2.0

        keypoint_center_x, keypoint_center_y = keypoint_summary["center"]
        kx1, ky1, kx2, ky2 = keypoint_summary["bbox"]
        keypoint_width = max(float(kx2 - kx1), 1.0)
        keypoint_height = max(float(ky2 - ky1), 1.0)
        keypoint_area = max(keypoint_width * keypoint_height, 1.0)
        keypoint_diag = float(np.hypot(keypoint_width, keypoint_height))

        center_distance = float(np.hypot(keypoint_center_x - face_center_x, keypoint_center_y - face_center_y))
        normalized_distance = center_distance / max(face_diag, keypoint_diag, 1.0)

        margin_x = face_width * 0.20
        margin_y = face_height * 0.20
        points = keypoint_summary["points"]
        outside_count = 0
        for x, y in points:
            if x < fx1 - margin_x or x > fx2 + margin_x or y < fy1 - margin_y or y > fy2 + margin_y:
                outside_count += 1
        outside_ratio = outside_count / max(len(points), 1)

        expected_face_to_keypoint_ratio = 4.0
        size_ratio = face_area / max(keypoint_area, 1.0)
        size_penalty = abs(float(np.log(max(size_ratio, 1e-6) / expected_face_to_keypoint_ratio)))

        detection_bonus = 0.15 * float(face.get("score", 0.0))
        return normalized_distance + (1.5 * outside_ratio) + (0.2 * size_penalty) - detection_bonus

    def _pose_keypoints(self, detection: dict) -> list:
        raw_keypoints = detection.get("pose_keypoints")
        if isinstance(raw_keypoints, str):
            try:
                decoded = json.loads(raw_keypoints)
                return decoded if isinstance(decoded, list) else []
            except Exception:
                return []
        return raw_keypoints if isinstance(raw_keypoints, list) else []

    def detect_gazes(
        self,
        frame: np.ndarray,
        face_map: dict[int, FaceDetection],
        device: str,
        gaze_point_method: str,
    ) -> dict[int, GazePoint]:
        if not face_map:
            return {}
        assert self.models.gazelle is not None
        assert self.models.gazelle_transform is not None

        height, width = frame.shape[:2]
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img_tensor = self.models.gazelle_transform(frame_pil).unsqueeze(0).to(device)
        sorted_faces = sorted(face_map.values(), key=lambda face: face.track_id)
        bboxes = [
            (face.x1 / width, face.y1 / height, face.x2 / width, face.y2 / height)
            for face in sorted_faces
        ]

        with torch.no_grad():
            output = self.models.gazelle({"images": img_tensor, "bboxes": [bboxes]})

        gaze_map: dict[int, GazePoint] = {}
        for index, face in enumerate(sorted_faces):
            heatmap = output["heatmap"][0][index].detach().cpu().numpy()
            heatmap = np.maximum(heatmap, 0)
            x_gaze, y_gaze = self.point_resolver.resolve(heatmap, width, height, gaze_point_method)
            gaze_map[face.track_id] = GazePoint(
                track_id=face.track_id,
                inout=float(output["inout"][0][index]),
                x_gaze=x_gaze,
                y_gaze=y_gaze,
                heatmap=heatmap.astype(np.float32),
                frame_width=width,
                frame_height=height,
            )
        return gaze_map

    def detect_offscreen_directions(
        self,
        frame: np.ndarray,
        face_map: dict[int, FaceDetection],
        gaze_map: dict[int, GazePoint],
        det_thresh: float,
        device: str,
        head_pose_detection_backend: str,
    ) -> dict[str, dict[str, float | str]]:
        if head_pose_detection_backend != "mobileone":
            raise ValueError(f"Unsupported head pose detection backend: {head_pose_detection_backend}")
        if not face_map or not gaze_map:
            return {}
        if self.models.mobile_gaze is None or self.models.mobile_gaze_transform is None:
            return {}

        crops: list[torch.Tensor] = []
        track_ids: list[str] = []
        for track_id, gaze in gaze_map.items():
            if gaze is None or gaze.inout > det_thresh:
                continue
            face = face_map.get(track_id)
            if face is None:
                continue
            crop = self._crop_face(frame, face)
            if crop is None:
                continue
            crops.append(self.models.mobile_gaze_transform(crop))
            track_ids.append(str(track_id))

        if not crops:
            return {}

        device_obj = torch.device(device)
        batch = torch.stack(crops).to(device_obj)
        idx_tensor = torch.arange(90, device=device_obj, dtype=torch.float32)

        with torch.no_grad():
            yaw_logits, pitch_logits = self.models.mobile_gaze(batch)

        yaw_probs = torch.softmax(yaw_logits, dim=1)
        pitch_probs = torch.softmax(pitch_logits, dim=1)
        yaw_deg = torch.sum(yaw_probs * idx_tensor, dim=1) * 4.0 - 180.0
        pitch_deg = torch.sum(pitch_probs * idx_tensor, dim=1) * 4.0 - 180.0

        directions: dict[str, dict[str, float | str]] = {}
        for index, track_id in enumerate(track_ids):
            yaw_value = float(yaw_deg[index].item())
            pitch_value = float(pitch_deg[index].item())
            directions[track_id] = {
                "direction": self._direction_from_angles(yaw_value, pitch_value),
                "yaw": yaw_value,
                "pitch": pitch_value,
            }
        return directions

    def _crop_face(self, frame: np.ndarray, face: FaceDetection) -> np.ndarray | None:
        height, width = frame.shape[:2]
        x1 = max(0, min(width, int(face.x1)))
        y1 = max(0, min(height, int(face.y1)))
        x2 = max(0, min(width, int(face.x2)))
        y2 = max(0, min(height, int(face.y2)))
        if x2 <= x1 or y2 <= y1:
            return None
        crop = frame[y1:y2, x1:x2]
        return crop if crop.size > 0 else None

    def _direction_from_angles(self, yaw_deg: float, pitch_deg: float) -> str:
        horizontal_threshold = 12.0
        vertical_threshold = 12.0

        horizontal = ""
        # Use person-centric left/right labels (not screen-centric).
        # Positive yaw means the face turns to the person's right.
        if yaw_deg >= horizontal_threshold:
            horizontal = "right"
        elif yaw_deg <= -horizontal_threshold:
            horizontal = "left"

        vertical = ""
        if pitch_deg >= vertical_threshold:
            vertical = "up"
        elif pitch_deg <= -vertical_threshold:
            vertical = "down"

        if vertical and horizontal:
            return f"{vertical} {horizontal}"
        if vertical:
            return vertical
        if horizontal:
            return horizontal
        # Near-center yaw/pitch means the person is likely facing the camera/front.
        return "front"

    def to_record(
        self,
        frame_idx: int,
        track_id: str,
        face: FaceDetection | None,
        gaze: GazePoint | None,
        offscreen_direction: str | None,
        offscreen_angles: tuple[float, float] | None,
    ) -> GazeRecord:
        if face is None:
            return GazeRecord(
                frame_idx=frame_idx,
                track_id=track_id,
                gaze_detected=False,
                inout=None,
                x_gaze=None,
                y_gaze=None,
                offscreen_direction=None,
                offscreen_yaw=None,
                offscreen_pitch=None,
            )
        return GazeRecord(
            frame_idx=frame_idx,
            track_id=track_id,
            gaze_detected=gaze is not None,
            inout=None if gaze is None else gaze.inout,
            x_gaze=None if gaze is None else gaze.x_gaze,
            y_gaze=None if gaze is None else gaze.y_gaze,
            offscreen_direction=offscreen_direction,
            offscreen_yaw=None if offscreen_angles is None else offscreen_angles[0],
            offscreen_pitch=None if offscreen_angles is None else offscreen_angles[1],
        )

    def _ensure_person_heatmap_dirs(self, context: MediaContext, detections: list[dict]) -> None:
        for detection in detections:
            if detection.get("cls") != "person":
                continue
            (context.heatmap_dir / f"person_{str(detection['track_id'])}").mkdir(parents=True, exist_ok=True)

    def _person_heatmap_frame_path(self, context: MediaContext, track_id: str, frame_idx: int) -> Path:
        return context.heatmap_dir / f"person_{track_id}" / f"{frame_idx:06d}.jpg"

    def _update_progress(self, progress_bar, step: int, total: int, label: str) -> None:
        update_progress(progress_bar, step, total, label)

    def _resolve_target_label(
        self,
        track_id: str,
        gaze: GazePoint | None,
        frame_objects: list[dict],
        face_map: dict[int, FaceDetection],
        det_thresh: float,
        gaze_target_radius: int,
        person_part_distance_scale: float,
        person_part_min_conf: float,
        selected_object_classes: list[str],
        offscreen_direction: str | None = None,
    ) -> str:
        if gaze is None:
            return "out of frame"
        if gaze.inout <= det_thresh:
            return offscreen_direction or "out of frame"

        x_gaze = gaze.x_gaze
        y_gaze = gaze.y_gaze

        candidates: list[tuple[int, str]] = []

        for other_track_id, face in face_map.items():
            if other_track_id == track_id:
                continue
            if self._point_hits_box(x_gaze, y_gaze, face.x1, face.y1, face.x2, face.y2, gaze_target_radius):
                face_area = (face.x2 - face.x1) * (face.y2 - face.y1)
                candidates.append((face_area, f"person {other_track_id}'s face"))

        for detection in frame_objects:
            x1 = int(detection["x1"])
            y1 = int(detection["y1"])
            x2 = int(detection["x2"])
            y2 = int(detection["y2"])
            if self._point_hits_box(x_gaze, y_gaze, x1, y1, x2, y2, gaze_target_radius):
                area = (x2 - x1) * (y2 - y1)
                if str(detection["cls"]) == "person":
                    part_label = resolve_person_part_label(
                        detection,
                        x_gaze,
                        y_gaze,
                        gaze_target_radius,
                        person_part_distance_scale,
                        person_part_min_conf,
                    )
                    if part_label is not None and part_label != "other":
                        candidates.append((area, f"person {str(detection['track_id'])}'s {part_label}"))
                elif str(detection["cls"]) in set(selected_object_classes):
                    candidates.append((area, str(detection.get("label", detection["cls"]))))

        if not candidates:
            return "other"
        _, target = min(candidates, key=lambda item: item[0])
        return target

    def _point_enabled(self, visualization_mode: str) -> bool:
        return visualization_mode in {"point", "both"}

    def _heatmap_enabled(self, visualization_mode: str) -> bool:
        return visualization_mode in {"heatmap", "both"}

    def _point_hits_box(
        self,
        x: int,
        y: int,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        radius: int,
    ) -> bool:
        if radius <= 0:
            return x1 <= x <= x2 and y1 <= y <= y2

        closest_x = min(max(x, x1), x2)
        closest_y = min(max(y, y1), y2)
        dx = x - closest_x
        dy = y - closest_y
        return (dx * dx) + (dy * dy) <= (radius * radius)

    def _should_draw_object(self, detection: dict, selected_object_classes: list[str]) -> bool:
        cls_name = str(detection.get("cls", ""))
        return cls_name == "person" or cls_name in set(selected_object_classes)

    def _filter_visible_objects(
        self,
        frame_objects: list[dict],
        selected_object_classes: list[str],
    ) -> list[dict]:
        selected_set = set(selected_object_classes)
        return [
            detection
            for detection in frame_objects
            if str(detection.get("cls", "")) == "person" or str(detection.get("cls", "")) in selected_set
        ]

    def _notify_skip(self, progress_bar, message: str) -> None:
        if progress_bar is not None:
            progress_bar.progress(0.0, text=message)
        print(message, flush=True)
