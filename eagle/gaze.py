from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image

from .annotate import FrameAnnotator
from .constants import GAZE_COLUMNS
from .models import ModelManager
from .temporal import GazeTemporalProcessor
from .types import FaceDetection, GazePoint, GazeRecord, MediaContext


class FaceGazeEstimator:
    """Estimate face and gaze results for sampled media frames."""

    def __init__(
        self,
        models: ModelManager,
        annotator: FrameAnnotator,
        temporal_processor: GazeTemporalProcessor,
    ) -> None:
        self.models = models
        self.annotator = annotator
        self.temporal_processor = temporal_processor

    def estimate(
        self,
        context: MediaContext,
        device: str,
        det_thresh: float,
        visualization_mode: str,
        heatmap_alpha: float,
        gaze_smoothing_window: int,
        progress_bar=None,
    ) -> pd.DataFrame:
        object_df = pd.read_csv(context.objects_path)
        if object_df.empty:
            empty_df = pd.DataFrame(columns=GAZE_COLUMNS)
            empty_df.to_csv(context.gaze_path, index=False)
            return empty_df

        records: list[dict] = []
        if context.media_type == "image":
            self._estimate_image(
                context,
                object_df,
                device,
                det_thresh,
                visualization_mode,
                heatmap_alpha,
                records,
                progress_bar,
            )
        else:
            self._estimate_video(
                context,
                object_df,
                device,
                det_thresh,
                visualization_mode,
                heatmap_alpha,
                gaze_smoothing_window,
                records,
                progress_bar,
            )

        gaze_df = pd.DataFrame(records, columns=GAZE_COLUMNS)
        gaze_df.to_csv(context.gaze_path, index=False)
        return gaze_df

    def _estimate_image(
        self,
        context: MediaContext,
        object_df: pd.DataFrame,
        device: str,
        det_thresh: float,
        visualization_mode: str,
        heatmap_alpha: float,
        records: list[dict],
        progress_bar=None,
    ) -> None:
        frame = cv2.imread(str(context.media_path))
        if frame is None:
            raise FileNotFoundError(f"Could not open image: {context.media_path}")

        point_enabled = self._point_enabled(visualization_mode)
        heatmap_enabled = self._heatmap_enabled(visualization_mode)
        frame_objects = object_df.to_dict(orient="records")
        face_map = self.detect_faces_for_frame(frame, frame_objects, det_thresh)
        gaze_map = self.detect_gazes(frame, face_map, device)

        if point_enabled:
            annotated_frame = frame.copy()
            for detection in frame_objects:
                track_id = int(detection["track_id"])
                face = face_map.get(track_id)
                raw_gaze = gaze_map.get(track_id)
                gaze = gaze_map.get(track_id)
                records.append(asdict(self.to_record(0, track_id, face, raw_gaze, gaze)))
                annotated_frame = self.annotator.draw_object(annotated_frame, detection)
                annotated_frame = self.annotator.draw_face_and_gaze_point(annotated_frame, face, gaze, det_thresh)
            cv2.imwrite(str(context.annotated_image_path), annotated_frame)
            cv2.imwrite(str(context.temp_dir / "0.jpg"), annotated_frame)
        if heatmap_enabled:
            self._ensure_person_heatmap_dirs(context, frame_objects)
            for detection in frame_objects:
                track_id = int(detection["track_id"])
                face = face_map.get(track_id)
                raw_gaze = gaze_map.get(track_id)
                gaze = gaze_map.get(track_id)
                if not point_enabled:
                    records.append(asdict(self.to_record(0, track_id, face, raw_gaze, gaze)))
                if detection["cls"] != "person":
                    continue
                person_frame = frame.copy()
                person_frame = self.annotator.draw_object(person_frame, detection)
                person_frame = self.annotator.draw_face_and_heatmap(
                    person_frame, face, gaze, det_thresh, heatmap_alpha
                )
                cv2.imwrite(str(self._person_heatmap_frame_path(context, track_id, 0)), person_frame)
        self._update_progress(progress_bar, 1, 1, "Detecting face & gaze...")

    def _estimate_video(
        self,
        context: MediaContext,
        object_df: pd.DataFrame,
        device: str,
        det_thresh: float,
        visualization_mode: str,
        heatmap_alpha: float,
        gaze_smoothing_window: int,
        records: list[dict],
        progress_bar=None,
    ) -> None:
        point_enabled = self._point_enabled(visualization_mode)
        heatmap_enabled = self._heatmap_enabled(visualization_mode)
        face_maps_by_frame: dict[int, dict[int, FaceDetection]] = {}
        sparse_gaze_by_frame: dict[int, dict[int, GazePoint]] = {}
        self._collect_sparse_gazes(
            context,
            object_df,
            device,
            det_thresh,
            face_maps_by_frame,
            sparse_gaze_by_frame,
            progress_bar,
        )
        dense_gaze_by_frame = self.temporal_processor.interpolate_and_smooth(
            frame_indices=context.object_frame_idx,
            face_maps_by_frame=face_maps_by_frame,
            sparse_gaze_by_frame=sparse_gaze_by_frame,
            smoothing_window=gaze_smoothing_window,
        )
        self._render_video_outputs(
            context,
            object_df,
            face_maps_by_frame,
            sparse_gaze_by_frame,
            dense_gaze_by_frame,
            det_thresh,
            visualization_mode,
            heatmap_alpha,
            records,
            progress_bar,
        )

    def _collect_sparse_gazes(
        self,
        context: MediaContext,
        object_df: pd.DataFrame,
        device: str,
        det_thresh: float,
        face_maps_by_frame: dict[int, dict[int, FaceDetection]],
        sparse_gaze_by_frame: dict[int, dict[int, GazePoint]],
        progress_bar=None,
    ) -> None:
        capture = cv2.VideoCapture(str(context.media_path))
        if not capture.isOpened():
            raise FileNotFoundError(f"Could not open video: {context.media_path}")

        object_frame_set = set(context.object_frame_idx)
        gaze_frame_set = set(context.gaze_frame_idx)
        expected_steps = max(1, len(context.gaze_frame_idx))
        gaze_step = 0

        try:
            for frame_idx in range(context.total_frames):
                ret, frame = capture.read()
                if not ret:
                    break
                if frame_idx not in object_frame_set:
                    continue

                frame_objects = object_df[object_df["frame_idx"] == frame_idx].to_dict(orient="records")
                face_map = self.detect_faces_for_frame(frame, frame_objects, det_thresh)
                face_maps_by_frame[frame_idx] = face_map
                if frame_idx in gaze_frame_set:
                    sparse_gaze_by_frame[frame_idx] = self.detect_gazes(frame, face_map, device)
                    gaze_step += 1
                    self._update_progress(progress_bar, gaze_step, expected_steps, "Detecting face & gaze...")
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
        records: list[dict],
        progress_bar=None,
    ) -> None:
        render_capture = cv2.VideoCapture(str(context.media_path))
        if not render_capture.isOpened():
            raise FileNotFoundError(f"Could not open video: {context.media_path}")

        point_enabled = self._point_enabled(visualization_mode)
        heatmap_enabled = self._heatmap_enabled(visualization_mode)
        object_frame_set = set(context.object_frame_idx)
        expected_steps = max(1, len(context.object_frame_idx))
        render_step = 0
        frame_name_width = len(str(context.total_frames - 1))
        if heatmap_enabled:
            self._ensure_person_heatmap_dirs(
                context,
                object_df.drop_duplicates("track_id").to_dict(orient="records"),
            )

        try:
            for frame_idx in range(context.total_frames):
                ret, frame = render_capture.read()
                if not ret:
                    break
                if frame_idx not in object_frame_set:
                    continue

                frame_objects = object_df[object_df["frame_idx"] == frame_idx].to_dict(orient="records")
                face_map = face_maps_by_frame.get(frame_idx, {})
                raw_gaze_map = sparse_gaze_by_frame.get(frame_idx, {})
                gaze_map = dense_gaze_by_frame.get(frame_idx, {})
                annotated_frame = frame.copy() if point_enabled else None

                for detection in frame_objects:
                    track_id = int(detection["track_id"])
                    face = face_map.get(track_id)
                    raw_gaze = raw_gaze_map.get(track_id)
                    gaze = gaze_map.get(track_id)
                    records.append(asdict(self.to_record(frame_idx, track_id, face, raw_gaze, gaze)))
                    if point_enabled:
                        assert annotated_frame is not None
                        annotated_frame = self.annotator.draw_object(annotated_frame, detection)
                        annotated_frame = self.annotator.draw_face_and_gaze_point(
                            annotated_frame, face, gaze, det_thresh
                        )
                    if heatmap_enabled:
                        if detection["cls"] != "person":
                            continue
                        person_frame = frame.copy()
                        person_frame = self.annotator.draw_object(person_frame, detection)
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

    def detect_faces_for_frame(
        self,
        frame: np.ndarray,
        detections: list[dict],
        det_thresh: float,
    ) -> dict[int, FaceDetection]:
        assert self.models.retinaface is not None
        person_detections = [d for d in detections if d["cls"] == "person"]
        if not person_detections:
            return {}

        raw_faces = self.models.retinaface.predict_jsons(frame)
        valid_faces = [face for face in raw_faces if float(face["score"]) >= det_thresh]
        face_map: dict[int, FaceDetection] = {}

        for detection in person_detections:
            x1, y1, x2, y2 = (int(detection[key]) for key in ["x1", "y1", "x2", "y2"])
            matching_faces = []
            for face in valid_faces:
                fx1, fy1, fx2, fy2 = map(int, face["bbox"])
                center_x = (fx1 + fx2) / 2
                center_y = (fy1 + fy2) / 2
                if x1 <= center_x <= x2 and y1 <= center_y <= y2:
                    matching_faces.append(face)
            if not matching_faces:
                continue
            top_face = min(matching_faces, key=lambda face: face["bbox"][1])
            fx1, fy1, fx2, fy2 = map(int, top_face["bbox"])
            track_id = int(detection["track_id"])
            face_map[track_id] = FaceDetection(
                track_id=track_id,
                conf=float(top_face["score"]),
                x1=fx1,
                y1=fy1,
                x2=fx2,
                y2=fy2,
            )
        return face_map

    def detect_gazes(self, frame: np.ndarray, face_map: dict[int, FaceDetection], device: str) -> dict[int, GazePoint]:
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
            y_hm, x_hm = np.unravel_index(np.argmax(heatmap), heatmap.shape)
            gaze_map[face.track_id] = GazePoint(
                track_id=face.track_id,
                inout=float(output["inout"][0][index]),
                x_gaze=int(x_hm / heatmap.shape[1] * width),
                y_gaze=int(y_hm / heatmap.shape[0] * height),
                heatmap=heatmap.astype(np.float32),
                frame_width=width,
                frame_height=height,
            )
        return gaze_map

    def to_record(
        self,
        frame_idx: int,
        track_id: int,
        face: FaceDetection | None,
        raw_gaze: GazePoint | None,
        gaze: GazePoint | None,
    ) -> GazeRecord:
        if face is None:
            return GazeRecord(
                frame_idx=frame_idx,
                track_id=track_id,
                face_detected=False,
                face_conf=None,
                face_x1=None,
                face_y1=None,
                face_x2=None,
                face_y2=None,
                raw_gaze_detected=False,
                raw_inout=None,
                raw_x_gaze=None,
                raw_y_gaze=None,
                gaze_detected=False,
                inout=None,
                x_gaze=None,
                y_gaze=None,
            )
        return GazeRecord(
            frame_idx=frame_idx,
            track_id=track_id,
            face_detected=True,
            face_conf=face.conf,
            face_x1=face.x1,
            face_y1=face.y1,
            face_x2=face.x2,
            face_y2=face.y2,
            raw_gaze_detected=raw_gaze is not None,
            raw_inout=None if raw_gaze is None else raw_gaze.inout,
            raw_x_gaze=None if raw_gaze is None else raw_gaze.x_gaze,
            raw_y_gaze=None if raw_gaze is None else raw_gaze.y_gaze,
            gaze_detected=gaze is not None,
            inout=None if gaze is None else gaze.inout,
            x_gaze=None if gaze is None else gaze.x_gaze,
            y_gaze=None if gaze is None else gaze.y_gaze,
        )

    def _ensure_person_heatmap_dirs(self, context: MediaContext, detections: list[dict]) -> None:
        for detection in detections:
            if detection.get("cls") != "person":
                continue
            (context.heatmap_dir / f"person_{int(detection['track_id'])}").mkdir(parents=True, exist_ok=True)

    def _person_heatmap_frame_path(self, context: MediaContext, track_id: int, frame_idx: int) -> Path:
        return context.heatmap_dir / f"person_{track_id}" / f"{frame_idx:06d}.jpg"

    def _update_progress(self, progress_bar, step: int, total: int, label: str) -> None:
        if progress_bar is None:
            return
        ratio = min(step / max(total, 1), 1.0)
        progress_bar.progress(ratio, text=f"{label} {round(ratio * 100)} %")

    def _point_enabled(self, visualization_mode: str) -> bool:
        return visualization_mode in {"point", "both"}

    def _heatmap_enabled(self, visualization_mode: str) -> bool:
        return visualization_mode in {"heatmap", "both"}
