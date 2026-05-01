import json
import math
from typing import Any

import pandas as pd
import torch
import yaml

from .constants import OBJECT_COLUMNS
from .models import ModelManager
from .progress import update_progress
from .temporal import ObjectTrackSmoother
from .types import AppPaths, MediaContext


class ObjectTracker:
    """Run pose tracking for persons and object tracking for non-person classes."""

    def __init__(self, models: ModelManager, paths: AppPaths, smoother: ObjectTrackSmoother) -> None:
        self.models = models
        self.paths = paths
        self.smoother = smoother

    def detect(
        self,
        context: MediaContext,
        device: str,
        det_thresh: float,
        smoothing_window: int,
        selected_object_classes: list[str],
        progress_bar=None,
    ) -> pd.DataFrame:
        needs_non_person_detections = any(cls_name != "person" for cls_name in selected_object_classes)
        raw_rows = self._run_detection(
            context,
            device,
            det_thresh,
            needs_non_person_detections,
            progress_bar,
        )
        self._reassign_non_person_track_ids(raw_rows)
        self._report_detection_coverage(context, raw_rows)
        self._update_progress(progress_bar, 0, 1, "Smoothing object tracks...")
        detections = self.smoother.smooth(raw_rows, context.total_frames, smoothing_window, context.media_type)
        self._update_progress(progress_bar, 1, 1, "Smoothing object tracks...")
        self._update_progress(progress_bar, 0, 1, "Saving objects.csv...")
        detections.to_csv(context.objects_path, index=False)
        self._update_progress(progress_bar, 1, 1, "Saving objects.csv...")
        try:
            with self.paths.botsort_runtime_path.open("r", encoding="utf-8") as file:
                tracker_config = yaml.safe_load(file) or {}
        except Exception:
            tracker_config = None
        with context.objects_meta_path.open("w", encoding="utf-8") as file:
            json.dump(
                {
                    "raw_detection_cache": True,
                    "media_path": str(context.media_path.resolve()),
                    "media_mtime_ns": context.media_path.stat().st_mtime_ns,
                    "det_thresh": float(det_thresh),
                    "object_stride": int(context.object_stride),
                    "object_smoothing_window": int(smoothing_window),
                    "person_detection_source": "pose",
                    "includes_non_person_detections": bool(needs_non_person_detections),
                    "tracker_config": tracker_config,
                },
                file,
                ensure_ascii=False,
                indent=2,
            )
        return detections

    def _run_detection(
        self,
        context: MediaContext,
        device: str,
        det_thresh: float,
        needs_non_person_detections: bool,
        progress_bar=None,
    ) -> list[dict[str, Any]]:
        if context.media_type == "image":
            return self._detect_image(context, device, det_thresh, needs_non_person_detections, progress_bar)
        return self._detect_video(context, device, det_thresh, needs_non_person_detections, progress_bar)

    def _detect_image(
        self,
        context: MediaContext,
        device: str,
        det_thresh: float,
        needs_non_person_detections: bool,
        progress_bar=None,
    ) -> list[dict[str, Any]]:
        raw_rows: list[dict[str, Any]] = []
        raw_rows.extend(self._detect_image_persons(context, device, det_thresh))
        if needs_non_person_detections:
            raw_rows.extend(self._detect_image_non_persons(context, device, det_thresh))
        else:
            self._notify_skip(
                progress_bar,
                "Skipping non-person object detection: only person annotations were requested.",
            )
        self._update_progress(progress_bar, 1, 1, "Detecting objects...")
        return raw_rows

    def _detect_video(
        self,
        context: MediaContext,
        device: str,
        det_thresh: float,
        needs_non_person_detections: bool,
        progress_bar=None,
    ) -> list[dict[str, Any]]:
        raw_rows: list[dict[str, Any]] = []
        expected_steps = max(1, len(context.object_frame_idx))
        stages = 1 + int(needs_non_person_detections)
        global_step = 0
        update_interval = max(1, expected_steps // 200)

        for result_index, result in enumerate(self._run_pose_track(context, device)):
            raw_rows.extend(self._pose_rows_from_result(result_index, result, context, det_thresh))
            global_step += 1
            if global_step == expected_steps or global_step % update_interval == 0:
                self._update_progress(progress_bar, global_step, expected_steps * stages, "Detecting persons (pose)...")

        if needs_non_person_detections:
            for result_index, result in enumerate(self._run_object_track(context, device)):
                raw_rows.extend(self._non_person_rows_from_result(result_index, result, context, det_thresh))
                global_step += 1
                if result_index + 1 == expected_steps or (result_index + 1) % update_interval == 0:
                    self._update_progress(
                        progress_bar,
                        global_step,
                        expected_steps * stages,
                        "Detecting non-person objects...",
                    )
        else:
            self._notify_skip(
                progress_bar,
                "Skipping non-person object detection: only person annotations were requested.",
            )
        return raw_rows

    def _reassign_non_person_track_ids(self, raw_rows: list[dict[str, Any]]) -> None:
        person_ids = {str(row["track_id"]) for row in raw_rows if str(row.get("cls", "")) == "person"}
        assigned_ids = set(person_ids)

        numeric_person_ids = [int(track_id) for track_id in person_ids if self._is_int_like(track_id)]
        next_track_id = max(numeric_person_ids, default=0) + 1
        remap: dict[tuple[str, str], str] = {}

        for row in raw_rows:
            cls_name = str(row.get("cls", ""))
            if cls_name == "person":
                continue

            original_track_id = str(row["track_id"])
            key = (cls_name, original_track_id)
            if key not in remap:
                candidate = original_track_id
                if candidate in assigned_ids:
                    while str(next_track_id) in assigned_ids:
                        next_track_id += 1
                    candidate = str(next_track_id)
                    next_track_id += 1
                remap[key] = candidate
                assigned_ids.add(candidate)

            row["track_id"] = remap[key]
            row["label"] = f"{cls_name} {row['track_id']}"

    def _detect_image_persons(self, context: MediaContext, device: str, det_thresh: float) -> list[dict[str, Any]]:
        assert self.models.yolo_pose is not None
        results = self.models.yolo_pose.predict(
            source=str(context.media_path),
            verbose=False,
            device=self._yolo_device(device),
        )
        if not results:
            return []
        return self._pose_rows_from_image_result(results[0], det_thresh)

    def _detect_image_non_persons(self, context: MediaContext, device: str, det_thresh: float) -> list[dict[str, Any]]:
        assert self.models.yolo is not None
        results = self.models.yolo.predict(
            source=str(context.media_path),
            verbose=False,
            device=self._yolo_device(device),
        )
        if not results:
            return []
        return self._non_person_rows_from_result(0, results[0], context, det_thresh)

    def _run_pose_track(self, context: MediaContext, device: str):
        assert self.models.yolo_pose is not None
        return self.models.yolo_pose.track(
            source=str(context.media_path),
            stream=True,
            verbose=False,
            tracker=str(self.paths.botsort_runtime_path),
            vid_stride=context.object_stride,
            device=self._yolo_device(device),
        )

    def _run_object_track(self, context: MediaContext, device: str):
        assert self.models.yolo is not None
        return self.models.yolo.track(
            source=str(context.media_path),
            stream=True,
            verbose=False,
            tracker=str(self.paths.botsort_runtime_path),
            vid_stride=context.object_stride,
            device=self._yolo_device(device),
        )

    def _yolo_device(self, device: str) -> str | torch.device:
        if device.startswith("cuda:"):
            return torch.device(device)
        return device

    def _pose_rows_from_result(
        self,
        result_index: int,
        result,
        context: MediaContext,
        det_thresh: float,
    ) -> list[dict[str, Any]]:
        raw_rows: list[dict[str, Any]] = []
        boxes = result.boxes
        if boxes is None or boxes.id is None:
            return raw_rows

        keypoint_triplets = self._keypoint_triplets(result)
        for box_index, (cls_id, track_id, conf, xyxy) in enumerate(
            zip(
                boxes.cls.tolist(),
                boxes.id.tolist(),
                boxes.conf.tolist(),
                boxes.xyxy.tolist(),
            )
        ):
            if track_id is None or conf < det_thresh:
                continue
            cls_name = result.names[int(cls_id)]
            if cls_name != "person":
                continue
            pose_keypoints = keypoint_triplets[box_index] if box_index < len(keypoint_triplets) else []
            raw_rows.append(
                {
                    "yolo_idx": result_index,
                    "frame_idx": min(result_index * context.object_stride, context.total_frames - 1),
                    "cls": "person",
                    "track_id": str(int(track_id)) if self._is_int_like(track_id) else str(track_id),
                    "source": "pose",
                    "conf": float(conf),
                    "x1": float(xyxy[0]),
                    "y1": float(xyxy[1]),
                    "x2": float(xyxy[2]),
                    "y2": float(xyxy[3]),
                    "pose_keypoints": json.dumps(pose_keypoints, ensure_ascii=False),
                    "label": f"person {track_id}",
                }
            )
        return raw_rows

    def _pose_rows_from_image_result(self, result, det_thresh: float) -> list[dict[str, Any]]:
        raw_rows: list[dict[str, Any]] = []
        boxes = result.boxes
        if boxes is None:
            return raw_rows

        keypoint_triplets = self._keypoint_triplets(result)
        for box_index, (cls_id, conf, xyxy) in enumerate(
            zip(
                boxes.cls.tolist(),
                boxes.conf.tolist(),
                boxes.xyxy.tolist(),
            ),
            start=1,
        ):
            if conf < det_thresh:
                continue
            cls_name = result.names[int(cls_id)]
            if cls_name != "person":
                continue
            pose_keypoints = keypoint_triplets[box_index - 1] if box_index - 1 < len(keypoint_triplets) else []
            raw_rows.append(
                {
                    "frame_idx": 0,
                    "cls": "person",
                    "track_id": str(box_index),
                    "source": "pose",
                    "conf": float(conf),
                    "x1": float(xyxy[0]),
                    "y1": float(xyxy[1]),
                    "x2": float(xyxy[2]),
                    "y2": float(xyxy[3]),
                    "pose_keypoints": json.dumps(pose_keypoints, ensure_ascii=False),
                    "label": f"person {box_index}",
                }
            )
        return raw_rows

    def _non_person_rows_from_result(
        self,
        result_index: int,
        result,
        context: MediaContext,
        det_thresh: float,
    ) -> list[dict[str, Any]]:
        raw_rows: list[dict[str, Any]] = []
        boxes = result.boxes
        if boxes is None:
            return raw_rows

        ids = boxes.id.tolist() if boxes.id is not None else [index + 1 for index in range(len(boxes.cls.tolist()))]
        for cls_id, track_id, conf, xyxy in zip(
            boxes.cls.tolist(),
            ids,
            boxes.conf.tolist(),
            boxes.xyxy.tolist(),
        ):
            if track_id is None or conf < det_thresh:
                continue
            cls_name = result.names[int(cls_id)]
            if cls_name == "person":
                continue
            track_label = str(int(track_id)) if self._is_int_like(track_id) else str(track_id)
            raw_rows.append(
                {
                    "yolo_idx": result_index,
                    "frame_idx": min(result_index * context.object_stride, context.total_frames - 1),
                    "cls": cls_name,
                    "track_id": track_label,
                    "source": "detect",
                    "conf": float(conf),
                    "x1": float(xyxy[0]),
                    "y1": float(xyxy[1]),
                    "x2": float(xyxy[2]),
                    "y2": float(xyxy[3]),
                    "pose_keypoints": None,
                    "label": f"{cls_name} {track_label}",
                }
            )
        return raw_rows

    def _keypoint_triplets(self, result) -> list[list[list[float | None]]]:
        if getattr(result, "keypoints", None) is None:
            return []
        xy_data = getattr(result.keypoints, "xy", None)
        conf_data = getattr(result.keypoints, "conf", None)
        if xy_data is None:
            return []

        xy = xy_data.cpu().numpy() if hasattr(xy_data, "cpu") else xy_data
        conf = conf_data.cpu().numpy() if hasattr(conf_data, "cpu") else conf_data
        triplets: list[list[list[float | None]]] = []
        for index, person_points in enumerate(xy):
            point_list: list[list[float | None]] = []
            for point_index, point in enumerate(person_points):
                point_conf = None
                if conf is not None and index < len(conf) and point_index < len(conf[index]):
                    point_conf = float(conf[index][point_index])
                point_list.append([float(point[0]), float(point[1]), point_conf])
            triplets.append(point_list)
        return triplets

    def _is_int_like(self, value: Any) -> bool:
        try:
            if value is None:
                return False
            numeric = float(value)
            return math.isfinite(numeric)
        except Exception:
            return False

    def _update_progress(self, progress_bar, step: int, total: int, label: str) -> None:
        update_progress(progress_bar, step, total, label)

    def _report_detection_coverage(self, context: MediaContext, raw_rows: list[dict[str, Any]]) -> None:
        """Print how many frame results Ultralytics actually yielded."""

        if context.media_type != "video":
            return

        yielded_steps = 0
        yolo_max = None
        if raw_rows:
            yolo_max = max(row.get("yolo_idx", 0) for row in raw_rows)
            yielded_steps = yolo_max + 1

        expected_steps = len(context.object_frame_idx)
        missing_steps = max(expected_steps - yielded_steps, 0)
        missing_ratio = 0.0 if expected_steps == 0 else missing_steps / expected_steps

        print(
            (
                "Tracking coverage: "
                f"readable_total_frames={context.total_frames}, "
                f"expected_object_steps={expected_steps}, "
                f"yielded_steps={yielded_steps}, "
                f"missing_steps={missing_steps}, "
                f"missing_ratio={missing_ratio:.2%}"
            ),
            flush=True,
        )

    def _notify_skip(self, progress_bar, message: str) -> None:
        if progress_bar is not None:
            progress_bar.progress(0.0, text=message)
        print(message, flush=True)
