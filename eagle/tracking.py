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

    def detect_persons(
        self,
        context: MediaContext,
        device: str,
        det_thresh: float,
        person_detection_backend: str,
        smoothing_window: int,
        max_switch_gap: int,
        progress_bar=None,
    ) -> pd.DataFrame:
        raw_rows = self._run_person_detection(context, device, det_thresh, progress_bar)
        self._report_detection_coverage(context, raw_rows, "person")
        self._update_progress(progress_bar, 0, 1, "Interpolating and smoothing persons...")
        detections = self.smoother.smooth(
            raw_rows,
            context.total_frames,
            smoothing_window,
            context.media_type,
            max_switch_gap=max_switch_gap,
        )
        self._update_progress(progress_bar, 1, 1, "Interpolating and smoothing persons...")
        self._update_progress(progress_bar, 0, 1, "Saving persons.csv...")
        detections.to_csv(context.persons_path, index=False)
        self._update_progress(progress_bar, 1, 1, "Saving persons.csv...")
        self._write_track_meta(
            context.persons_meta_path,
            context,
            det_thresh,
            context.person_stride,
            smoothing_window,
            {
                "detection_stage": "persons",
                "person_detection_source": "pose",
                "pose_keypoint_assignment": "bbox_geometry_v1",
                "backend": person_detection_backend,
                "max_switch_gap": int(max_switch_gap),
            },
        )
        return detections

    def detect_objects(
        self,
        context: MediaContext,
        device: str,
        object_detection_backend: str,
        det_thresh: float,
        smoothing_window: int,
        selected_object_classes: list[str],
        progress_bar=None,
    ) -> pd.DataFrame:
        needs_non_person_detections = any(cls_name != "person" for cls_name in selected_object_classes)
        if needs_non_person_detections:
            raw_rows = self._run_non_person_detection(context, device, det_thresh, progress_bar)
            self._report_detection_coverage(context, raw_rows, "non-person object")
        else:
            raw_rows = []
            self._notify_skip(
                progress_bar,
                "Skipping non-person object detection: only person annotations were requested.",
            )
        self._update_progress(progress_bar, 0, 1, "Interpolating and smoothing objects...")
        detections = self.smoother.smooth(raw_rows, context.total_frames, smoothing_window, context.media_type)
        self._update_progress(progress_bar, 1, 1, "Interpolating and smoothing objects...")
        self._update_progress(progress_bar, 0, 1, "Saving objects.csv...")
        detections.to_csv(context.objects_path, index=False)
        self._update_progress(progress_bar, 1, 1, "Saving objects.csv...")
        self._write_track_meta(
            context.objects_meta_path,
            context,
            det_thresh,
            context.object_stride,
            smoothing_window,
            {
                "detection_stage": "objects",
                "backend": object_detection_backend,
                "includes_non_person_detections": bool(needs_non_person_detections),
            },
        )
        return detections

    def _run_person_detection(
        self,
        context: MediaContext,
        device: str,
        det_thresh: float,
        progress_bar=None,
    ) -> list[dict[str, Any]]:
        if context.media_type == "image":
            raw_rows = self._detect_image_persons(context, device, det_thresh)
            self._update_progress(progress_bar, 1, 1, "Detecting persons (pose)...")
            return raw_rows

        raw_rows: list[dict[str, Any]] = []
        expected_steps = max(1, len(context.person_frame_idx))
        update_interval = max(1, expected_steps // 200)
        for result_index, result in enumerate(self._run_pose_track(context, device)):
            raw_rows.extend(self._pose_rows_from_result(result_index, result, context, det_thresh))
            step = result_index + 1
            if step == expected_steps or step % update_interval == 0:
                self._update_progress(progress_bar, step, expected_steps, "Detecting persons (pose)...")
        return raw_rows

    def _run_non_person_detection(
        self,
        context: MediaContext,
        device: str,
        det_thresh: float,
        progress_bar=None,
    ) -> list[dict[str, Any]]:
        if context.media_type == "image":
            raw_rows = self._detect_image_non_persons(context, device, det_thresh)
            self._update_progress(progress_bar, 1, 1, "Detecting non-person objects...")
            return raw_rows

        raw_rows: list[dict[str, Any]] = []
        expected_steps = max(1, len(context.object_frame_idx))
        update_interval = max(1, expected_steps // 200)
        for result_index, result in enumerate(self._run_object_track(context, device)):
            raw_rows.extend(self._non_person_rows_from_result(result_index, result, context, det_thresh))
            step = result_index + 1
            if step == expected_steps or step % update_interval == 0:
                self._update_progress(progress_bar, step, expected_steps, "Detecting non-person objects...")
        return raw_rows

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
            vid_stride=context.person_stride,
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
        assigned_keypoints = self._assign_keypoints_to_boxes(boxes.xyxy.tolist(), keypoint_triplets)
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
            pose_keypoints = assigned_keypoints.get(box_index, [])
            raw_rows.append(
                {
                    "yolo_idx": result_index,
                    "frame_idx": min(result_index * context.person_stride, context.total_frames - 1),
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
        assigned_keypoints = self._assign_keypoints_to_boxes(boxes.xyxy.tolist(), keypoint_triplets)
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
            pose_keypoints = assigned_keypoints.get(box_index - 1, [])
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

    def _assign_keypoints_to_boxes(
        self,
        boxes_xyxy: list[list[float]],
        keypoint_triplets: list[list[list[float | None]]],
    ) -> dict[int, list[list[float | None]]]:
        if not boxes_xyxy or not keypoint_triplets:
            return {}

        candidates: list[tuple[float, int, int]] = []
        for box_index, xyxy in enumerate(boxes_xyxy):
            for keypoint_index, keypoints in enumerate(keypoint_triplets):
                score = self._score_keypoints_for_box(keypoints, xyxy)
                if score is None:
                    continue
                candidates.append((score, box_index, keypoint_index))

        assigned_boxes: set[int] = set()
        assigned_keypoints: set[int] = set()
        output: dict[int, list[list[float | None]]] = {}
        for _, box_index, keypoint_index in sorted(candidates, key=lambda item: item[0], reverse=True):
            if box_index in assigned_boxes or keypoint_index in assigned_keypoints:
                continue
            assigned_boxes.add(box_index)
            assigned_keypoints.add(keypoint_index)
            output[box_index] = keypoint_triplets[keypoint_index]
        return output

    def _score_keypoints_for_box(self, keypoints: list[list[float | None]], xyxy: list[float]) -> float | None:
        x1, y1, x2, y2 = map(float, xyxy)
        width = max(x2 - x1, 1.0)
        height = max(y2 - y1, 1.0)
        diagonal = max(math.hypot(width, height), 1.0)

        valid_points: list[tuple[float, float]] = []
        for point in keypoints:
            if not isinstance(point, (list, tuple)) or len(point) < 2:
                continue
            px = self._finite_float(point[0])
            py = self._finite_float(point[1])
            if px is None or py is None:
                continue
            valid_points.append((px, py))
        if not valid_points:
            return None

        inside_count = sum(1 for px, py in valid_points if x1 <= px <= x2 and y1 <= py <= y2)
        inside_ratio = inside_count / len(valid_points)
        keypoint_center_x = sum(px for px, _ in valid_points) / len(valid_points)
        keypoint_center_y = sum(py for _, py in valid_points) / len(valid_points)
        box_center_x = (x1 + x2) / 2.0
        box_center_y = (y1 + y2) / 2.0
        center_distance = math.hypot(keypoint_center_x - box_center_x, keypoint_center_y - box_center_y)
        if inside_ratio < 0.50 or center_distance > diagonal * 0.75:
            return None
        return inside_ratio - center_distance / diagonal

    def _finite_float(self, value: Any) -> float | None:
        try:
            numeric = float(value)
        except Exception:
            return None
        return numeric if math.isfinite(numeric) else None

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

    def _write_track_meta(
        self,
        meta_path,
        context: MediaContext,
        det_thresh: float,
        stride: int,
        smoothing_window: int,
        extra: dict[str, Any],
    ) -> None:
        try:
            with self.paths.botsort_runtime_path.open("r", encoding="utf-8") as file:
                tracker_config = yaml.safe_load(file) or {}
        except Exception:
            tracker_config = None
        with meta_path.open("w", encoding="utf-8") as file:
            meta = {
                "raw_detection_cache": True,
                "media_path": str(context.media_path.resolve()),
                "media_mtime_ns": context.media_path.stat().st_mtime_ns,
                "det_thresh": float(det_thresh),
                "stride": int(stride),
                "smoothing_window": int(smoothing_window),
                "tracker_config": tracker_config,
            }
            meta.update(extra)
            json.dump(meta, file, ensure_ascii=False, indent=2)

    def _report_detection_coverage(self, context: MediaContext, raw_rows: list[dict[str, Any]], label: str = "tracking") -> None:
        """Print how many frame results Ultralytics actually yielded."""

        if context.media_type != "video":
            return

        yielded_steps = 0
        yolo_max = None
        if raw_rows:
            yolo_max = max(row.get("yolo_idx", 0) for row in raw_rows)
            yielded_steps = yolo_max + 1

        expected_steps = len(context.person_frame_idx) if label == "person" else len(context.object_frame_idx)
        missing_steps = max(expected_steps - yielded_steps, 0)
        missing_ratio = 0.0 if expected_steps == 0 else missing_steps / expected_steps

        print(
            (
                "Tracking coverage: "
                f"stage={label}, "
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
