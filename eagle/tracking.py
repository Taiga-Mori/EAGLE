from typing import Any

import pandas as pd

from .constants import OBJECT_COLUMNS
from .models import ModelManager
from .temporal import ObjectTrackSmoother
from .types import AppPaths, MediaContext


class ObjectTracker:
    """Run YOLO/BoT-SORT and return a normalized object table."""

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
        person_only_mode: bool,
        progress_bar=None,
    ) -> pd.DataFrame:
        raw_rows = self._run_detection(context, device, det_thresh, person_only_mode, progress_bar)
        self._report_detection_coverage(context, raw_rows)
        detections = self.smoother.smooth(raw_rows, context.total_frames, smoothing_window, context.media_type)
        detections.to_csv(context.objects_path, index=False)
        return detections

    def _run_detection(
        self,
        context: MediaContext,
        device: str,
        det_thresh: float,
        person_only_mode: bool,
        progress_bar=None,
    ) -> list[dict[str, Any]]:
        if context.media_type == "image":
            return self._detect_image(context, device, det_thresh, person_only_mode, progress_bar)
        return self._detect_video(context, device, det_thresh, person_only_mode, progress_bar)

    def _detect_image(
        self,
        context: MediaContext,
        device: str,
        det_thresh: float,
        person_only_mode: bool,
        progress_bar=None,
    ) -> list[dict[str, Any]]:
        assert self.models.yolo is not None
        results = self.models.yolo.predict(source=str(context.media_path), verbose=True, device=device)
        raw_rows: list[dict[str, Any]] = []
        if results:
            result = results[0]
            boxes = result.boxes
            if boxes is not None:
                for index, (cls_id, conf, xyxy) in enumerate(
                    zip(boxes.cls.tolist(), boxes.conf.tolist(), boxes.xyxy.tolist()),
                    start=1,
                ):
                    if conf < det_thresh:
                        continue
                    cls_name = result.names[int(cls_id)]
                    if person_only_mode and cls_name != "person":
                        continue
                    raw_rows.append(
                        {
                            "frame_idx": 0,
                            "cls": cls_name,
                            "track_id": index,
                            "conf": float(conf),
                            "x1": int(round(xyxy[0])),
                            "y1": int(round(xyxy[1])),
                            "x2": int(round(xyxy[2])),
                            "y2": int(round(xyxy[3])),
                            "label": f"{cls_name} {index}",
                        }
                    )
        self._update_progress(progress_bar, 1, 1, "Detecting objects...")
        return raw_rows

    def _detect_video(
        self,
        context: MediaContext,
        device: str,
        det_thresh: float,
        person_only_mode: bool,
        progress_bar=None,
    ) -> list[dict[str, Any]]:
        assert self.models.yolo is not None
        results = self.models.yolo.track(
            source=str(context.media_path),
            stream=True,
            verbose=True,
            tracker=str(self.paths.botsort_runtime_path),
            vid_stride=context.object_stride,
            device=device,
        )
        raw_rows: list[dict[str, Any]] = []
        expected_steps = max(1, len(context.object_frame_idx))

        for result_index, result in enumerate(results):
            boxes = result.boxes
            if boxes is not None and boxes.id is not None:
                for cls_id, track_id, conf, xyxy in zip(
                    boxes.cls.tolist(),
                    boxes.id.tolist(),
                    boxes.conf.tolist(),
                    boxes.xyxy.tolist(),
                ):
                    if track_id is None or conf < det_thresh:
                        continue
                    cls_name = result.names[int(cls_id)]
                    if person_only_mode and cls_name != "person":
                        continue
                    raw_rows.append(
                        {
                            "yolo_idx": result_index,
                            "frame_idx": min(result_index * context.object_stride, context.total_frames - 1),
                            "cls": cls_name,
                            "track_id": int(track_id),
                            "conf": float(conf),
                            "x1": float(xyxy[0]),
                            "y1": float(xyxy[1]),
                            "x2": float(xyxy[2]),
                            "y2": float(xyxy[3]),
                        }
                    )
            self._update_progress(progress_bar, result_index + 1, expected_steps, "Detecting objects...")
        return raw_rows

    def _update_progress(self, progress_bar, step: int, total: int, label: str) -> None:
        if progress_bar is None:
            return
        ratio = min(step / max(total, 1), 1.0)
        progress_bar.progress(ratio, text=f"{label} {round(ratio * 100)} %")

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
                "YOLO coverage: "
                f"readable_total_frames={context.total_frames}, "
                f"expected_object_steps={expected_steps}, "
                f"ultralytics_yielded_steps={yielded_steps}, "
                f"missing_steps={missing_steps}, "
                f"missing_ratio={missing_ratio:.2%}"
            ),
            flush=True,
        )
