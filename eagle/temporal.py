import cv2
import numpy as np
import pandas as pd

from .constants import OBJECT_COLUMNS
from .types import FaceDetection, GazePoint


class GazePointResolver:
    """Convert gaze heatmaps into image-space gaze points."""

    def resolve(self, heatmap: np.ndarray, frame_width: int, frame_height: int, method: str) -> tuple[int, int]:
        heatmap = np.maximum(heatmap.astype(np.float32), 0)
        if heatmap.size == 0:
            return 0, 0

        if method == "argmax":
            y_hm, x_hm = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        elif method == "center_of_mass":
            x_hm, y_hm = self._weighted_centroid(heatmap)
        elif method == "peak_centroid":
            x_hm, y_hm = self._peak_centroid(heatmap)
        elif method == "peak_region_centroid":
            x_hm, y_hm = self._peak_region_centroid(heatmap)
        elif method == "soft_argmax":
            x_hm, y_hm = self._soft_argmax(heatmap)
        else:
            raise ValueError(f"Unsupported gaze point method: {method}")

        x_gaze = int(np.clip(round(x_hm / heatmap.shape[1] * frame_width), 0, max(frame_width - 1, 0)))
        y_gaze = int(np.clip(round(y_hm / heatmap.shape[0] * frame_height), 0, max(frame_height - 1, 0)))
        return x_gaze, y_gaze

    def _weighted_centroid(self, heatmap: np.ndarray) -> tuple[float, float]:
        total = float(heatmap.sum())
        if total <= 0:
            y_hm, x_hm = np.unravel_index(np.argmax(heatmap), heatmap.shape)
            return float(x_hm), float(y_hm)

        y_idx, x_idx = np.indices(heatmap.shape, dtype=np.float32)
        x_hm = float((heatmap * x_idx).sum() / total)
        y_hm = float((heatmap * y_idx).sum() / total)
        return x_hm, y_hm

    def _peak_centroid(self, heatmap: np.ndarray, relative_threshold: float = 0.5) -> tuple[float, float]:
        peak = float(heatmap.max())
        if peak <= 0:
            y_hm, x_hm = np.unravel_index(np.argmax(heatmap), heatmap.shape)
            return float(x_hm), float(y_hm)

        masked = np.where(heatmap >= peak * relative_threshold, heatmap, 0.0).astype(np.float32)
        return self._weighted_centroid(masked)

    def _peak_region_centroid(self, heatmap: np.ndarray, relative_threshold: float = 0.5) -> tuple[float, float]:
        peak = float(heatmap.max())
        if peak <= 0:
            y_hm, x_hm = np.unravel_index(np.argmax(heatmap), heatmap.shape)
            return float(x_hm), float(y_hm)

        thresholded = (heatmap >= peak * relative_threshold).astype(np.uint8)
        peak_y, peak_x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        num_labels, labels = cv2.connectedComponents(thresholded)
        if num_labels <= 1:
            return self._weighted_centroid(heatmap)

        peak_label = int(labels[peak_y, peak_x])
        if peak_label == 0:
            return float(peak_x), float(peak_y)

        region_mask = (labels == peak_label).astype(np.float32)
        masked = heatmap * region_mask
        return self._weighted_centroid(masked)

    def _soft_argmax(self, heatmap: np.ndarray, temperature: float = 12.0) -> tuple[float, float]:
        peak = float(heatmap.max())
        if peak <= 0:
            y_hm, x_hm = np.unravel_index(np.argmax(heatmap), heatmap.shape)
            return float(x_hm), float(y_hm)

        normalized = heatmap / peak
        logits = temperature * normalized
        logits = logits - float(logits.max())
        weights = np.exp(logits).astype(np.float32)
        return self._weighted_centroid(weights)


class ObjectTrackSmoother:
    """Fill missing object frames and smooth track geometry."""

    def smooth(self, raw_rows: list[dict], total_frames: int, window: int, media_type: str) -> pd.DataFrame:
        if not raw_rows:
            return pd.DataFrame(columns=OBJECT_COLUMNS)
        if media_type == "image":
            output = pd.DataFrame(raw_rows)
            if "source" not in output.columns:
                output["source"] = "detect"
            if "pose_keypoints" not in output.columns:
                output["pose_keypoints"] = None
            return output[OBJECT_COLUMNS].sort_values(["frame_idx", "track_id"]).reset_index(drop=True)

        for row in raw_rows:
            row["frame_idx"] = int(row["frame_idx"])
            row.pop("yolo_idx", None)

        detections = pd.DataFrame(raw_rows).sort_values(["track_id", "frame_idx"])
        smoothed_groups = []
        bbox_cols = ["x1", "y1", "x2", "y2"]
        extra_cols = [column for column in ["source", "pose_keypoints"] if column in detections.columns]

        detections["track_id"] = detections["track_id"].astype(str)
        for track_id, group in detections.groupby("track_id", sort=False):
            group = group.sort_values(["frame_idx", "conf"], ascending=[True, False])
            group = group.drop_duplicates(subset="frame_idx", keep="first")
            frame_range = range(int(group["frame_idx"].min()), int(group["frame_idx"].max()) + 1)
            group = group.set_index("frame_idx").reindex(frame_range)
            group["track_id"] = str(track_id)
            classes = group["cls"].dropna()
            if classes.empty:
                continue
            group["cls"] = str(classes.mode().iloc[0])
            group[bbox_cols] = group[bbox_cols].interpolate(method="linear", limit_direction="both")
            group["conf"] = group["conf"].interpolate(method="linear", limit_direction="both")
            if "source" in extra_cols:
                non_null_sources = group["source"].dropna().astype(str)
                group["source"] = str(non_null_sources.mode().iloc[0]) if not non_null_sources.empty else "detect"
            if "pose_keypoints" in extra_cols:
                group["pose_keypoints"] = group["pose_keypoints"].ffill().bfill()
            if window > 1:
                group[bbox_cols] = group[bbox_cols].rolling(window=window, min_periods=1, center=True).mean()
                group["conf"] = group["conf"].rolling(window=window, min_periods=1, center=True).mean()
            group = group.dropna(subset=bbox_cols).reset_index().rename(columns={"index": "frame_idx"})
            group["label"] = group.apply(lambda row: f"{row['cls']} {row['track_id']}", axis=1)
            smoothed_groups.append(group)

        if not smoothed_groups:
            return pd.DataFrame(columns=OBJECT_COLUMNS)

        output = pd.concat(smoothed_groups, ignore_index=True).sort_values(["frame_idx", "track_id"])
        for column in ["frame_idx", "x1", "y1", "x2", "y2"]:
            output[column] = output[column].round().astype(int)
        output["track_id"] = output["track_id"].astype(str)
        output["conf"] = output["conf"].astype(float)
        if "source" not in output.columns:
            output["source"] = "detect"
        if "pose_keypoints" not in output.columns:
            output["pose_keypoints"] = None
        return output[OBJECT_COLUMNS]


class GazeTemporalProcessor:
    """Interpolate sparse gaze results and smooth them over time."""

    def __init__(self, point_resolver: GazePointResolver | None = None) -> None:
        self.point_resolver = point_resolver or GazePointResolver()

    def interpolate_faces(
        self,
        frame_indices: list[int],
        raw_face_maps_by_frame: dict[int, dict[int, FaceDetection]],
        object_df: pd.DataFrame,
        smoothing_window: int = 1,
    ) -> dict[int, dict[int, FaceDetection]]:
        dense_face_maps_by_frame: dict[int, dict[int, FaceDetection]] = {frame_idx: {} for frame_idx in frame_indices}
        person_df = object_df[object_df["cls"] == "person"].copy()
        if person_df.empty:
            return dense_face_maps_by_frame

        person_df["track_id"] = person_df["track_id"].astype(str)
        frame_index_set = set(frame_indices)

        for track_id, group in person_df.groupby("track_id", sort=False):
            track_frames = sorted(int(frame_idx) for frame_idx in group["frame_idx"].tolist() if int(frame_idx) in frame_index_set)
            if not track_frames:
                continue

            sparse_faces = [
                (frame_idx, raw_face_maps_by_frame[frame_idx][track_id])
                for frame_idx in track_frames
                if track_id in raw_face_maps_by_frame.get(frame_idx, {})
            ]
            if not sparse_faces:
                continue

            dense_faces = self._interpolate_face_track(track_id, track_frames, sparse_faces)
            if smoothing_window > 1:
                dense_faces = self._smooth_face_track(dense_faces, smoothing_window)
            for frame_idx, face in dense_faces.items():
                dense_face_maps_by_frame.setdefault(frame_idx, {})[track_id] = face

        return dense_face_maps_by_frame

    def interpolate_and_smooth(
        self,
        frame_indices: list[int],
        face_maps_by_frame: dict[int, dict[int, object]],
        sparse_gaze_by_frame: dict[int, dict[int, GazePoint]],
        smoothing_window: int,
        point_method: str,
    ) -> dict[int, dict[int, GazePoint]]:
        dense_gaze_by_frame: dict[int, dict[int, GazePoint]] = {frame_idx: {} for frame_idx in frame_indices}
        track_to_frames: dict[int, list[int]] = {}
        for frame_idx in frame_indices:
            for track_id in face_maps_by_frame.get(frame_idx, {}):
                track_to_frames.setdefault(track_id, []).append(frame_idx)

        for track_id, track_frames in track_to_frames.items():
            sparse_points = [
                (frame_idx, sparse_gaze_by_frame[frame_idx][track_id])
                for frame_idx in track_frames
                if track_id in sparse_gaze_by_frame.get(frame_idx, {})
            ]
            if not sparse_points:
                continue
            dense_series = self._interpolate_track(track_id, track_frames, sparse_points, point_method)
            if smoothing_window > 1:
                dense_series = self._smooth_track(dense_series, smoothing_window, point_method)
            for frame_idx, gaze in dense_series.items():
                dense_gaze_by_frame[frame_idx][track_id] = gaze

        return dense_gaze_by_frame

    def _interpolate_track(
        self,
        track_id: str,
        track_frames: list[int],
        sparse_points: list[tuple[int, GazePoint]],
        point_method: str,
    ) -> dict[int, GazePoint]:
        dense_series: dict[int, GazePoint] = {}
        sparse_frames = [frame_idx for frame_idx, _ in sparse_points]

        for frame_idx in track_frames:
            prev_idx = max((i for i, sparse_frame in enumerate(sparse_frames) if sparse_frame <= frame_idx), default=None)
            next_idx = min((i for i, sparse_frame in enumerate(sparse_frames) if sparse_frame >= frame_idx), default=None)

            if prev_idx is None and next_idx is None:
                continue
            if prev_idx is None:
                dense_series[frame_idx] = sparse_points[next_idx][1]
                continue
            if next_idx is None:
                dense_series[frame_idx] = sparse_points[prev_idx][1]
                continue

            prev_frame, prev_gaze = sparse_points[prev_idx]
            next_frame, next_gaze = sparse_points[next_idx]
            if prev_frame == next_frame:
                dense_series[frame_idx] = prev_gaze
                continue

            ratio = (frame_idx - prev_frame) / (next_frame - prev_frame)
            heatmap = ((1.0 - ratio) * prev_gaze.heatmap) + (ratio * next_gaze.heatmap)
            x_gaze, y_gaze = self.point_resolver.resolve(
                heatmap,
                prev_gaze.frame_width,
                prev_gaze.frame_height,
                point_method,
            )
            dense_series[frame_idx] = GazePoint(
                track_id=track_id,
                inout=float((1.0 - ratio) * prev_gaze.inout + ratio * next_gaze.inout),
                x_gaze=x_gaze,
                y_gaze=y_gaze,
                heatmap=heatmap.astype(np.float32),
                frame_width=prev_gaze.frame_width,
                frame_height=prev_gaze.frame_height,
            )
        return dense_series

    def _interpolate_face_track(
        self,
        track_id: str,
        track_frames: list[int],
        sparse_faces: list[tuple[int, FaceDetection]],
    ) -> dict[int, FaceDetection]:
        dense_series: dict[int, FaceDetection] = {}
        sparse_frames = [frame_idx for frame_idx, _ in sparse_faces]

        for frame_idx in track_frames:
            prev_idx = max((i for i, sparse_frame in enumerate(sparse_frames) if sparse_frame <= frame_idx), default=None)
            next_idx = min((i for i, sparse_frame in enumerate(sparse_frames) if sparse_frame >= frame_idx), default=None)

            if prev_idx is None and next_idx is None:
                continue
            if prev_idx is None:
                dense_series[frame_idx] = sparse_faces[next_idx][1]
                continue
            if next_idx is None:
                dense_series[frame_idx] = sparse_faces[prev_idx][1]
                continue

            prev_frame, prev_face = sparse_faces[prev_idx]
            next_frame, next_face = sparse_faces[next_idx]
            if prev_frame == next_frame:
                dense_series[frame_idx] = prev_face
                continue

            ratio = (frame_idx - prev_frame) / (next_frame - prev_frame)
            dense_series[frame_idx] = FaceDetection(
                track_id=track_id,
                conf=float((1.0 - ratio) * prev_face.conf + ratio * next_face.conf),
                x1=int(round((1.0 - ratio) * prev_face.x1 + ratio * next_face.x1)),
                y1=int(round((1.0 - ratio) * prev_face.y1 + ratio * next_face.y1)),
                x2=int(round((1.0 - ratio) * prev_face.x2 + ratio * next_face.x2)),
                y2=int(round((1.0 - ratio) * prev_face.y2 + ratio * next_face.y2)),
            )

        return dense_series

    def _smooth_face_track(self, dense_series: dict[int, FaceDetection], window: int) -> dict[int, FaceDetection]:
        ordered_frames = sorted(dense_series)
        half_window = window // 2
        smoothed_series: dict[int, FaceDetection] = {}

        for position, frame_idx in enumerate(ordered_frames):
            left = max(0, position - half_window)
            right = min(len(ordered_frames), position + half_window + 1)
            window_faces = [dense_series[idx] for idx in ordered_frames[left:right]]
            smoothed_series[frame_idx] = FaceDetection(
                track_id=dense_series[frame_idx].track_id,
                conf=float(np.mean([face.conf for face in window_faces])),
                x1=int(round(np.mean([face.x1 for face in window_faces]))),
                y1=int(round(np.mean([face.y1 for face in window_faces]))),
                x2=int(round(np.mean([face.x2 for face in window_faces]))),
                y2=int(round(np.mean([face.y2 for face in window_faces]))),
            )

        return smoothed_series

    def _smooth_track(self, dense_series: dict[int, GazePoint], window: int, point_method: str) -> dict[int, GazePoint]:
        ordered_frames = sorted(dense_series)
        half_window = window // 2
        smoothed_series: dict[int, GazePoint] = {}

        for position, frame_idx in enumerate(ordered_frames):
            left = max(0, position - half_window)
            right = min(len(ordered_frames), position + half_window + 1)
            window_points = [dense_series[idx] for idx in ordered_frames[left:right]]
            mean_heatmap = np.mean([point.heatmap for point in window_points], axis=0, dtype=np.float32)
            frame_width = window_points[0].frame_width
            frame_height = window_points[0].frame_height
            x_gaze, y_gaze = self.point_resolver.resolve(mean_heatmap, frame_width, frame_height, point_method)
            smoothed_series[frame_idx] = GazePoint(
                track_id=dense_series[frame_idx].track_id,
                inout=float(np.mean([point.inout for point in window_points])),
                x_gaze=x_gaze,
                y_gaze=y_gaze,
                heatmap=mean_heatmap.astype(np.float32),
                frame_width=frame_width,
                frame_height=frame_height,
            )
        return smoothed_series
