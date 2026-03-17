import numpy as np
import pandas as pd

from .constants import OBJECT_COLUMNS
from .types import GazePoint


class ObjectTrackSmoother:
    """Fill missing object frames and smooth track geometry."""

    def smooth(self, raw_rows: list[dict], total_frames: int, window: int, media_type: str) -> pd.DataFrame:
        if not raw_rows:
            return pd.DataFrame(columns=OBJECT_COLUMNS)
        if media_type == "image":
            output = pd.DataFrame(raw_rows)
            return output[OBJECT_COLUMNS].sort_values(["frame_idx", "track_id"]).reset_index(drop=True)

        for row in raw_rows:
            row["frame_idx"] = int(row["frame_idx"])
            row.pop("yolo_idx", None)

        detections = pd.DataFrame(raw_rows).sort_values(["track_id", "frame_idx"])
        smoothed_groups = []
        bbox_cols = ["x1", "y1", "x2", "y2"]

        for track_id, group in detections.groupby("track_id"):
            group = group.sort_values("frame_idx")
            frame_range = range(int(group["frame_idx"].min()), int(group["frame_idx"].max()) + 1)
            group = group.set_index("frame_idx").reindex(frame_range)
            group["track_id"] = int(track_id)
            classes = group["cls"].dropna()
            if classes.empty:
                continue
            group["cls"] = str(classes.mode().iloc[0])
            group[bbox_cols] = group[bbox_cols].interpolate(method="linear", limit_direction="both")
            group["conf"] = group["conf"].interpolate(method="linear", limit_direction="both")
            if window > 1:
                group[bbox_cols] = group[bbox_cols].rolling(window=window, min_periods=1, center=True).mean()
                group["conf"] = group["conf"].rolling(window=window, min_periods=1, center=True).mean()
            group = group.dropna(subset=bbox_cols).reset_index().rename(columns={"index": "frame_idx"})
            group["label"] = group.apply(lambda row: f"{row['cls']} {int(row['track_id'])}", axis=1)
            smoothed_groups.append(group)

        if not smoothed_groups:
            return pd.DataFrame(columns=OBJECT_COLUMNS)

        output = pd.concat(smoothed_groups, ignore_index=True).sort_values(["frame_idx", "track_id"])
        for column in ["frame_idx", "track_id", "x1", "y1", "x2", "y2"]:
            output[column] = output[column].round().astype(int)
        output["conf"] = output["conf"].astype(float)
        return output[OBJECT_COLUMNS]


class GazeTemporalProcessor:
    """Interpolate sparse gaze results and smooth them over time."""

    def interpolate_and_smooth(
        self,
        frame_indices: list[int],
        face_maps_by_frame: dict[int, dict[int, object]],
        sparse_gaze_by_frame: dict[int, dict[int, GazePoint]],
        smoothing_window: int,
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
            dense_series = self._interpolate_track(track_id, track_frames, sparse_points)
            if smoothing_window > 1:
                dense_series = self._smooth_track(dense_series, smoothing_window)
            for frame_idx, gaze in dense_series.items():
                dense_gaze_by_frame[frame_idx][track_id] = gaze

        return dense_gaze_by_frame

    def _interpolate_track(
        self,
        track_id: int,
        track_frames: list[int],
        sparse_points: list[tuple[int, GazePoint]],
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
            dense_series[frame_idx] = GazePoint(
                track_id=track_id,
                inout=float((1.0 - ratio) * prev_gaze.inout + ratio * next_gaze.inout),
                x_gaze=int(round((1.0 - ratio) * prev_gaze.x_gaze + ratio * next_gaze.x_gaze)),
                y_gaze=int(round((1.0 - ratio) * prev_gaze.y_gaze + ratio * next_gaze.y_gaze)),
                heatmap=heatmap.astype(np.float32),
                frame_width=prev_gaze.frame_width,
                frame_height=prev_gaze.frame_height,
            )
        return dense_series

    def _smooth_track(self, dense_series: dict[int, GazePoint], window: int) -> dict[int, GazePoint]:
        ordered_frames = sorted(dense_series)
        half_window = window // 2
        smoothed_series: dict[int, GazePoint] = {}

        for position, frame_idx in enumerate(ordered_frames):
            left = max(0, position - half_window)
            right = min(len(ordered_frames), position + half_window + 1)
            window_points = [dense_series[idx] for idx in ordered_frames[left:right]]
            mean_heatmap = np.mean([point.heatmap for point in window_points], axis=0, dtype=np.float32)
            max_index = np.argmax(mean_heatmap)
            y_hm, x_hm = np.unravel_index(max_index, mean_heatmap.shape)
            frame_width = window_points[0].frame_width
            frame_height = window_points[0].frame_height
            smoothed_series[frame_idx] = GazePoint(
                track_id=dense_series[frame_idx].track_id,
                inout=float(np.mean([point.inout for point in window_points])),
                x_gaze=int(x_hm / mean_heatmap.shape[1] * frame_width),
                y_gaze=int(y_hm / mean_heatmap.shape[0] * frame_height),
                heatmap=mean_heatmap.astype(np.float32),
                frame_width=frame_width,
                frame_height=frame_height,
            )
        return smoothed_series
