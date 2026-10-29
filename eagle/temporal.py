import bisect
import cv2
import numpy as np
import pandas as pd
import json

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

    def smooth(
        self,
        raw_rows: list[dict],
        total_frames: int,
        window: int,
        media_type: str,
        max_switch_gap: int = 0,
    ) -> pd.DataFrame:
        if not raw_rows:
            return pd.DataFrame(columns=OBJECT_COLUMNS)
        if media_type == "image":
            output = pd.DataFrame(raw_rows)
            if "object_detected" not in output.columns:
                output["object_detected"] = True
            if "source" not in output.columns:
                output["source"] = "detect"
            if "pose_keypoints" not in output.columns:
                output["pose_keypoints"] = None
            return output[OBJECT_COLUMNS].sort_values(["frame_idx", "track_id"]).reset_index(drop=True)

        for row in raw_rows:
            row["frame_idx"] = int(row["frame_idx"])
            row.pop("yolo_idx", None)

        detections = pd.DataFrame(raw_rows).sort_values(["track_id", "frame_idx"])
        detections["object_detected"] = True
        smoothed_groups = []
        bbox_cols = ["x1", "y1", "x2", "y2"]
        extra_cols = [column for column in ["source", "pose_keypoints"] if column in detections.columns]

        detections["track_id"] = detections["track_id"].astype(str)
        for track_id, group in detections.groupby("track_id", sort=False):
            group = group.sort_values(["frame_idx", "conf"], ascending=[True, False])
            group = group.drop_duplicates(subset="frame_idx", keep="first")
            if max_switch_gap > 0 and str(group["cls"].dropna().astype(str).mode().iloc[0]) == "person":
                group = self._remove_short_switch_islands(group, max_switch_gap)
                if group.empty:
                    continue
            frame_range = range(int(group["frame_idx"].min()), int(group["frame_idx"].max()) + 1)
            group = group.set_index("frame_idx").reindex(frame_range)
            group["track_id"] = str(track_id)
            group["object_detected"] = group["object_detected"].where(group["object_detected"].notna(), False).astype(bool)
            classes = group["cls"].dropna()
            if classes.empty:
                continue
            group["cls"] = str(classes.mode().iloc[0])
            geometry_df, keypoint_count = self._geometry_dataframe(group, bbox_cols)
            geometry_df = geometry_df.interpolate(method="linear", limit_direction="both")
            if window > 1:
                geometry_df = geometry_df.rolling(window=window, min_periods=1, center=True).mean()
            group[bbox_cols] = geometry_df[bbox_cols]
            group["conf"] = geometry_df["conf"]
            if "pose_keypoints" in extra_cols:
                group["pose_keypoints"] = self._serialize_pose_keypoints(geometry_df, keypoint_count)
            if "source" in extra_cols:
                non_null_sources = group["source"].dropna().astype(str)
                group["source"] = str(non_null_sources.mode().iloc[0]) if not non_null_sources.empty else "detect"
            group = group.dropna(subset=bbox_cols).reset_index().rename(columns={"index": "frame_idx"})
            group["label"] = group.apply(lambda row: f"{row['cls']} {row['track_id']}", axis=1)
            smoothed_groups.append(group)

        if not smoothed_groups:
            return pd.DataFrame(columns=OBJECT_COLUMNS)

        output = pd.concat(smoothed_groups, ignore_index=True).sort_values(["frame_idx", "track_id"])
        for column in ["frame_idx", "x1", "y1", "x2", "y2"]:
            output[column] = output[column].round().astype(int)
        output["track_id"] = output["track_id"].astype(str)
        output["object_detected"] = output["object_detected"].where(output["object_detected"].notna(), False).astype(bool)
        output["conf"] = output["conf"].astype(float)
        if "source" not in output.columns:
            output["source"] = "detect"
        if "pose_keypoints" not in output.columns:
            output["pose_keypoints"] = None
        return output[OBJECT_COLUMNS]

    def _geometry_dataframe(self, group: pd.DataFrame, bbox_cols: list[str]) -> tuple[pd.DataFrame, int]:
        geometry = group[bbox_cols + ["conf"]].astype(float).copy()
        if "pose_keypoints" not in group.columns:
            return geometry, 0

        parsed = [self._parse_pose_keypoints(value) for value in group["pose_keypoints"].tolist()]
        keypoint_count = max((len(points) for points in parsed), default=0)
        if keypoint_count == 0:
            return geometry, 0

        for keypoint_index in range(keypoint_count):
            for value_index, value_name in enumerate(["x", "y", "conf"]):
                column = f"{keypoint_index}_{value_name}"
                values: list[float] = []
                for points in parsed:
                    if keypoint_index >= len(points):
                        values.append(np.nan)
                        continue
                    value = points[keypoint_index][value_index]
                    values.append(float(value) if value is not None and np.isfinite(float(value)) else np.nan)
                geometry[column] = values
        return geometry, keypoint_count

    def _serialize_pose_keypoints(self, geometry_df: pd.DataFrame, keypoint_count: int) -> pd.Series:
        if keypoint_count == 0:
            return pd.Series([None] * len(geometry_df), index=geometry_df.index)
        output: list[str | None] = []
        for row_index in geometry_df.index:
            keypoints: list[list[float | None]] = []
            for keypoint_index in range(keypoint_count):
                x = geometry_df.at[row_index, f"{keypoint_index}_x"]
                y = geometry_df.at[row_index, f"{keypoint_index}_y"]
                conf = geometry_df.at[row_index, f"{keypoint_index}_conf"]
                if pd.isna(x) or pd.isna(y):
                    keypoints.append([None, None, None if pd.isna(conf) else float(conf)])
                    continue
                keypoints.append(
                    [
                        float(x),
                        float(y),
                        None if pd.isna(conf) else float(conf),
                    ]
                )
            output.append(json.dumps(keypoints, ensure_ascii=False))
        return pd.Series(output, index=geometry_df.index)

    def _parse_pose_keypoints(self, value) -> list[list[float | None]]:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return []
        payload = value
        if isinstance(value, str):
            if not value.strip():
                return []
            try:
                payload = json.loads(value)
            except Exception:
                return []
        if not isinstance(payload, list):
            return []

        keypoints: list[list[float | None]] = []
        for point in payload:
            if not isinstance(point, (list, tuple)) or len(point) < 2:
                keypoints.append([None, None, None])
                continue
            x = self._finite_or_none(point[0])
            y = self._finite_or_none(point[1])
            conf = self._finite_or_none(point[2]) if len(point) >= 3 else None
            keypoints.append([x, y, conf])
        return keypoints

    def _finite_or_none(self, value) -> float | None:
        if value is None:
            return None
        try:
            numeric = float(value)
        except Exception:
            return None
        return numeric if np.isfinite(numeric) else None

    def _remove_short_switch_islands(self, group: pd.DataFrame, max_switch_gap: int) -> pd.DataFrame:
        if len(group) < 5:
            return group

        ordered = group.sort_values("frame_idx").reset_index(drop=True)
        centers = self._track_geometry_centers(ordered)
        widths = np.maximum(ordered["x2"].astype(float).to_numpy() - ordered["x1"].astype(float).to_numpy(), 1.0)
        heights = np.maximum(ordered["y2"].astype(float).to_numpy() - ordered["y1"].astype(float).to_numpy(), 1.0)
        diagonals = np.hypot(widths, heights)
        frame_indices = ordered["frame_idx"].astype(int).to_numpy()

        remove_positions: set[int] = set()
        index = 1
        while index < len(ordered) - 1:
            best_end: int | None = None
            for end in range(index, len(ordered) - 1):
                if int(frame_indices[end] - frame_indices[index] + 1) > int(max_switch_gap):
                    break
                before = index - 1
                after = end + 1
                reference_diag = max(float(np.median([diagonals[before], diagonals[after]])), 1.0)
                before_after_distance = float(np.linalg.norm(centers[before] - centers[after]))
                if before_after_distance > reference_diag * 0.50:
                    continue
                anchor = (centers[before] + centers[after]) / 2.0
                island_distances = np.linalg.norm(centers[index : end + 1] - anchor, axis=1)
                if float(np.median(island_distances)) <= reference_diag * 0.75:
                    continue
                best_end = end

            if best_end is None:
                index += 1
                continue
            remove_positions.update(range(index, best_end + 1))
            index = best_end + 1

        if not remove_positions:
            return group
        kept = ordered.drop(index=sorted(remove_positions)).copy()
        return kept.reset_index(drop=True)

    def _track_geometry_centers(self, group: pd.DataFrame) -> np.ndarray:
        bbox_centers = np.column_stack(
            [
                (group["x1"].astype(float).to_numpy() + group["x2"].astype(float).to_numpy()) / 2.0,
                (group["y1"].astype(float).to_numpy() + group["y2"].astype(float).to_numpy()) / 2.0,
            ]
        )
        if "pose_keypoints" not in group.columns:
            return bbox_centers

        centers = bbox_centers.copy()
        for row_index, value in enumerate(group["pose_keypoints"].tolist()):
            keypoints = self._parse_pose_keypoints(value)
            valid_points = [(point[0], point[1]) for point in keypoints if point[0] is not None and point[1] is not None]
            if not valid_points:
                continue
            centers[row_index] = np.asarray(valid_points, dtype=float).mean(axis=0)
        return centers



class PersonTrackPostprocessor:
    """Repair person tracks after tracking.

    Duplicates: two tracks that box the same person in at least 80% of their common frames are merged into
    the longer one. A frame counts when IoU >= duplicate_iou, or when one box lies inside the other (e.g. an
    upper-body box inside a full-body box) and their head keypoints coincide. ID switches: a track that starts shortly after another ends
    (within stitch_max_gap frames, or overlapping it by up to half of that) near where it ended and with a
    similar box size continues that track. Both steps repeat until nothing changes, since stitched tracks
    can turn out to duplicate others. Gaps inside merged tracks are filled by linear interpolation
    (object_detected=False).
    """

    DUPLICATE_FRAME_RATIO = 0.8
    STITCH_MAX_DISTANCE = 0.5  # box center distance relative to the box diagonal
    STITCH_MAX_SIZE_RATIO = 2.0
    MAX_ROUNDS = 10
    CONTAINMENT = 0.9  # intersection over the smaller box
    HEAD_DISTANCE = 0.15  # head keypoint distance relative to the smaller box diagonal
    HEAD_KEYPOINTS = range(5)  # COCO nose, eyes, ears
    HEAD_MIN_CONF = 0.3

    def __init__(self, smoother: ObjectTrackSmoother) -> None:
        self.smoother = smoother

    def process(
        self,
        tracks: pd.DataFrame,
        duplicate_iou: float,
        stitch_max_gap: int,
    ) -> tuple[pd.DataFrame, list[dict]]:
        if tracks.empty:
            return tracks, []
        tracks = tracks.copy()
        tracks["track_id"] = tracks["track_id"].astype(str)
        merges: list[dict] = []
        for _ in range(self.MAX_ROUNDS):
            merged_before = len(merges)
            tracks = self._merge_duplicates(tracks, duplicate_iou, merges)
            tracks = self._stitch_fragments(tracks, stitch_max_gap, merges)
            if len(merges) == merged_before:
                break
        tracks = self._fill_gaps(tracks)
        return tracks.sort_values(["frame_idx", "track_id"]).reset_index(drop=True)[OBJECT_COLUMNS], merges

    @staticmethod
    def _box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        x1 = np.maximum(a[:, 0], b[:, 0])
        y1 = np.maximum(a[:, 1], b[:, 1])
        x2 = np.minimum(a[:, 2], b[:, 2])
        y2 = np.minimum(a[:, 3], b[:, 3])
        inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
        area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
        area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        return inter / np.maximum(area_a + area_b - inter, 1e-6)

    def _head_points(self, group: pd.DataFrame) -> pd.DataFrame:
        points = []
        for value in group["pose_keypoints"].tolist() if "pose_keypoints" in group.columns else [None] * len(group):
            keypoints = self.smoother._parse_pose_keypoints(value)
            valid = [
                (keypoints[i][0], keypoints[i][1])
                for i in self.HEAD_KEYPOINTS
                if i < len(keypoints) and keypoints[i][0] is not None and (keypoints[i][2] or 0.0) >= self.HEAD_MIN_CONF
            ]
            points.append(np.mean(valid, axis=0) if valid else (np.nan, np.nan))
        return pd.DataFrame(points, index=group["frame_idx"].astype(int).to_numpy(), columns=["hx", "hy"])

    def _duplicate_frames(self, a: pd.DataFrame, b: pd.DataFrame, duplicate_iou: float) -> np.ndarray:
        box_a, box_b = a[["x1", "y1", "x2", "y2"]].to_numpy(), b[["x1", "y1", "x2", "y2"]].to_numpy()
        iou = self._box_iou(box_a, box_b)
        inter_w = np.clip(np.minimum(box_a[:, 2], box_b[:, 2]) - np.maximum(box_a[:, 0], box_b[:, 0]), 0, None)
        inter_h = np.clip(np.minimum(box_a[:, 3], box_b[:, 3]) - np.maximum(box_a[:, 1], box_b[:, 1]), 0, None)
        area_a = (box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1])
        area_b = (box_b[:, 2] - box_b[:, 0]) * (box_b[:, 3] - box_b[:, 1])
        smaller = np.where(area_a <= area_b, 0, 1)
        containment = inter_w * inter_h / np.maximum(np.minimum(area_a, area_b), 1e-6)
        small_boxes = np.where(smaller[:, None] == 0, box_a, box_b)
        diagonal = np.hypot(small_boxes[:, 2] - small_boxes[:, 0], small_boxes[:, 3] - small_boxes[:, 1])
        head_distance = np.hypot(a["hx"].to_numpy() - b["hx"].to_numpy(), a["hy"].to_numpy() - b["hy"].to_numpy())
        same_head = head_distance <= self.HEAD_DISTANCE * diagonal  # False when a head is missing (NaN)
        return (iou >= duplicate_iou) | ((containment >= self.CONTAINMENT) & same_head)

    def _merge_duplicates(self, tracks: pd.DataFrame, duplicate_iou: float, merges: list[dict]) -> pd.DataFrame:
        boxes = {
            track_id: pd.concat(
                [group.set_index(group["frame_idx"].astype(int))[["x1", "y1", "x2", "y2"]].astype(float), self._head_points(group)],
                axis=1,
            ).sort_index()
            for track_id, group in tracks.groupby("track_id")
        }
        pairs = []
        ids = sorted(boxes)
        for i, a in enumerate(ids):
            for b in ids[i + 1 :]:
                if boxes[a].index[0] > boxes[b].index[-1] or boxes[b].index[0] > boxes[a].index[-1]:
                    continue
                common = boxes[a].index.intersection(boxes[b].index)
                if len(common) == 0:
                    continue
                duplicate = self._duplicate_frames(boxes[a].loc[common], boxes[b].loc[common], duplicate_iou)
                if float(np.mean(duplicate)) >= self.DUPLICATE_FRAME_RATIO:
                    pairs.append((len(common), a, b))

        root = {track_id: track_id for track_id in ids}

        def find(track_id: str) -> str:
            while root[track_id] != track_id:
                track_id = root[track_id]
            return track_id

        for _, a, b in sorted(pairs, reverse=True):
            a, b = find(a), find(b)
            if a == b:
                continue
            sizes = tracks["track_id"].value_counts()
            keep, drop = (a, b) if sizes.get(a, 0) >= sizes.get(b, 0) else (b, a)
            keep_frames = set(tracks.loc[tracks["track_id"] == keep, "frame_idx"].astype(int))
            drop_rows = tracks["track_id"] == drop
            overlapping = drop_rows & tracks["frame_idx"].astype(int).isin(keep_frames)
            tracks = tracks[~overlapping].copy()
            tracks.loc[tracks["track_id"] == drop, "track_id"] = keep
            root[drop] = keep
            merges.append({"track_id": drop, "merged_into": keep, "reason": "duplicate"})
        return tracks

    def _stitch_fragments(self, tracks: pd.DataFrame, stitch_max_gap: int, merges: list[dict]) -> pd.DataFrame:
        if stitch_max_gap <= 0:
            return tracks
        ends = {}
        for track_id, group in tracks.groupby("track_id"):
            group = group.sort_values("frame_idx")
            first, last = group.iloc[0], group.iloc[-1]
            ends[track_id] = (int(first["frame_idx"]), int(last["frame_idx"]), first, last)

        candidates = []
        for a, (_, a_end, _, a_last) in ends.items():
            for b, (b_start, _, b_first, _) in ends.items():
                gap = b_start - a_end
                if a == b or gap > stitch_max_gap or gap < -(stitch_max_gap // 2) or b_start <= ends[a][0]:
                    continue
                width_a, height_a = a_last["x2"] - a_last["x1"], a_last["y2"] - a_last["y1"]
                width_b, height_b = b_first["x2"] - b_first["x1"], b_first["y2"] - b_first["y1"]
                diagonal = max(float(np.hypot(width_a, height_a)), 1.0)
                distance = float(
                    np.hypot(
                        (a_last["x1"] + a_last["x2"] - b_first["x1"] - b_first["x2"]) / 2.0,
                        (a_last["y1"] + a_last["y2"] - b_first["y1"] - b_first["y2"]) / 2.0,
                    )
                )
                size_ratio = max(width_a * height_a, 1.0) / max(width_b * height_b, 1.0)
                if distance > self.STITCH_MAX_DISTANCE * diagonal:
                    continue
                if not 1.0 / self.STITCH_MAX_SIZE_RATIO <= size_ratio <= self.STITCH_MAX_SIZE_RATIO:
                    continue
                candidates.append((distance / diagonal + abs(gap) / stitch_max_gap, a, b))

        successor: dict[str, str] = {}
        predecessor: dict[str, str] = {}
        for _, a, b in sorted(candidates):
            if a in successor or b in predecessor:
                continue
            successor[a] = b
            predecessor[b] = a

        # Where a successor overlaps its predecessor, the predecessor's boxes are kept.
        for b, a in predecessor.items():
            tracks = tracks[~((tracks["track_id"] == b) & (tracks["frame_idx"].astype(int) <= ends[a][1]))]
        rename = {}
        for b in predecessor:
            head = b
            while head in predecessor:
                head = predecessor[head]
            rename[b] = head
        for b, head in sorted(rename.items(), key=lambda item: ends[item[0]][0]):
            merges.append({"track_id": b, "merged_into": head, "reason": "id_switch"})
        tracks = tracks.copy()
        tracks["track_id"] = tracks["track_id"].map(lambda track_id: rename.get(track_id, track_id))
        return tracks

    def _fill_gaps(self, tracks: pd.DataFrame) -> pd.DataFrame:
        bbox_cols = ["x1", "y1", "x2", "y2"]
        groups = []
        for track_id, group in tracks.groupby("track_id", sort=False):
            group = group.sort_values("frame_idx").drop_duplicates(subset="frame_idx", keep="first")
            frame_range = range(int(group["frame_idx"].min()), int(group["frame_idx"].max()) + 1)
            if len(group) == len(frame_range):
                groups.append(group)
                continue
            group = group.set_index("frame_idx").reindex(frame_range)
            group["track_id"] = str(track_id)
            geometry_df, keypoint_count = self.smoother._geometry_dataframe(group, bbox_cols)
            geometry_df = geometry_df.interpolate(method="linear", limit_direction="both")
            group[bbox_cols] = geometry_df[bbox_cols].round()
            group["conf"] = geometry_df["conf"]
            if "pose_keypoints" in group.columns:
                group["pose_keypoints"] = self.smoother._serialize_pose_keypoints(geometry_df, keypoint_count)
            group["object_detected"] = group["object_detected"].where(group["object_detected"].notna(), False).astype(bool)
            for column in ["cls", "source"]:
                group[column] = group[column].ffill().bfill()
            group = group.reset_index().rename(columns={"index": "frame_idx"})
            groups.append(group)
        output = pd.concat(groups, ignore_index=True)
        output["track_id"] = output["track_id"].astype(str)
        for column in ["frame_idx", "x1", "y1", "x2", "y2"]:
            output[column] = output[column].round().astype(int)
        output["label"] = output["cls"].astype(str) + " " + output["track_id"]
        return output


class GazeTemporalProcessor:
    """Interpolate sparse face and gaze results over time and smooth face boxes."""

    def __init__(self, point_resolver: GazePointResolver | None = None) -> None:
        self.point_resolver = point_resolver or GazePointResolver()

    def interpolate_faces(
        self,
        frame_indices: list[int],
        sampled_frame_indices: list[int],
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
        sampled_frame_set = set(sampled_frame_indices)

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

            sampled_track_frames = sorted(
                frame_idx
                for frame_idx in track_frames
                if frame_idx in sampled_frame_set
            )
            dense_faces = self._interpolate_face_track_between_successful_samples(
                track_id,
                track_frames,
                sparse_faces,
                sampled_track_frames,
            )
            if smoothing_window > 1:
                dense_faces = self._smooth_face_track(dense_faces, smoothing_window)
            for frame_idx, face in dense_faces.items():
                dense_face_maps_by_frame.setdefault(frame_idx, {})[track_id] = face

        return dense_face_maps_by_frame

    def _interpolate_face_track_between_successful_samples(
        self,
        track_id: str,
        track_frames: list[int],
        sparse_faces: list[tuple[int, FaceDetection]],
        sampled_track_frames: list[int],
    ) -> dict[int, FaceDetection]:
        dense_series: dict[int, FaceDetection] = {frame_idx: face for frame_idx, face in sparse_faces}
        if len(sampled_track_frames) < 2:
            return dense_series

        face_by_frame = dict(sparse_faces)
        track_frame_set = set(track_frames)
        for prev_sample, next_sample in zip(sampled_track_frames, sampled_track_frames[1:]):
            if prev_sample not in face_by_frame or next_sample not in face_by_frame:
                continue
            prev_face = face_by_frame[prev_sample]
            next_face = face_by_frame[next_sample]
            frame_gap = next_sample - prev_sample
            if frame_gap <= 1:
                continue
            for frame_idx in range(prev_sample + 1, next_sample):
                if frame_idx not in track_frame_set:
                    continue
                ratio = (frame_idx - prev_sample) / frame_gap
                dense_series[frame_idx] = FaceDetection(
                    track_id=track_id,
                    conf=float((1.0 - ratio) * prev_face.conf + ratio * next_face.conf),
                    x1=int(round((1.0 - ratio) * prev_face.x1 + ratio * next_face.x1)),
                    y1=int(round((1.0 - ratio) * prev_face.y1 + ratio * next_face.y1)),
                    x2=int(round((1.0 - ratio) * prev_face.x2 + ratio * next_face.x2)),
                    y2=int(round((1.0 - ratio) * prev_face.y2 + ratio * next_face.y2)),
                )
        return dense_series

    def interpolate(
        self,
        frame_indices: list[int],
        face_maps_by_frame: dict[int, dict[int, object]],
        sparse_gaze_by_frame: dict[int, dict[int, GazePoint]],
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
            # sparse_frames is sorted, so bisect finds the neighbouring samples without a linear scan.
            right = bisect.bisect_right(sparse_frames, frame_idx)
            prev_idx = right - 1 if right > 0 else None
            left = bisect.bisect_left(sparse_frames, frame_idx)
            next_idx = left if left < len(sparse_frames) else None

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

    def _smooth_face_track(self, dense_series: dict[int, FaceDetection], window: int) -> dict[int, FaceDetection]:
        ordered_frames = sorted(dense_series)
        half_window = window // 2
        smoothed_series: dict[int, FaceDetection] = {}

        for segment_frames in self._contiguous_frame_segments(ordered_frames):
            for position, frame_idx in enumerate(segment_frames):
                left = max(0, position - half_window)
                right = min(len(segment_frames), position + half_window + 1)
                window_faces = [dense_series[idx] for idx in segment_frames[left:right]]
                smoothed_series[frame_idx] = FaceDetection(
                    track_id=dense_series[frame_idx].track_id,
                    conf=float(np.mean([face.conf for face in window_faces])),
                    x1=int(round(np.mean([face.x1 for face in window_faces]))),
                    y1=int(round(np.mean([face.y1 for face in window_faces]))),
                    x2=int(round(np.mean([face.x2 for face in window_faces]))),
                    y2=int(round(np.mean([face.y2 for face in window_faces]))),
                )

        return smoothed_series

    def _contiguous_frame_segments(self, frame_indices: list[int]) -> list[list[int]]:
        if not frame_indices:
            return []
        segments: list[list[int]] = [[frame_indices[0]]]
        for frame_idx in frame_indices[1:]:
            if frame_idx == segments[-1][-1] + 1:
                segments[-1].append(frame_idx)
                continue
            segments.append([frame_idx])
        return segments
