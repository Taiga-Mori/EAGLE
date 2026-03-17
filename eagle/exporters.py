import shutil
import subprocess
from pathlib import Path

import cv2
import pandas as pd

from .constants import ANNOTATION_COLUMNS
from .types import AppPaths, MediaContext


class AnnotationExporter:
    """Export visualization media and ELAN-compatible annotations."""

    def __init__(self, paths: AppPaths) -> None:
        self.paths = paths

    def make_video(self, context: MediaContext, visualization_mode: str) -> Path | list[Path]:
        if context.media_type != "video":
            raise RuntimeError("make_video() is only available for video input.")
        if visualization_mode == "heatmap":
            return self.make_heatmap_videos(context)
        if visualization_mode == "both":
            outputs: list[Path] = [self._make_point_video(context)]
            outputs.extend(self.make_heatmap_videos(context))
            return outputs

        return self._make_point_video(context)

    def _make_point_video(self, context: MediaContext) -> Path:
        silent_video = context.output_dir / "temp.mp4"
        output_video = context.output_dir / "all_points.mp4"
        subprocess.run(
            [
                str(self.paths.ffmpeg_path),
                "-y",
                "-framerate",
                str(context.object_target_fps),
                "-pattern_type",
                "glob",
                "-i",
                str(context.temp_dir / "*.jpg"),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(silent_video),
            ],
            check=True,
        )

        merge_command = [
            str(self.paths.ffmpeg_path),
            "-y",
            "-i",
            str(silent_video),
            "-i",
            str(context.media_path),
            "-c:v",
            "copy",
            "-map",
            "0:v:0",
        ]
        if self._video_has_audio_stream(context.media_path):
            merge_command.extend(["-c:a", "copy", "-map", "1:a:0"])
        else:
            merge_command.append("-an")
        merge_command.append(str(output_video))
        subprocess.run(merge_command, check=True)
        silent_video.unlink(missing_ok=True)
        shutil.rmtree(context.temp_dir, ignore_errors=True)
        return output_video

    def make_image(self, context: MediaContext, visualization_mode: str) -> Path | list[Path]:
        if context.media_type != "image":
            raise RuntimeError("make_image() is only available for image input.")
        if visualization_mode == "heatmap":
            outputs = self.make_heatmap_images(context)
            shutil.rmtree(context.temp_dir, ignore_errors=True)
            return outputs
        if visualization_mode == "both":
            outputs: list[Path] = []
            if not context.annotated_image_path.exists():
                raise RuntimeError("Run det_faces_and_gaze() before make_image().")
            outputs.append(context.annotated_image_path)
            outputs.extend(self.make_heatmap_images(context))
            shutil.rmtree(context.temp_dir, ignore_errors=True)
            return outputs
        if not context.annotated_image_path.exists():
            raise RuntimeError("Run det_faces_and_gaze() before make_image().")
        shutil.rmtree(context.temp_dir, ignore_errors=True)
        return context.annotated_image_path

    def make_heatmap_videos(self, context: MediaContext) -> list[Path]:
        outputs: list[Path] = []
        for person_dir in sorted(path for path in context.heatmap_dir.iterdir() if path.is_dir()):
            if not any(person_dir.glob("*.jpg")):
                continue
            silent_video = person_dir / "temp.mp4"
            output_video = context.output_dir / f"{person_dir.name}_heatmap.mp4"
            subprocess.run(
                [
                    str(self.paths.ffmpeg_path),
                    "-y",
                    "-framerate",
                    str(context.object_target_fps),
                    "-pattern_type",
                    "glob",
                    "-i",
                    str(person_dir / "*.jpg"),
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    str(silent_video),
                ],
                check=True,
            )
            silent_video.replace(output_video)
            shutil.rmtree(person_dir, ignore_errors=True)
            outputs.append(output_video)
        shutil.rmtree(context.heatmap_dir, ignore_errors=True)
        return outputs

    def make_heatmap_images(self, context: MediaContext) -> list[Path]:
        outputs: list[Path] = []
        for person_dir in sorted(path for path in context.heatmap_dir.iterdir() if path.is_dir()):
            source_paths = sorted(person_dir.glob("*.jpg"))
            if not source_paths:
                continue
            output_path = context.output_dir / f"{person_dir.name}_heatmap.jpg"
            source_paths[0].replace(output_path)
            outputs.append(output_path)
        shutil.rmtree(context.heatmap_dir, ignore_errors=True)
        return outputs

    def make_elan_csv(self, context: MediaContext, det_thresh: float) -> pd.DataFrame:
        gaze_df = pd.read_csv(context.gaze_path)
        object_df = pd.read_csv(context.objects_path)
        if gaze_df.empty or object_df.empty:
            empty_df = pd.DataFrame(columns=ANNOTATION_COLUMNS)
            empty_df.to_csv(context.annotation_path, index=False)
            return empty_df

        object_df["label"] = object_df["label"].astype(str).str.replace(r"^(person \d+)$", r"\1's body", regex=True)
        object_df["area"] = (object_df["x2"] - object_df["x1"]) * (object_df["y2"] - object_df["y1"])
        face_df = gaze_df[gaze_df["face_detected"] == True].copy()
        face_df["face_area"] = (face_df["face_x2"] - face_df["face_x1"]) * (face_df["face_y2"] - face_df["face_y1"])

        target_rows = []
        valid_gaze_df = gaze_df[gaze_df["face_detected"] == True].copy()
        for _, gaze_row in valid_gaze_df.iterrows():
            target_rows.append(
                {
                    "frame_idx": int(gaze_row["frame_idx"]),
                    "track_id": int(gaze_row["track_id"]),
                    "target": self._resolve_target(gaze_row, object_df, face_df, det_thresh),
                }
            )

        target_df = pd.DataFrame(target_rows)
        if target_df.empty:
            empty_df = pd.DataFrame(columns=ANNOTATION_COLUMNS)
            empty_df.to_csv(context.annotation_path, index=False)
            return empty_df

        if context.media_type == "image":
            annotation_df = pd.DataFrame(
                [
                    {
                        "tier": f"person {int(row['track_id'])}",
                        "start_time": 0.0,
                        "end_time": 1.0,
                        "gaze": row["target"],
                    }
                    for _, row in target_df.iterrows()
                ],
                columns=ANNOTATION_COLUMNS,
            )
            annotation_df.to_csv(context.annotation_path, index=False)
            return annotation_df

        segments = []
        for track_id, group in target_df.groupby("track_id"):
            group = group.sort_values("frame_idx")
            start_frame = None
            previous_target = None
            for _, row in group.iterrows():
                current_frame = int(row["frame_idx"])
                current_target = str(row["target"])
                if previous_target is None:
                    start_frame = current_frame
                    previous_target = current_target
                    continue
                if current_target != previous_target:
                    segments.append(
                        {
                            "tier": f"person {int(track_id)}",
                            "start_time": start_frame / context.fps,
                            "end_time": current_frame / context.fps,
                            "gaze": previous_target,
                        }
                    )
                    start_frame = current_frame
                    previous_target = current_target
            segments.append(
                {
                    "tier": f"person {int(track_id)}",
                    "start_time": start_frame / context.fps,
                    "end_time": (context.total_frames - 1) / context.fps,
                    "gaze": previous_target,
                }
            )

        annotation_df = pd.DataFrame(segments, columns=ANNOTATION_COLUMNS)
        annotation_df.to_csv(context.annotation_path, index=False)
        return annotation_df

    def _resolve_target(
        self,
        gaze_row: pd.Series,
        object_df: pd.DataFrame,
        face_df: pd.DataFrame,
        det_thresh: float,
    ) -> str:
        if pd.isna(gaze_row["inout"]) or float(gaze_row["inout"]) <= det_thresh:
            return "out of frame"
        if pd.isna(gaze_row["x_gaze"]) or pd.isna(gaze_row["y_gaze"]):
            return "other"

        face_hits = face_df[
            (face_df["frame_idx"] == int(gaze_row["frame_idx"]))
            & (face_df["track_id"] != int(gaze_row["track_id"]))
            & (face_df["face_x1"] <= gaze_row["x_gaze"])
            & (gaze_row["x_gaze"] <= face_df["face_x2"])
            & (face_df["face_y1"] <= gaze_row["y_gaze"])
            & (gaze_row["y_gaze"] <= face_df["face_y2"])
        ]
        if not face_hits.empty:
            hit = face_hits.sort_values("face_area").iloc[0]
            return f"person {int(hit['track_id'])}'s face"

        hits = object_df[
            (object_df["frame_idx"] == int(gaze_row["frame_idx"]))
            & (object_df["x1"] <= gaze_row["x_gaze"])
            & (gaze_row["x_gaze"] <= object_df["x2"])
            & (object_df["y1"] <= gaze_row["y_gaze"])
            & (gaze_row["y_gaze"] <= object_df["y2"])
        ]
        if hits.empty:
            return "other"
        hit = hits.sort_values("area").iloc[0]
        if str(hit["cls"]) == "person":
            return f"person {int(hit['track_id'])}'s body"
        return str(hit["label"])

    def _video_has_audio_stream(self, video_path: Path) -> bool:
        audio_prop = getattr(cv2, "CAP_PROP_AUDIO_TOTAL_STREAMS", None)
        if audio_prop is None:
            return False
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            return False
        try:
            return int(capture.get(audio_prop)) > 0
        finally:
            capture.release()
