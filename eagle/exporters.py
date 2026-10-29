import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .body_parts import resolve_person_part_label
from .constants import ANNOTATION_COLUMNS
from .types import AppPaths, MediaContext


class PipedVideoWriter:
    """Stream BGR frames into an ffmpeg process so no per-frame images touch the disk."""

    def __init__(self, command: list[str], output_path: Path, purpose: str) -> None:
        self.command = command
        self.output_path = output_path
        self.purpose = purpose
        # ffmpeg's log goes to a file: an undrained stderr pipe would fill up and stall the encoder.
        self._log = tempfile.TemporaryFile()
        self._process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self._log)

    def write(self, frame: np.ndarray) -> None:
        assert self._process.stdin is not None
        try:
            self._process.stdin.write(np.ascontiguousarray(frame).tobytes())
        except (BrokenPipeError, OSError) as exc:
            self._process.wait()
            raise self._error() from exc

    def close(self) -> None:
        assert self._process.stdin is not None
        try:
            self._process.stdin.close()
        except (BrokenPipeError, OSError):
            pass
        if self._process.wait() != 0:
            raise self._error()
        self._log.close()

    def abort(self) -> None:
        self._process.kill()
        self._process.wait()
        self._log.close()
        self.output_path.unlink(missing_ok=True)

    def _error(self) -> RuntimeError:
        self._log.seek(0)
        details = self._log.read().decode("utf-8", errors="replace").strip() or "ffmpeg did not print an error message."
        self._log.close()
        self.output_path.unlink(missing_ok=True)
        return RuntimeError(
            f"ffmpeg failed while {self.purpose}.\n"
            f"Exit code: {self._process.returncode}\n"
            f"Command: {' '.join(self.command)}\n"
            f"Details: {details}"
        )


class AnnotationExporter:
    """Export visualization media and ELAN-compatible annotations."""

    def __init__(self, paths: AppPaths) -> None:
        self.paths = paths
        self._available_ffmpeg_encoders: set[str] | None = None
        self._ffmpeg_executable: str | None = None
        self._piped_encoders: dict[tuple[int, int], str] = {}

    @staticmethod
    def point_video_path(context: MediaContext) -> Path:
        return context.temp_dir / "temp.mp4"

    @staticmethod
    def heatmap_video_path(context: MediaContext, track_id: str) -> Path:
        return context.heatmap_dir / f"person_{track_id}.mp4"

    def open_video_writer(
        self,
        output_path: Path,
        width: int,
        height: int,
        fps: float,
        purpose: str,
        hardware: bool = True,
    ) -> PipedVideoWriter:
        encoder = self._piped_h264_encoder(width, height, fps) if hardware else "libx264"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        return PipedVideoWriter(
            self._raw_input_command(width, height, fps) + self._h264_encoder_args(encoder) + [str(output_path)],
            output_path,
            f"{purpose} with {encoder}",
        )

    def _raw_input_command(self, width: int, height: int, fps: float) -> list[str]:
        return [
            self._resolve_ffmpeg_executable(),
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{width}x{height}",
            "-framerate",
            str(fps),
            "-i",
            "pipe:0",
        ]

    def _resolve_ffmpeg_executable(self) -> str:
        if self._ffmpeg_executable is None:
            self._ffmpeg_executable = str(self.paths.ffmpeg_path)
            try:
                subprocess.run([self._ffmpeg_executable, "-version"], check=True, capture_output=True)
            except Exception:
                fallback_ffmpeg = shutil.which("ffmpeg")
                if fallback_ffmpeg is not None:
                    self._ffmpeg_executable = fallback_ffmpeg
        return self._ffmpeg_executable

    def _piped_h264_encoder(self, width: int, height: int, fps: float) -> str:
        # A streamed encode cannot be redone from saved frames, so try the hardware encoder on two
        # blank frames first and use libx264 if it does not work for this frame size.
        if (width, height) not in self._piped_encoders:
            encoder = self._preferred_h264_encoder() or "libx264"
            if encoder != "libx264":
                probe = self._raw_input_command(width, height, fps) + self._h264_encoder_args(encoder) + ["-f", "null", "-"]
                try:
                    subprocess.run(
                        probe,
                        input=bytes(width * height * 3 * 2),
                        check=True,
                        capture_output=True,
                        timeout=60,
                    )
                except Exception as exc:
                    print(f"Hardware video encoder {encoder} is unavailable; using libx264.\n{exc}", flush=True)
                    encoder = "libx264"
            self._piped_encoders[(width, height)] = encoder
        return self._piped_encoders[(width, height)]

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
        # det_gaze() streams the annotated frames into temp_dir; here the original audio is muxed in
        # and the finished file moved. Long ffmpeg writes straight onto slow/network-backed output
        # folders (e.g. WSL /mnt/c) can fail mid-file.
        silent_video = self.point_video_path(context)
        merged_video = context.temp_dir / "all_points.mp4"
        output_video = context.output_dir / "all_points.mp4"
        if not silent_video.exists():
            raise RuntimeError("Run det_faces() and det_gaze() before make_video().")

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
        merge_command.append(str(merged_video))
        self._run_ffmpeg(merge_command, "merging the original audio into the annotated point video")
        silent_video.unlink(missing_ok=True)
        output_video.unlink(missing_ok=True)
        shutil.move(str(merged_video), str(output_video))
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
                raise RuntimeError("Run det_faces() and det_gaze() before make_image().")
            outputs.append(context.annotated_image_path)
            outputs.extend(self.make_heatmap_images(context))
            shutil.rmtree(context.temp_dir, ignore_errors=True)
            return outputs
        if not context.annotated_image_path.exists():
            raise RuntimeError("Run det_faces() and det_gaze() before make_image().")
        shutil.rmtree(context.temp_dir, ignore_errors=True)
        return context.annotated_image_path

    def make_heatmap_videos(self, context: MediaContext) -> list[Path]:
        outputs: list[Path] = []
        for heatmap_video in sorted(context.heatmap_dir.glob("person_*.mp4")):
            output_video = context.output_dir / f"{heatmap_video.stem}_heatmap.mp4"
            output_video.unlink(missing_ok=True)
            shutil.move(str(heatmap_video), str(output_video))
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

    def make_elan_csv(
        self,
        context: MediaContext,
        det_thresh: float,
        gaze_target_radius: int,
        person_part_distance_scale: float,
        person_part_min_conf: float,
        selected_object_classes: list[str],
        label_smoothing_sec: float = 0.0,
    ) -> pd.DataFrame:
        context.annotation_path.parent.mkdir(parents=True, exist_ok=True)
        gaze_df = pd.read_csv(context.gaze_path)
        face_df = pd.read_csv(context.faces_path)
        object_frames = []
        if context.persons_path.exists():
            object_frames.append(pd.read_csv(context.persons_path))
        if context.objects_path.exists():
            object_frames.append(pd.read_csv(context.objects_path))
        object_frames = [frame.dropna(how="all") for frame in object_frames]
        object_frames = [frame for frame in object_frames if not frame.empty]
        object_df = pd.concat(object_frames, ignore_index=True) if object_frames else pd.DataFrame()
        if gaze_df.empty or face_df.empty or object_df.empty:
            empty_df = pd.DataFrame(columns=ANNOTATION_COLUMNS)
            self._write_annotation_csv(empty_df, context)
            return empty_df

        object_df["label"] = object_df.apply(
            lambda row: f"person {row['track_id']}" if str(row["cls"]) == "person" else str(row["label"]),
            axis=1,
        )
        object_df["area"] = (object_df["x2"] - object_df["x1"]) * (object_df["y2"] - object_df["y1"])
        face_df = face_df[face_df["face_x1"].notna()].copy()
        face_df["face_area"] = (face_df["face_x2"] - face_df["face_x1"]) * (face_df["face_y2"] - face_df["face_y1"])

        target_rows = []
        face_keys = set(zip(face_df["frame_idx"].astype(int), face_df["track_id"].astype(str)))
        valid_gaze_rows = []
        for _, gaze_row in gaze_df.iterrows():
            key = (int(gaze_row["frame_idx"]), str(gaze_row["track_id"]))
            if key in face_keys:
                valid_gaze_rows.append(gaze_row)
        valid_gaze_df = pd.DataFrame(valid_gaze_rows)
        # Split per frame once; filtering the full tables for every gaze row is quadratic on long videos.
        face_by_frame = {int(frame_idx): group for frame_idx, group in face_df.groupby("frame_idx")}
        object_by_frame = {int(frame_idx): group for frame_idx, group in object_df.groupby("frame_idx")}
        empty_face_df = face_df.iloc[0:0]
        empty_object_df = object_df.iloc[0:0]
        for _, gaze_row in valid_gaze_df.iterrows():
            frame_idx = int(gaze_row["frame_idx"])
            target_rows.append(
                {
                    "frame_idx": frame_idx,
                    "track_id": str(gaze_row["track_id"]),
                    "target": self._resolve_target(
                        gaze_row,
                        object_by_frame.get(frame_idx, empty_object_df),
                        face_by_frame.get(frame_idx, empty_face_df),
                        det_thresh,
                        gaze_target_radius,
                        person_part_distance_scale,
                        person_part_min_conf,
                        selected_object_classes,
                    ),
                }
            )

        target_df = pd.DataFrame(target_rows)
        if target_df.empty:
            empty_df = pd.DataFrame(columns=ANNOTATION_COLUMNS)
            self._write_annotation_csv(empty_df, context)
            return empty_df

        if context.media_type == "image":
            annotation_df = pd.DataFrame(
                [
                    {
                        "tier": self._gaze_tier_label(row["track_id"]),
                        "start_time": 0.0,
                        "end_time": 1.0,
                        "gaze": row["target"],
                    }
                    for _, row in target_df.iterrows()
                ],
                columns=ANNOTATION_COLUMNS,
            )
            self._write_annotation_csv(annotation_df, context)
            return annotation_df

        label_window = int(round(label_smoothing_sec * context.fps)) | 1
        segments = []
        for track_id, group in target_df.groupby("track_id"):
            group = group.sort_values("frame_idx")
            if label_window > 1:
                group = self._smooth_track_labels(group, label_window)
            start_frame = None
            previous_frame = None
            previous_target = None
            for _, row in group.iterrows():
                current_frame = int(row["frame_idx"])
                current_target = str(row["target"])
                if previous_target is None:
                    start_frame = current_frame
                    previous_frame = current_frame
                    previous_target = current_target
                    continue
                has_frame_gap = previous_frame is not None and current_frame > previous_frame + 1
                if has_frame_gap or current_target != previous_target:
                    segments.append(
                        {
                            "tier": self._gaze_tier_label(track_id),
                            "start_time": start_frame / context.fps,
                            "end_time": (previous_frame + 1) / context.fps,
                            "gaze": previous_target,
                        }
                    )
                    start_frame = current_frame
                    previous_target = current_target
                previous_frame = current_frame
            segments.append(
                {
                    "tier": self._gaze_tier_label(track_id),
                    "start_time": start_frame / context.fps,
                    "end_time": (previous_frame + 1) / context.fps,
                    "gaze": previous_target,
                }
            )

        annotation_df = pd.DataFrame(segments, columns=ANNOTATION_COLUMNS)
        self._write_annotation_csv(annotation_df, context)
        return annotation_df

    @classmethod
    def _smooth_track_labels(cls, group: pd.DataFrame, window: int) -> pd.DataFrame:
        """Mode-filter one track's frame-wise targets within each run of consecutive frames."""
        frames = group["frame_idx"].astype(int).to_numpy()
        targets = group["target"].astype(str).tolist()
        smoothed: list[str] = []
        run_start = 0
        for index in range(1, len(frames) + 1):
            if index == len(frames) or frames[index] != frames[index - 1] + 1:
                smoothed.extend(cls._mode_filter(targets[run_start:index], window))
                run_start = index
        group = group.copy()
        group["target"] = smoothed
        return group

    @staticmethod
    def _mode_filter(labels: list[str], window: int) -> list[str]:
        """Most frequent label in a centered window; ties keep the current label."""
        half = window // 2
        counts = Counter(labels[:half])
        smoothed = []
        for index, label in enumerate(labels):
            if index + half < len(labels):
                counts[labels[index + half]] += 1
            if index - half - 1 >= 0:
                counts[labels[index - half - 1]] -= 1
            best = max(counts.values())
            smoothed.append(label if counts[label] == best else max(counts, key=counts.__getitem__))
        return smoothed

    def _write_annotation_csv(self, annotation_df: pd.DataFrame, context: MediaContext) -> None:
        annotation_df.to_csv(context.annotation_path, index=False)
        if not context.annotation_path.exists():
            raise RuntimeError(f"Failed to write annotation CSV: {context.annotation_path}")

    def _gaze_tier_label(self, track_id: object) -> str:
        return f"person {str(track_id)}_Gaze"

    def _resolve_target(
        self,
        gaze_row: pd.Series,
        object_df: pd.DataFrame,
        face_df: pd.DataFrame,
        det_thresh: float,
        gaze_target_radius: int,
        person_part_distance_scale: float,
        person_part_min_conf: float,
        selected_object_classes: list[str],
    ) -> str:
        if pd.isna(gaze_row["inout"]):
            return "out of frame"
        if float(gaze_row["inout"]) <= det_thresh:
            if "offscreen_direction" in gaze_row and pd.notna(gaze_row["offscreen_direction"]):
                return str(gaze_row["offscreen_direction"])
            return "out of frame"
        if pd.isna(gaze_row["x_gaze"]) or pd.isna(gaze_row["y_gaze"]):
            return "other"

        candidates: list[tuple[float, str]] = []

        face_hits = face_df[
            (face_df["frame_idx"] == int(gaze_row["frame_idx"]))
            & (face_df["track_id"].astype(str) != str(gaze_row["track_id"]))
        ]
        for _, hit in face_hits.iterrows():
            if self._point_hits_box(
                int(gaze_row["x_gaze"]),
                int(gaze_row["y_gaze"]),
                int(hit["face_x1"]),
                int(hit["face_y1"]),
                int(hit["face_x2"]),
                int(hit["face_y2"]),
                gaze_target_radius,
            ):
                candidates.append((float(hit["face_area"]), f"person {hit['track_id']}'s face"))

        hits = object_df[
            (object_df["frame_idx"] == int(gaze_row["frame_idx"]))
        ]
        for _, hit in hits.iterrows():
            if self._point_hits_box(
                int(gaze_row["x_gaze"]),
                int(gaze_row["y_gaze"]),
                int(hit["x1"]),
                int(hit["y1"]),
                int(hit["x2"]),
                int(hit["y2"]),
                gaze_target_radius,
            ):
                if str(hit["cls"]) == "person":
                    part_label = resolve_person_part_label(
                        hit.to_dict(),
                        int(gaze_row["x_gaze"]),
                        int(gaze_row["y_gaze"]),
                        gaze_target_radius,
                        person_part_distance_scale,
                        person_part_min_conf,
                    )
                    if part_label is not None and part_label != "other":
                        candidates.append((float(hit["area"]), f"person {hit['track_id']}'s {part_label}"))
                elif str(hit["cls"]) in set(selected_object_classes):
                    candidates.append((float(hit["area"]), str(hit["label"])))

        if not candidates:
            return "other"
        _, target = min(candidates, key=lambda item: item[0])
        return target

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

    def _run_ffmpeg(self, command: list[str], purpose: str) -> None:
        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"Failed while {purpose} because ffmpeg was not found.\n"
                f"Expected ffmpeg path: {self.paths.ffmpeg_path}"
            ) from exc
        except subprocess.CalledProcessError as exc:
            return_code = int(exc.returncode)
            unsigned_code = return_code & 0xFFFFFFFF
            fallback_command = self._ffmpeg_path_fallback_command(command)
            if unsigned_code == 0xC0000135 and fallback_command is not None:
                try:
                    subprocess.run(fallback_command, check=True, capture_output=True, text=True)
                    return
                except subprocess.CalledProcessError:
                    pass
            stderr = (exc.stderr or "").strip()
            stdout = (exc.stdout or "").strip()
            details = stderr or stdout or "ffmpeg did not print an error message."
            dll_hint = ""
            if unsigned_code == 0xC0000135:
                dll_hint = (
                    "\n\nOn Windows, exit code 0xC0000135 usually means ffmpeg.exe could not start because "
                    "a required DLL was not found. Use a static Windows ffmpeg build, or place all required "
                    "ffmpeg DLLs next to ffmpeg.exe."
                )
            raise RuntimeError(
                f"ffmpeg failed while {purpose}.\n"
                f"Exit code: {return_code} (0x{unsigned_code:08X})\n"
                f"Command: {' '.join(command)}\n"
                f"Details: {details}"
                f"{dll_hint}"
            ) from exc

    def _ffmpeg_path_fallback_command(self, command: list[str]) -> list[str] | None:
        fallback_ffmpeg = shutil.which("ffmpeg")
        if fallback_ffmpeg is None:
            return None
        if Path(fallback_ffmpeg).resolve() == Path(command[0]).resolve():
            return None
        return [fallback_ffmpeg, *command[1:]]

    def _preferred_h264_encoder(self) -> str | None:
        if str(sys.platform).startswith("darwin") and self._ffmpeg_encoder_available("h264_videotoolbox"):
            return "h264_videotoolbox"
        if self._ffmpeg_encoder_available("h264_nvenc"):
            return "h264_nvenc"
        return None

    def _h264_encoder_args(self, encoder: str) -> list[str]:
        if encoder == "h264_videotoolbox":
            return ["-c:v", "h264_videotoolbox", "-allow_sw", "1", "-pix_fmt", "yuv420p"]
        if encoder == "h264_nvenc":
            return ["-c:v", "h264_nvenc", "-pix_fmt", "yuv420p"]
        return ["-c:v", "libx264", "-pix_fmt", "yuv420p"]

    def _ffmpeg_encoder_available(self, encoder: str) -> bool:
        if self._available_ffmpeg_encoders is None:
            self._available_ffmpeg_encoders = self._detect_ffmpeg_encoders()
        return encoder in self._available_ffmpeg_encoders

    def _detect_ffmpeg_encoders(self) -> set[str]:
        command = [str(self.paths.ffmpeg_path), "-hide_banner", "-encoders"]
        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True)
        except Exception:
            fallback_ffmpeg = shutil.which("ffmpeg")
            if fallback_ffmpeg is None:
                return set()
            try:
                result = subprocess.run(
                    [fallback_ffmpeg, "-hide_banner", "-encoders"],
                    check=True,
                    capture_output=True,
                    text=True,
                )
            except Exception:
                return set()

        encoders: set[str] = set()
        for line in (result.stdout or "").splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[0].startswith("V"):
                encoders.add(parts[1])
        return encoders
