import colorsys
import hashlib
import math
from typing import Any

import cv2
import numpy as np

from .body_parts import build_person_attention_regions
from .types import FaceDetection, GazePoint


class FrameAnnotator:
    """Draw object, face, gaze, and heatmap overlays."""

    def id_to_color(self, track_id: str) -> tuple[int, int, int]:
        digest = hashlib.sha1(str(track_id).encode("utf-8")).digest()
        hue_seed = int.from_bytes(digest[:8], "big") / float(1 << 64)
        hue = (hue_seed * 0.61803398875) % 1.0
        r, g, b = colorsys.hsv_to_rgb(hue, 0.9, 0.95)
        return int(b * 255), int(g * 255), int(r * 255)

    def draw_object(self, frame: np.ndarray, detection: dict[str, Any]) -> np.ndarray:
        cls_name = str(detection["cls"])
        track_id = str(detection["track_id"])
        x1, y1, x2, y2 = (int(detection[key]) for key in ["x1", "y1", "x2", "y2"])
        color = self.id_to_color(track_id)
        text = str(detection.get("label") or f"{cls_name} {track_id}")
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        self._draw_label(frame, text, x1, y1, y2, color, line_index=0)
        return frame

    def draw_person_keypoints(
        self,
        frame: np.ndarray,
        detection: dict[str, Any],
        person_part_distance_scale: float = 0.22,
        person_part_min_conf: float = 0.0,
    ) -> np.ndarray:
        if str(detection.get("cls", "")) != "person":
            return frame
        color = self.id_to_color(str(detection["track_id"]))
        overlay = frame.copy()
        for region in build_person_attention_regions(
            detection,
            distance_scale=person_part_distance_scale,
            min_conf=person_part_min_conf,
        ):
            if region["kind"] == "point":
                cv2.circle(
                    overlay,
                    (int(round(region["center"][0])), int(round(region["center"][1]))),
                    int(region["radius"]),
                    color,
                    -1,
                )
            elif region["kind"] == "segment":
                cv2.line(
                    overlay,
                    (int(round(region["start"][0])), int(round(region["start"][1]))),
                    (int(round(region["end"][0])), int(round(region["end"][1]))),
                    color,
                    max(1, int(region["radius"] * 2)),
                )
        cv2.addWeighted(overlay, 0.12, frame, 0.88, 0.0, dst=frame)
        return frame

    def draw_face_and_gaze_point(
        self,
        frame: np.ndarray,
        face: FaceDetection | None,
        gaze: GazePoint | None,
        det_thresh: float,
        gaze_target_radius: int = 0,
        offscreen_angles: tuple[float, float] | None = None,
    ) -> np.ndarray:
        if face is None:
            return frame
        color = self.id_to_color(face.track_id)
        cv2.rectangle(frame, (face.x1, face.y1), (face.x2, face.y2), color, 2)
        center_x = (face.x1 + face.x2) // 2
        center_y = (face.y1 + face.y2) // 2
        if gaze is not None and gaze.inout >= det_thresh:
            if gaze_target_radius > 0:
                cv2.circle(frame, (gaze.x_gaze, gaze.y_gaze), gaze_target_radius, color, 2)
                cv2.circle(frame, (gaze.x_gaze, gaze.y_gaze), 4, color, -1)
            else:
                cv2.circle(frame, (gaze.x_gaze, gaze.y_gaze), 6, color, -1)
            cv2.line(frame, (center_x, center_y), (gaze.x_gaze, gaze.y_gaze), color, 2)
        elif offscreen_angles is not None:
            endpoint = self._line_endpoint_from_angles(
                frame.shape[1],
                frame.shape[0],
                center_x,
                center_y,
                offscreen_angles[0],
                offscreen_angles[1],
            )
            if endpoint is not None:
                cv2.line(frame, (center_x, center_y), endpoint, color, 2)
        return frame

    def _line_endpoint_from_angles(
        self,
        frame_width: int,
        frame_height: int,
        center_x: int,
        center_y: int,
        yaw_deg: float,
        pitch_deg: float,
    ) -> tuple[int, int] | None:
        yaw_rad = math.radians(yaw_deg)
        pitch_rad = math.radians(pitch_deg)
        dx = -math.sin(yaw_rad) * math.cos(pitch_rad)
        dy = -math.sin(pitch_rad)
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            return None

        candidates: list[float] = []
        if dx > 1e-6:
            candidates.append((frame_width - 1 - center_x) / dx)
        elif dx < -1e-6:
            candidates.append((0 - center_x) / dx)
        if dy > 1e-6:
            candidates.append((frame_height - 1 - center_y) / dy)
        elif dy < -1e-6:
            candidates.append((0 - center_y) / dy)

        positive = [value for value in candidates if value > 0]
        if not positive:
            return None
        scale = min(positive)
        end_x = int(round(center_x + dx * scale))
        end_y = int(round(center_y + dy * scale))
        end_x = max(0, min(frame_width - 1, end_x))
        end_y = max(0, min(frame_height - 1, end_y))
        return end_x, end_y

    def draw_face_and_heatmap(
        self,
        frame: np.ndarray,
        face: FaceDetection | None,
        gaze: GazePoint | None,
        det_thresh: float,
        alpha: float,
    ) -> np.ndarray:
        if face is None:
            return frame
        color = self.id_to_color(face.track_id)
        cv2.rectangle(frame, (face.x1, face.y1), (face.x2, face.y2), color, 2)
        if gaze is None:
            return frame
        normalized = gaze.heatmap.astype(np.float32)
        max_value = float(normalized.max())
        if max_value > 0:
            normalized = normalized / max_value
        heatmap_uint8 = np.uint8(np.clip(normalized * 255, 0, 255))
        heatmap_uint8 = cv2.resize(heatmap_uint8, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_CUBIC)
        colored_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        blended = cv2.addWeighted(colored_heatmap, alpha, frame, 1.0 - alpha, 0.0)
        cv2.rectangle(blended, (face.x1, face.y1), (face.x2, face.y2), color, 2)
        return blended

    def draw_gaze_target_label(
        self,
        frame: np.ndarray,
        track_id: str,
        face: FaceDetection | None,
        target_label: str,
    ) -> np.ndarray:
        if face is None:
            return frame
        color = self.id_to_color(track_id)
        self._draw_text_only_label(frame, target_label, face.x1, face.y1, face.y2, color, line_index=0)
        return frame

    def _draw_label(
        self,
        frame: np.ndarray,
        text: str,
        x1: int,
        y1: int,
        y2: int,
        color: tuple[int, int, int],
        line_index: int = 0,
    ) -> None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        line_height = text_height + baseline + 4
        top_y2 = y1 - line_index * line_height
        top_y1 = top_y2 - text_height - baseline
        if top_y1 >= 0:
            label_y1 = top_y1
            label_y2 = top_y2
        else:
            label_y1 = min(frame.shape[0] - text_height - baseline, y2 + line_index * line_height)
            label_y2 = label_y1 + text_height + baseline
        cv2.rectangle(frame, (x1, label_y1), (x1 + text_width, label_y2), color, -1)
        cv2.putText(
            frame,
            text,
            (x1, label_y2 - baseline),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    def _draw_text_only_label(
        self,
        frame: np.ndarray,
        text: str,
        x1: int,
        y1: int,
        y2: int,
        color: tuple[int, int, int],
        line_index: int = 0,
    ) -> None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 2
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        vertical_padding = 2
        line_height = text_height + baseline + vertical_padding
        top_y2 = y1 - vertical_padding - line_index * line_height
        top_y1 = top_y2 - text_height - baseline
        if top_y1 >= 0:
            text_y = top_y2 - baseline
        else:
            bottom_y1 = min(
                frame.shape[0] - text_height - baseline,
                y2 + vertical_padding + line_index * line_height,
            )
            text_y = bottom_y1 + text_height
        cv2.putText(
            frame,
            text,
            (x1, text_y),
            font,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )
