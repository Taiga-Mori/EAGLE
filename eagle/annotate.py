import colorsys
from typing import Any

import cv2
import numpy as np

from .types import FaceDetection, GazePoint


class FrameAnnotator:
    """Draw object, face, gaze, and heatmap overlays."""

    def id_to_color(self, track_id: int) -> tuple[int, int, int]:
        hue = (int(track_id) * 0.61803398875) % 1.0
        r, g, b = colorsys.hsv_to_rgb(hue, 0.9, 0.95)
        return int(b * 255), int(g * 255), int(r * 255)

    def draw_object(self, frame: np.ndarray, detection: dict[str, Any]) -> np.ndarray:
        cls_name = str(detection["cls"])
        track_id = int(detection["track_id"])
        x1, y1, x2, y2 = (int(detection[key]) for key in ["x1", "y1", "x2", "y2"])
        color = self.id_to_color(track_id)
        text = f"{cls_name} {track_id}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        (text_width, text_height), baseline = cv2.getTextSize(text, font, 0.5, 1)
        label_y1 = max(0, y1 - text_height - baseline)
        label_y2 = label_y1 + text_height + baseline
        cv2.rectangle(frame, (x1, label_y1), (x1 + text_width, label_y2), color, -1)
        cv2.putText(frame, text, (x1, label_y2 - baseline), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        return frame

    def draw_face_and_gaze_point(
        self,
        frame: np.ndarray,
        face: FaceDetection | None,
        gaze: GazePoint | None,
        det_thresh: float,
    ) -> np.ndarray:
        if face is None:
            return frame
        color = self.id_to_color(face.track_id)
        cv2.rectangle(frame, (face.x1, face.y1), (face.x2, face.y2), color, 2)
        if gaze is not None and gaze.inout >= det_thresh:
            center_x = (face.x1 + face.x2) // 2
            center_y = (face.y1 + face.y2) // 2
            cv2.circle(frame, (gaze.x_gaze, gaze.y_gaze), 6, color, -1)
            cv2.line(frame, (center_x, center_y), (gaze.x_gaze, gaze.y_gaze), color, 2)
        return frame

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
        if gaze is None or gaze.inout < det_thresh:
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
