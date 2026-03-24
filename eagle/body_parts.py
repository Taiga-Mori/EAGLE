import json
import math
from typing import Any

import cv2
import numpy as np


LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_ELBOW = 7
RIGHT_ELBOW = 8
LEFT_WRIST = 9
RIGHT_WRIST = 10
LEFT_HIP = 11
RIGHT_HIP = 12
LEFT_KNEE = 13
RIGHT_KNEE = 14
LEFT_ANKLE = 15
RIGHT_ANKLE = 16

LEFT_ARM_CHAIN = [LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST]
RIGHT_ARM_CHAIN = [RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST]
ARM_HAND_KEYPOINTS = [LEFT_ELBOW, RIGHT_ELBOW, LEFT_WRIST, RIGHT_WRIST]
UPPER_BODY_KEYPOINTS = [LEFT_SHOULDER, RIGHT_SHOULDER]
LOWER_BODY_KEYPOINTS = [LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE, LEFT_ANKLE, RIGHT_ANKLE]


def parse_pose_keypoints(raw_value: Any) -> list[tuple[float, float, float]]:
    """Parse stored pose keypoints into `(x, y, conf)` tuples."""

    if raw_value is None:
        return []
    if isinstance(raw_value, float) and math.isnan(raw_value):
        return []

    payload = raw_value
    if isinstance(raw_value, str):
        if not raw_value.strip():
            return []
        try:
            payload = json.loads(raw_value)
        except Exception:
            return []

    if not isinstance(payload, list):
        return []

    keypoints: list[tuple[float, float, float]] = []
    for item in payload:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            keypoints.append((0.0, 0.0, 0.0))
            continue
        x = float(item[0])
        y = float(item[1])
        conf = float(item[2]) if len(item) >= 3 and item[2] is not None else 1.0
        keypoints.append((x, y, conf))
    return keypoints


def build_person_part_shapes(detection: dict[str, Any], min_conf: float = 0.2) -> dict[str, dict[str, Any]]:
    """Build approximate body-part shapes from COCO pose keypoints."""

    keypoints = parse_pose_keypoints(detection.get("pose_keypoints"))
    if not keypoints:
        return _fallback_part_shapes(detection)

    x1 = int(detection["x1"])
    y1 = int(detection["y1"])
    x2 = int(detection["x2"])
    y2 = int(detection["y2"])
    width = max(x2 - x1, 1)
    height = max(y2 - y1, 1)

    def point(index: int) -> tuple[float, float] | None:
        if index >= len(keypoints):
            return None
        px, py, conf = keypoints[index]
        if conf < min_conf:
            return None
        return (float(px), float(py))

    def extended_palm(chain: list[int], extension_scale: float = 0.45) -> tuple[float, float] | None:
        shoulder = point(chain[0])
        elbow = point(chain[1])
        wrist = point(chain[2])
        if elbow is None or wrist is None:
            return None
        return (
            wrist[0] + (wrist[0] - elbow[0]) * extension_scale,
            wrist[1] + (wrist[1] - elbow[1]) * extension_scale,
        )

    def polygon_area(points: list[tuple[float, float]]) -> float:
        if len(points) < 3:
            return 0.0
        area = 0.0
        for index, current in enumerate(points):
            nxt = points[(index + 1) % len(points)]
            area += current[0] * nxt[1] - nxt[0] * current[1]
        return abs(area) * 0.5

    arm_radius = max(width * 0.045, height * 0.03, 6.0)
    palm_radius = max(width * 0.06, height * 0.04, 8.0)

    arm_segments: list[tuple[tuple[float, float], tuple[float, float], float]] = []
    palm_circles: list[tuple[tuple[float, float], float]] = []
    for chain in (LEFT_ARM_CHAIN, RIGHT_ARM_CHAIN):
        shoulder = point(chain[0])
        elbow = point(chain[1])
        wrist = point(chain[2])
        if shoulder is not None and elbow is not None:
            arm_segments.append((shoulder, elbow, arm_radius))
        if elbow is not None and wrist is not None:
            arm_segments.append((elbow, wrist, arm_radius))
        palm = extended_palm(chain)
        if wrist is not None and palm is not None:
            arm_segments.append((wrist, palm, arm_radius * 0.9))
            palm_circles.append((palm, palm_radius))
            palm_circles.append((wrist, palm_radius * 0.8))

    shapes: dict[str, dict[str, Any]] = {}
    if arm_segments or palm_circles:
        arm_area = sum((_segment_length(start, end) * (2.0 * radius)) for start, end, radius in arm_segments)
        arm_area += sum((math.pi * (radius ** 2)) for _, radius in palm_circles)
        shapes["arm/hand"] = {
            "kind": "compound",
            "segments": arm_segments,
            "circles": palm_circles,
            "area": arm_area,
        }

    left_shoulder = point(LEFT_SHOULDER)
    right_shoulder = point(RIGHT_SHOULDER)
    left_hip = point(LEFT_HIP)
    right_hip = point(RIGHT_HIP)
    upper_polygon = [pt for pt in [left_shoulder, right_shoulder, right_hip, left_hip] if pt is not None]
    if len(upper_polygon) == 4:
        shapes["upper body"] = {
            "kind": "polygon",
            "points": upper_polygon,
            "area": polygon_area(upper_polygon),
        }
    else:
        shapes["upper body"] = _fallback_part_shapes(detection)["upper body"]

    left_knee = point(LEFT_KNEE)
    right_knee = point(RIGHT_KNEE)
    left_ankle = point(LEFT_ANKLE)
    right_ankle = point(RIGHT_ANKLE)
    lower_polygon = [pt for pt in [left_hip, right_hip, right_ankle or right_knee, left_ankle or left_knee] if pt is not None]
    if len(lower_polygon) == 4:
        shapes["lower body"] = {
            "kind": "polygon",
            "points": lower_polygon,
            "area": polygon_area(lower_polygon),
        }
    else:
        shapes["lower body"] = _fallback_part_shapes(detection)["lower body"]

    return shapes


def resolve_person_part_label(
    detection: dict[str, Any],
    x: int,
    y: int,
    extra_radius: int = 0,
    distance_scale: float = 0.22,
    min_conf: float = 0.2,
) -> str | None:
    """Resolve a body-part label from thick skeleton line/point regions inside one person box."""

    x1 = int(detection["x1"])
    y1 = int(detection["y1"])
    x2 = int(detection["x2"])
    y2 = int(detection["y2"])
    if not (x1 <= x <= x2 and y1 <= y <= y2):
        return None

    regions = build_person_attention_regions(detection, extra_radius, distance_scale, min_conf)
    if not regions:
        return _fallback_person_part_label(detection, x, y)
    candidates: list[tuple[float, str]] = []
    for region in regions:
        distance = _distance_to_region((float(x), float(y)), region)
        if distance <= float(region["radius"]):
            candidates.append((distance, str(region["label"])))
    if not candidates:
        return "other"
    return min(candidates, key=lambda item: item[0])[1]


def person_part_distance_threshold(
    detection: dict[str, Any],
    extra_radius: int = 0,
    distance_scale: float = 0.22,
) -> float:
    """Return the keypoint-distance threshold used for body-part assignment."""

    x1 = int(detection["x1"])
    y1 = int(detection["y1"])
    x2 = int(detection["x2"])
    y2 = int(detection["y2"])
    width = max(x2 - x1, 1)
    height = max(y2 - y1, 1)
    return max(24.0, min(width, height) * float(distance_scale)) + float(max(extra_radius, 0))


def build_person_attention_regions(
    detection: dict[str, Any],
    extra_radius: int = 0,
    distance_scale: float = 0.22,
    min_conf: float = 0.2,
) -> list[dict[str, Any]]:
    """Return thick line/circle regions used for body-part assignment."""

    keypoints = parse_pose_keypoints(detection.get("pose_keypoints"))
    if not keypoints:
        return []

    radius = int(round(person_part_distance_threshold(detection, extra_radius, distance_scale)))
    regions: list[dict[str, Any]] = []

    def point(index: int) -> tuple[float, float] | None:
        if index >= len(keypoints):
            return None
        px, py, conf = keypoints[index]
        if conf < min_conf:
            return None
        return (float(px), float(py))

    def add_point(label: str, pt: tuple[float, float] | None) -> None:
        if pt is None:
            return
        regions.append({"label": label, "kind": "point", "center": pt, "radius": radius})

    def add_segment(label: str, start: tuple[float, float] | None, end: tuple[float, float] | None) -> None:
        if start is None or end is None:
            return
        regions.append({"label": label, "kind": "segment", "start": start, "end": end, "radius": radius})

    left_shoulder = point(LEFT_SHOULDER)
    right_shoulder = point(RIGHT_SHOULDER)
    left_elbow = point(LEFT_ELBOW)
    right_elbow = point(RIGHT_ELBOW)
    left_wrist = point(LEFT_WRIST)
    right_wrist = point(RIGHT_WRIST)
    left_hip = point(LEFT_HIP)
    right_hip = point(RIGHT_HIP)
    left_knee = point(LEFT_KNEE)
    right_knee = point(RIGHT_KNEE)
    left_ankle = point(LEFT_ANKLE)
    right_ankle = point(RIGHT_ANKLE)

    add_segment("arm/hand", left_shoulder, left_elbow)
    add_segment("arm/hand", left_elbow, left_wrist)
    add_segment("arm/hand", right_shoulder, right_elbow)
    add_segment("arm/hand", right_elbow, right_wrist)
    add_point("arm/hand", left_elbow)
    add_point("arm/hand", left_wrist)
    add_point("arm/hand", right_elbow)
    add_point("arm/hand", right_wrist)

    add_segment("upper body", left_shoulder, right_shoulder)
    add_segment("upper body", left_shoulder, left_hip)
    add_segment("upper body", right_shoulder, right_hip)
    add_segment("upper body", left_hip, right_hip)
    add_point("upper body", left_shoulder)
    add_point("upper body", right_shoulder)

    add_segment("lower body", left_hip, left_knee)
    add_segment("lower body", left_knee, left_ankle)
    add_segment("lower body", right_hip, right_knee)
    add_segment("lower body", right_knee, right_ankle)
    add_segment("lower body", left_hip, right_hip)
    add_point("lower body", left_hip)
    add_point("lower body", right_hip)
    add_point("lower body", left_knee)
    add_point("lower body", right_knee)
    add_point("lower body", left_ankle)
    add_point("lower body", right_ankle)
    return regions


def point_hits_part_shape(x: int, y: int, shape: dict[str, Any], extra_radius: int = 0) -> bool:
    """Return whether a point or point-radius overlaps a part shape."""

    kind = str(shape.get("kind", ""))
    if kind == "polygon":
        contour = np.array(shape.get("points", []), dtype=np.float32)
        if len(contour) < 3:
            return False
        distance = cv2.pointPolygonTest(contour, (float(x), float(y)), True)
        return distance >= -float(extra_radius)

    if kind == "compound":
        for start, end, radius in shape.get("segments", []):
            if _distance_to_segment((float(x), float(y)), start, end) <= float(radius) + float(extra_radius):
                return True
        for center, radius in shape.get("circles", []):
            if math.hypot(float(x) - center[0], float(y) - center[1]) <= float(radius) + float(extra_radius):
                return True
        return False

    return False


def part_shape_area(shape: dict[str, Any]) -> float:
    """Return the comparison area for one part shape."""

    return float(shape.get("area", 0.0))


def _fallback_person_part_label(detection: dict[str, Any], x: int, y: int) -> str:
    x1 = int(detection["x1"])
    y1 = int(detection["y1"])
    x2 = int(detection["x2"])
    y2 = int(detection["y2"])
    width = max(x2 - x1, 1)
    height = max(y2 - y1, 1)
    left_band = x1 + width * 0.22
    right_band = x2 - width * 0.22
    lower_start = y1 + height * 0.55
    if y >= lower_start:
        return "lower body"
    if x <= left_band or x >= right_band:
        return "arm/hand"
    return "upper body"


def _fallback_part_shapes(detection: dict[str, Any]) -> dict[str, dict[str, Any]]:
    x1 = int(detection["x1"])
    y1 = int(detection["y1"])
    x2 = int(detection["x2"])
    y2 = int(detection["y2"])
    width = max(x2 - x1, 1)
    height = max(y2 - y1, 1)
    mid_y = y1 + height * 0.5
    upper_polygon = [(x1, y1), (x2, y1), (x2, mid_y), (x1, mid_y)]
    lower_polygon = [(x1, mid_y), (x2, mid_y), (x2, y2), (x1, y2)]
    arm_band_y1 = y1 + height * 0.2
    arm_band_y2 = y1 + height * 0.62
    return {
        "arm/hand": {
            "kind": "compound",
            "segments": [
                ((x1, arm_band_y1), (x1, arm_band_y2), max(width * 0.08, 8.0)),
                ((x2, arm_band_y1), (x2, arm_band_y2), max(width * 0.08, 8.0)),
            ],
            "circles": [],
            "area": float(height * max(width * 0.16, 16.0) * 2.0),
        },
        "upper body": {
            "kind": "polygon",
            "points": upper_polygon,
            "area": float((x2 - x1) * (mid_y - y1)),
        },
        "lower body": {
            "kind": "polygon",
            "points": lower_polygon,
            "area": float((x2 - x1) * (y2 - mid_y)),
        },
    }


def _segment_length(start: tuple[float, float], end: tuple[float, float]) -> float:
    return math.hypot(end[0] - start[0], end[1] - start[1])


def _distance_to_segment(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> float:
    px, py = point
    x1, y1 = start
    x2, y2 = end
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return math.hypot(px - x1, py - y1)
    t = ((px - x1) * dx + (py - y1) * dy) / float((dx * dx) + (dy * dy))
    t = min(1.0, max(0.0, t))
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    return math.hypot(px - proj_x, py - proj_y)


def _distance_to_region(point: tuple[float, float], region: dict[str, Any]) -> float:
    kind = str(region.get("kind", ""))
    if kind == "point":
        center = region["center"]
        return math.hypot(point[0] - center[0], point[1] - center[1])
    if kind == "segment":
        return _distance_to_segment(point, region["start"], region["end"])
    return float("inf")
