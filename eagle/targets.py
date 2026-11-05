"""Decide what a gaze point lands on.

One implementation serves both the labels drawn on the annotated video and the ELAN export, so the
two cannot drift apart.
"""

import math

from .body_parts import resolve_person_part_label


def resolve_gaze_target(
    track_id: object,
    inout: float | None,
    x_gaze: float | None,
    y_gaze: float | None,
    offscreen_direction: str | None,
    frame_faces: list[dict],
    frame_objects: list[dict],
    det_thresh: float,
    gaze_target_radius: int,
    person_part_distance_scale: float,
    person_part_min_conf: float,
    selected_object_classes: set[str],
) -> str:
    """Return the gaze target label of one person in one frame.

    frame_faces holds that frame's faces (track_id, face_x1..face_y2) and frame_objects its tracked
    persons and objects (cls, track_id, x1..y2, label, pose_keypoints). Among everything the gaze point
    hits, the smallest box wins: another person's face, a body part of a person, or a selected object.
    """
    if _missing(inout):
        return "out of frame"
    if float(inout) <= det_thresh:
        return "out of frame" if _missing(offscreen_direction) else str(offscreen_direction)
    if _missing(x_gaze) or _missing(y_gaze):
        return "other"
    x, y = int(x_gaze), int(y_gaze)

    candidates: list[tuple[float, str]] = []
    for face in frame_faces:
        if str(face["track_id"]) == str(track_id):
            continue
        if point_hits_box(x, y, face["face_x1"], face["face_y1"], face["face_x2"], face["face_y2"], gaze_target_radius):
            area = (face["face_x2"] - face["face_x1"]) * (face["face_y2"] - face["face_y1"])
            candidates.append((float(area), f"person {face['track_id']}'s face"))

    for obj in frame_objects:
        if not point_hits_box(x, y, obj["x1"], obj["y1"], obj["x2"], obj["y2"], gaze_target_radius):
            continue
        area = float((obj["x2"] - obj["x1"]) * (obj["y2"] - obj["y1"]))
        if str(obj["cls"]) == "person":
            part_label = resolve_person_part_label(
                obj, x, y, gaze_target_radius, person_part_distance_scale, person_part_min_conf
            )
            if part_label is not None and part_label != "other":
                candidates.append((area, f"person {obj['track_id']}'s {part_label}"))
        elif str(obj["cls"]) in selected_object_classes:
            candidates.append((area, str(obj.get("label", obj["cls"]))))

    if not candidates:
        return "other"
    _, target = min(candidates, key=lambda item: item[0])
    return target


def point_hits_box(x: int, y: int, x1, y1, x2, y2, radius: int) -> bool:
    """Whether the point is inside the box, or within `radius` pixels of it when radius > 0."""
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    if radius <= 0:
        return x1 <= x <= x2 and y1 <= y <= y2
    dx = x - min(max(x, x1), x2)
    dy = y - min(max(y, y1), y2)
    return dx * dx + dy * dy <= radius * radius


def _missing(value) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))
