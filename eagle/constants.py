OBJECT_COLUMNS = [
    "frame_idx",
    "cls",
    "track_id",
    "conf",
    "x1",
    "y1",
    "x2",
    "y2",
    "label",
]

GAZE_COLUMNS = [
    "frame_idx",
    "track_id",
    "face_detected",
    "face_conf",
    "face_x1",
    "face_y1",
    "face_x2",
    "face_y2",
    "raw_gaze_detected",
    "raw_inout",
    "raw_x_gaze",
    "raw_y_gaze",
    "gaze_detected",
    "inout",
    "x_gaze",
    "y_gaze",
]

ANNOTATION_COLUMNS = ["tier", "start_time", "end_time", "gaze"]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".wmv", ".webm"}
VISUALIZATION_MODES = {"point", "heatmap", "both"}
