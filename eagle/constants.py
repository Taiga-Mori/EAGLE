OBJECT_COLUMNS = [
    "frame_idx",
    "cls",
    "track_id",
    "object_detected",
    "source",
    "conf",
    "x1",
    "y1",
    "x2",
    "y2",
    "pose_keypoints",
    "label",
]

COCO_OBJECT_CLASSES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]

FACE_COLUMNS = [
    "frame_idx",
    "track_id",
    "face_detected",
    "face_conf",
    "face_x1",
    "face_y1",
    "face_x2",
    "face_y2",
]

GAZE_COLUMNS = [
    "frame_idx",
    "track_id",
    "gaze_detected",
    "inout",
    "x_gaze",
    "y_gaze",
    "offscreen_direction",
    "offscreen_yaw",
    "offscreen_pitch",
]

ANNOTATION_COLUMNS = ["tier", "start_time", "end_time", "gaze"]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".wmv", ".webm"}
VISUALIZATION_MODES = {"point", "heatmap", "both"}
GAZE_POINT_METHODS = {"argmax", "center_of_mass", "peak_centroid", "peak_region_centroid", "soft_argmax"}
FACE_DETECTION_BACKENDS = {"mediapipe", "retinaface"}
DEFAULT_FACE_DETECTION_BACKEND = "retinaface"
PERSON_DETECTION_BACKENDS = {
    "yolo26x-pose": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26x-pose.pt",
}
DEFAULT_PERSON_DETECTION_BACKEND = "yolo26x-pose"
YOLO_OBJECT_MODELS = {
    "yolo26n": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt",
    "yolo26s": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s.pt",
    "yolo26m": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt",
    "yolo26l": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26l.pt",
    "yolo26x": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26x.pt",
}
OBJECT_DETECTION_BACKENDS = YOLO_OBJECT_MODELS
DEFAULT_OBJECT_DETECTION_BACKEND = "yolo26x"
GAZE_DETECTION_BACKENDS = {"gazelle"}
DEFAULT_GAZE_DETECTION_BACKEND = "gazelle"
HEAD_POSE_DETECTION_BACKENDS = {"l2cs", "mobileone"}
DEFAULT_HEAD_POSE_DETECTION_BACKEND = "l2cs"


def discover_custom_gaze_models(custom_models_dir) -> dict[str, str]:
    """Discover custom GazeLLE .pt files in ~/.EAGLE/custom/gaze/"""
    if not custom_models_dir.exists():
        return {}
    models = {}
    for pt_file in custom_models_dir.glob("*.pt"):
        model_name = pt_file.stem
        models[model_name] = str(pt_file)
    return models


def discover_custom_person_models(custom_models_dir) -> dict[str, str]:
    """Discover custom person detection .pt files in ~/.EAGLE/custom/person/"""
    if not custom_models_dir.exists():
        return {}
    models = {}
    for pt_file in custom_models_dir.glob("*.pt"):
        model_name = pt_file.stem
        models[model_name] = str(pt_file)
    return models


def discover_custom_object_models(custom_models_dir) -> dict[str, str]:
    """Discover custom object detection .pt files in ~/.EAGLE/custom/object/"""
    if not custom_models_dir.exists():
        return {}
    models = {}
    for pt_file in custom_models_dir.glob("*.pt"):
        model_name = pt_file.stem
        models[model_name] = str(pt_file)
    return models


def discover_custom_face_models(custom_models_dir) -> dict[str, str]:
    """Discover custom face detection .pt files in ~/.EAGLE/custom/face/"""
    if not custom_models_dir.exists():
        return {}
    models = {}
    for pt_file in custom_models_dir.glob("*.pt"):
        model_name = pt_file.stem
        models[model_name] = str(pt_file)
    return models


def discover_custom_headpose_models(custom_models_dir) -> dict[str, str]:
    """Discover custom head pose detection .pt files in ~/.EAGLE/custom/headpose/"""
    if not custom_models_dir.exists():
        return {}
    models = {}
    for pt_file in custom_models_dir.glob("*.pt"):
        model_name = pt_file.stem
        models[model_name] = str(pt_file)
    return models
