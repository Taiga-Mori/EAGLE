from pathlib import Path
import base64
from functools import lru_cache
import os
import errno
import signal
import socket
import subprocess
import sys
import time
import webbrowser

import cv2
import yaml

from eagle import EAGLE
from eagle.constants import (
    COCO_OBJECT_CLASSES,
    DEFAULT_FACE_DETECTION_BACKEND,
    DEFAULT_GAZE_DETECTION_BACKEND,
    DEFAULT_HEAD_POSE_DETECTION_BACKEND,
    DEFAULT_OBJECT_DETECTION_BACKEND,
    DEFAULT_PERSON_DETECTION_BACKEND,
    FACE_DETECTION_BACKENDS,
    GAZE_DETECTION_BACKENDS,
    HEAD_POSE_DETECTION_BACKENDS,
    OBJECT_DETECTION_BACKENDS,
    PERSON_DETECTION_BACKENDS,
)


APP_PATH = Path(__file__).resolve()
RUNTIME_ROOT = Path(sys._MEIPASS) if hasattr(sys, "_MEIPASS") else APP_PATH.parent
APP_DIR = APP_PATH.parent
APP_ICON_PATH = RUNTIME_ROOT / "assets" / "icon_trans.png"
STREAMLIT_CHILD_ENV = "EAGLE_STREAMLIT_CHILD"
STREAMLIT_PORT_ENV = "EAGLE_STREAMLIT_PORT"
MEDIA_FILE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
    ".webm",
    ".m4v",
}


def raise_file_descriptor_limit(min_soft_limit: int = 4096) -> None:
    if sys.platform == "win32":
        return
    try:
        import resource

        soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
        target_soft_limit = max(int(soft_limit), int(min_soft_limit))
        if hard_limit != resource.RLIM_INFINITY:
            target_soft_limit = min(target_soft_limit, int(hard_limit))
        if target_soft_limit > soft_limit:
            resource.setrlimit(resource.RLIMIT_NOFILE, (target_soft_limit, hard_limit))
    except Exception:
        # Keep startup resilient if the platform denies raising the limit.
        pass


def terminate_current_process() -> None:
    try:
        os.kill(os.getpid(), signal.SIGTERM)
    except Exception:
        os._exit(0)


def detect_media_type(annotator: EAGLE, input_path: Path | None) -> str | None:
    if input_path is None:
        return None
    try:
        return annotator.config_manager.detect_media_type(input_path)
    except ValueError:
        return None


def resolve_video_fps(input_path: Path) -> float:
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open input: {input_path}")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
    finally:
        capture.release()
    if fps <= 0:
        raise RuntimeError(f"Invalid FPS metadata for {input_path}")
    return fps


def output_paths_to_lines(output_paths) -> list[str]:
    if isinstance(output_paths, list):
        return [str(path) for path in output_paths]
    return [str(output_paths)]


def load_botsort_defaults(annotator: EAGLE) -> dict:
    with annotator.paths.botsort_template_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def normalize_selected_classes(selected_classes: list[str] | None) -> list[str]:
    if not selected_classes:
        return list(COCO_OBJECT_CLASSES)
    return [cls_name for cls_name in COCO_OBJECT_CLASSES if cls_name in selected_classes]


def get_streamlit():
    import streamlit as st

    return st


def reset_to_start(st) -> None:
    st.session_state.state = "start"


def browse_input_file() -> tuple[Path | None, str | None]:
    if sys.platform == "win32":
        try:
            import tkinter as tk
            from tkinter import filedialog

            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            selected = filedialog.askopenfilename(
                title="Select input image or video",
                filetypes=[
                    (
                        "Media files",
                        "*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp *.mp4 *.mov *.avi *.mkv *.webm *.m4v",
                    ),
                    ("All files", "*.*"),
                ],
            )
            root.destroy()
        except Exception as exc:
            return None, str(exc)
        if not selected:
            return None, "No file was selected."
        return Path(selected), None

    if sys.platform != "darwin":
        return None, "Native file browsing is only available on macOS and Windows."
    try:
        result = subprocess.run(
            [
                "osascript",
                "-e",
                (
                    'POSIX path of (choose file with prompt "Select input image or video" '
                    'of type {"public.image", "public.movie", "public.video"})'
                ),
            ],
            check=True,
            capture_output=True,
            text=True,
            close_fds=True,
        )
    except subprocess.CalledProcessError as exc:
        error_text = (exc.stderr or exc.stdout or str(exc)).strip()
        return None, error_text or "The native file dialog could not be opened."
    except OSError as exc:
        if exc.errno == errno.EMFILE:
            return (
                None,
                "Too many open files. The packaged app hit the macOS file descriptor limit while opening the native file dialog.",
            )
        return None, str(exc)
    except Exception as exc:
        return None, str(exc)
    selected = result.stdout.strip()
    if not selected:
        return None, "No file was selected."
    return Path(selected), None


def default_output_dir(input_path: Path | None) -> Path:
    if input_path is None:
        return APP_DIR / "output"
    if os.access(input_path.parent, os.W_OK):
        return input_path.parent / input_path.stem
    return APP_DIR / "output" / input_path.stem


def render_header(st) -> None:
    if APP_ICON_PATH.exists():
        col1, col2 = st.columns([2, 5], vertical_alignment="center")
        with col1:
            encoded = load_icon_base64()
            st.markdown(
                (
                    "<div style='padding-top: 0.25rem;'>"
                    f"<img src='data:image/png;base64,{encoded}' "
                    "style='width: 220px; max-width: 100%; height: auto; image-rendering: -webkit-optimize-contrast;' />"
                    "</div>"
                ),
                unsafe_allow_html=True,
            )
        with col2:
            st.title("EAGLE")
            st.caption("End-to-end Automatic Gaze LabEling tool for interaction studies")
        return
    st.title("EAGLE")
    st.caption("End-to-end Automatic Gaze LabEling tool for interaction studies")


def render_quit_button(st) -> None:
    col_left, col_right = st.columns([5, 1])
    with col_right:
        if st.button("Quit EAGLE", use_container_width=True):
            terminate_current_process()


@lru_cache(maxsize=1)
def load_icon_base64() -> str:
    return base64.b64encode(APP_ICON_PATH.read_bytes()).decode("ascii")


def is_running_under_streamlit() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
    except Exception:
        return False
    return get_script_run_ctx() is not None


def resolve_streamlit_script_path() -> Path:
    candidate = RUNTIME_ROOT / "app.py"
    if candidate.exists():
        return candidate
    return APP_PATH


def launch_streamlit_server(port: int) -> int:
    import streamlit.web.cli as stcli

    sys.argv = [
        "streamlit",
        "run",
        str(resolve_streamlit_script_path()),
        "--server.port",
        str(port),
        "--server.headless=true",
        "--global.developmentMode=false",
    ]
    return stcli.main()


def find_available_port(start_port: int = 8501, max_tries: int = 20) -> int:
    for port in range(start_port, start_port + max_tries):
        if not _is_port_available(port):
            continue
        return port
    raise RuntimeError(f"Could not find an available port in {start_port}-{start_port + max_tries - 1}")


def _is_port_available(port: int) -> bool:
    checks = [
        (socket.AF_INET, ("127.0.0.1", port)),
        (socket.AF_INET, ("0.0.0.0", port)),
        (socket.AF_INET6, ("::1", port)),
        (socket.AF_INET6, ("::", port)),
    ]
    for family, address in checks:
        try:
            with socket.socket(family, socket.SOCK_STREAM) as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind(address)
        except OSError:
            return False
    return True


def wait_for_port(port: int, timeout_seconds: float = 15.0) -> bool:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return True
        except OSError:
            time.sleep(0.2)
    return False


def launch_frozen_app_server(port: int) -> None:
    env = os.environ.copy()
    env[STREAMLIT_CHILD_ENV] = "1"
    env[STREAMLIT_PORT_ENV] = str(port)
    child = subprocess.Popen(
        [sys.executable],
        cwd=str(RUNTIME_ROOT),
        env=env,
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    _keep_launcher_alive(child, port)


def _keep_launcher_alive(child: subprocess.Popen, port: int) -> None:
    def terminate_child(*_args) -> None:
        if child.poll() is None:
            child.terminate()
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, terminate_child)
    signal.signal(signal.SIGINT, terminate_child)

    if wait_for_port(port):
        url = f"http://localhost:{port}"
        try:
            if sys.platform == "darwin":
                subprocess.Popen(["open", url])
            else:
                webbrowser.open(url)
        except Exception:
            webbrowser.open(url)

    try:
        while child.poll() is None:
            time.sleep(0.5)
    finally:
        if child.poll() is None:
            child.terminate()


def main() -> None:
    raise_file_descriptor_limit()
    st = get_streamlit()
    st.set_page_config(page_title="EAGLE", layout="centered")
    render_header(st)
    render_quit_button(st)

    if "annotator" not in st.session_state:
        st.session_state.annotator = EAGLE()
    if "state" not in st.session_state:
        st.session_state.state = "start"
    if "results" not in st.session_state:
        st.session_state.results = None
    if "error_message" not in st.session_state:
        st.session_state.error_message = None
    if "selected_input_path" not in st.session_state:
        st.session_state.selected_input_path = ""
    if "browse_error_message" not in st.session_state:
        st.session_state.browse_error_message = None

    annotator: EAGLE = st.session_state.annotator
    botsort_defaults = load_botsort_defaults(annotator)

    if st.session_state.state == "start":
        st.subheader("Basic Settings")
        col_input_path, col_browse = st.columns([5, 1])
        with col_input_path:
            input_path_str = st.text_input(
                "Input file",
                value=st.session_state.selected_input_path,
                disabled=False,
            )
        with col_browse:
            st.write("")
            st.write("")
            if st.button("Browse", use_container_width=True):
                selected_path, browse_error = browse_input_file()
                if selected_path is not None:
                    st.session_state.selected_input_path = str(selected_path)
                    st.session_state.browse_error_message = None
                    st.rerun()
                if browse_error and browse_error != "No file was selected.":
                    st.session_state.browse_error_message = browse_error
        browse_error_message = st.session_state.get("browse_error_message")
        if browse_error_message:
            st.warning(
                "Native Browse did not complete. "
                f"You can still paste a path or use the built-in file browser below.\n\nDetails: {browse_error_message}"
            )

        if sys.platform not in {"darwin", "win32"}:
            st.caption("Linux containers do not support the native Browse dialog. Enter a mounted path manually.")

        input_path = Path(input_path_str) if input_path_str else None
        output_dir_default = default_output_dir(input_path)
        output_parent_dir = output_dir_default.parent
        output_name = st.text_input(
            "Output folder name",
            value=str(st.session_state.get("output_name", output_dir_default.name)),
        ).strip()
        output_dir = output_parent_dir / (output_name or output_dir_default.name)
        media_type = detect_media_type(annotator, input_path)

        selected_device = st.session_state.get("device", annotator.device_options[0])
        if selected_device == "cuda" and "cuda:0" in annotator.device_options:
            selected_device = "cuda:0"
        if selected_device not in annotator.device_options:
            selected_device = annotator.device_options[0]

        device = st.selectbox(
            "Device",
            options=annotator.device_options,
            index=annotator.device_options.index(selected_device),
        )

        if input_path is None:
            st.info("Choose an input file to begin.")
        elif media_type is None:
            st.warning("Supported inputs are common image and video files.")
        elif media_type is not None:
            st.info(f"Detected input type: {media_type}")
            st.caption(f"Input: `{input_path}`")
            st.caption(f"Output parent: `{output_parent_dir}`")
            st.caption(f"Final output directory: `{output_dir}`")

        with st.expander("Detailed Settings", expanded=False):
            video_fps = None
            if media_type == "video" and input_path is not None:
                try:
                    video_fps = resolve_video_fps(input_path)
                except Exception:
                    video_fps = None

            def fps_input(label: str, state_key: str) -> float | None:
                if video_fps is None:
                    return None
                raw_value = st.session_state.get(state_key)
                saved_value = float(video_fps if raw_value is None else raw_value)
                saved_value = min(max(saved_value, 0.1), video_fps)
                return float(
                    st.number_input(
                        label,
                        min_value=0.1,
                        max_value=float(video_fps),
                        step=1.0,
                        value=saved_value,
                    )
                )

            st.subheader("Pipeline Settings")
            person_det_thresh = float(st.session_state.get("person_det_thresh", 0.5))
            object_det_thresh = float(st.session_state.get("object_det_thresh", 0.5))
            face_det_thresh = float(st.session_state.get("face_det_thresh", 0.5))
            gaze_det_thresh = float(st.session_state.get("gaze_det_thresh", 0.5))

            person_detection_backend = DEFAULT_PERSON_DETECTION_BACKEND
            object_detection_backend = DEFAULT_OBJECT_DETECTION_BACKEND
            face_detection_backend = DEFAULT_FACE_DETECTION_BACKEND
            gaze_detection_backend = DEFAULT_GAZE_DETECTION_BACKEND
            head_pose_detection_backend = DEFAULT_HEAD_POSE_DETECTION_BACKEND
            person_target_fps = video_fps
            object_target_fps = video_fps
            face_target_fps = video_fps
            gaze_target_fps = video_fps
            head_pose_target_fps = video_fps

            person_smoothing_window = int(st.session_state.get("person_smoothing_window", 5))
            object_smoothing_window = int(st.session_state.get("object_smoothing_window", 5))
            face_smoothing_window = int(st.session_state.get("face_smoothing_window", 5))
            gaze_smoothing_window = int(st.session_state.get("gaze_smoothing_window", 5))
            person_part_distance_scale = float(st.session_state.get("person_part_distance_scale", 0.10))
            person_part_min_conf = float(st.session_state.get("person_part_min_conf", 0.0))
            selected_object_classes = normalize_selected_classes(
                st.session_state.get("selected_object_classes", list(COCO_OBJECT_CLASSES))
            )

            person_tab, object_tab, face_tab, gaze_tab = st.tabs(["Person", "Object", "Face", "Gaze"])
            with person_tab:
                person_detection_options = list(PERSON_DETECTION_BACKENDS)
                saved_person_detection_backend = st.session_state.get(
                    "person_detection_backend",
                    DEFAULT_PERSON_DETECTION_BACKEND,
                )
                if saved_person_detection_backend not in PERSON_DETECTION_BACKENDS:
                    saved_person_detection_backend = DEFAULT_PERSON_DETECTION_BACKEND
                person_detection_backend = st.selectbox(
                    "Person detection backend",
                    options=person_detection_options,
                    index=person_detection_options.index(saved_person_detection_backend),
                )
                person_det_thresh = float(
                    st.slider("Person threshold", 0.0, 1.0, person_det_thresh, 0.05)
                )
                if media_type == "video":
                    person_target_fps = fps_input("Person frame FPS", "person_target_fps")
                person_smoothing_window = int(
                    st.number_input(
                        "Person smoothing window",
                        min_value=1,
                        step=1,
                        value=int(st.session_state.get("person_smoothing_window", 5)),
                    )
                )
                person_part_distance_scale = float(
                    st.number_input(
                        "Person part distance scale",
                        min_value=0.01,
                        step=0.01,
                        value=float(st.session_state.get("person_part_distance_scale", 0.10)),
                        help="Scales how far from each keypoint a gaze can be and still count as that body part.",
                    )
                )
                person_part_min_conf = float(
                    st.slider(
                        "Person part keypoint min confidence",
                        min_value=0.0,
                        max_value=1.0,
                        value=float(st.session_state.get("person_part_min_conf", 0.0)),
                        step=0.05,
                    )
                )
                reuse_cached_persons = st.checkbox(
                    "Reuse existing persons.csv when available",
                    value=bool(st.session_state.get("reuse_cached_persons", False)),
                )

                st.subheader("BoT-SORT")
                col5, col6 = st.columns(2)
                with col5:
                    track_high_thresh = float(
                        st.slider(
                            "track_high_thresh",
                            min_value=0.0,
                            max_value=1.0,
                            value=float(
                                st.session_state.get(
                                    "track_high_thresh",
                                    botsort_defaults.get("track_high_thresh", 0.25),
                                )
                            ),
                            step=0.05,
                        )
                    )
                    new_track_thresh = float(
                        st.slider(
                            "new_track_thresh",
                            min_value=0.0,
                            max_value=1.0,
                            value=float(
                                st.session_state.get(
                                    "new_track_thresh",
                                    botsort_defaults.get("new_track_thresh", 0.25),
                                )
                            ),
                            step=0.05,
                        )
                    )
                    match_thresh = float(
                        st.slider(
                            "match_thresh",
                            min_value=0.0,
                            max_value=1.0,
                            value=float(st.session_state.get("match_thresh", botsort_defaults.get("match_thresh", 0.8))),
                            step=0.05,
                        )
                    )
                with col6:
                    track_low_thresh = float(
                        st.slider(
                            "track_low_thresh",
                            min_value=0.0,
                            max_value=1.0,
                            value=float(st.session_state.get("track_low_thresh", botsort_defaults.get("track_low_thresh", 0.1))),
                            step=0.05,
                        )
                    )
                    track_buffer = int(
                        st.number_input(
                            "track_buffer",
                            min_value=1,
                            step=1,
                            value=int(st.session_state.get("track_buffer", botsort_defaults.get("track_buffer", 30))),
                        )
                    )
                    with_reid = st.checkbox(
                        "Enable ReID",
                        value=bool(st.session_state.get("with_reid", botsort_defaults.get("with_reid", False))),
                    )

                proximity_thresh = float(
                    st.slider(
                        "proximity_thresh",
                        min_value=0.0,
                        max_value=1.0,
                        value=float(st.session_state.get("proximity_thresh", botsort_defaults.get("proximity_thresh", 0.5))),
                        step=0.05,
                    )
                )
                appearance_thresh = float(
                    st.slider(
                        "appearance_thresh",
                        min_value=0.0,
                        max_value=1.0,
                        value=float(st.session_state.get("appearance_thresh", botsort_defaults.get("appearance_thresh", 0.8))),
                        step=0.05,
                    )
                )

            with object_tab:
                object_detection_options = list(OBJECT_DETECTION_BACKENDS)
                saved_object_detection_backend = st.session_state.get(
                    "object_detection_backend",
                    DEFAULT_OBJECT_DETECTION_BACKEND,
                )
                if saved_object_detection_backend not in OBJECT_DETECTION_BACKENDS:
                    saved_object_detection_backend = DEFAULT_OBJECT_DETECTION_BACKEND
                object_detection_backend = st.selectbox(
                    "Object detection backend",
                    options=object_detection_options,
                    index=object_detection_options.index(saved_object_detection_backend),
                    help="Only the selected YOLO object weights are kept in ~/.EAGLE/.",
                )
                object_det_thresh = float(
                    st.slider("Object threshold", 0.0, 1.0, object_det_thresh, 0.05)
                )
                if media_type == "video":
                    object_target_fps = fps_input("Object frame FPS", "object_target_fps")
                object_smoothing_window = int(
                    st.number_input(
                        "Object smoothing window",
                        min_value=1,
                        step=1,
                        value=int(st.session_state.get("object_smoothing_window", 5)),
                    )
                )
                reuse_cached_objects = st.checkbox(
                    "Reuse existing objects.csv when available",
                    value=bool(st.session_state.get("reuse_cached_objects", False)),
                )
                saved_selected_classes = normalize_selected_classes(
                    st.session_state.get("selected_object_classes", list(COCO_OBJECT_CLASSES))
                )
                alphabetized_object_classes = sorted(COCO_OBJECT_CLASSES)
                track_all_classes = st.checkbox(
                    "Track all object classes",
                    value=len(saved_selected_classes) == len(COCO_OBJECT_CLASSES),
                    help="Leave this on for the default behavior. Turn it off only if you want to limit tracked objects.",
                )
                if track_all_classes:
                    selected_object_classes = list(COCO_OBJECT_CLASSES)
                else:
                    selected_class_defaults = (
                        saved_selected_classes
                        if len(saved_selected_classes) != len(COCO_OBJECT_CLASSES)
                        else ["person"]
                    )
                    selected_object_classes = st.multiselect(
                        "Object classes to keep",
                        options=alphabetized_object_classes,
                        default=sorted(selected_class_defaults),
                        placeholder="Choose object classes",
                    )
                    if not selected_object_classes:
                        st.warning("Select at least one object class.")
                if len(selected_object_classes) == len(COCO_OBJECT_CLASSES):
                    st.caption("Currently tracking all YOLO object classes.")
                else:
                    preview = ", ".join(selected_object_classes[:8])
                    suffix = "" if len(selected_object_classes) <= 8 else ", ..."
                    st.caption(f"Tracking {len(selected_object_classes)} classes: {preview}{suffix}")

            with face_tab:
                face_detection_options = ["retinaface", "mediapipe"]
                saved_face_detection_backend = st.session_state.get(
                    "face_detection_backend",
                    DEFAULT_FACE_DETECTION_BACKEND,
                )
                if saved_face_detection_backend not in FACE_DETECTION_BACKENDS:
                    saved_face_detection_backend = DEFAULT_FACE_DETECTION_BACKEND
                face_detection_backend = st.selectbox(
                    "Face detection backend",
                    options=face_detection_options,
                    index=face_detection_options.index(saved_face_detection_backend),
                )
                face_det_thresh = float(
                    st.slider("Face threshold", 0.0, 1.0, face_det_thresh, 0.05)
                )
                if media_type == "video":
                    face_target_fps = fps_input("Face frame FPS", "face_target_fps")
                face_smoothing_window = int(
                    st.number_input(
                        "Face smoothing window",
                        min_value=1,
                        step=1,
                        value=int(st.session_state.get("face_smoothing_window", 5)),
                    )
                )
                reuse_cached_faces = st.checkbox(
                    "Reuse existing faces.csv when available",
                    value=bool(st.session_state.get("reuse_cached_faces", False)),
                )

            with gaze_tab:
                gaze_detection_options = list(GAZE_DETECTION_BACKENDS)
                saved_gaze_detection_backend = st.session_state.get(
                    "gaze_detection_backend",
                    DEFAULT_GAZE_DETECTION_BACKEND,
                )
                if saved_gaze_detection_backend not in GAZE_DETECTION_BACKENDS:
                    saved_gaze_detection_backend = DEFAULT_GAZE_DETECTION_BACKEND
                gaze_detection_backend = st.selectbox(
                    "Gaze detection backend",
                    options=gaze_detection_options,
                    index=gaze_detection_options.index(saved_gaze_detection_backend),
                )
                head_pose_detection_options = list(HEAD_POSE_DETECTION_BACKENDS)
                saved_head_pose_detection_backend = st.session_state.get(
                    "head_pose_detection_backend",
                    DEFAULT_HEAD_POSE_DETECTION_BACKEND,
                )
                if saved_head_pose_detection_backend not in HEAD_POSE_DETECTION_BACKENDS:
                    saved_head_pose_detection_backend = DEFAULT_HEAD_POSE_DETECTION_BACKEND
                head_pose_detection_backend = st.selectbox(
                    "Head pose detection backend",
                    options=head_pose_detection_options,
                    index=head_pose_detection_options.index(saved_head_pose_detection_backend),
                    help="Currently only MobileOne is implemented.",
                )
                gaze_det_thresh = float(
                    st.slider("Gaze threshold", 0.0, 1.0, gaze_det_thresh, 0.05)
                )
                if media_type == "video":
                    col_gaze_fps, col_head_fps = st.columns(2)
                    with col_gaze_fps:
                        gaze_target_fps = fps_input("Gaze frame FPS", "gaze_target_fps")
                    with col_head_fps:
                        head_pose_target_fps = fps_input("Head pose frame FPS", "head_pose_target_fps")
                gaze_smoothing_window = int(
                    st.number_input(
                        "Gaze smoothing window",
                        min_value=1,
                        step=1,
                        value=int(st.session_state.get("gaze_smoothing_window", 5)),
                    )
                )
                visualization_mode = st.selectbox(
                    "Visualization mode",
                    options=["both", "point", "heatmap"],
                    index=["both", "point", "heatmap"].index(st.session_state.get("visualization_mode", "both")),
                )
                heatmap_alpha = float(
                    st.slider(
                        "Heatmap alpha",
                        min_value=0.0,
                        max_value=1.0,
                        value=float(st.session_state.get("heatmap_alpha", 0.35)),
                        step=0.05,
                    )
                )
                gaze_target_radius = int(
                    st.number_input(
                        "Gaze target radius (px)",
                        min_value=0,
                        step=1,
                        value=int(st.session_state.get("gaze_target_radius", 15)),
                        help="0 means point-only target assignment. Larger values use a circle around the gaze point.",
                    )
                )
                reuse_cached_gaze = st.checkbox(
                    "Reuse existing gaze.csv and heatmaps.npz when available",
                    value=bool(st.session_state.get("reuse_cached_gaze", False)),
                )

        tracker_updates = {
            "track_high_thresh": track_high_thresh,
            "track_low_thresh": track_low_thresh,
            "new_track_thresh": new_track_thresh,
            "track_buffer": track_buffer,
            "match_thresh": match_thresh,
            "with_reid": with_reid,
            "proximity_thresh": proximity_thresh,
            "appearance_thresh": appearance_thresh,
        }

        st.session_state.input_path = "" if input_path is None else str(input_path)
        st.session_state.output_dir = str(output_dir)
        st.session_state.output_name = output_dir.name
        st.session_state.person_det_thresh = person_det_thresh
        st.session_state.object_det_thresh = object_det_thresh
        st.session_state.face_det_thresh = face_det_thresh
        st.session_state.gaze_det_thresh = gaze_det_thresh
        st.session_state.device = device
        st.session_state.person_detection_backend = person_detection_backend
        st.session_state.object_detection_backend = object_detection_backend
        st.session_state.visualization_mode = visualization_mode
        st.session_state.face_detection_backend = face_detection_backend
        st.session_state.gaze_detection_backend = gaze_detection_backend
        st.session_state.head_pose_detection_backend = head_pose_detection_backend
        st.session_state.heatmap_alpha = heatmap_alpha
        st.session_state.gaze_target_radius = gaze_target_radius
        st.session_state.person_part_distance_scale = person_part_distance_scale
        st.session_state.person_part_min_conf = person_part_min_conf
        st.session_state.reuse_cached_objects = reuse_cached_objects
        st.session_state.reuse_cached_persons = reuse_cached_persons
        st.session_state.reuse_cached_faces = reuse_cached_faces
        st.session_state.reuse_cached_gaze = reuse_cached_gaze
        st.session_state.selected_object_classes = selected_object_classes
        st.session_state.person_smoothing_window = person_smoothing_window
        st.session_state.object_smoothing_window = object_smoothing_window
        st.session_state.face_smoothing_window = face_smoothing_window
        st.session_state.gaze_smoothing_window = gaze_smoothing_window
        st.session_state.person_target_fps = person_target_fps
        st.session_state.object_target_fps = object_target_fps
        st.session_state.face_target_fps = face_target_fps
        st.session_state.gaze_target_fps = gaze_target_fps
        st.session_state.head_pose_target_fps = head_pose_target_fps
        st.session_state.track_high_thresh = track_high_thresh
        st.session_state.track_low_thresh = track_low_thresh
        st.session_state.new_track_thresh = new_track_thresh
        st.session_state.track_buffer = track_buffer
        st.session_state.match_thresh = match_thresh
        st.session_state.with_reid = with_reid
        st.session_state.proximity_thresh = proximity_thresh
        st.session_state.appearance_thresh = appearance_thresh

        run_disabled = input_path is None or not selected_object_classes
        if st.button("Run Pipeline", type="primary", disabled=run_disabled):
            st.session_state.state = "processing"
            st.rerun()

    elif st.session_state.state == "processing":
        progress_bar = st.progress(0, text="Preparing...")
        try:
            input_path = Path(st.session_state.input_path)
            output_dir = Path(st.session_state.output_dir)
            media_type = annotator.config_manager.detect_media_type(input_path)

            person_target_fps = st.session_state.person_target_fps if media_type == "video" else None
            object_target_fps = st.session_state.object_target_fps if media_type == "video" else None
            face_target_fps = st.session_state.face_target_fps if media_type == "video" else None
            gaze_target_fps = st.session_state.gaze_target_fps if media_type == "video" else None
            head_pose_target_fps = st.session_state.head_pose_target_fps if media_type == "video" else None

            tracker_updates = {
                "track_high_thresh": st.session_state.track_high_thresh,
                "track_low_thresh": st.session_state.track_low_thresh,
                "new_track_thresh": st.session_state.new_track_thresh,
                "track_buffer": st.session_state.track_buffer,
                "match_thresh": st.session_state.match_thresh,
                "with_reid": st.session_state.with_reid,
                "proximity_thresh": st.session_state.proximity_thresh,
                "appearance_thresh": st.session_state.appearance_thresh,
            }

            annotator.preprocess(
                input_path=input_path,
                output_dir=output_dir,
                person_target_fps=person_target_fps,
                object_target_fps=object_target_fps,
                face_target_fps=face_target_fps,
                gaze_target_fps=gaze_target_fps,
                head_pose_target_fps=head_pose_target_fps,
                person_det_thresh=st.session_state.person_det_thresh,
                object_det_thresh=st.session_state.object_det_thresh,
                face_det_thresh=st.session_state.face_det_thresh,
                gaze_det_thresh=st.session_state.gaze_det_thresh,
                person_detection_backend=st.session_state.person_detection_backend,
                object_detection_backend=st.session_state.object_detection_backend,
                gaze_detection_backend=st.session_state.gaze_detection_backend,
                head_pose_detection_backend=st.session_state.head_pose_detection_backend,
                updates=tracker_updates,
                device=st.session_state.device,
                visualization_mode=st.session_state.visualization_mode,
                heatmap_alpha=st.session_state.heatmap_alpha,
                face_detection_backend=st.session_state.face_detection_backend,
                gaze_target_radius=st.session_state.gaze_target_radius,
                person_part_distance_scale=st.session_state.person_part_distance_scale,
                person_part_min_conf=st.session_state.person_part_min_conf,
                person_smoothing_window=st.session_state.person_smoothing_window,
                object_smoothing_window=st.session_state.object_smoothing_window,
                face_smoothing_window=st.session_state.face_smoothing_window,
                gaze_smoothing_window=st.session_state.gaze_smoothing_window,
                selected_object_classes=st.session_state.selected_object_classes,
                reuse_cached_persons=st.session_state.reuse_cached_persons,
                reuse_cached_objects=st.session_state.reuse_cached_objects,
                reuse_cached_faces=st.session_state.reuse_cached_faces,
                reuse_cached_gaze=st.session_state.reuse_cached_gaze,
            )
            st.session_state.results = annotator.run_all(progress_bar=progress_bar)
            st.session_state.error_message = None
            st.session_state.state = "end"
            st.rerun()
        except Exception as error:
            st.session_state.error_message = str(error)
            st.session_state.state = "error"
            st.rerun()

    elif st.session_state.state == "error":
        st.error(st.session_state.error_message or "Unknown error")
        if st.button("Back"):
            reset_to_start(st)
            st.rerun()

    elif st.session_state.state == "end":
        st.success("Completed")
        results = st.session_state.results or {}
        context = annotator.context

        if context is not None:
            st.write(f"Persons CSV: `{context.persons_path}`")
            st.write(f"Objects CSV: `{context.objects_path}`")
            st.write(f"Faces CSV: `{context.faces_path}`")
            st.write(f"Gaze CSV: `{context.gaze_path}`")
            st.write(f"Annotation CSV: `{context.annotation_path}`")

        output_paths = results.get("media_output_paths")
        if output_paths is not None:
            st.write("Visualization outputs:")
            for line in output_paths_to_lines(output_paths):
                st.write(f"- `{line}`")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Run Another File"):
                reset_to_start(st)
                st.rerun()
        with col2:
            if st.button("Keep Settings and Edit"):
                reset_to_start(st)
                st.rerun()


if __name__ == "__main__":
    if is_running_under_streamlit():
        main()
    elif os.environ.get(STREAMLIT_CHILD_ENV) == "1":
        port = int(os.environ.get(STREAMLIT_PORT_ENV, "8501"))
        raise SystemExit(launch_streamlit_server(port))
    elif getattr(sys, "frozen", False):
        port = find_available_port()
        if port != 8501:
            print(f"Port 8501 is busy. Launching Streamlit on port {port}.", flush=True)
        launch_frozen_app_server(port)
    else:
        port = find_available_port()
        if port != 8501:
            print(f"Port 8501 is busy. Launching Streamlit on port {port}.", flush=True)
        raise SystemExit(launch_streamlit_server(port))
