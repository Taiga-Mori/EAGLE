from pathlib import Path
import base64
from functools import lru_cache
import os
import signal
import socket
import subprocess
import sys
import time
import webbrowser

import cv2
import yaml

from eagle import EAGLE
from eagle.constants import COCO_OBJECT_CLASSES


APP_PATH = Path(__file__).resolve()
RUNTIME_ROOT = Path(sys._MEIPASS) if hasattr(sys, "_MEIPASS") else APP_PATH.parent
APP_DIR = APP_PATH.parent
APP_ICON_PATH = RUNTIME_ROOT / "assets" / "icon_trans.png"
STREAMLIT_CHILD_ENV = "EAGLE_STREAMLIT_CHILD"
STREAMLIT_PORT_ENV = "EAGLE_STREAMLIT_PORT"


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


def load_json_file(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        import json

        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def build_tracker_config_snapshot(botsort_defaults: dict, tracker_updates: dict) -> dict:
    snapshot = dict(botsort_defaults)
    snapshot.update(tracker_updates)
    return snapshot


def get_object_cache_mismatches(
    output_dir: Path,
    input_path: Path | None,
    media_type: str | None,
    det_thresh: float,
    object_frame_interval: int,
    object_smoothing_window: int,
    tracker_config: dict,
    selected_object_classes: list[str],
) -> list[str]:
    meta = load_json_file(output_dir / ".objects_meta.json")
    objects_path = output_dir / "objects.csv"
    if meta is None or not objects_path.exists():
        return []

    reasons: list[str] = []
    if meta.get("raw_detection_cache") is not True:
        reasons.append("cache was created with an older format")
        return reasons
    if input_path is not None:
        if meta.get("media_path") != str(input_path.resolve()):
            reasons.append("input file is different")
        elif int(meta.get("media_mtime_ns", -1)) != input_path.stat().st_mtime_ns:
            reasons.append("input file timestamp is different")
    expected_stride = 1 if media_type != "video" else int(object_frame_interval)
    if abs(float(meta.get("det_thresh", -1.0)) - float(det_thresh)) > 1e-9:
        reasons.append("detection threshold is different")
    if int(meta.get("object_stride", -1)) != expected_stride:
        reasons.append("object frame interval is different")
    if int(meta.get("object_smoothing_window", -1)) != int(object_smoothing_window):
        reasons.append("object smoothing window is different")
    if str(meta.get("person_detection_source", "")) != "pose":
        reasons.append("cached objects.csv was created before pose-based person tracking")
    requested_non_person_detections = any(cls_name != "person" for cls_name in selected_object_classes)
    cached_includes_non_person = bool(meta.get("includes_non_person_detections", True))
    if requested_non_person_detections and not cached_includes_non_person:
        reasons.append("cached objects.csv does not include non-person detections")
    if meta.get("tracker_config") != tracker_config:
        reasons.append("BoT-SORT settings are different")
    return reasons


def get_gaze_cache_mismatches(
    output_dir: Path,
    input_path: Path | None,
    media_type: str | None,
    det_thresh: float,
    gaze_frame_interval: int,
    gaze_smoothing_window: int,
) -> list[str]:
    meta = load_json_file(output_dir / ".gaze_meta.json")
    gaze_path = output_dir / "gaze.csv"
    heatmap_path = output_dir / "gaze_heatmaps.npz"
    if meta is None or not gaze_path.exists() or not heatmap_path.exists():
        return []

    reasons: list[str] = []
    if input_path is not None:
        if meta.get("media_path") != str(input_path.resolve()):
            reasons.append("input file is different")
        elif int(meta.get("media_mtime_ns", -1)) != input_path.stat().st_mtime_ns:
            reasons.append("input file timestamp is different")
    expected_stride = 1 if media_type != "video" else int(gaze_frame_interval)
    if abs(float(meta.get("det_thresh", -1.0)) - float(det_thresh)) > 1e-9:
        reasons.append("detection threshold is different")
    if int(meta.get("gaze_stride", -1)) != expected_stride:
        reasons.append("gaze frame interval is different")
    if int(meta.get("gaze_smoothing_window", -1)) != int(gaze_smoothing_window):
        reasons.append("gaze smoothing window is different")
    return reasons


def get_streamlit():
    import streamlit as st

    return st


def reset_to_start(st) -> None:
    st.session_state.state = "start"


def browse_input_file() -> Path | None:
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
        )
    except subprocess.CalledProcessError:
        return None
    selected = result.stdout.strip()
    if not selected:
        return None
    return Path(selected)


def default_output_dir(input_path: Path | None) -> Path:
    if input_path is None:
        return APP_DIR / "output"
    return input_path.parent / input_path.stem


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
    st = get_streamlit()
    st.set_page_config(page_title="EAGLE", layout="centered")
    render_header(st)

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

    annotator: EAGLE = st.session_state.annotator
    botsort_defaults = load_botsort_defaults(annotator)

    if st.session_state.state == "start":
        st.subheader("Basic Settings")
        col_input_path, col_browse = st.columns([5, 1])
        with col_input_path:
            input_path_str = st.text_input(
                "Input file",
                value=st.session_state.selected_input_path,
                disabled=True,
            )
        with col_browse:
            st.write("")
            st.write("")
            if st.button("Browse", use_container_width=True):
                selected_path = browse_input_file()
                if selected_path is not None:
                    st.session_state.selected_input_path = str(selected_path)
                    st.rerun()

        input_path = Path(input_path_str) if input_path_str else None
        output_dir_default = default_output_dir(input_path)
        output_parent_dir = output_dir_default.parent
        output_name = st.text_input(
            "Output folder name",
            value=str(st.session_state.get("output_name", output_dir_default.name)),
        ).strip()
        output_dir = output_parent_dir / (output_name or output_dir_default.name)
        media_type = detect_media_type(annotator, input_path)

        device = st.selectbox(
            "Device",
            options=annotator.device_options,
            index=annotator.device_options.index(st.session_state.get("device", annotator.device_options[0])),
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
            st.subheader("Inference")
            col1, col2 = st.columns(2)
            with col1:
                det_thresh = float(
                    st.slider(
                        "Detection threshold",
                        min_value=0.0,
                        max_value=1.0,
                        value=float(st.session_state.get("det_thresh", 0.5)),
                        step=0.05,
                    )
                )
            with col2:
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
                    value=int(st.session_state.get("gaze_target_radius", 0)),
                    help="0 means point-only target assignment. Larger values use a circle around the gaze point.",
                )
            )
            person_part_distance_scale = float(
                st.number_input(
                    "Person part distance scale",
                    min_value=0.01,
                    step=0.01,
                    value=float(st.session_state.get("person_part_distance_scale", 0.22)),
                    help="Scales how far from each keypoint a gaze can be and still count as that body part.",
                )
            )
            reuse_cached_objects = st.checkbox(
                "Reuse existing objects.csv when available",
                value=bool(st.session_state.get("reuse_cached_objects", False)),
            )
            reuse_cached_gaze = st.checkbox(
                "Reuse existing gaze.csv and heatmaps.npz when available",
                value=bool(st.session_state.get("reuse_cached_gaze", False)),
            )

            saved_selected_classes = normalize_selected_classes(
                st.session_state.get("selected_object_classes", list(COCO_OBJECT_CLASSES))
            )

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
                st.caption("Pick only the object classes you want to keep.")
                selected_object_classes = st.multiselect(
                    "Object classes to keep",
                    options=COCO_OBJECT_CLASSES,
                    default=selected_class_defaults,
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
            st.subheader("Temporal Settings")
            object_smoothing_window = int(
                st.number_input(
                    "Object smoothing window",
                    min_value=1,
                    step=1,
                    value=int(st.session_state.get("object_smoothing_window", 5)),
                )
            )
            gaze_smoothing_window = int(
                st.number_input(
                    "Gaze smoothing window",
                    min_value=1,
                    step=1,
                    value=int(st.session_state.get("gaze_smoothing_window", 5)),
                )
            )

            object_frame_interval = 1
            gaze_frame_interval = 1
            if media_type == "video":
                col3, col4 = st.columns(2)
                with col3:
                    object_frame_interval = int(
                        st.number_input(
                            "Object frame interval",
                            min_value=1,
                            step=1,
                            value=int(st.session_state.get("object_frame_interval", 1)),
                        )
                    )
                with col4:
                    gaze_frame_interval = int(
                        st.number_input(
                            "Gaze frame interval",
                            min_value=1,
                            step=1,
                            value=int(st.session_state.get("gaze_frame_interval", 1)),
                        )
                    )
                if gaze_frame_interval < object_frame_interval:
                    st.warning("Gaze frame interval should be greater than or equal to object frame interval.")

            st.subheader("BoT-SORT")
            col5, col6 = st.columns(2)
            with col5:
                track_high_thresh = float(
                    st.slider(
                        "track_high_thresh",
                        min_value=0.0,
                        max_value=1.0,
                        value=float(st.session_state.get("track_high_thresh", botsort_defaults.get("track_high_thresh", 0.25))),
                        step=0.05,
                    )
                )
                new_track_thresh = float(
                    st.slider(
                        "new_track_thresh",
                        min_value=0.0,
                        max_value=1.0,
                        value=float(st.session_state.get("new_track_thresh", botsort_defaults.get("new_track_thresh", 0.25))),
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
        tracker_snapshot = build_tracker_config_snapshot(botsort_defaults, tracker_updates)

        object_cache_mismatches = (
            get_object_cache_mismatches(
                output_dir,
                input_path,
                media_type,
                det_thresh,
                object_frame_interval,
                object_smoothing_window,
                tracker_snapshot,
                selected_object_classes,
            )
            if reuse_cached_objects
            else []
        )
        gaze_cache_mismatches = (
            get_gaze_cache_mismatches(
                output_dir,
                input_path,
                media_type,
                det_thresh,
                gaze_frame_interval,
                gaze_smoothing_window,
            )
            if reuse_cached_gaze
            else []
        )

        force_reuse_cached_objects = False
        force_reuse_cached_gaze = False
        if object_cache_mismatches:
            st.warning(
                "Cached objects.csv was created with different settings. "
                "EAGLE will recompute it unless you force reuse."
            )
            for reason in object_cache_mismatches:
                st.caption(f"- objects cache mismatch: {reason}")
            force_reuse_cached_objects = st.checkbox(
                "Force reuse cached objects.csv anyway",
                value=bool(st.session_state.get("force_reuse_cached_objects", False)),
            )
        if gaze_cache_mismatches:
            st.warning(
                "Cached gaze.csv / gaze_heatmaps.npz was created with different settings. "
                "EAGLE will recompute it unless you force reuse."
            )
            for reason in gaze_cache_mismatches:
                st.caption(f"- gaze cache mismatch: {reason}")
            force_reuse_cached_gaze = st.checkbox(
                "Force reuse cached gaze.csv and heatmaps.npz anyway",
                value=bool(st.session_state.get("force_reuse_cached_gaze", False)),
            )

        st.session_state.input_path = "" if input_path is None else str(input_path)
        st.session_state.output_dir = str(output_dir)
        st.session_state.output_name = output_dir.name
        st.session_state.det_thresh = det_thresh
        st.session_state.device = device
        st.session_state.visualization_mode = visualization_mode
        st.session_state.heatmap_alpha = heatmap_alpha
        st.session_state.gaze_target_radius = gaze_target_radius
        st.session_state.person_part_distance_scale = person_part_distance_scale
        st.session_state.reuse_cached_objects = reuse_cached_objects
        st.session_state.reuse_cached_gaze = reuse_cached_gaze
        st.session_state.force_reuse_cached_objects = force_reuse_cached_objects
        st.session_state.force_reuse_cached_gaze = force_reuse_cached_gaze
        st.session_state.selected_object_classes = selected_object_classes
        st.session_state.object_smoothing_window = object_smoothing_window
        st.session_state.gaze_smoothing_window = gaze_smoothing_window
        st.session_state.object_frame_interval = object_frame_interval
        st.session_state.gaze_frame_interval = gaze_frame_interval
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
        status = st.empty()
        try:
            input_path = Path(st.session_state.input_path)
            output_dir = Path(st.session_state.output_dir)
            media_type = annotator.config_manager.detect_media_type(input_path)

            object_target_fps = None
            gaze_target_fps = None
            if media_type == "video":
                fps = resolve_video_fps(input_path)
                object_target_fps = fps / max(int(st.session_state.object_frame_interval), 1)
                gaze_target_fps = fps / max(int(st.session_state.gaze_frame_interval), 1)

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

            status.write("Loading models and preparing pipeline...")
            annotator.preprocess(
                input_path=input_path,
                output_dir=output_dir,
                object_target_fps=object_target_fps,
                gaze_target_fps=gaze_target_fps,
                det_thresh=st.session_state.det_thresh,
                updates=tracker_updates,
                device=st.session_state.device,
                visualization_mode=st.session_state.visualization_mode,
                heatmap_alpha=st.session_state.heatmap_alpha,
                gaze_target_radius=st.session_state.gaze_target_radius,
                person_part_distance_scale=st.session_state.person_part_distance_scale,
                object_smoothing_window=st.session_state.object_smoothing_window,
                gaze_smoothing_window=st.session_state.gaze_smoothing_window,
                selected_object_classes=st.session_state.selected_object_classes,
                reuse_cached_objects=st.session_state.reuse_cached_objects,
                reuse_cached_gaze=st.session_state.reuse_cached_gaze,
                force_reuse_cached_objects=st.session_state.force_reuse_cached_objects,
                force_reuse_cached_gaze=st.session_state.force_reuse_cached_gaze,
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
            st.write(f"Objects CSV: `{context.objects_path}`")
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
