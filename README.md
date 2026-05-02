# EAGLE
<p align="center">
  <em><strong>E</strong>nd-to-end <strong>A</strong>utomatic <strong>G</strong>aze <strong>L</strong>ab<strong>E</strong>ling tool for interaction studies</em>
</p>


<p align="center">
  <img src="assets/icon_name_trans.png" alt="EAGLE app icon" width="280" />
</p>

## Languages

- [English](README.md)
- [日本語](README.ja.md)
- [简体中文](README.zh.md)
- [Español](README.es.md)

## Overview
EAGLE is a Streamlit-based gaze annotation support tool for image and video analysis. It combines:

- Person tracking with YOLO pose
- Non-person object tracking with YOLO detection
- Face detection with RetinaFace by default, with MediaPipe also available
- Gaze heatmap estimation with GAZELLE
- Off-screen direction estimation with MobileOne gaze
- CSV export plus annotated image/video export

EAGLE is an annotation assistance tool, not guaranteed ground truth. You must review and validate all outputs before using them in research, analysis, reporting, or decision-making.

## Example Outputs
<p align="center">
  <img src="assets/book.gif" alt="EAGLE example output: book" width="31%" />
  <img src="assets/interview.gif" alt="EAGLE example output: interview" width="31%" />
  <img src="assets/three.gif" alt="EAGLE example output: three people" width="31%" />
</p>

## GUI Example
<p align="center">
  <img src="assets/GUI.png" alt="EAGLE GUI example" width="90%" />
</p>

## What The Current System Does
For each input image or video, the current pipeline can:

- Detect and track persons from pose detections
- Optionally keep all COCO object classes or only selected classes
- Detect one face per tracked person
- Estimate gaze heatmaps and gaze points
- Infer off-screen direction labels such as `left`, `up right`, `down`, or `front` (person-centric orientation)
- Generate ELAN-style gaze segments in `annotation.csv`
- Reuse phase caches when settings still match

For person detections, EAGLE also uses pose keypoints to assign gaze to body parts such as face, head, torso, arms, or legs when possible.

## Supported Inputs
- Images: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tif`, `.tiff`, `.webp`
- Videos: `.mp4`, `.mov`, `.avi`, `.mkv`, `.m4v`, `.wmv`, `.webm`

## Download
Prebuilt application bundles are available from GitHub Releases:

- https://github.com/Taiga-Mori/EAGLE/releases

## Setup
Create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Recommended Python version: `3.10`

## Launching The App
Recommended:

```bash
python app.py
```

This launcher starts Streamlit for you and opens the app in a browser. If port `8501` is unavailable, it tries nearby ports automatically.

You can also run Streamlit directly:

```bash
venv/bin/streamlit run app.py
```

## First Run And Model Download
On first use, EAGLE may download and cache:

- Selected YOLO object weights: one of `yolo26n.pt`, `yolo26s.pt`, `yolo26m.pt`, `yolo26l.pt`, or `yolo26x.pt`
- `yolo26x-pose.pt`
- RetinaFace pretrained weights
- GAZELLE torch.hub files
- `mobileone_s0.pt`

If first-run loading fails, check:

- The machine is online
- GitHub / model hosting endpoints are reachable
- The current user can write to `~/.EAGLE/`

## App Workflow
1. Launch the app with `python app.py`.
2. Set `Input file`:
   - macOS: click `Browse`.
   - Linux/Docker: use `Container File Browser` or type a mounted path.
3. Select one input image or video.
4. Confirm the detected media type.
5. Edit `Output folder name` if needed.
6. Optionally open `Detailed Settings`.
7. Click `Run Pipeline`.
8. Review the output paths shown at the end.

Output directory behavior:

- If the input directory is writable, the output parent is the same as the input file's parent.
- If the input directory is read-only (common with `:ro` Docker mounts), the output falls back to `/app/output/<input_stem>`.
- The default output folder name is the input stem.
- Example:
  - Input: `/data/session01/test.mp4`
  - Writable parent: `/data/session01/test`
  - Read-only parent fallback: `/app/output/test`

## Main Settings

### Basic Settings
- `Input file`
  - macOS: read-only field populated by `Browse`.
  - Linux/Docker: editable field (manual path entry supported).
- `Output folder name`
  - Folder created next to the input file.
- `Device`
  - Uses `cuda:0`, `cuda:1`, ... when multiple NVIDIA GPUs are available (otherwise `mps` or `cpu`).

### Inference
- `Person detection backend`
  - `yolo26x-pose`.
- `Object detection backend`
  - Selects the YOLO26 detection backend (`yolo26n`, `yolo26s`, `yolo26m`, `yolo26l`, or `yolo26x`). EAGLE keeps only the selected object weights in `~/.EAGLE/`.
- `Face detection backend`
  - `retinaface` by default, or `mediapipe` when you want to use MediaPipe.
- `Gaze detection backend`
  - `gazelle`.
- `Head pose detection backend`
  - `mobileone`. This setting is prepared for future head-pose model backends.
- `Person threshold`, `Object threshold`, `Face threshold`, `Gaze threshold`
  - Separate thresholds for person pose tracking, non-person object tracking, face detection, and gaze in/out interpretation.
- `Visualization mode`
  - `both`, `point`, or `heatmap`
- `Heatmap alpha`
  - Overlay strength for heatmap outputs.
- `Gaze target radius (px)`
  - `0` means point-only target assignment. Larger values use a circular target area.
- `Person part distance scale`
  - Controls how far from a person's keypoints gaze can still count as a body part target.
- `Person part keypoint min confidence`
  - Minimum pose keypoint confidence used for body-part assignment. Default is `0.0`.
- `Reuse existing persons.csv/objects.csv/faces.csv/gaze.csv when available`
  - Reuses each phase cache separately when metadata still matches the current run.
- `Track all object classes`
  - When off, you can explicitly choose which COCO classes to keep.

### Temporal Settings
- `Person smoothing window`
  - Smoothing window for person boxes.
- `Object smoothing window`
  - Smoothing window for tracked object boxes.
- `Face smoothing window`
  - Smoothing window for interpolated face boxes.
- `Gaze smoothing window`
  - Smoothing window for gaze estimates and off-screen direction angles.
- `Person frame FPS`, `Object frame FPS`, `Face frame FPS`, `Gaze frame FPS`, `Head pose frame FPS`
  - For videos only. Each stage runs at the selected target FPS and then writes its own intermediate output.

Important:

- Later-stage FPS values can be changed without invalidating earlier-stage caches when their own settings are unchanged.
  Tracks, faces, gaze, and head pose fields are interpolated or smoothed after their stage runs.
- Internally, target FPS values are converted to frame strides from the source FPS.

### BoT-SORT
The UI exposes these tracker settings from [`config/botsort.yaml`](config/botsort.yaml):

- `track_high_thresh`
- `track_low_thresh`
- `new_track_thresh`
- `track_buffer`
- `match_thresh`
- `Enable ReID`
- `proximity_thresh`
- `appearance_thresh`

## Speed Tips
- Use `cuda:*` on NVIDIA GPUs, or `mps` on Apple Silicon when available.
- Lower stage FPS values, especially for stages other than gaze, to reduce repeated detection work.
- If faces are mostly frontal, try `mediapipe` for the face detection backend.

## Output Files
Current outputs can include:

- `persons.csv`
  - Smoothed person tracking results
- `.persons_meta.json`
  - Metadata used to validate person cache reuse
- `objects.csv`
  - Smoothed non-person object tracking results
- `.objects_meta.json`
  - Metadata used to validate object cache reuse
- `faces.csv`
  - Face detection boxes assigned to person tracks
- `.faces_meta.json`
  - Metadata used to validate face cache reuse
- `gaze.csv`
  - Smoothed gaze points, in/out probabilities, and off-screen direction fields
- `gaze_heatmaps.npz`
  - Cached dense gaze heatmaps
- `.gaze_meta.json`
  - Metadata used to validate gaze cache reuse
- `annotation.csv`
  - ELAN-style gaze segments or single-image labels
- `all_points.jpg`
  - Point visualization for image input
- `all_points.mp4`
  - Point visualization for video input
- `person_<track_id>_heatmap.jpg`
  - Per-person heatmap output for image input
- `person_<track_id>_heatmap.mp4`
  - Per-person heatmap output for video input

Temporary folders such as `temp/` and `heatmaps/` are removed after export.

## CSV Contents
`persons.csv` columns:

- `frame_idx`
- `cls`
- `track_id`
- `object_detected`
- `source`
- `conf`
- `x1`, `y1`, `x2`, `y2`
- `pose_keypoints`
- `label`

`objects.csv` columns:

- `frame_idx`
- `cls`
- `track_id`
- `object_detected`
- `source`
- `conf`
- `x1`, `y1`, `x2`, `y2`
- `pose_keypoints`
- `label`

`faces.csv` columns:

- `frame_idx`
- `track_id`
- `face_detected`
- `face_conf`
- `face_x1`, `face_y1`, `face_x2`, `face_y2`

`gaze.csv` columns:

- `frame_idx`
- `track_id`
- `gaze_detected`
- `inout`
- `x_gaze`, `y_gaze`
- `offscreen_direction`
- `offscreen_yaw`
- `offscreen_pitch`

`annotation.csv` columns:

- `tier`
- `start_time`
- `end_time`
- `gaze`

## Development Entry Points
- [`app.py`](app.py)
  - Streamlit UI and launcher
- [`main.py`](main.py)
  - Minimal local smoke-test entry point
- [`eagle/pipeline.py`](eagle/pipeline.py)
  - Main Python API facade

Minimal code usage:

```python
from eagle import EAGLE

eagle = EAGLE()
eagle.preprocess(
    input_path="input.mp4",
    output_dir="output_dir",
    person_target_fps=15,
    object_target_fps=15,
    face_target_fps=15,
    gaze_target_fps=15,
    head_pose_target_fps=15,
    person_det_thresh=0.5,
    object_det_thresh=0.5,
    face_det_thresh=0.5,
    gaze_det_thresh=0.5,
    person_detection_backend="yolo26x-pose",
    object_detection_backend="yolo26x",
    face_detection_backend="retinaface",
    gaze_detection_backend="gazelle",
    head_pose_detection_backend="mobileone",
    device="cpu",
    visualization_mode="both",
    person_part_min_conf=0.0,
)
results = eagle.run_all()
```

## Disclaimer
- Use this software at your own risk.
- EAGLE does not guarantee correctness of detections, gaze estimates, target assignments, or exported annotations.
- Final responsibility for review and correction remains with the user.

## License
This project is licensed under `AGPL-3.0-or-later`.
See [LICENSE](LICENSE) for the repository license text.

## Acknowledgements
- Ultralytics YOLO
  - https://docs.ultralytics.com/
- BoT-SORT
  - https://github.com/NirAharon/BoT-SORT
- MediaPipe
  - https://developers.google.com/mediapipe
- RetinaFace
  - https://github.com/serengil/retinaface
- GAZELLE
  - https://github.com/fkryan/gazelle
- MobileOne gaze-estimation weights
  - https://github.com/yakhyo/gaze-estimation
