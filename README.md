# EAGLE
<p align="center">
  <strong>E</strong>nd-to-end <strong>A</strong>utomatic <strong>G</strong>aze <strong>L</strong>ab<strong>E</strong>ling tool for interaction studies
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
- Face detection with MediaPipe by default, with RetinaFace also available
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
- Reuse cached `objects.csv` and cached gaze outputs when settings still match
- Force reuse old caches even when the app detects a settings mismatch

For person detections, EAGLE also uses pose keypoints to assign gaze to body parts such as face, head, torso, arms, or legs when possible.

## Supported Inputs
- Images: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tif`, `.tiff`, `.webp`
- Videos: `.mp4`, `.mov`, `.avi`, `.mkv`, `.m4v`, `.wmv`, `.webm`

## Current Runtime Notes
- On macOS, `Browse` opens a native file dialog through `osascript`.
- On Linux (including Docker), use `Container File Browser` or enter a mounted path manually.
- The core pipeline itself is regular Python code under [`eagle/`](eagle/).
- Bundled FFmpeg binaries are included for macOS and Windows.
- Model files are cached under `~/.EAGLE/`.

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
- RetinaFace pretrained weights when the RetinaFace backend is selected
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
- `Detection threshold`
  - Shared threshold used for object filtering, face filtering, and gaze in/out interpretation.
- `YOLO object model`
  - Selects the YOLO26 detection model (`yolo26n`, `yolo26s`, `yolo26m`, `yolo26l`, or `yolo26x`). EAGLE keeps only the selected object weights in `~/.EAGLE/`.
- `Face detection backend`
  - `mediapipe` by default, or `retinaface` when you want to use RetinaFace.
- `Off-screen direction backend`
  - `mobileone`. This setting is prepared for future head-direction model backends.
- `Visualization mode`
  - `both`, `point`, or `heatmap`
- `Heatmap alpha`
  - Overlay strength for heatmap outputs.
- `Gaze target radius (px)`
  - `0` means point-only target assignment. Larger values use a circular target area.
- `Person part distance scale`
  - Controls how far from a person's keypoints gaze can still count as a body part target.
- `Reuse existing objects.csv when available`
  - Reuses cached object detections when metadata still matches the current run.
- `Reuse existing gaze.csv and heatmaps.npz when available`
  - Reuses cached gaze outputs when metadata still matches the current run.
- `Track all object classes`
  - When off, you can explicitly choose which COCO classes to keep.

### Temporal Settings
- `Object smoothing window`
  - Smoothing window for tracked object boxes.
- `Face smoothing window`
  - Smoothing window for interpolated face boxes.
- `Gaze smoothing window`
  - Smoothing window for gaze estimates and off-screen direction angles.
- `Object frame interval`
  - For videos only. Object detection/tracking runs every Nth frame.
- `Gaze frame interval`
  - For videos only. Gaze estimation runs every Nth frame and is then interpolated/smoothed.

Important:

- `Gaze frame interval` can be smaller than `Object frame interval`.
  Object tracks are linearly interpolated between detection frames, and video outputs are rendered at the source FPS.
- Internally, video settings are converted to target FPS values from the source FPS.

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

## Cache Behavior
EAGLE writes cache metadata files and checks them before reuse:

- `objects.csv` with `.objects_meta.json`
- `faces.csv` with `.faces_meta.json`
- `gaze.csv` with `gaze_heatmaps.npz` and `.gaze_meta.json`

Object cache reuse depends on:

- Input file path
- Input file timestamp
- Detection threshold
- YOLO object model
- Object frame interval
- Object smoothing window
- Selected object classes
- BoT-SORT settings
- Whether the cache was created with the current pose-based person tracking format

Gaze cache reuse depends on:

- Input file path
- Input file timestamp
- Detection threshold
- Face detection backend
- Off-screen direction backend
- Gaze frame interval
- Face smoothing window
- Gaze smoothing window
- `faces.csv` timestamp

If the app detects a mismatch, it warns you and offers a `Force reuse` checkbox.

## Output Files
Current outputs can include:

- `objects.csv`
  - Smoothed object/person tracking results
- `.objects_meta.json`
  - Metadata used to validate object cache reuse
- `faces.csv`
  - Face detection boxes assigned to person tracks
- `.faces_meta.json`
  - Metadata used to validate face cache reuse
- `gaze.csv`
  - Raw gaze values, processed gaze points, and off-screen direction fields
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
`objects.csv` columns:

- `frame_idx`
- `cls`
- `track_id`
- `source`
- `conf`
- `x1`, `y1`, `x2`, `y2`
- `pose_keypoints`
- `label`

`gaze.csv` columns:

- `frame_idx`
- `track_id`
- `raw_gaze_detected`
- `raw_inout`
- `raw_x_gaze`, `raw_y_gaze`
- `gaze_detected`
- `inout`
- `x_gaze`, `y_gaze`
- `offscreen_direction`
- `offscreen_yaw`
- `offscreen_pitch`

`faces.csv` columns:

- `frame_idx`
- `track_id`
- `face_detected`
- `face_conf`
- `face_x1`, `face_y1`, `face_x2`, `face_y2`

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
    det_thresh=0.5,
    device="cpu",
    visualization_mode="both",
)
results = eagle.run_all()
```

## Validation
Basic syntax validation:

```bash
python -m py_compile main.py app.py eagle/*.py
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
