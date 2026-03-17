# EAGLE
EAGLE: End-to-end Automatic Gaze LabEling tool for interaction studies

<p align="center">
  <img src="assets/icon_name_trans.png" alt="EAGLE app icon" width="280" />
</p>

## English

### Overview
EAGLE is a gaze-annotation pipeline for interaction studies. It detects people and other objects, estimates face locations, predicts gaze heatmaps and gaze points, and exports results as CSV files plus annotated images or videos.

The current app supports:
- Image input and video input
- Point visualization and per-person heatmap visualization
- Object tracking focused on all classes or persons only
- Reuse of existing `objects.csv` when desired
- Output of `objects.csv`, `gaze.csv`, `annotation.csv`, `all_points.*`, and `person_*_heatmap.*`

### Launching the App
The app is designed so that this command is enough:

```bash
python app.py
```

When run this way, `app.py` launches Streamlit automatically using the project virtual environment:

```bash
/Users/taigamori/Works/EAGLE/venv/bin/streamlit
```

If port `8501` is already in use, the app will try another nearby port automatically.

You can also run Streamlit directly if you prefer:

```bash
/Users/taigamori/Works/EAGLE/venv/bin/streamlit run app.py
```

### App Workflow
1. Launch the app with `python app.py`.
2. Click `Browse`.
3. Select an input image or video file from the native file picker.
4. Confirm the detected media type shown in the UI.
5. Choose the output folder name.

Output behavior:
- The parent output directory is automatically set to the same directory as the input file.
- You choose only the name of the output folder.
- Example:
  - Input: `/data/session01/test.mp4`
  - Output folder name: `test_output`
  - Final output directory: `/data/session01/test_output`

6. Adjust inference and tracking settings if needed.
7. Click `Run Pipeline`.
8. Wait until the app finishes and shows the output file list.

### Main Settings

#### Input / Output
- `Browse`
  - Opens a native file selection dialog.
- `Output folder name`
  - Name of the directory created next to the input file.

#### Inference
- `Detection threshold`
  - Confidence threshold used for object and face-related filtering.
- `Device`
  - Inference device such as `cpu`, `cuda`, or `mps`, depending on availability.
- `Visualization mode`
  - `both`: output both point visualization and heatmap visualization
  - `point`: output only point visualization
  - `heatmap`: output only per-person heatmap visualization
- `Heatmap alpha`
  - Transparency of the heatmap overlay.
- `Track persons only`
  - If enabled, only detections with class `person` are kept for object tracking output.
- `Reuse existing objects.csv when available`
  - If enabled, the pipeline will reuse an existing `objects.csv` in the output directory instead of rerunning object detection.

#### Temporal Settings
- `Object smoothing window`
  - Temporal smoothing window for tracked object boxes.
- `Gaze smoothing window`
  - Temporal smoothing window for gaze heatmaps and gaze points.
- `Object frame interval`
  - For video only. Run object detection/tracking every Nth frame.
- `Gaze frame interval`
  - For video only. Run gaze estimation every Nth frame, then interpolate and smooth in between.

#### BoT-SORT
The app exposes common BoT-SORT settings directly in the UI:
- `track_high_thresh`
- `track_low_thresh`
- `new_track_thresh`
- `track_buffer`
- `match_thresh`
- `Enable ReID`
- `proximity_thresh`
- `appearance_thresh`

Their default values come from [`config/botsort.yaml`](/Users/taigamori/Works/EAGLE/config/botsort.yaml).

### Output Files
Depending on the input type and visualization mode, the app writes files such as:

- `objects.csv`
  - Object detection and tracking results
- `gaze.csv`
  - Face detections and gaze estimates
  - Includes both raw sampled gaze values and processed interpolated/smoothed values
- `annotation.csv`
  - ELAN-style gaze annotation summary
- `all_points.jpg`
  - Point visualization for image input
- `all_points.mp4`
  - Point visualization for video input
- `person_1_heatmap.jpg`, `person_2_heatmap.jpg`, ...
  - Per-person heatmap outputs for image input
- `person_1_heatmap.mp4`, `person_2_heatmap.mp4`, ...
  - Per-person heatmap outputs for video input

Temporary directories such as `temp` and `heatmaps` are cleaned up after export, so only final deliverables remain.

### Notes
- The file browser currently uses a macOS-native dialog through `osascript`.
- If you rerun the same output directory with `Reuse existing objects.csv when available` enabled, object detection may be skipped and prior object results will be reused.
- For heatmap mode, each person is exported separately because multiple overlaid heatmaps are difficult to interpret.
- For video input, gaze is not necessarily estimated on every frame. Sparse predictions are interpolated and smoothed over time.

### Validation
The codebase has been syntax-checked with:

```bash
python -m py_compile main.py app.py eagle/*.py
```

### Acknowledgements
This app is built on top of several excellent open-source models and projects. We are very grateful to their authors and maintainers.

- Ultralytics YOLO for object detection and tracking integration
  - https://docs.ultralytics.com/
- BoT-SORT for multi-object tracking
  - https://github.com/NirAharon/BoT-SORT
- RetinaFace for face detection
  - https://github.com/serengil/retinaface
- GAZELLE / Gaze-LLE for gaze target estimation
  - https://github.com/fkryan/gazelle
- DINOv2 as the visual backbone used by GAZELLE
  - https://github.com/facebookresearch/dinov2

We sincerely appreciate the research community and open-source contributors whose work made this application possible.

---

## 日本語

### 概要
EAGLE は、相互作用研究向けの視線アノテーション支援ツールです。人物や物体の検出・追跡、顔検出、視線ヒートマップ推定、視線点推定を行い、CSV と注釈付き画像・動画を出力します。

現在のアプリでは、以下に対応しています。
- 画像入力と動画入力
- 点の可視化と人物ごとのヒートマップ可視化
- 全クラス追跡または人物限定追跡
- 既存 `objects.csv` の再利用
- `objects.csv`、`gaze.csv`、`annotation.csv`、`all_points.*`、`person_*_heatmap.*` の出力

### アプリの起動方法
基本的には以下で起動できます。

```bash
python app.py
```

この実行では、`app.py` がプロジェクトの仮想環境にある Streamlit を自動で起動します。

```bash
/Users/taigamori/Works/EAGLE/venv/bin/streamlit
```

もし `8501` ポートが使用中なら、近い別ポートを自動で探して起動します。

必要であれば、直接 Streamlit を起動してもかまいません。

```bash
/Users/taigamori/Works/EAGLE/venv/bin/streamlit run app.py
```

### アプリの使い方
1. `python app.py` で起動します。
2. `Browse` を押します。
3. ネイティブのファイル選択ダイアログから画像または動画を選びます。
4. UI 上で入力種別が正しく判定されていることを確認します。
5. 出力フォルダ名を指定します。

出力先の挙動:
- 出力先の親ディレクトリは、入力ファイルと同じディレクトリに自動設定されます。
- ユーザーは、その中に作るフォルダ名だけを指定します。
- 例:
  - 入力: `/data/session01/test.mp4`
  - 出力フォルダ名: `test_output`
  - 最終出力先: `/data/session01/test_output`

6. 必要に応じて推論・追跡設定を調整します。
7. `Run Pipeline` を押します。
8. 処理完了後、出力ファイル一覧を確認します。

### 主な設定項目

#### Input / Output
- `Browse`
  - ネイティブのファイル選択ダイアログを開きます。
- `Output folder name`
  - 入力ファイルと同じ階層に作る出力フォルダ名です。

#### Inference
- `Detection threshold`
  - 物体や顔の信頼度フィルタに使う閾値です。
- `Device`
  - `cpu`、`cuda`、`mps` など、利用可能な推論デバイスです。
- `Visualization mode`
  - `both`: 点とヒートマップの両方を出力
  - `point`: 点のみ出力
  - `heatmap`: 人物ごとのヒートマップのみ出力
- `Heatmap alpha`
  - ヒートマップ重ね描きの透明度です。
- `Track persons only`
  - 有効にすると、物体追跡結果として `person` クラスのみを残します。
- `Reuse existing objects.csv when available`
  - 有効にすると、出力先に既存の `objects.csv` がある場合は物体検出をスキップして再利用します。

#### Temporal Settings
- `Object smoothing window`
  - 物体 bbox の時間方向平滑化窓です。
- `Gaze smoothing window`
  - 視線ヒートマップと視線点の時間方向平滑化窓です。
- `Object frame interval`
  - 動画のみ。物体検出・追跡を何フレームおきに実行するかを指定します。
- `Gaze frame interval`
  - 動画のみ。視線推定を何フレームおきに実行するかを指定します。中間フレームは補間・平滑化されます。

#### BoT-SORT
UI から以下の主要パラメータを直接調整できます。
- `track_high_thresh`
- `track_low_thresh`
- `new_track_thresh`
- `track_buffer`
- `match_thresh`
- `Enable ReID`
- `proximity_thresh`
- `appearance_thresh`

これらの初期値は [`config/botsort.yaml`](/Users/taigamori/Works/EAGLE/config/botsort.yaml) から読み込まれます。

### 出力ファイル
入力種別と可視化モードに応じて、以下のようなファイルが出力されます。

- `objects.csv`
  - 物体検出・追跡結果
- `gaze.csv`
  - 顔検出と視線推定結果
  - 生のサンプル値と、補間・平滑化後の値の両方を含みます
- `annotation.csv`
  - ELAN 形式に近い視線アノテーション要約
- `all_points.jpg`
  - 画像入力時の点可視化結果
- `all_points.mp4`
  - 動画入力時の点可視化結果
- `person_1_heatmap.jpg`, `person_2_heatmap.jpg`, ...
  - 画像入力時の人物別ヒートマップ結果
- `person_1_heatmap.mp4`, `person_2_heatmap.mp4`, ...
  - 動画入力時の人物別ヒートマップ結果

`temp` や `heatmaps` のような一時ディレクトリは出力完了後に削除され、最終成果物だけが残るようになっています。

### 注意点
- 現在のファイル選択ダイアログは、macOS の `osascript` を使ったネイティブ選択を利用しています。
- 同じ出力ディレクトリを再実行し、`Reuse existing objects.csv when available` を有効にしている場合、物体検出をスキップして前回の `objects.csv` を再利用することがあります。
- ヒートマップは、複数人分を同一画面に重ねると見づらいため、人物ごとに分けて出力します。
- 動画では、視線を毎フレーム必ず推定しているわけではありません。疎に推定した結果を時間方向に補間・平滑化しています。

### 動作確認
コード全体の文法確認には以下を使っています。

```bash
python -m py_compile main.py app.py eagle/*.py
```

### 謝辞
このアプリは、複数の優れたオープンソースモデル・プロジェクトを土台として成り立っています。著者のみなさま、メンテナのみなさまに深く感謝します。

- Ultralytics YOLO
  - 物体検出と追跡統合
  - https://docs.ultralytics.com/
- BoT-SORT
  - 多物体追跡
  - https://github.com/NirAharon/BoT-SORT
- RetinaFace
  - 顔検出
  - https://github.com/serengil/retinaface
- GAZELLE / Gaze-LLE
  - 視線先推定
  - https://github.com/fkryan/gazelle
- DINOv2
  - GAZELLE で使われる視覚バックボーン
  - https://github.com/facebookresearch/dinov2

このアプリは、こうした研究とオープンソースの積み重ねの上に成り立っています。素晴らしい公開にあらためて感謝します。
