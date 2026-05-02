# EAGLE
<p align="center">
  <em><strong>E</strong>nd-to-end <strong>A</strong>utomatic <strong>G</strong>aze <strong>L</strong>ab<strong>E</strong>ling tool for interaction studies</em>
</p>

<p align="center">
  <img src="assets/icon_name_trans.png" alt="EAGLE app icon" width="280" />
</p>

## 言語

- [English](README.md)
- [日本語](README.ja.md)
- [简体中文](README.zh.md)
- [Español](README.es.md)

## 概要
EAGLE は、画像・動画解析向けの Streamlit ベース視線アノテーション支援ツールです。次の処理を組み合わせます。

- YOLO pose による人物追跡
- YOLO detection による非人物オブジェクト追跡
- デフォルトの RetinaFace、または選択可能な MediaPipe による顔検出
- GAZELLE による視線ヒートマップ推定
- MobileOne gaze による画面外方向推定
- CSV 出力と、注釈付き画像・動画の出力

EAGLE はアノテーション支援ツールであり、正解データを保証するものではありません。研究、分析、報告、意思決定に使用する前に、すべての出力を必ず確認・検証してください。

## 出力例
<p align="center">
  <img src="assets/book.gif" alt="EAGLE example output: book" width="31%" />
  <img src="assets/interview.gif" alt="EAGLE example output: interview" width="31%" />
  <img src="assets/three.gif" alt="EAGLE example output: three people" width="31%" />
</p>

## GUI例
<p align="center">
  <img src="assets/GUI.png" alt="EAGLE GUI example" width="90%" />
</p>

## 現在のシステムでできること
入力画像または動画ごとに、現在のパイプラインは次を実行できます。

- pose 検出から人物を検出・追跡する
- すべての COCO オブジェクトクラスを保持する、または選択したクラスだけを保持する
- 追跡中の人物ごとに顔を 1 つ検出する
- 視線ヒートマップと視線点を推定する
- `left`、`up right`、`down`、`front` などの画面外方向ラベルを推定する（人物中心の向き）
- `annotation.csv` に ELAN 風の視線区間を生成する
- 設定が一致する場合、各フェーズのキャッシュを再利用する

人物検出では、EAGLE は pose keypoint も使い、可能な場合は視線を face、head、torso、arms、legs などの身体部位に割り当てます。

## 対応入力
- 画像: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tif`, `.tiff`, `.webp`
- 動画: `.mp4`, `.mov`, `.avi`, `.mkv`, `.m4v`, `.wmv`, `.webm`

## ダウンロード
ビルド済みアプリバンドルは GitHub Releases から取得できます。

- https://github.com/Taiga-Mori/EAGLE/releases

## セットアップ
仮想環境を作成して依存関係をインストールします。

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

推奨 Python バージョン: `3.10`

## アプリの起動
推奨:

```bash
python app.py
```

このランチャーは Streamlit を起動し、ブラウザでアプリを開きます。ポート `8501` が使えない場合は、近いポートを自動的に試します。

Streamlit を直接起動することもできます。

```bash
venv/bin/streamlit run app.py
```

## 初回起動とモデルダウンロード
初回利用時、EAGLE は必要に応じて次をダウンロードしてキャッシュします。

- 選択した YOLO object 重み: `yolo26n.pt`, `yolo26s.pt`, `yolo26m.pt`, `yolo26l.pt`, `yolo26x.pt` のいずれか
- `yolo26x-pose.pt`
- RetinaFace の学習済み重み
- GAZELLE の torch.hub ファイル
- `mobileone_s0.pt`

初回ロードに失敗した場合は、次を確認してください。

- マシンがオンラインであること
- GitHub / モデル配布エンドポイントに到達できること
- 現在のユーザーが `~/.EAGLE/` に書き込めること

## アプリのワークフロー
1. `python app.py` でアプリを起動します。
2. `Input file` を設定します。
   - macOS: `Browse` をクリックします。
   - Linux/Docker: `Container File Browser` を使うか、マウント済みパスを入力します。
3. 入力画像または動画を 1 つ選択します。
4. 検出されたメディア種別を確認します。
5. 必要に応じて `Output folder name` を編集します。
6. 必要に応じて `Detailed Settings` を開きます。
7. `Run Pipeline` をクリックします。
8. 最後に表示される出力パスを確認します。

出力ディレクトリの挙動:

- 入力ディレクトリに書き込み可能な場合、出力先の親ディレクトリは入力ファイルの親と同じです。
- 入力ディレクトリが読み取り専用の場合（Docker の `:ro` マウントでよくあります）、出力先は `/app/output/<input_stem>` にフォールバックします。
- デフォルトの出力フォルダ名は入力ファイルの stem です。
- 例:
  - 入力: `/data/session01/test.mp4`
  - 書き込み可能な親: `/data/session01/test`
  - 読み取り専用親のフォールバック: `/app/output/test`

## 主な設定

### Basic Settings
- `Input file`
  - macOS: `Browse` によってセットされる読み取り専用欄です。
  - Linux/Docker: 編集可能な欄です（手入力でパス指定できます）。
- `Output folder name`
  - 入力ファイルの隣に作られるフォルダ名です。
- `Device`
  - NVIDIA GPU が複数ある場合は `cuda:0`, `cuda:1`, ... を使います（それ以外は `mps` または `cpu`）。

### Inference
- `Person detection backend`
  - `yolo26x-pose` です。
- `Object detection backend`
  - YOLO26 detection backend（`yolo26n`, `yolo26s`, `yolo26m`, `yolo26l`, `yolo26x`）を選択します。EAGLE は `~/.EAGLE/` に選択中の object 重みだけを残します。
- `Face detection backend`
  - デフォルトは `retinaface` です。MediaPipe を使いたい場合は `mediapipe` を選びます。
- `Gaze detection backend`
  - `gazelle` です。
- `Head pose detection backend`
  - `mobileone` です。将来の頭部姿勢推定 backend 追加に備えた設定です。
- `Person threshold`, `Object threshold`, `Face threshold`, `Gaze threshold`
  - 人物姿勢追跡、人物以外のオブジェクト追跡、顔検出、視線 in/out 解釈に個別に使うしきい値です。
- `Visualization mode`
  - `both`, `point`, `heatmap`
- `Heatmap alpha`
  - ヒートマップ出力の重ね合わせ強度です。
- `Gaze target radius (px)`
  - `0` は点のみでターゲット割り当てを行います。大きい値では視線点の周囲円を使います。
- `Person part distance scale`
  - 視線が人物の keypoint からどれくらい離れていても身体部位として扱うかを制御します。
- `Person part keypoint min confidence`
  - 身体部位判定に使う pose keypoint confidence の下限です。デフォルトは `0.0` です。
- `Reuse existing persons.csv/objects.csv/faces.csv/gaze.csv when available`
  - メタデータが現在の実行と一致する場合、各フェーズのキャッシュを個別に再利用します。
- `Track all object classes`
  - オフにすると、保持する COCO クラスを明示的に選択できます。

### Temporal Settings
- `Person smoothing window`
  - 人物 bbox に対する平滑化窓です。
- `Object smoothing window`
  - 追跡オブジェクトの bbox に対する平滑化窓です。
- `Face smoothing window`
  - 補間済みの顔 bbox に対する平滑化窓です。
- `Gaze smoothing window`
  - 視線推定と画面外方向角度に対する平滑化窓です。
- `Person frame FPS`, `Object frame FPS`, `Face frame FPS`, `Gaze frame FPS`, `Head pose frame FPS`
  - 動画のみ。各ステージを指定した target FPS で実行し、それぞれの中間生成物を書き出します。

重要:

- 後段ステージの FPS は、前段ステージの設定が同じなら前段キャッシュを無効化せずに変更できます。Track、face、gaze、head pose は各ステージ実行後に補間または平滑化されます。
- 内部的には、target FPS 値は元動画の FPS から frame stride に変換されます。

### BoT-SORT
UI では [`config/botsort.yaml`](config/botsort.yaml) の以下の tracker 設定を変更できます。

- `track_high_thresh`
- `track_low_thresh`
- `new_track_thresh`
- `track_buffer`
- `match_thresh`
- `Enable ReID`
- `proximity_thresh`
- `appearance_thresh`

## 高速化のヒント
- NVIDIA GPU が使える環境では `cuda:*`、Apple Silicon では利用可能なら `mps` を使ってください。
- 各ステージの FPS を下げると処理量を減らせます。特に視線以外のステージで効果が出やすいです。
- 顔がほぼ正面向きなら、face detection backend は `mediapipe` も試してください。

## 出力ファイル
現在の出力には次が含まれます。

- `persons.csv`
  - 平滑化済みの人物追跡結果
- `.persons_meta.json`
  - person cache 再利用検証用のメタデータ
- `objects.csv`
  - 平滑化済みの非人物オブジェクト追跡結果
- `.objects_meta.json`
  - object cache 再利用検証用のメタデータ
- `faces.csv`
  - 人物 track に対応付けられた顔検出 bbox
- `.faces_meta.json`
  - face cache 再利用検証用のメタデータ
- `gaze.csv`
  - 平滑化済みの視線点、in/out 確率、画面外方向フィールド
- `gaze_heatmaps.npz`
  - dense な視線ヒートマップのキャッシュ
- `.gaze_meta.json`
  - gaze cache 再利用検証用のメタデータ
- `annotation.csv`
  - ELAN 風の視線区間、または単一画像ラベル
- `all_points.jpg`
  - 画像入力用の point 可視化
- `all_points.mp4`
  - 動画入力用の point 可視化
- `person_<track_id>_heatmap.jpg`
  - 画像入力用の人物別ヒートマップ出力
- `person_<track_id>_heatmap.mp4`
  - 動画入力用の人物別ヒートマップ出力

`temp/` や `heatmaps/` などの一時フォルダは export 後に削除されます。

## CSV の内容
`persons.csv` の列:

- `frame_idx`
- `cls`
- `track_id`
- `object_detected`
- `source`
- `conf`
- `x1`, `y1`, `x2`, `y2`
- `pose_keypoints`
- `label`

`objects.csv` の列:

- `frame_idx`
- `cls`
- `track_id`
- `object_detected`
- `source`
- `conf`
- `x1`, `y1`, `x2`, `y2`
- `pose_keypoints`
- `label`

`faces.csv` の列:

- `frame_idx`
- `track_id`
- `face_detected`
- `face_conf`
- `face_x1`, `face_y1`, `face_x2`, `face_y2`

`gaze.csv` の列:

- `frame_idx`
- `track_id`
- `gaze_detected`
- `inout`
- `x_gaze`, `y_gaze`
- `offscreen_direction`
- `offscreen_yaw`
- `offscreen_pitch`

`annotation.csv` の列:

- `tier`
- `start_time`
- `end_time`
- `gaze`

## 開発用エントリポイント
- [`app.py`](app.py)
  - Streamlit UI とランチャー
- [`main.py`](main.py)
  - ローカル用の最小スモークテスト入口
- [`eagle/pipeline.py`](eagle/pipeline.py)
  - メインの Python API facade

最小コード例:

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

## 免責事項
- このソフトウェアは自己責任で使用してください。
- EAGLE は検出、視線推定、ターゲット割り当て、エクスポートされた注釈の正確性を保証しません。
- 最終的な確認と修正の責任はユーザーにあります。

## ライセンス
このプロジェクトは `AGPL-3.0-or-later` でライセンスされています。リポジトリのライセンス本文は [LICENSE](LICENSE) を参照してください。

## 謝辞
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
