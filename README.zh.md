# EAGLE
<p align="center">
  <em><strong>E</strong>nd-to-end <strong>A</strong>utomatic <strong>G</strong>aze <strong>L</strong>ab<strong>E</strong>ling tool for interaction studies</em>
</p>

<p align="center">
  <img src="assets/icon_name_trans.png" alt="EAGLE app icon" width="280" />
</p>

## 语言

- [English](README.md)
- [日本語](README.ja.md)
- [简体中文](README.zh.md)
- [Español](README.es.md)

## 概述
EAGLE 是一个基于 Streamlit 的视线标注辅助工具，用于图像和视频分析。它结合了：

- 使用 YOLO pose 进行人物跟踪
- 使用 YOLO detection 进行非人物对象跟踪
- 默认使用 RetinaFace 进行人脸检测，也可以选择 MediaPipe
- 使用 GAZELLE 进行视线热力图估计
- 使用 MobileOne gaze 进行画面外方向估计
- CSV 导出以及带标注的图像/视频导出

EAGLE 是标注辅助工具，并不保证输出就是真实标注。在用于研究、分析、报告或决策之前，必须检查并验证所有输出。

## 输出示例
<p align="center">
  <img src="assets/book.gif" alt="EAGLE example output: book" width="31%" />
  <img src="assets/interview.gif" alt="EAGLE example output: interview" width="31%" />
  <img src="assets/three.gif" alt="EAGLE example output: three people" width="31%" />
</p>

## GUI 示例
<p align="center">
  <img src="assets/GUI.png" alt="EAGLE GUI example" width="90%" />
</p>

## 当前系统的功能
对于每个输入图像或视频，当前流程可以：

- 从姿态检测结果中检测并跟踪人物
- 可选择保留所有 COCO 对象类别，或只保留选定类别
- 为每个被跟踪的人物检测一个人脸
- 估计视线热力图和视线点
- 推断 `left`、`up right`、`down`、`front` 等画面外方向标签（以人物自身方向为基准）
- 在 `annotation.csv` 中生成 ELAN 风格的视线片段
- 当设置仍然匹配时，复用各阶段缓存

对于人物检测，EAGLE 还会使用 pose keypoints，在可能时把视线分配到 face、head、torso、arms、legs 等身体部位。

## 支持的输入
- 图像: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tif`, `.tiff`, `.webp`
- 视频: `.mp4`, `.mov`, `.avi`, `.mkv`, `.m4v`, `.wmv`, `.webm`

## 下载
预构建应用包可从 GitHub Releases 获取：

- https://github.com/Taiga-Mori/EAGLE/releases

## 安装
创建虚拟环境并安装依赖：

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

推荐 Python 版本：`3.10`

## 启动应用
推荐方式：

```bash
python app.py
```

该启动器会为你启动 Streamlit，并在浏览器中打开应用。如果端口 `8501` 不可用，它会自动尝试附近端口。

也可以直接运行 Streamlit：

```bash
venv/bin/streamlit run app.py
```

## 首次运行和模型下载
首次使用时，EAGLE 可能会下载并缓存：

- 选定的 YOLO object 权重：`yolo26n.pt`、`yolo26s.pt`、`yolo26m.pt`、`yolo26l.pt` 或 `yolo26x.pt`
- `yolo26x-pose.pt`
- RetinaFace 预训练权重
- GAZELLE torch.hub 文件
- `mobileone_s0.pt`

如果首次加载失败，请检查：

- 机器是否联网
- 是否能访问 GitHub / 模型托管端点
- 当前用户是否可以写入 `~/.EAGLE/`

## 应用工作流
1. 使用 `python app.py` 启动应用。
2. 设置 `Input file`：
   - macOS/Windows：点击 `Browse`。
   - Linux/Docker：使用 `Container File Browser`，或输入已挂载路径。
3. 选择一个输入图像或视频。
4. 确认检测到的媒体类型。
5. 如有需要，编辑 `Output folder name`。
6. 如有需要，打开 `Detailed Settings`。
7. 点击 `Run Pipeline`。
8. 查看最后显示的输出路径。

输出目录行为：

- 如果输入目录可写，输出父目录与输入文件的父目录相同。
- 如果输入目录只读（Docker `:ro` 挂载中常见），输出会回退到 `/app/output/<input_stem>`。
- 默认输出文件夹名是输入文件的 stem。
- 示例：
  - 输入: `/data/session01/test.mp4`
  - 可写父目录: `/data/session01/test`
  - 只读父目录回退: `/app/output/test`

## 主要设置

### Basic Settings
- `Input file`
  - macOS/Windows：可编辑字段，也可以由 `Browse` 填入。
  - Linux/Docker：可编辑字段（支持手动输入路径）。
- `Output folder name`
  - 在输入文件旁边创建的文件夹名称。
- `Device`
  - 当有多个 NVIDIA GPU 时使用 `cuda:0`, `cuda:1`, ...（否则使用 `mps` 或 `cpu`）。

### Inference
- `Person detection backend`
  - `yolo26x-pose`。
- `Object detection backend`
  - 选择 YOLO26 detection backend（`yolo26n`、`yolo26s`、`yolo26m`、`yolo26l` 或 `yolo26x`）。EAGLE 在 `~/.EAGLE/` 中只保留当前选定的 object 权重。
- `Face detection backend`
  - 默认是 `retinaface`；如果想使用 MediaPipe，可以选择 `mediapipe`。
- `Gaze detection backend`
  - `gazelle`。
- `Head pose detection backend`
  - `mobileone`。这个设置是为了以后扩展头部姿态估计 backend。
- `Person threshold`, `Object threshold`, `Face threshold`, `Gaze threshold`
  - 分别用于人物姿态跟踪、非人物对象跟踪、人脸检测和视线 in/out 解释的阈值。
- `Visualization mode`
  - `both`, `point`, `heatmap`
- `Heatmap alpha`
  - 热力图输出的叠加强度。
- `Gaze target radius (px)`
  - `0` 表示只用点进行目标分配。更大的值会使用视线点周围的圆形区域。
- `Person part distance scale`
  - 控制视线距离人物 keypoint 多远时仍可被视为对应身体部位。
- `Person part keypoint min confidence`
  - 用于身体部位分配的 pose keypoint confidence 下限。默认值为 `0.0`。
- `Reuse existing persons.csv/objects.csv/faces.csv/gaze.csv when available`
  - 当元数据仍匹配当前运行时，分别复用各阶段缓存。
- `Track all object classes`
  - 关闭时，可以明确选择要保留的 COCO 类别。

### Temporal Settings
- `Person smoothing window`
  - 人物框的平滑窗口。
- `Object smoothing window`
  - 跟踪对象框的平滑窗口。
- `Face smoothing window`
  - 插值后人脸框的平滑窗口。
- `Gaze smoothing window`
  - 视线估计和画面外方向角度的平滑窗口。
- `Person frame FPS`, `Object frame FPS`, `Face frame FPS`, `Gaze frame FPS`, `Head pose frame FPS`
  - 仅视频。每个阶段按所选 target FPS 运行，并写出自己的中间输出。

重要：

- 后续阶段的 FPS 可以在自身设置不变时调整，而不会使前序阶段缓存失效。Track、face、gaze 和 head pose 会在各阶段运行后进行插值或平滑。
- 在内部，target FPS 会根据源 FPS 转换为 frame stride。

### BoT-SORT
UI 暴露了 [`config/botsort.yaml`](config/botsort.yaml) 中的这些 tracker 设置：

- `track_high_thresh`
- `track_low_thresh`
- `new_track_thresh`
- `track_buffer`
- `match_thresh`
- `Enable ReID`
- `proximity_thresh`
- `appearance_thresh`

## 加速提示
- 在 NVIDIA GPU 上使用 `cuda:*`；在 Apple Silicon 上如可用则使用 `mps`。
- 降低各阶段 FPS 可以减少重复检测工作，尤其是 gaze 以外的阶段。
- 如果人脸大多是正面，可以尝试将 face detection backend 设为 `mediapipe`。

## 输出文件
当前输出可能包括：

- `persons.csv`
  - 平滑后的人物跟踪结果
- `.persons_meta.json`
  - 用于验证 person cache 复用的元数据
- `objects.csv`
  - 平滑后的非人物对象跟踪结果
- `.objects_meta.json`
  - 用于验证 object cache 复用的元数据
- `faces.csv`
  - 分配到人物 track 的人脸检测框
- `.faces_meta.json`
  - 用于验证 face cache 复用的元数据
- `gaze.csv`
  - 平滑后的视线点、in/out 概率和画面外方向字段
- `gaze_heatmaps.npz`
  - 缓存的 dense 视线热力图
- `.gaze_meta.json`
  - 用于验证 gaze cache 复用的元数据
- `annotation.csv`
  - ELAN 风格的视线片段或单图像标签
- `all_points.jpg`
  - 图像输入的点可视化
- `all_points.mp4`
  - 视频输入的点可视化
- `person_<track_id>_heatmap.jpg`
  - 图像输入的人物级热力图输出
- `person_<track_id>_heatmap.mp4`
  - 视频输入的人物级热力图输出

`temp/` 和 `heatmaps/` 等临时文件夹会在导出后删除。

## CSV 内容
`persons.csv` 列：

- `frame_idx`
- `cls`
- `track_id`
- `object_detected`
- `source`
- `conf`
- `x1`, `y1`, `x2`, `y2`
- `pose_keypoints`
- `label`

`objects.csv` 列：

- `frame_idx`
- `cls`
- `track_id`
- `object_detected`
- `source`
- `conf`
- `x1`, `y1`, `x2`, `y2`
- `pose_keypoints`
- `label`

`faces.csv` 列：

- `frame_idx`
- `track_id`
- `face_detected`
- `face_conf`
- `face_x1`, `face_y1`, `face_x2`, `face_y2`

`gaze.csv` 列：

- `frame_idx`
- `track_id`
- `gaze_detected`
- `inout`
- `x_gaze`, `y_gaze`
- `offscreen_direction`
- `offscreen_yaw`
- `offscreen_pitch`

`annotation.csv` 列：

- `tier`
- `start_time`
- `end_time`
- `gaze`

## 开发入口
- [`app.py`](app.py)
  - Streamlit UI 和启动器
- [`main.py`](main.py)
  - 最小本地 smoke-test 入口
- [`eagle/pipeline.py`](eagle/pipeline.py)
  - 主 Python API facade

最小代码用法：

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

## 免责声明
- 使用本软件需自行承担风险。
- EAGLE 不保证检测、视线估计、目标分配或导出标注的正确性。
- 最终审查和修正责任由用户承担。

## 许可证
本项目采用 `AGPL-3.0-or-later` 许可证。仓库许可证正文见 [LICENSE](LICENSE)。

## 致谢
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
