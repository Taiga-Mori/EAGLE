from pathlib import Path

from eagle import EAGLE
from eagle.progress import ConsoleProgress, format_elapsed


INPUT_DIR = Path("/home/mori/Data/ADOS/TD2026")


def iter_target_videos(input_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in input_dir.glob("*.mp4")
        if path.is_file() and path.name.endswith("VA.mp4")
    )


def process_video(eagle: EAGLE, input_path: Path) -> dict:
    output_dir = input_path.with_suffix("")

    eagle.preprocess(
        input_path=input_path,
        output_dir=output_dir,
        person_target_fps=15,
        object_target_fps=15,
        face_target_fps=15,
        gaze_target_fps=15,
        head_pose_target_fps=15,
        person_det_thresh=0.3,
        object_det_thresh=0.5,
        face_det_thresh=0.5,
        gaze_det_thresh=0.5,
        person_detection_backend="yolo26x-pose",
        object_detection_backend="yolo26x",
        gaze_detection_backend="gazelle",
        head_pose_detection_backend="l2cs",
        device="cuda:4",
        visualization_mode="both",
        heatmap_alpha=0.35,
        face_detection_backend="retinaface",
        gaze_point_method="peak_region_centroid",
        gaze_target_radius=15,
        person_part_distance_scale=0.10,
        person_part_min_conf=0.0,
        face_fallback_min_size_scale=1.2,
        person_smoothing_window=5,
        person_max_switch_gap=15,
        object_smoothing_window=5,
        face_smoothing_window=5,
        gaze_smoothing_window=5,
        selected_object_classes=["person"],
        reuse_cached_persons=False,
        reuse_cached_objects=False,
        reuse_cached_faces=False,
        reuse_cached_gaze=False,
    )
    eagle.update_botsort_yaml(
        {
            "track_high_thresh": 0.7,
            "track_low_thresh": 0.2,
            "new_track_thresh": 0.5,
            "track_buffer": 150,
            "match_thresh": 0.9,
            "with_reid": False,
        }
    )
    progress = ConsoleProgress()
    print(f"Starting pipeline: {input_path}", flush=True)
    results = eagle.run_all(progress_bar=progress)

    print("Persons CSV:", eagle.context.persons_path)
    print("Objects CSV:", eagle.context.objects_path)
    print("Faces CSV:", eagle.context.faces_path)
    print("Gaze CSV:", eagle.context.gaze_path)
    print("Annotation CSV:", eagle.context.annotation_path)
    print("Visualization Output:", results["media_output_paths"])
    print("Elapsed:", format_elapsed(float(results["elapsed_seconds"])))
    return results


if __name__ == "__main__":
    """
    English: Batch runner for mp4 files ending with VA.mp4 in INPUT_DIR.
    日本語: INPUT_DIR 内の VA.mp4 で終わる mp4 をまとめて処理するための実行ファイル。
    """

    input_paths = iter_target_videos(INPUT_DIR)
    if not input_paths:
        raise FileNotFoundError(f"No mp4 files ending with VA.mp4 found in {INPUT_DIR}")

    eagle = EAGLE()
    failures = []
    total = len(input_paths)
    print(f"Found {total} target videos in {INPUT_DIR}", flush=True)
    for index, input_path in enumerate(input_paths, start=1):
        print(f"\n[{index}/{total}] {input_path}", flush=True)
        try:
            process_video(eagle, input_path)
        except Exception as error:
            failures.append((input_path, error))
            print(f"Failed: {input_path}", flush=True)
            print(f"Error: {error}", flush=True)

    succeeded = total - len(failures)
    print(f"\nCompleted batch: {succeeded}/{total} succeeded.", flush=True)
    if failures:
        print("Failures:", flush=True)
        for failed_path, error in failures:
            print(f"- {failed_path}: {error}", flush=True)
        raise SystemExit(1)
