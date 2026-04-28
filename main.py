from eagle import EAGLE
from eagle.progress import ConsoleProgress


if __name__ == "__main__":
    """
    English: Minimal manual smoke test for local development.
    Change the paths and runtime options as needed before running.
    日本語: ローカル開発用の最小スモークテスト。
    実行前にパスや設定を必要に応じて変更してください。
    """

    eagle = EAGLE()
    eagle.preprocess(
        input_path="",
        output_dir="",
        object_target_fps=15,
        gaze_target_fps=15,
        det_thresh=0.5,
        device="cpu",
        visualization_mode="both",
        heatmap_alpha=0.35,
        gaze_point_method="peak_region_centroid",
        gaze_target_radius=15,
        person_part_distance_scale=0.10,
        object_smoothing_window=5,
        gaze_smoothing_window=5,
        selected_object_classes=["person"],
        reuse_cached_gaze=False,
    )
    eagle.update_botsort_yaml(
        {
            "track_high_thresh": 0.7,
            "track_low_thresh": 0.2,
            "new_track_thresh": 0.8,
            "track_buffer": 60,
            "match_thresh": 0.9,
            "with_reid": False,
        }
    )
    progress = ConsoleProgress()
    print("Starting pipeline...", flush=True)
    results = eagle.run_all(progress_bar=progress)

    print("Objects CSV:", eagle.context.objects_path)
    print("Gaze CSV:", eagle.context.gaze_path)
    print("Annotation CSV:", eagle.context.annotation_path)
    print("Visualization Output:", results["media_output_paths"])
