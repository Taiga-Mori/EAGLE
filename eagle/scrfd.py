from pathlib import Path

import cv2
import numpy as np


class SCRFDFaceDetector:
    """SCRFD face detector (InsightFace ONNX export with box + keypoint heads)."""

    def __init__(self, model_path: Path | str, device: str, max_size: int = 1280, nms_thresh: float = 0.4) -> None:
        import onnxruntime as ort

        providers = ["CPUExecutionProvider"]
        if device.startswith("cuda"):
            device_id = int(device.split(":", 1)[1]) if ":" in device else 0
            providers = [("CUDAExecutionProvider", {"device_id": device_id}), "CPUExecutionProvider"]
        options = ort.SessionOptions()
        options.log_severity_level = 3
        self.session = ort.InferenceSession(str(model_path), sess_options=options, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        num_outputs = len(self.output_names)
        if num_outputs not in (6, 9):
            raise RuntimeError(f"Unsupported SCRFD model: expected 6 or 9 outputs, got {num_outputs}")
        self.use_kps = num_outputs == 9
        self.strides = [8, 16, 32]
        self.num_anchors = 2
        self.max_size = int(max_size)
        self.nms_thresh = float(nms_thresh)
        self._center_cache: dict[tuple[int, int, int], np.ndarray] = {}

    def _anchor_centers(self, height: int, width: int, stride: int) -> np.ndarray:
        key = (height, width, stride)
        centers = self._center_cache.get(key)
        if centers is None:
            grid = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
            centers = (grid * stride).reshape(-1, 2)
            centers = np.repeat(centers, self.num_anchors, axis=0)
            self._center_cache[key] = centers
        return centers

    def detect(self, frame: np.ndarray, score_thresh: float) -> list[dict]:
        """Detect faces in a BGR frame. Returns RetinaFace-style dicts with bbox, score and landmarks."""
        height, width = frame.shape[:2]
        scale = min(1.0, self.max_size / float(max(height, width))) if self.max_size > 0 else 1.0
        resized_w = max(32, int(round(width * scale)))
        resized_h = max(32, int(round(height * scale)))
        input_w = ((resized_w + 31) // 32) * 32
        input_h = ((resized_h + 31) // 32) * 32
        resized = cv2.resize(frame, (resized_w, resized_h)) if scale != 1.0 else frame
        padded = np.zeros((input_h, input_w, 3), dtype=np.uint8)
        padded[:resized_h, :resized_w] = resized
        blob = cv2.dnn.blobFromImage(padded, 1.0 / 128.0, (input_w, input_h), (127.5, 127.5, 127.5), swapRB=True)
        outputs = self.session.run(self.output_names, {self.input_name: blob})

        fmc = len(self.strides)
        all_scores, all_boxes, all_kps = [], [], []
        for index, stride in enumerate(self.strides):
            scores = outputs[index].reshape(-1)
            keep = np.where(scores >= score_thresh)[0]
            if keep.size == 0:
                continue
            centers = self._anchor_centers(input_h // stride, input_w // stride, stride)[keep]
            distances = outputs[index + fmc].reshape(-1, 4)[keep] * stride
            boxes = np.concatenate([centers - distances[:, :2], centers + distances[:, 2:]], axis=1)
            all_scores.append(scores[keep])
            all_boxes.append(boxes)
            if self.use_kps:
                kps = outputs[index + fmc * 2].reshape(-1, 10)[keep] * stride
                kps = kps.reshape(-1, 5, 2) + centers[:, None, :]
                all_kps.append(kps)
        if not all_scores:
            return []

        scores = np.concatenate(all_scores)
        boxes = np.concatenate(all_boxes) / scale
        kps = np.concatenate(all_kps) / scale if all_kps else None
        nms_boxes = [[float(x1), float(y1), float(x2 - x1), float(y2 - y1)] for x1, y1, x2, y2 in boxes]
        keep = cv2.dnn.NMSBoxes(nms_boxes, scores.astype(float).tolist(), float(score_thresh), self.nms_thresh)
        keep = np.array(keep, dtype=int).reshape(-1)

        faces: list[dict] = []
        for index in keep[np.argsort(-scores[keep])]:
            x1, y1, x2, y2 = boxes[index]
            x1 = int(np.clip(x1, 0, width - 1))
            y1 = int(np.clip(y1, 0, height - 1))
            x2 = int(np.clip(x2, x1 + 1, width))
            y2 = int(np.clip(y2, y1 + 1, height))
            face = {"bbox": [x1, y1, x2, y2], "score": float(scores[index])}
            if kps is not None:
                face["landmarks"] = kps[index].tolist()
            faces.append(face)
        return faces
