import os
import logging
import shutil
import ssl
import sys
import warnings
from contextlib import contextmanager
from pathlib import Path
from urllib.request import urlopen

os.environ.setdefault("YOLO_VERBOSE", "False")
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
warnings.filterwarnings("ignore", message="Error fetching version info.*", module="albumentations.*")

import torch
import torch.nn.functional as F
from torchvision import transforms
from retinaface.pre_trained_models import get_model
from ultralytics import YOLO

try:
    from ultralytics.utils import LOGGER as ULTRALYTICS_LOGGER
except Exception:
    ULTRALYTICS_LOGGER = None

from .mobile_gaze import mobileone_s0_gaze
from .constants import (
    DEFAULT_FACE_DETECTION_BACKEND,
    DEFAULT_GAZE_DETECTION_BACKEND,
    DEFAULT_HEAD_POSE_DETECTION_BACKEND,
    DEFAULT_OBJECT_DETECTION_BACKEND,
    DEFAULT_PERSON_DETECTION_BACKEND,
    GAZE_DETECTION_BACKENDS,
    HEAD_POSE_DETECTION_BACKENDS,
    OBJECT_DETECTION_BACKENDS,
    PERSON_DETECTION_BACKENDS,
)
from .types import AppPaths

MEDIAPIPE_FACE_DETECTOR_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
)


@contextmanager
def _suppress_native_stderr():
    """Temporarily silence native libraries that write directly to stderr."""
    saved_stderr_fd = None
    try:
        sys.stderr.flush()
        stderr_fd = sys.stderr.fileno()
        saved_stderr_fd = os.dup(stderr_fd)
        with open(os.devnull, "w") as devnull:
            os.dup2(devnull.fileno(), stderr_fd)
            yield
    except Exception:
        raise
    finally:
        if saved_stderr_fd is not None:
            os.dup2(saved_stderr_fd, stderr_fd)
            os.close(saved_stderr_fd)


class ModelManager:
    """Download, cache, and serve external ML models."""

    def __init__(self, paths: AppPaths) -> None:
        self.paths = paths
        self.yolo: YOLO | None = None
        self.yolo_pose: YOLO | None = None
        self.retinaface = None
        self.mediapipe_face_detector = None
        self.mediapipe_face_detector_api: str | None = None
        self.gazelle = None
        self.gazelle_transform = None
        self.mobile_gaze = None
        self.mobile_gaze_transform = None
        self.loaded_device: str | None = None
        self.loaded_object_detection_backend: str | None = None
        self.loaded_person_detection_backend: str | None = None
        self._configure_download_environment()

    def _configure_download_environment(self) -> None:
        os.environ.setdefault("TORCH_HOME", str(self.paths.torch_home))
        cache_dir = self.paths.app_dir / "cache"
        matplotlib_cache_dir = cache_dir / "matplotlib"
        cache_dir.mkdir(parents=True, exist_ok=True)
        matplotlib_cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))
        os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache_dir))
        if ULTRALYTICS_LOGGER is not None:
            ULTRALYTICS_LOGGER.setLevel(logging.ERROR)
        self._ensure_torch_attention_compat()
        try:
            import certifi

            cert_path = certifi.where()
            os.environ.setdefault("SSL_CERT_FILE", cert_path)
            os.environ.setdefault("REQUESTS_CA_BUNDLE", cert_path)
            self._ssl_context = ssl.create_default_context(cafile=cert_path)
        except Exception:
            self._ssl_context = None
        torch.hub.set_dir(str(self.paths.torch_hub_dir))

    def _ensure_torch_attention_compat(self) -> None:
        """Provide torch<2.0 compatibility for models expecting SDP attention."""
        if hasattr(F, "scaled_dot_product_attention"):
            return

        def _scaled_dot_product_attention(
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            attn_mask: torch.Tensor | None = None,
            dropout_p: float = 0.0,
            is_causal: bool = False,
            scale: float | None = None,
        ) -> torch.Tensor:
            d_k = query.size(-1)
            scale_factor = (1.0 / (d_k**0.5)) if scale is None else float(scale)
            attn = torch.matmul(query, key.transpose(-2, -1)) * scale_factor

            if is_causal:
                q_len = query.size(-2)
                k_len = key.size(-2)
                causal_mask = torch.ones((q_len, k_len), device=query.device, dtype=torch.bool).triu(1)
                attn = attn.masked_fill(causal_mask, float("-inf"))

            if attn_mask is not None:
                if attn_mask.dtype == torch.bool:
                    attn = attn.masked_fill(~attn_mask, float("-inf"))
                else:
                    attn = attn + attn_mask

            attn = torch.softmax(attn, dim=-1)
            if dropout_p and dropout_p > 0.0:
                attn = torch.dropout(attn, dropout_p, train=False)
            return torch.matmul(attn, value)

        F.scaled_dot_product_attention = _scaled_dot_product_attention

    def yolo_object_path(self, backend: str) -> Path:
        return self.paths.app_dir / f"{backend}.pt"

    def ensure_yolo_object_weights(self, backend: str) -> None:
        if backend not in OBJECT_DETECTION_BACKENDS:
            raise ValueError(f"Unsupported object detection backend '{backend}'.")
        self._remove_unselected_yolo_object_weights(backend)
        self._ensure_download(
            self.yolo_object_path(backend),
            OBJECT_DETECTION_BACKENDS[backend],
            f"YOLO object weights ({backend})",
        )

    def _remove_unselected_yolo_object_weights(self, selected_backend: str) -> None:
        selected_path = self.yolo_object_path(selected_backend)
        for backend in OBJECT_DETECTION_BACKENDS:
            candidate = self.yolo_object_path(backend)
            if candidate == selected_path or not candidate.exists():
                continue
            try:
                candidate.unlink()
                print(f"Removed unused YOLO object weights: {candidate}", flush=True)
            except Exception as exc:
                raise RuntimeError(f"Failed to remove unused YOLO object weights: {candidate}\nOriginal error: {exc}") from exc

    def ensure_person_detection_weights(self, backend: str) -> None:
        if backend not in PERSON_DETECTION_BACKENDS:
            raise ValueError(f"Unsupported person detection backend '{backend}'.")
        self._ensure_download(
            self.paths.yolo_pose_path,
            PERSON_DETECTION_BACKENDS[backend],
            f"Person detection weights ({backend})",
        )

    def ensure_mobile_gaze_weights(self) -> None:
        self._ensure_download(
            self.paths.mobile_gaze_path,
            "https://github.com/yakhyo/gaze-estimation/releases/download/weights/mobileone_s0.pt",
            "MobileGaze weights",
        )

    def ensure_mediapipe_face_detector_weights(self) -> None:
        self._ensure_download(
            self.paths.mediapipe_face_detector_path,
            MEDIAPIPE_FACE_DETECTOR_URL,
            "MediaPipe face detector model",
        )

    def _ensure_download(self, destination, url: str, label: str) -> None:
        if destination.is_file():
            return
        try:
            with urlopen(url, context=self._ssl_context) as response, destination.open("wb") as output:
                shutil.copyfileobj(response, output)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to download {label} on first run.\n"
                f"Expected cache path: {destination}\n"
                "Please make sure this machine is connected to the internet and try again.\n"
                f"Original error: {exc}"
            ) from exc

    def _load_mediapipe_face_detector(self) -> None:
        try:
            import mediapipe as mp

            if hasattr(mp, "solutions"):
                with _suppress_native_stderr():
                    self.mediapipe_face_detector = mp.solutions.face_detection.FaceDetection(
                        model_selection=1,
                        min_detection_confidence=0.0,
                    )
                self.mediapipe_face_detector_api = "solutions"
                return

            self.ensure_mediapipe_face_detector_weights()
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision

            options = mp_vision.FaceDetectorOptions(
                base_options=mp_python.BaseOptions(
                    model_asset_path=str(self.paths.mediapipe_face_detector_path),
                ),
                running_mode=mp_vision.RunningMode.IMAGE,
                min_detection_confidence=0.0,
            )
            with _suppress_native_stderr():
                self.mediapipe_face_detector = mp_vision.FaceDetector.create_from_options(options)
            self.mediapipe_face_detector_api = "tasks"
        except Exception as exc:
            raise RuntimeError(
                "Failed to initialize MediaPipe face detection.\n"
                "Install mediapipe or select RetinaFace as the face detection backend.\n"
                f"Original error: {exc}"
            ) from exc

    def load(
        self,
        device: str,
        person_detection_backend: str = DEFAULT_PERSON_DETECTION_BACKEND,
        object_detection_backend: str = DEFAULT_OBJECT_DETECTION_BACKEND,
        face_detection_backend: str = DEFAULT_FACE_DETECTION_BACKEND,
        gaze_detection_backend: str = DEFAULT_GAZE_DETECTION_BACKEND,
        head_pose_detection_backend: str = DEFAULT_HEAD_POSE_DETECTION_BACKEND,
    ) -> None:
        if device.startswith("cuda:"):
            cuda_device = torch.device(device)
            if cuda_device.index is None or cuda_device.index >= torch.cuda.device_count():
                raise ValueError(
                    f"Unsupported CUDA device '{device}'. Available CUDA devices: "
                    f"{', '.join(f'cuda:{idx}' for idx in range(torch.cuda.device_count()))}"
                )
            try:
                torch.cuda.set_device(cuda_device)
                probe = torch.ones((1,), device=cuda_device)
                _ = probe + 1
                torch.cuda.synchronize(cuda_device)
            except Exception as exc:
                raise RuntimeError(
                    f"CUDA device '{device}' is visible but cannot run kernels with the installed PyTorch build.\n"
                    "On Windows, this usually means the PyTorch CUDA wheel does not support this GPU's compute "
                    "capability, or the NVIDIA driver/PyTorch CUDA runtime combination is incompatible.\n"
                    "Install a PyTorch build that supports this GPU, update the NVIDIA driver, or select 'cpu'.\n"
                    f"Original error: {exc}"
                ) from exc
            print(f"Using CUDA device {device}: {torch.cuda.get_device_name(cuda_device.index)}", flush=True)
        if gaze_detection_backend not in GAZE_DETECTION_BACKENDS:
            raise ValueError(f"Unsupported gaze detection backend '{gaze_detection_backend}'.")
        if head_pose_detection_backend not in HEAD_POSE_DETECTION_BACKENDS:
            raise ValueError(f"Unsupported head pose detection backend '{head_pose_detection_backend}'.")
        self.ensure_yolo_object_weights(object_detection_backend)
        self.ensure_person_detection_weights(person_detection_backend)
        if head_pose_detection_backend == "mobileone":
            self.ensure_mobile_gaze_weights()
        if self.yolo is None or self.loaded_object_detection_backend != object_detection_backend:
            self.yolo = YOLO(self.yolo_object_path(object_detection_backend))
            self.loaded_object_detection_backend = object_detection_backend
        if self.yolo_pose is None or self.loaded_person_detection_backend != person_detection_backend:
            self.yolo_pose = YOLO(self.paths.yolo_pose_path)
            self.loaded_person_detection_backend = person_detection_backend
        if face_detection_backend == "mediapipe" and self.mediapipe_face_detector is None:
            self._load_mediapipe_face_detector()
        if face_detection_backend == "retinaface" and (self.retinaface is None or self.loaded_device != device):
            try:
                self.retinaface = get_model("resnet50_2020-07-20", max_size=2048, device=device)
                self.retinaface.eval()
            except Exception as exc:
                raise RuntimeError(
                    "Failed to load RetinaFace on first run.\n"
                    "RetinaFace may need to download pretrained weights.\n"
                    "Please make sure this machine is connected to the internet and try again.\n"
                    f"Original error: {exc}"
                ) from exc
        if self.gazelle is None or self.gazelle_transform is None:
            try:
                self.gazelle, self.gazelle_transform = torch.hub.load(
                    "fkryan/gazelle",
                    "gazelle_dinov2_vitl14_inout",
                    source="github",
                    skip_validation=True,
                    trust_repo=True,
                )
                self.gazelle.eval()
            except Exception as exc:
                raise RuntimeError(
                    "Failed to load the GAZELLE model on first run.\n"
                    "GAZELLE is loaded through torch.hub and may need to download model files.\n"
                    f"Torch hub cache directory: {self.paths.torch_hub_dir}\n"
                    "Please make sure this machine is connected to the internet and try again.\n"
                    f"Original error: {exc}"
                ) from exc
        if head_pose_detection_backend == "mobileone" and (
            self.mobile_gaze is None or self.mobile_gaze_transform is None or self.loaded_device != device
        ):
            try:
                model = mobileone_s0_gaze(num_classes=90)
                state_dict = torch.load(self.paths.mobile_gaze_path, map_location=device)
                model.load_state_dict(state_dict)
                model.eval()
                self.mobile_gaze = model.to(device)
                self.mobile_gaze_transform = transforms.Compose(
                    [
                        transforms.ToPILImage(),
                        transforms.Resize((448, 448)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ]
                )
            except Exception as exc:
                raise RuntimeError(
                    "Failed to load the MobileGaze model on first run.\n"
                    f"Expected weight path: {self.paths.mobile_gaze_path}\n"
                    "Please make sure this machine is connected to the internet and try again.\n"
                    f"Original error: {exc}"
                ) from exc
        self.gazelle.to(device)
        self.loaded_device = device
