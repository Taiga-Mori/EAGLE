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

import numpy as np
import torch
import torch.nn.functional as F
from filelock import FileLock
from torchvision import transforms
from iglovikov_helper_functions.utils.image_utils import pad_to_size, unpad_from_size
from retinaface.box_utils import decode, decode_landm
from retinaface.pre_trained_models import get_model
from torchvision.ops import nms
from ultralytics import YOLO

try:
    from ultralytics.utils import LOGGER as ULTRALYTICS_LOGGER
except Exception:
    ULTRALYTICS_LOGGER = None

from .l2cs import l2cs_resnet50
from .mobile_gaze import mobileone_s0_gaze
from .gat_gaze import GAT_CHECKPOINT_URL, load_gat_model
from .scrfd import SCRFDFaceDetector
from .coreml import PaGECoreML, RetinaFaceCoreML, coreml_enabled, yolo_coreml_path
from .gaze_models import (
    GAZELLE_DINOV3_MODELS,
    PAGE_MODELS,
    PaGEGazeModel,
    TransformGazeModel,
    load_gazelle_dinov3_model,
    load_page_model,
)
from .constants import (
    FACE_CROP_SIZE,
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
SCRFD_MODEL_PACK_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
SCRFD_MODEL_PACK_MEMBER = "det_10g.onnx"
SCRFD_MAX_SIZE = 960
L2CS_GAZE_URL = (
    "https://huggingface.co/dorni/SpeakerVid-5M-data-curation-models/resolve/"
    "5c6e04d7fa3321e6228e79162f8ec98466bf308a/L2CSNet_gaze360.pkl?download=true"
)


def retinaface_detect_batch(
    retinaface, images: list[np.ndarray], confidence_threshold: float = 0.7, nms_threshold: float = 0.4
) -> list[list[dict]]:
    """Run RetinaFace on several images in one forward pass.

    Matches retinaface's predict_jsons, which takes one image per call, and runs the network in
    Core ML when ModelManager attached a conversion, otherwise in whatever precision its weights are in. On MPS, 3 head crops in one batch took 30 ms each against
    43 ms one at a time.
    """
    if not images:
        return []
    max_size = retinaface.max_size
    padded = [pad_to_size(target_size=(max_size, max_size), image=retinaface.transform(image=image)["image"]) for image in images]
    batch = torch.from_numpy(np.stack([item["image"] for item in padded]).transpose(0, 3, 1, 2))
    with torch.inference_mode():
        coreml = getattr(retinaface, "coreml", None)
        if coreml is not None:
            loc, conf, land = coreml(batch.numpy())
            prior_box = retinaface.prior_box.cpu()
        else:
            weight_dtype = next(retinaface.model.parameters()).dtype
            loc, conf, land = retinaface.model(batch.to(retinaface.device, weight_dtype))
            prior_box = retinaface.prior_box
        loc, conf, land = loc.float(), conf.float().softmax(dim=-1), land.float()
        scale = torch.tensor([max_size, max_size], device=loc.device, dtype=torch.float32)
        results: list[list[dict]] = []
        for index, (image, item) in enumerate(zip(images, padded)):
            boxes = decode(loc[index], prior_box, retinaface.variance) * scale.repeat(2)
            landmarks = decode_landm(land[index], prior_box, retinaface.variance) * scale.repeat(5)
            scores = conf[index][:, 1]
            valid = torch.where(scores > confidence_threshold)[0]
            boxes, landmarks, scores = boxes[valid], landmarks[valid], scores[valid]
            order = scores.argsort(descending=True)
            boxes, landmarks, scores = boxes[order], landmarks[order], scores[order]
            keep = nms(boxes, scores, nms_threshold)
            results.append(_retinaface_annotations(image, item["pads"], max_size, boxes[keep].int(), landmarks[keep], scores[keep]))
    return results


def _retinaface_annotations(image, pads, max_size: int, boxes, landmarks, scores) -> list[dict]:
    if boxes.shape[0] == 0:
        return []
    original_height, original_width = image.shape[:2]
    unpadded = unpad_from_size(pads, bboxes=boxes.cpu().numpy(), keypoints=landmarks.cpu().numpy().reshape([-1, 2]))
    resize_coeff = max(original_height, original_width) / max_size
    boxes = (unpadded["bboxes"] * resize_coeff).astype(int)
    landmarks = (unpadded["keypoints"].reshape(-1, 10) * resize_coeff).astype(int)
    scores = scores.cpu().numpy().astype(np.float64)
    annotations = []
    for box_id, bbox in enumerate(boxes):
        x_min, y_min, x_max, y_max = bbox
        x_min = np.clip(x_min, 0, original_width - 1)
        x_max = np.clip(x_max, x_min + 1, original_width - 1)
        y_min = np.clip(y_min, 0, original_height - 1)
        y_max = np.clip(y_max, y_min + 1, original_height - 1)
        if x_min >= x_max or y_min >= y_max:
            continue
        # Like predict_jsons, report the unclipped box; the clipping only drops degenerate ones.
        annotations.append({"bbox": bbox.tolist(), "score": scores[box_id], "landmarks": landmarks[box_id].reshape(-1, 2).tolist()})
    return annotations


def _install_mps_attention() -> None:
    """Route scaled_dot_product_attention on MPS tensors through plain matmul + softmax.

    On an M3 with PyTorch 2.10, the MPS SDPA kernel is about 2.5x slower than the same computation
    written out with matmuls, which makes the gaze models 25-30% slower end to end. It also reuses the
    scale of its first call for later calls with the same shapes, ignoring their `scale` argument.
    The models look F.scaled_dot_product_attention up at call time, so replacing the function covers
    them. Only float32 inputs are routed here: in float16 the unfused logits overflow to NaN in the
    DINOv3 and PaGE backbones. Other devices, dropout and grouped-query attention keep the original.
    """
    original = F.scaled_dot_product_attention
    if getattr(original, "_eagle_mps_attention", False):
        return

    def scaled_dot_product_attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        scale: float | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if (
            query.device.type != "mps"
            or query.dtype != torch.float32
            or dropout_p > 0.0
            or kwargs.get("enable_gqa", False)
        ):
            return original(
                query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale, **kwargs
            )
        scale_factor = query.size(-1) ** -0.5 if scale is None else scale
        attn = (query * scale_factor) @ key.transpose(-2, -1)
        if is_causal:
            causal_mask = torch.ones(attn.shape[-2:], device=attn.device, dtype=torch.bool).triu(1)
            attn = attn.masked_fill(causal_mask, float("-inf"))
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn = attn.masked_fill(~attn_mask, float("-inf"))
            else:
                attn = attn + attn_mask
        return attn.softmax(dim=-1) @ value

    scaled_dot_product_attention._eagle_mps_attention = True
    F.scaled_dot_product_attention = scaled_dot_product_attention


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


def _prefer_cached_torch_hub_repos() -> None:
    """Make torch.hub use a cached repo without asking GitHub for its default branch.

    For "owner/repo" without a ref, torch.hub checks on GitHub whether the branch is main or master on
    every load, and only falls back to the cache when there is no connection at all: a transient
    server error (e.g. 504) made loading an already downloaded GAZELLE fail. GAZELLE also loads DINOv2
    through torch.hub internally, so this is patched in torch.hub rather than at EAGLE's call sites.
    """
    original = torch.hub._parse_repo_info
    if getattr(original, "eagle_prefers_cache", False):
        return

    def parse_repo_info(github: str):
        if ":" not in github:
            owner, name = github.split("/")
            for ref in ("main", "master"):
                if os.path.isdir(os.path.join(torch.hub.get_dir(), f"{owner}_{name}_{ref}")):
                    return owner, name, ref
        return original(github)

    parse_repo_info.eagle_prefers_cache = True
    torch.hub._parse_repo_info = parse_repo_info


class ModelManager:
    """Download, cache, and serve external ML models."""

    def __init__(self, paths: AppPaths) -> None:
        self.paths = paths
        self.yolo: YOLO | None = None
        self.yolo_pose: YOLO | None = None
        self.retinaface = None
        self.retinaface_max_size: int | None = None
        self.scrfd: SCRFDFaceDetector | None = None
        self.mediapipe_face_detector = None
        self.mediapipe_face_detector_api: str | None = None
        self.gaze_model: TransformGazeModel | PaGEGazeModel | None = None
        self.loaded_gaze_detection_backend: str | None = None
        self.l2cs_gaze = None
        self.l2cs_gaze_transform = None
        self.gat_gaze = None
        self.mobile_gaze = None
        self.mobile_gaze_transform = None
        self.loaded_device: str | None = None
        self.loaded_object_detection_backend: str | None = None
        self.loaded_person_detection_backend: str | None = None
        self._yolo_used_in_half: set[str] = set()
        self._yolo_coreml: dict[Path, YOLO] = {}
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
        _install_mps_attention()
        try:
            import certifi

            cert_path = certifi.where()
            os.environ.setdefault("SSL_CERT_FILE", cert_path)
            os.environ.setdefault("REQUESTS_CA_BUNDLE", cert_path)
            self._ssl_context = ssl.create_default_context(cafile=cert_path)
        except Exception:
            self._ssl_context = None
        torch.hub.set_dir(str(self.paths.torch_hub_dir))
        _prefer_cached_torch_hub_repos()

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

    def yolo_model(self, name: str, half: bool) -> YOLO:
        """Return the "yolo" or "yolo_pose" model ready for one run at the given precision.

        ultralytics converts a model to float16 in place, and converting it back keeps the rounded
        weights, so a model used in float16 is reloaded before a float32 run. Its predictor is dropped
        too: ultralytics would otherwise keep the first run's device and precision.
        """
        if not half and name in self._yolo_used_in_half:
            path = self.paths.yolo_pose_path if name == "yolo_pose" else self.yolo_object_path(self.loaded_object_detection_backend)
            setattr(self, name, YOLO(path))
            self._yolo_used_in_half.discard(name)
        if half:
            self._yolo_used_in_half.add(name)
        model = getattr(self, name)
        assert model is not None
        model.predictor = None
        return model

    def yolo_coreml_model(self, name: str, input_shape: tuple[int, int]) -> YOLO:
        """Return a Core ML export of the "yolo" or "yolo_pose" model for this input shape."""
        pt_path = self.paths.yolo_pose_path if name == "yolo_pose" else self.yolo_object_path(self.loaded_object_detection_backend)
        path = yolo_coreml_path(pt_path, self.paths.app_dir, input_shape)
        if path not in self._yolo_coreml:
            self._yolo_coreml[path] = YOLO(str(path), task="pose" if name == "yolo_pose" else "detect")
        model = self._yolo_coreml[path]
        model.predictor = None
        return model

    def yolo_object_path(self, backend: str) -> Path:
        return self.paths.app_dir / f"{backend}.pt"

    def ensure_yolo_object_weights(self, backend: str) -> None:
        if backend not in OBJECT_DETECTION_BACKENDS:
            raise ValueError(f"Unsupported object detection backend '{backend}'.")
        self._ensure_download(
            self.yolo_object_path(backend),
            OBJECT_DETECTION_BACKENDS[backend],
            f"YOLO object weights ({backend})",
        )

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

    def ensure_l2cs_gaze_weights(self) -> None:
        if self.paths.l2cs_gaze_path.is_file():
            return
        try:
            from huggingface_hub import hf_hub_download

            downloaded_path = hf_hub_download(
                repo_id="dorni/SpeakerVid-5M-data-curation-models",
                filename="L2CSNet_gaze360.pkl",
                revision="5c6e04d7fa3321e6228e79162f8ec98466bf308a",
                cache_dir=str(self.paths.app_dir / "huggingface"),
            )
            shutil.copyfile(downloaded_path, self.paths.l2cs_gaze_path)
        except Exception:
            self._ensure_download(
                self.paths.l2cs_gaze_path,
                L2CS_GAZE_URL,
                "L2CS-Net Gaze360 weights",
            )

    def ensure_mediapipe_face_detector_weights(self) -> None:
        self._ensure_download(
            self.paths.mediapipe_face_detector_path,
            MEDIAPIPE_FACE_DETECTOR_URL,
            "MediaPipe face detector model",
        )

    def ensure_scrfd_weights(self) -> None:
        destination = self.paths.scrfd_face_detector_path
        if destination.is_file():
            return
        import zipfile

        archive_path = destination.with_suffix(".zip.part")
        try:
            self._ensure_download(archive_path, SCRFD_MODEL_PACK_URL, "SCRFD face detector model pack")
            with zipfile.ZipFile(archive_path) as archive:
                member = next(name for name in archive.namelist() if name.endswith(SCRFD_MODEL_PACK_MEMBER))
                temp_path = destination.with_suffix(".onnx.part")
                with archive.open(member) as source, temp_path.open("wb") as output:
                    shutil.copyfileobj(source, output)
                temp_path.replace(destination)
        finally:
            archive_path.unlink(missing_ok=True)

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

    @property
    def gat_gaze_path(self) -> Path:
        return self.paths.app_dir / "gat_stwsge_gaze360_gf.ckpt"

    def gazelle_dinov3_weights_path(self, backend: str) -> Path:
        return self.paths.app_dir / f"{backend}.pt"

    def _load_gaze_model(self, gaze_detection_backend: str, device: str) -> TransformGazeModel | PaGEGazeModel:
        if gaze_detection_backend in GAZELLE_DINOV3_MODELS:
            weights_path = self.gazelle_dinov3_weights_path(gaze_detection_backend)
            self._ensure_download(
                weights_path,
                GAZELLE_DINOV3_MODELS[gaze_detection_backend]["url"],
                f"GAZELLE-DINOv3 weights ({gaze_detection_backend})",
            )
            try:
                return load_gazelle_dinov3_model(gaze_detection_backend, weights_path, self.paths.torch_hub_dir)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to load the GAZELLE-DINOv3 model '{gaze_detection_backend}'.\n"
                    f"Weight path: {weights_path}\n"
                    f"DINOv3 code is downloaded to: {self.paths.torch_hub_dir}\n"
                    f"Original error: {exc}"
                ) from exc
        if gaze_detection_backend in PAGE_MODELS:
            try:
                return load_page_model(gaze_detection_backend, self.paths.app_dir / "huggingface")
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to load the PaGE model '{gaze_detection_backend}' from HuggingFace.\n"
                    "Please make sure transformers is installed and this machine is connected to the internet.\n"
                    f"Original error: {exc}"
                ) from exc

        is_custom_model = gaze_detection_backend != "gazelle"
        try:
            _, gazelle_transform = torch.hub.load(
                "fkryan/gazelle",
                "gazelle_dinov2_vitl14_inout",
                source="github",
                skip_validation=True,
                trust_repo=True,
            )
            if is_custom_model:
                from .constants import discover_custom_gaze_models
                custom_models = discover_custom_gaze_models(self.paths.custom_gaze_models_dir)
                if gaze_detection_backend not in custom_models:
                    raise FileNotFoundError(f"Custom gaze model '{gaze_detection_backend}' not found in {self.paths.custom_gaze_models_dir}")
                model_path = custom_models[gaze_detection_backend]
                loaded = torch.load(model_path, map_location=device, weights_only=False)
                if isinstance(loaded, dict):
                    arch = "gazelle_dinov2_vitb14_inout"
                    if "vitl14" in model_path.lower():
                        arch = "gazelle_dinov2_vitl14_inout"
                    gazelle, _ = torch.hub.load(
                        "fkryan/gazelle",
                        arch,
                        source="github",
                        skip_validation=True,
                        trust_repo=True,
                    )
                    gazelle.load_state_dict(loaded, strict=False)
                else:
                    gazelle = loaded
            else:
                gazelle, _ = torch.hub.load(
                    "fkryan/gazelle",
                    "gazelle_dinov2_vitl14_inout",
                    source="github",
                    skip_validation=True,
                    trust_repo=True,
                )
            gazelle.eval()
            return TransformGazeModel(gazelle, gazelle_transform)
        except Exception as exc:
            if is_custom_model:
                raise RuntimeError(
                    f"Failed to load custom gaze model '{gaze_detection_backend}' from {self.paths.custom_gaze_models_dir}.\n"
                    f"Original error: {exc}"
                ) from exc
            else:
                raise RuntimeError(
                    "Failed to load the GAZELLE model on first run.\n"
                    "GAZELLE is loaded through torch.hub and may need to download model files.\n"
                    f"Torch hub cache directory: {self.paths.torch_hub_dir}\n"
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
        gaze_half_precision: bool = False,
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
        from .constants import discover_custom_gaze_models
        custom_gaze_models = discover_custom_gaze_models(self.paths.custom_gaze_models_dir)
        valid_gaze_backends = set(GAZE_DETECTION_BACKENDS) | set(custom_gaze_models.keys())
        if gaze_detection_backend not in valid_gaze_backends:
            raise ValueError(f"Unsupported gaze detection backend '{gaze_detection_backend}'. Available: {sorted(valid_gaze_backends)}")
        if head_pose_detection_backend not in HEAD_POSE_DETECTION_BACKENDS:
            raise ValueError(f"Unsupported head pose detection backend '{head_pose_detection_backend}'.")
        self.ensure_yolo_object_weights(object_detection_backend)
        self.ensure_person_detection_weights(person_detection_backend)
        if head_pose_detection_backend == "l2cs":
            self.ensure_l2cs_gaze_weights()
        if head_pose_detection_backend == "mobileone":
            self.ensure_mobile_gaze_weights()
        if head_pose_detection_backend == "gat":
            self._ensure_download(self.gat_gaze_path, GAT_CHECKPOINT_URL, "GaT (ST-WSGE) gaze weights")
        if self.yolo is None or self.loaded_object_detection_backend != object_detection_backend:
            self.yolo = YOLO(self.yolo_object_path(object_detection_backend))
            self.loaded_object_detection_backend = object_detection_backend
            self._yolo_used_in_half.discard("yolo")
        if self.yolo_pose is None or self.loaded_person_detection_backend != person_detection_backend:
            self.yolo_pose = YOLO(self.paths.yolo_pose_path)
            self.loaded_person_detection_backend = person_detection_backend
            self._yolo_used_in_half.discard("yolo_pose")
        if face_detection_backend == "mediapipe" and self.mediapipe_face_detector is None:
            self._load_mediapipe_face_detector()
        # RetinaFace pads every input to a max_size square; it only ever sees head crops.
        retinaface_max_size = FACE_CROP_SIZE
        if face_detection_backend == "retinaface" and (
            self.retinaface is None or self.loaded_device != device or self.retinaface_max_size != retinaface_max_size
        ):
            try:
                # retinaface re-extracts its zipped weights on every load, so concurrent EAGLE processes can
                # read a half-written file; loading one process at a time avoids that.
                self.paths.torch_hub_dir.mkdir(parents=True, exist_ok=True)
                with FileLock(str(self.paths.torch_hub_dir / "retinaface.lock")):
                    self.retinaface = get_model("resnet50_2020-07-20", max_size=retinaface_max_size, device=device)
                self.retinaface_max_size = retinaface_max_size
                self.retinaface.eval()
                self.retinaface.coreml = None
                if coreml_enabled(device):
                    self.retinaface.coreml = RetinaFaceCoreML(self.retinaface.model, retinaface_max_size, self.paths.app_dir)
                elif device == "mps":
                    # float16 found the same faces as float32 on 180 head crops (mean box IoU 0.9995) and
                    # cut a 512x512 crop from 78 to 49 ms.
                    self.retinaface.model.half()
            except Exception as exc:
                raise RuntimeError(
                    "Failed to load RetinaFace on first run.\n"
                    "RetinaFace may need to download pretrained weights.\n"
                    "Please make sure this machine is connected to the internet and try again.\n"
                    f"Original error: {exc}"
                ) from exc
        if face_detection_backend == "scrfd" and (self.scrfd is None or self.loaded_device != device):
            try:
                self.ensure_scrfd_weights()
                self.scrfd = SCRFDFaceDetector(self.paths.scrfd_face_detector_path, device, max_size=SCRFD_MAX_SIZE)
            except Exception as exc:
                raise RuntimeError(
                    "Failed to load the SCRFD face detector.\n"
                    "SCRFD needs onnxruntime (onnxruntime-gpu for CUDA) and downloads its weights on first run.\n"
                    f"Original error: {exc}"
                ) from exc
        uses_coreml_backbones = isinstance(self.gaze_model, PaGEGazeModel) and self.gaze_model.uses_coreml
        if (
            self.gaze_model is None
            or self.loaded_gaze_detection_backend != gaze_detection_backend
            or (uses_coreml_backbones and not coreml_enabled(device))
        ):
            self.gaze_model = None
            self.gaze_model = self._load_gaze_model(gaze_detection_backend, device)
            self.loaded_gaze_detection_backend = gaze_detection_backend
        self.gaze_model.half_precision = gaze_half_precision
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
        if head_pose_detection_backend == "l2cs" and (
            self.l2cs_gaze is None or self.l2cs_gaze_transform is None or self.loaded_device != device
        ):
            try:
                model = l2cs_resnet50(num_bins=90)
                state_dict = torch.load(self.paths.l2cs_gaze_path, map_location=device)
                if isinstance(state_dict, dict) and "state_dict" in state_dict:
                    state_dict = state_dict["state_dict"]
                if isinstance(state_dict, dict):
                    state_dict = {
                        str(key).removeprefix("module."): value
                        for key, value in state_dict.items()
                    }
                model.load_state_dict(state_dict)
                model.eval()
                self.l2cs_gaze = model.to(device)
                self.l2cs_gaze_transform = transforms.Compose(
                    [
                        transforms.ToPILImage(),
                        transforms.Resize((448, 448)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ]
                )
            except Exception as exc:
                raise RuntimeError(
                    "Failed to load the L2CS-Net model on first run.\n"
                    f"Expected weight path: {self.paths.l2cs_gaze_path}\n"
                    "Please make sure this machine is connected to the internet and try again.\n"
                    f"Original error: {exc}"
                ) from exc
        if head_pose_detection_backend == "gat" and (self.gat_gaze is None or self.loaded_device != device):
            try:
                self.gat_gaze = load_gat_model(self.gat_gaze_path, self.paths.torch_hub_dir).to(device)
            except Exception as exc:
                raise RuntimeError(
                    "Failed to load the GaT (ST-WSGE) gaze model.\n"
                    f"Weight path: {self.gat_gaze_path}\n"
                    "GaT needs the einops package and downloads the Swin3D code from facebookresearch/omnivore.\n"
                    f"Original error: {exc}"
                ) from exc
        if (
            isinstance(self.gaze_model, PaGEGazeModel)
            and self.gaze_model.coreml is None
            and coreml_enabled(device)
            and PaGECoreML.supports(self.gaze_model.model)
        ):
            revision = PAGE_MODELS[gaze_detection_backend]["revision"][:12]
            self.gaze_model.coreml = PaGECoreML(self.gaze_model.model, f"{gaze_detection_backend}-{revision}", self.paths.app_dir)
        self.gaze_model.to(device)
        self.loaded_device = device
