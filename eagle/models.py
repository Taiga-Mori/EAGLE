import os
import logging
import shutil
import ssl
import warnings
from urllib.request import urlopen

os.environ.setdefault("YOLO_VERBOSE", "False")
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
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
from .types import AppPaths


class ModelManager:
    """Download, cache, and serve external ML models."""

    def __init__(self, paths: AppPaths) -> None:
        self.paths = paths
        self.yolo: YOLO | None = None
        self.yolo_pose: YOLO | None = None
        self.retinaface = None
        self.gazelle = None
        self.gazelle_transform = None
        self.mobile_gaze = None
        self.mobile_gaze_transform = None
        self.loaded_device: str | None = None
        self._configure_download_environment()

    def _configure_download_environment(self) -> None:
        os.environ.setdefault("TORCH_HOME", str(self.paths.torch_home))
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

    def ensure_yolo_weights(self) -> None:
        self._ensure_download(
            self.paths.yolo_path,
            "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26x.pt",
            "YOLO weights",
        )

    def ensure_yolo_pose_weights(self) -> None:
        self._ensure_download(
            self.paths.yolo_pose_path,
            "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26x-pose.pt",
            "YOLO pose weights",
        )

    def ensure_mobile_gaze_weights(self) -> None:
        self._ensure_download(
            self.paths.mobile_gaze_path,
            "https://github.com/yakhyo/gaze-estimation/releases/download/weights/mobileone_s0.pt",
            "MobileGaze weights",
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

    def load(self, device: str) -> None:
        if device.startswith("cuda:"):
            try:
                torch.cuda.set_device(torch.device(device))
            except Exception:
                # Keep existing behavior if runtime cannot switch devices explicitly.
                pass
        self.ensure_yolo_weights()
        self.ensure_yolo_pose_weights()
        self.ensure_mobile_gaze_weights()
        if self.yolo is None:
            self.yolo = YOLO(self.paths.yolo_path)
        if self.yolo_pose is None:
            self.yolo_pose = YOLO(self.paths.yolo_pose_path)
        if self.retinaface is None or self.loaded_device != device:
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
        if self.mobile_gaze is None or self.mobile_gaze_transform is None or self.loaded_device != device:
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
