import shutil
from urllib.request import urlopen

import torch
from retinaface.pre_trained_models import get_model
from ultralytics import YOLO

from .types import AppPaths


class ModelManager:
    """Download, cache, and serve external ML models."""

    def __init__(self, paths: AppPaths) -> None:
        self.paths = paths
        self.yolo: YOLO | None = None
        self.retinaface = None
        self.gazelle = None
        self.gazelle_transform = None
        self.loaded_device: str | None = None

    def ensure_yolo_weights(self) -> None:
        if self.paths.yolo_path.is_file():
            return
        try:
            with urlopen(
                "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26x.pt"
            ) as response, self.paths.yolo_path.open("wb") as output:
                shutil.copyfileobj(response, output)
        except Exception as exc:
            raise RuntimeError(f"Failed to download YOLO weights to {self.paths.yolo_path}") from exc

    def load(self, device: str) -> None:
        self.ensure_yolo_weights()
        if self.yolo is None:
            self.yolo = YOLO(self.paths.yolo_path)
        if self.retinaface is None or self.loaded_device != device:
            self.retinaface = get_model("resnet50_2020-07-20", max_size=2048, device=device)
            self.retinaface.eval()
        if self.gazelle is None or self.gazelle_transform is None:
            try:
                self.gazelle, self.gazelle_transform = torch.hub.load(
                    "fkryan/gazelle",
                    "gazelle_dinov2_vitl14_inout",
                    trust_repo=True,
                )
                self.gazelle.eval()
            except Exception as exc:
                raise RuntimeError("Failed to load the GAZELLE model from torch.hub") from exc
        self.gazelle.to(device)
        self.loaded_device = device
