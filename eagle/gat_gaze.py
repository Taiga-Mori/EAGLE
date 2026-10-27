"""GaT (ST-WSGE) 3D gaze direction estimation from head crops.

Model: "Enhancing 3D Gaze Estimation in the Wild using Weak Supervision with Gaze Following Labels"
(Vuillecard and Odobez, CVPR 2025), https://github.com/idiap/gaze3d. The checkpoint (trained on Gaze360 and
GazeFollow) and the Swin3D encoder code (facebookresearch/omnivore) are downloaded at run time from pinned
commits; both are licensed for non-commercial use only (CC BY-NC 4.0).

Unlike face-crop gaze models, GaT works from a head crop, so it also predicts gaze when the face is
seen from the side or from behind.
"""

import importlib.util
import math
import sys
import types
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

GAZE3D_COMMIT = "f2c13de3ae1910878a25fe15fc809d9d03e83d8b"
GAT_CHECKPOINT_URL = (
    f"https://media.githubusercontent.com/media/idiap/gaze3d/{GAZE3D_COMMIT}/checkpoints/gat_stwsge_gaze360_gf.ckpt"
)
OMNIVORE_COMMIT = "1d55abdc8dfc7bd5cbf69316841ab804d0acf1ca"
OMNIVORE_ARCHIVE_URL = f"https://github.com/facebookresearch/omnivore/archive/{OMNIVORE_COMMIT}.zip"
IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]
INPUT_SIZE = 224


class _Swin3DEncoder(nn.Module):
    def __init__(self, trunk: nn.Module) -> None:
        super().__init__()
        self.encoder = trunk
        self.output_features = trunk.num_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, ["stage3"])[0]


class _GazeHead(nn.Module):
    def __init__(self, in_features: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(in_features, 3)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers[0](x)


class GaTGaze(nn.Module):
    """Image-mode GaT: B x 3 x 224 x 224 head crops -> B x 3 unit gaze vectors (camera coordinates)."""

    def __init__(self, trunk: nn.Module) -> None:
        super().__init__()
        self.encoder = _Swin3DEncoder(trunk)
        self.decoder = nn.ModuleDict({"gaze": _GazeHead(self.encoder.output_features)})

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # A single image is duplicated along time, as in the original image inference.
        x = images.unsqueeze(2).repeat(1, 1, 2, 1, 1)  # B x C x T=2 x H x W
        x = self.encoder(x).mean(dim=[-2, -1])  # B x F x T/2
        gaze = self.decoder["gaze"](x[:, :, 0])
        return torch.nn.functional.normalize(gaze, p=2, dim=1, eps=1e-8)


def _ensure_omnivore(torch_hub_dir: Path) -> Path:
    repo_dir = Path(torch_hub_dir) / f"facebookresearch_omnivore_{OMNIVORE_COMMIT}"
    swin_path = repo_dir / "omnivision" / "models" / "swin_transformer.py"
    if not swin_path.is_file():
        archive_path = Path(torch_hub_dir) / f"omnivore_{OMNIVORE_COMMIT}.zip"
        torch.hub.download_url_to_file(OMNIVORE_ARCHIVE_URL, str(archive_path), progress=False)
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(torch_hub_dir)
        (Path(torch_hub_dir) / f"omnivore-{OMNIVORE_COMMIT}").replace(repo_dir)
        archive_path.unlink()
    return swin_path


def _build_swin3d_tiny(torch_hub_dir: Path) -> nn.Module:
    """Build Omnivore's Swin3D-T trunk without importing the omnivision package (which needs hydra etc.).

    The only omnivision dependency of the module is a checkpoint loader used for pretrained weights,
    which is not needed here because all weights come from the GaT checkpoint.
    """
    swin_path = _ensure_omnivore(torch_hub_dir)
    stub = types.ModuleType("omnivision.utils.checkpoint")
    stub.load_and_broadcast_checkpoint_list = lambda *args, **kwargs: (_ for _ in ()).throw(
        RuntimeError("Pretrained Swin3D weights are not used by EAGLE.")
    )
    saved = {name: sys.modules.get(name) for name in ("omnivision", "omnivision.utils", "omnivision.utils.checkpoint")}
    sys.modules.setdefault("omnivision", types.ModuleType("omnivision"))
    sys.modules.setdefault("omnivision.utils", types.ModuleType("omnivision.utils"))
    sys.modules["omnivision.utils.checkpoint"] = stub
    try:
        spec = importlib.util.spec_from_file_location("eagle_omnivore_swin_transformer", swin_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        for name, previous in saved.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous
    # Same configuration as omnivore_swinT in omnivore/models/omnivore_model.py.
    return module.SwinTransformer3D(
        pretrained2d=False,
        patch_size=(2, 4, 4),
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=(8, 7, 7),
        drop_path_rate=0.2,
        patch_norm=True,
        depth_mode="summed_rgb_d_tokens",
    )


def load_gat_model(checkpoint_path: Path, torch_hub_dir: Path) -> GaTGaze:
    model = GaTGaze(_build_swin3d_tiny(torch_hub_dir))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    return model


def head_box_from_face_box(x1: float, y1: float, x2: float, y2: float, width: int, height: int) -> tuple[float, float, float, float]:
    """Approximate the head box GaT was trained with (whole head incl. hair) from EAGLE's face/head box.

    The box is made square, enlarged by 30%, shifted up to include the top of the head, and then passed
    through GaT's own reshape (square, 10% smaller, anchored at the top).
    """
    side = max(x2 - x1, y2 - y1) * 1.3
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0 - 0.1 * side
    hx1, hy1, hx2, hy2 = cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2
    # GaT's BboxReshape(square=True, ratio=-0.1, adjust_top=True)
    size = side * 0.9
    hx1 = (hx1 + hx2) / 2 - size / 2
    hy1 = hy1 + side * 0.1
    return hx1, hy1, hx1 + size, hy1 + size


def preprocess_head_crop(frame_rgb: torch.Tensor, box: tuple[float, float, float, float]) -> torch.Tensor:
    """frame_rgb: uint8 tensor C x H x W. Returns a normalized 3 x 224 x 224 crop (zero-padded outside the frame)."""
    x1, y1, x2, y2 = box
    left, top = int(math.floor(x1)), int(math.floor(y1))
    w, h = max(1, int(round(x2 - x1))), max(1, int(round(y2 - y1)))
    crop = TF.resized_crop(frame_rgb, top, left, h, w, [INPUT_SIZE, INPUT_SIZE], antialias=True)
    crop = crop.float() / 255.0
    return TF.normalize(crop, IMG_MEAN, IMG_STD)


def gaze_vector_to_yaw_pitch_deg(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert Gaze360-convention unit vectors (camera looks along -z) to yaw/pitch in degrees.

    Same convention as GaT's cartesial2spherical: yaw = atan2(x, -z), pitch = asin(y).
    """
    v = vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-8)
    yaw = np.degrees(np.arctan2(v[:, 0], -v[:, 2]))
    pitch = np.degrees(np.arcsin(np.clip(v[:, 1], -1.0, 1.0)))
    return yaw, pitch
