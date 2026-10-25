"""Gaze target estimation models behind a common ``predict`` interface.

Every model takes a full RGB frame plus normalized head boxes and returns one gaze
heatmap and one in-frame probability per head:

- ``TransformGazeModel``: GAZELLE-style models (official GAZELLE, custom fine-tunes,
  and GAZELLE-DINOv3) that take ``{"images", "bboxes"}``.
- ``PaGEGazeModel``: PaGE, which additionally needs a crop of each head.
"""

import math
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from PIL import Image
from timm.models.vision_transformer import Block
from torchvision import transforms

# Meta's DINOv3 code at the commit PINTO0309/gazelle-dinov3 was trained against.
DINOV3_COMMIT = "7bf81b2a0eb0e330dbc84a5d3d31d86ed3cdbd84"
DINOV3_ARCHIVE_URL = f"https://github.com/facebookresearch/dinov3/archive/{DINOV3_COMMIT}.zip"

GAZELLE_DINOV3_MODELS = {
    "gazelle_dinov3_vitb16": {
        "dinov3_arch": "dinov3_vitb16",
        "url": "https://github.com/PINTO0309/gazelle-dinov3/releases/download/weights/gazelle_dinov3_vitb16_inout.pt",
        "in_size": (640, 640),
    },
}

# PaGE weights and their remote modeling code, pinned so updates upstream cannot
# silently change the code that runs with trust_remote_code.
PAGE_CODE_REPO_REVISION = "7147c67c440a110f050eb1017de50c7a1925c496"
PAGE_MODELS = {
    "page_vitb": {
        "repo_id": "Octopus1/page-vitb",
        "revision": "b991d8bcb2890a97db611a2080ca250ecd943952",
    },
}


class TransformGazeModel:
    """Wrap a GAZELLE-style model and its image transform."""

    def __init__(self, model: nn.Module, transform) -> None:
        self.model = model
        self.transform = transform

    def to(self, device: str) -> "TransformGazeModel":
        self.model.to(device)
        return self

    def predict(self, frame_pil: Image.Image, bboxes: list[tuple[float, float, float, float]], device: str):
        img_tensor = self.transform(frame_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            output = self.model({"images": img_tensor, "bboxes": [bboxes]})
        return output["heatmap"][0], output["inout"][0]


class PaGEGazeModel:
    """Wrap a PaGE HuggingFace model, which needs a head crop for every box."""

    def __init__(self, model: nn.Module, processor) -> None:
        self.model = model
        self.processor = processor

    def to(self, device: str) -> "PaGEGazeModel":
        self.model.to(device)
        return self

    def predict(self, frame_pil: Image.Image, bboxes: list[tuple[float, float, float, float]], device: str):
        width, height = frame_pil.size
        head_crops = []
        for x1, y1, x2, y2 in bboxes:
            left = min(max(int(round(x1 * width)), 0), width - 1)
            top = min(max(int(round(y1 * height)), 0), height - 1)
            right = max(min(int(round(x2 * width)), width), left + 1)
            bottom = max(min(int(round(y2 * height)), height), top + 1)
            head_crops.append(frame_pil.crop((left, top, right, bottom)))
        inputs = self.processor(frame_pil, head_crops=head_crops, bboxes=[bboxes])
        inputs = {
            "images": inputs["images"].to(device),
            "head_images": [head_images.to(device) for head_images in inputs["head_images"]],
            "bboxes": inputs["bboxes"],
        }
        with torch.inference_mode():
            output = self.model(inputs)
        return output["heatmap"][0], output["inout"][0]


def load_page_model(backend: str, cache_dir: Path) -> PaGEGazeModel:
    try:
        from transformers import AutoImageProcessor, AutoModel
        from transformers.utils import logging as hf_logging
    except ImportError as exc:
        raise RuntimeError("PaGE requires transformers. Install it with: pip install transformers==5.6.2") from exc

    spec = PAGE_MODELS[backend]
    kwargs = {
        "revision": spec["revision"],
        "code_revision": PAGE_CODE_REPO_REVISION,
        "trust_remote_code": True,
        "cache_dir": str(cache_dir),
    }
    previous_verbosity = hf_logging.get_verbosity()
    hf_logging.set_verbosity_error()
    try:
        model = AutoModel.from_pretrained(spec["repo_id"], **kwargs).eval()
        processor = AutoImageProcessor.from_pretrained(spec["repo_id"], **kwargs)
    finally:
        hf_logging.set_verbosity(previous_verbosity)
    return PaGEGazeModel(model, processor)


def load_gazelle_dinov3_model(backend: str, weights_path: Path, torch_hub_dir: Path) -> TransformGazeModel:
    spec = GAZELLE_DINOV3_MODELS[backend]
    dinov3 = _build_dinov3(spec["dinov3_arch"], torch_hub_dir)
    model = GazeLLEDinov3(_Dinov3Backbone(dinov3), in_size=spec["in_size"])
    checkpoint = _load_checkpoint(weights_path)
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.Resize(spec["in_size"]),
        ]
    )
    return TransformGazeModel(model, transform)


def _load_checkpoint(weights_path: Path) -> dict:
    """Load a gazelle-dinov3 training checkpoint without unpickling arbitrary objects.

    The checkpoints also store numpy RNG state, so numpy array types are allowlisted.
    """
    from numpy._core.multiarray import _reconstruct

    safe_globals = [(_reconstruct, "numpy.core.multiarray._reconstruct"), np.ndarray, np.dtype]
    safe_globals += [getattr(np.dtypes, name) for name in dir(np.dtypes) if name.endswith("DType")]
    with torch.serialization.safe_globals(safe_globals):
        return torch.load(weights_path, map_location="cpu", weights_only=True)


def _build_dinov3(arch: str, torch_hub_dir: Path) -> nn.Module:
    """Build an uninitialized DINOv3 ViT from Meta's code; weights come from the checkpoint.

    torch.hub.load cannot be used because DINOv3's hubconf also imports its
    segmentation heads, which need extra packages.
    """
    repo_dir = Path(torch_hub_dir) / f"facebookresearch_dinov3_{DINOV3_COMMIT}"
    if not (repo_dir / "dinov3" / "hub" / "backbones.py").is_file():
        archive_path = Path(torch_hub_dir) / f"dinov3_{DINOV3_COMMIT}.zip"
        torch.hub.download_url_to_file(DINOV3_ARCHIVE_URL, str(archive_path), progress=False)
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(torch_hub_dir)
        (Path(torch_hub_dir) / f"dinov3-{DINOV3_COMMIT}").replace(repo_dir)
        archive_path.unlink()
    if str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))
    from dinov3.hub import backbones

    model = getattr(backbones, arch)(pretrained=False)
    del model.head
    return model


# The modules below are adapted from PINTO0309/gazelle-dinov3 (MIT License,
# Copyright (c) 2025 Katsuya Hyodo), keeping parameter names compatible with its checkpoints.


class _Dinov3Backbone(nn.Module):
    def __init__(self, dinov3: nn.Module) -> None:
        super().__init__()
        self.dinov3 = dinov3

    def get_dimension(self) -> int:
        return self.dinov3.embed_dim

    def get_out_size(self, in_size: tuple[int, int]) -> tuple[int, int]:
        return in_size[0] // self.dinov3.patch_size, in_size[1] // self.dinov3.patch_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_h, out_w = self.get_out_size(x.shape[-2:])
        tokens = self.dinov3.forward_features(x)["x_norm_patchtokens"]
        return tokens.transpose(1, 2).reshape(tokens.shape[0], tokens.shape[2], out_h, out_w)


class GazeLLEDinov3(nn.Module):
    def __init__(self, backbone: _Dinov3Backbone, dim=256, num_layers=3, in_size=(640, 640), out_size=(64, 64)):
        super().__init__()
        self.backbone = backbone
        self.featmap_h, self.featmap_w = backbone.get_out_size(in_size)
        self.out_size = out_size
        self.linear = nn.Conv2d(backbone.get_dimension(), dim, 1)
        self.head_token = nn.Embedding(1, dim)
        self.register_buffer("pos_embed", _positionalencoding2d(dim, self.featmap_h, self.featmap_w))
        self.inout_token = nn.Embedding(1, dim)
        self.transformer = nn.Sequential(
            *[Block(dim=dim, num_heads=8, mlp_ratio=4, drop_path=0.1) for _ in range(num_layers)]
        )
        self.heatmap_head = nn.Sequential(
            nn.ConvTranspose2d(dim, dim, kernel_size=2, stride=2),
            nn.Conv2d(dim, 1, kernel_size=1, bias=False),
        )
        self.inout_head = nn.Sequential(nn.Linear(dim, 1))

    def forward(self, input):
        num_ppl_per_img = [len(bbox_list) for bbox_list in input["bboxes"]]
        x = self.linear(self.backbone(input["images"])) + self.pos_embed
        x = torch.cat([x[i : i + 1].repeat(count, 1, 1, 1) for i, count in enumerate(num_ppl_per_img)], dim=0)
        head_maps = torch.cat(self._head_maps(input["bboxes"]), dim=0).to(x.device)
        x = x + head_maps.unsqueeze(dim=1) * self.head_token.weight.unsqueeze(-1).unsqueeze(-1)
        x = x.flatten(start_dim=2).permute(0, 2, 1)
        x = torch.cat([self.inout_token.weight.unsqueeze(dim=0).repeat(x.shape[0], 1, 1), x], dim=1)
        x = self.transformer(x)

        inout = torch.sigmoid(self.inout_head(x[:, 0, :]).squeeze(dim=-1))
        x = x[:, 1:, :].reshape(x.shape[0], self.featmap_h, self.featmap_w, x.shape[2]).permute(0, 3, 1, 2)
        heatmap = torch.sigmoid(self.heatmap_head(x)).squeeze(dim=1)
        heatmap = TF.resize(heatmap, self.out_size, antialias=False)
        return {
            "heatmap": list(torch.split(heatmap, num_ppl_per_img)),
            "inout": list(torch.split(inout, num_ppl_per_img)),
        }

    def _head_maps(self, bboxes):
        head_maps = []
        for bbox_list in bboxes:
            img_head_maps = []
            for xmin, ymin, xmax, ymax in bbox_list:
                head_map = torch.zeros((self.featmap_h, self.featmap_w))
                head_map[
                    round(ymin * self.featmap_h) : round(ymax * self.featmap_h),
                    round(xmin * self.featmap_w) : round(xmax * self.featmap_w),
                ] = 1
                img_head_maps.append(head_map)
            head_maps.append(torch.stack(img_head_maps))
        return head_maps


def _positionalencoding2d(d_model: int, height: int, width: int) -> torch.Tensor:
    pe = torch.zeros(d_model, height, width)
    d_model = d_model // 2
    div_term = torch.exp(torch.arange(0.0, d_model, 2) * -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0.0, width).unsqueeze(1)
    pos_h = torch.arange(0.0, height).unsqueeze(1)
    pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model + 1 :: 2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    return pe
