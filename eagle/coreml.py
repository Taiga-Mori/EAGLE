"""Core ML versions of the heaviest models, used on Apple silicon (device "mps").

On an M3 MacBook Air, YOLO26x ran in 17 ms per 384x640 frame in Core ML (Neural Engine and GPU)
against 57 ms in PyTorch MPS float16, and Core ML on the GPU cut PaGE by about 35% against PyTorch
MPS float32 with the same heatmap peaks. Models are converted on first use and cached under
~/.EAGLE/coreml. PaGE stays off the Neural Engine: there its features came out wrong (cosine
similarity 0.77), most likely from float16 overflow.

Set EAGLE_DISABLE_COREML=1 to run everything in PyTorch instead.
"""

import copy
import logging
import math
import os
import shutil
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
from filelock import FileLock
from torch import nn

COREML_DIR_NAME = "coreml"
# Longest side of the YOLO input, as in ultralytics' default imgsz.
YOLO_INPUT_SIZE = 640
YOLO_STRIDE = 32


def coreml_enabled(device: str) -> bool:
    if device != "mps" or sys.platform != "darwin" or os.environ.get("EAGLE_DISABLE_COREML") == "1":
        return False
    return _import_coremltools() is not None


def _import_coremltools():
    # coremltools logs warnings about untested torch and scikit-learn versions on import.
    logging.getLogger("coremltools").setLevel(logging.ERROR)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import coremltools as ct
    except Exception:
        return None
    return ct


def yolo_input_shape(width: int, height: int) -> tuple[int, int]:
    """Return the (height, width) input that ultralytics uses for frames of this size.

    PyTorch YOLO letterboxes a 1280x720 video to 384x640 rather than 640x640. A Core ML model has a
    fixed input, so it is exported for the video's own aspect ratio to see the same pixels.
    """
    scale = YOLO_INPUT_SIZE / max(width, height)
    shape = [math.ceil(height * scale / YOLO_STRIDE) * YOLO_STRIDE, math.ceil(width * scale / YOLO_STRIDE) * YOLO_STRIDE]
    return min(shape[0], YOLO_INPUT_SIZE), min(shape[1], YOLO_INPUT_SIZE)


def yolo_coreml_path(pt_path: Path, app_dir: Path, input_shape: tuple[int, int]) -> Path:
    """Return a Core ML export of the YOLO weights for this input shape, exporting it on first use."""
    from ultralytics import YOLO

    height, width = input_shape
    coreml_dir = app_dir / COREML_DIR_NAME
    target = coreml_dir / f"{pt_path.stem}-{height}x{width}.mlpackage"
    if target.exists():
        return target
    coreml_dir.mkdir(parents=True, exist_ok=True)
    with FileLock(str(coreml_dir / f"{target.name}.lock")):
        if target.exists():
            return target
        print(f"Converting {pt_path.name} to Core ML for {width}x{height} input (first run only)...", flush=True)
        # ultralytics writes the export next to the weights, so export from a private copy and move the
        # finished package into place; an interrupted export then never leaves a broken cache.
        work_dir = coreml_dir / f"{target.stem}.tmp"
        shutil.rmtree(work_dir, ignore_errors=True)
        work_dir.mkdir()
        try:
            work_pt = work_dir / pt_path.name
            shutil.copy2(pt_path, work_pt)
            exported = Path(YOLO(str(work_pt)).export(format="coreml", imgsz=[height, width], verbose=False))
            exported.rename(target)
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)
    return target


class CoreMLBackbone(nn.Module):
    """Stand-in for a PaGE backbone that runs a Core ML conversion of it one image at a time."""

    def __init__(self, mlmodel, input_name: str, output_name: str) -> None:
        super().__init__()
        self.mlmodel = mlmodel
        self.input_name = input_name
        self.output_name = output_name

    def forward(self, x):
        if isinstance(x, (list, tuple)):
            # PaGE passes the head stream as a one-element list.
            assert len(x) == 1
            x = x[0]
        images = x.detach().to("cpu", torch.float32).numpy()
        features = [
            self.mlmodel.predict({self.input_name: images[index : index + 1]})[self.output_name]
            for index in range(images.shape[0])
        ]
        return torch.from_numpy(np.concatenate(features)).to(device=x.device, dtype=x.dtype)


def install_page_coreml_backbones(page_model: nn.Module, cache_key: str, app_dir: Path) -> None:
    """Replace PaGE's scene and head backbones with Core ML conversions running on the GPU.

    Both are needed: interleaved runs on 3-person frames took 777 ms per frame in PyTorch, 570 ms
    with only the scene backbone in Core ML and 502 ms with both.
    """
    for attribute in ("scene_branch_backbone", "head_branch_backbone"):
        backbone = getattr(page_model, attribute)
        if isinstance(backbone, CoreMLBackbone):
            continue
        height, width = backbone.in_size
        path = app_dir / COREML_DIR_NAME / f"{cache_key}-{attribute}-{height}x{width}.mlpackage"
        _convert_backbone(backbone, path, (1, 3, height, width))
        setattr(page_model, attribute, _load_backbone(path))


def _convert_backbone(backbone: nn.Module, path: Path, input_shape: tuple[int, ...]) -> None:
    if path.exists():
        return
    ct = _import_coremltools()
    path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(path.parent / f"{path.name}.lock")):
        if path.exists():
            return
        print(f"Converting the PaGE backbone to Core ML as {path.name} (first run only)...", flush=True)
        model = copy.deepcopy(backbone).to("cpu", torch.float32).eval()
        example = torch.zeros(input_shape)
        with torch.no_grad(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            traced = torch.jit.trace(model, example, check_trace=False)
            mlmodel = ct.convert(
                traced,
                inputs=[ct.TensorType(name="x", shape=input_shape)],
                compute_precision=ct.precision.FLOAT16,
                minimum_deployment_target=ct.target.macOS15,
                convert_to="mlprogram",
            )
        work_path = path.with_name(f"{path.stem}.tmp.mlpackage")
        shutil.rmtree(work_path, ignore_errors=True)
        mlmodel.save(str(work_path))
        work_path.rename(path)


def _load_backbone(path: Path) -> CoreMLBackbone:
    ct = _import_coremltools()
    mlmodel = ct.models.MLModel(str(path), compute_units=ct.ComputeUnit.CPU_AND_GPU)
    description = mlmodel.get_spec().description
    return CoreMLBackbone(mlmodel, description.input[0].name, description.output[0].name)
