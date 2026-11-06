"""Core ML versions of the heaviest models, used on Apple silicon (device "mps").

On an M3 MacBook Air, YOLO26x ran in 17 ms per 384x640 frame in Core ML (Neural Engine and GPU)
against 57 ms in PyTorch MPS float16. RetinaFace and PaGE gain as much; see their classes. Models are
converted on first use and cached under ~/.EAGLE/coreml.

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


def _convert(
    module: nn.Module,
    path: Path,
    example_inputs: tuple[torch.Tensor, ...],
    input_names: list[str],
    output_names: list[str],
    label: str,
) -> None:
    """Trace a module on the CPU in float32 and save it as a float16 Core ML program."""
    if path.exists():
        return
    ct = _import_coremltools()
    path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(path.parent / f"{path.name}.lock")):
        if path.exists():
            return
        print(f"Converting {label} to Core ML (first run only)...", flush=True)
        with torch.no_grad(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            traced = torch.jit.trace(module.eval(), example_inputs, check_trace=False)
            mlmodel = ct.convert(
                traced,
                inputs=[ct.TensorType(name=name, shape=tuple(x.shape)) for name, x in zip(input_names, example_inputs)],
                outputs=[ct.TensorType(name=name) for name in output_names],
                compute_precision=ct.precision.FLOAT16,
                minimum_deployment_target=ct.target.macOS15,
                convert_to="mlprogram",
            )
        work_path = path.with_name(f"{path.stem}.tmp.mlpackage")
        shutil.rmtree(work_path, ignore_errors=True)
        mlmodel.save(str(work_path))
        work_path.rename(path)


def _load(path: Path, compute_units: str):
    ct = _import_coremltools()
    return ct.models.MLModel(str(path), compute_units=getattr(ct.ComputeUnit, compute_units))


def _predict_all(mlmodel, inputs: list[dict]) -> list[dict]:
    """Predict a list of inputs in one Core ML batch request.

    The programs have a fixed batch of one (DINOv3's attention does not convert with a flexible batch),
    but a batch request still lets Core ML overlap the calls: on an M3 it cut 10-15% per image against
    calling predict in a loop. Threads do not help: predict holds the GIL and the GPU is already busy.
    """
    if not inputs:
        return []
    if len(inputs) == 1:
        return [mlmodel.predict(inputs[0])]
    return mlmodel.predict(inputs)


class RetinaFaceCoreML:
    """RetinaFace converted to Core ML for one max_size x max_size image at a time.

    On the Neural Engine a 512x512 head crop took 10 ms, against 30 ms for PyTorch MPS in float16 with
    three crops per batch; face scores moved by at most 0.002.
    """

    def __init__(self, network: nn.Module, max_size: int, app_dir: Path) -> None:
        path = app_dir / COREML_DIR_NAME / f"retinaface-resnet50_2020-07-20-{max_size}x{max_size}.mlpackage"
        if not path.exists():
            network = copy.deepcopy(network).to("cpu", torch.float32)
            for module in network.modules():
                if isinstance(module, nn.LeakyReLU):
                    # retinaface sets an int slope of 0, which coremltools rejects.
                    module.negative_slope = float(module.negative_slope)
                    module.inplace = False
            _convert(network, path, (torch.zeros(1, 3, max_size, max_size),), ["x"], ["loc", "conf", "land"], "RetinaFace")
        self.mlmodel = _load(path, "ALL")

    def __call__(self, images: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        outputs = _predict_all(self.mlmodel, [{"x": images[index : index + 1]} for index in range(images.shape[0])])
        return tuple(torch.from_numpy(np.concatenate([output[name] for output in outputs])) for name in ("loc", "conf", "land"))


class _PaGESceneEncoder(nn.Module):
    """PaGE's per-frame part: scene backbone, projection, front tokens and scene self-attention."""

    def __init__(self, page_model: nn.Module) -> None:
        super().__init__()
        self.page = page_model

    def forward(self, images):
        page = self.page
        tokens = page.scene_proj(page.scene_branch_backbone(images)).flatten(start_dim=2).permute(0, 2, 1)
        batch = tokens.shape[0]
        if page.n_reg_tokens > 0:
            tokens = torch.cat([page.scene_register_tokens.expand(batch, -1, -1), tokens], dim=1)
        tokens = torch.cat([page.scene_inout_token.expand(batch, -1, -1), tokens], dim=1)
        return page.scene_self_attn_layers(tokens)


class _PaGEPersonDecoder(nn.Module):
    """PaGE's per-person part, up to the heatmap and in/out logits before resizing and sigmoid."""

    def __init__(self, page_model: nn.Module) -> None:
        super().__init__()
        self.page = page_model

    def forward(self, head_images, scene_tokens, head_rects):
        page = self.page
        tokens = page.head_proj(page.head_branch_backbone(head_images)).flatten(start_dim=2).permute(0, 2, 1)
        batch = tokens.shape[0]
        if page.n_reg_tokens > 0:
            tokens = torch.cat([page.head_register_tokens.expand(batch, -1, -1), tokens], dim=1)
        tokens = torch.cat([page.head_inout_token.expand(batch, -1, -1), tokens], dim=1)
        tokens = page.head_self_attn_layers(tokens)
        out = page.scene_head_interaction_layers({"scene_tokens": scene_tokens, "head_tokens": tokens, "head_rects": head_rects})
        inout = page.inout_head(torch.cat((out["scene_tokens"][:, 0, :], out["head_tokens"][:, 0, :]), dim=1)).squeeze(dim=-1)
        scene_patch_tokens = out["scene_tokens"][:, page.n_front_tokens :, :]
        featmap = scene_patch_tokens.reshape(batch, page.scene_featmap_h, page.scene_featmap_w, -1).permute(0, 3, 1, 2)
        return page.heatmap_head(featmap).squeeze(dim=1), inout


class PaGECoreML:
    """PaGE run as two Core ML programs on the GPU: one per frame and one per person.

    PaGE without a head prompt gives every person in a frame the same scene tokens up to the
    scene-head interaction, so that part runs once per frame. On an M3 a 3-person frame took about
    210 ms (88 ms per frame plus 40 ms per person) with the same heatmap peaks as PyTorch. Converting
    with a fixed batch of 4 saved only 5-10%, so the programs take one image at a time. PaGE stays off the
    Neural Engine: there its heatmap peaks moved by up to 38 cells, most likely from float16 overflow.
    """

    @staticmethod
    def supports(page_model: nn.Module) -> bool:
        return page_model.pos_encoding == "rope" and not page_model.use_head_prompt and page_model.inout

    def __init__(self, page_model: nn.Module, cache_key: str, app_dir: Path) -> None:
        coreml_dir = app_dir / COREML_DIR_NAME
        scene_h, scene_w = page_model.scene_branch_backbone.in_size
        head_h, head_w = page_model.head_branch_backbone.in_size
        scene_path = coreml_dir / f"{cache_key}-scene-{scene_h}x{scene_w}.mlpackage"
        person_path = coreml_dir / f"{cache_key}-person-{head_h}x{head_w}.mlpackage"
        if not (scene_path.exists() and person_path.exists()):
            page_cpu = copy.deepcopy(page_model).to("cpu", torch.float32).eval()
            scene_images = torch.zeros(1, 3, scene_h, scene_w)
            with torch.no_grad():
                scene_tokens = _PaGESceneEncoder(page_cpu)(scene_images)
            # Any head rectangle inside the feature map works for tracing; it only fixes the shapes.
            head_rects = torch.tensor([[2.0, 2.0, 6.0, 6.0]])
            _convert(_PaGESceneEncoder(page_cpu), scene_path, (scene_images,), ["images"], ["scene_tokens"], "PaGE (scene part)")
            _convert(
                _PaGEPersonDecoder(page_cpu),
                person_path,
                (torch.zeros(1, 3, head_h, head_w), scene_tokens, head_rects),
                ["head_images", "scene_tokens", "head_rects"],
                ["heatmap", "inout"],
                "PaGE (person part)",
            )
        self.scene = _load(scene_path, "CPU_AND_GPU")
        self.person = _load(person_path, "CPU_AND_GPU")

    def __call__(self, images: torch.Tensor, head_images: torch.Tensor, head_rects: torch.Tensor, owners: list[int]):
        """Return heatmap and in/out logits for every person; owners[i] is person i's frame index."""
        images = images.to("cpu", torch.float32).numpy()
        head_images = head_images.to("cpu", torch.float32).numpy()
        head_rects = head_rects.to("cpu", torch.float32).numpy()
        scene_outputs = _predict_all(self.scene, [{"images": images[index : index + 1]} for index in range(images.shape[0])])
        person_outputs = _predict_all(
            self.person,
            [
                {
                    "head_images": head_images[person : person + 1],
                    "scene_tokens": scene_outputs[owner]["scene_tokens"],
                    "head_rects": head_rects[person : person + 1],
                }
                for person, owner in enumerate(owners)
            ],
        )
        heatmaps = np.concatenate([output["heatmap"] for output in person_outputs])
        inouts = np.concatenate([output["inout"] for output in person_outputs])
        return torch.from_numpy(heatmaps), torch.from_numpy(inouts)
