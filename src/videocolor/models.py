"""Pinned DDColor inference without training code or remote Python execution.

Importing this module never downloads weights. Only ``create_colorizer`` may
contact Hugging Face, and frames are always processed locally.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .color import render_gamut_safe


LOGGER = logging.getLogger(__name__)
ARCHITECTURE_REVISION = "2adb63f2656ac41cbdf7b894cddd94121a3faf13"
INPUT_SIZES = (256, 512)


class ModelError(RuntimeError):
    """An actionable failure to initialize or run the colorization model."""


@dataclass(frozen=True)
class ModelSpec:
    repository: str
    revision: str
    sha256: str
    size_bytes: int
    encoder: str


MODEL_SPECS = {
    "tiny": ModelSpec(
        repository="piddnad/ddcolor_paper_tiny",
        revision="cf9fd99c1d7472689ec7413441c1b799a51866a3",
        sha256="8a1277bc90a1bfbb6d2d83933a9a6bc821931879ca93e26e4fcec12165d41fce",
        size_bytes=220372845,
        encoder="convnext-t",
    ),
    "large": ModelSpec(
        repository="piddnad/ddcolor_modelscope",
        revision="060f67494e31883a4b13cb27f889f3154847ada4",
        sha256="d81711971ec59200da26d5e8a1afae8dd3778d495ea8ad7a7dadc769f403f7e7",
        size_bytes=911914869,
        encoder="convnext-l",
    ),
}


def _resolve_device(requested: str, torch_module: Any) -> str:
    if requested not in ("auto", "cpu", "cuda"):
        raise ValueError("device must be 'auto', 'cpu', or 'cuda'.")
    available = torch_module.cuda.is_available()
    if requested == "cuda" and not available:
        raise ModelError(
            "CUDA is unavailable. Select CPU, or install a CUDA-enabled PyTorch "
            "build and a compatible NVIDIA driver."
        )
    return "cuda" if requested != "cpu" and available else "cpu"


def _validate_frame(frame_bgr: np.ndarray) -> None:
    if not isinstance(frame_bgr, np.ndarray):
        raise ValueError("frame_bgr must be a NumPy array containing a BGR frame.")
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError("frame_bgr must have shape (height, width, 3).")
    if frame_bgr.dtype != np.uint8:
        raise ValueError("frame_bgr must have dtype uint8, with values from 0 to 255.")
    if frame_bgr.shape[0] < 1 or frame_bgr.shape[1] < 1:
        raise ValueError("frame_bgr cannot be empty.")


def _verify_checkpoint(path: Path, spec: ModelSpec) -> None:
    """Reject partial or changed cached weights before deserializing anything."""
    if path.stat().st_size != spec.size_bytes:
        raise ModelError(
            "Cached model has an unexpected size. Remove that model's cached "
            "checkpoint and download it again."
        )
    digest = hashlib.sha256()
    with path.open("rb") as checkpoint:
        for chunk in iter(lambda: checkpoint.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != spec.sha256:
        raise ModelError(
            "Cached model failed its SHA-256 integrity check. Remove that model's "
            "cached checkpoint and download it again."
        )


def _load_network(checkpoint: Path, spec: ModelSpec, input_size: int, torch_module: Any) -> Any:
    """Assign mapped checkpoint tensors without allocating duplicate parameters.

    See PyTorch's official ``module_load_state_dict_tips`` recipe. All supported
    checkpoints use the zip-based torch.save format required by mmap. The
    caller must verify the pinned file hash before calling this helper.
    """
    from ._vendor.ddcolor.model import DDColor

    # Metadata-only construction runs shape inference and installs the normal
    # encoder/spectral-normalization hooks, without random parameter storage.
    with torch_module.device("meta"):
        network = DDColor(
            encoder_name=spec.encoder,
            decoder_name="MultiScaleColorDecoder",
            input_size=(input_size, input_size),
            num_output_channels=2,
            last_norm="Spectral",
            do_normalize=False,
            num_queries=100,
            num_scales=3,
            dec_layers=9,
        )
    weights = torch_module.load(
        checkpoint, map_location="cpu", weights_only=True, mmap=True,
    )
    if isinstance(weights, dict) and "params" in weights:
        weights = weights["params"]
    network.load_state_dict(weights, strict=True, assign=True)
    # A strict match is required; no random/uninitialized fallback is accepted.
    if any(tensor.is_meta for tensor in (*network.parameters(), *network.buffers())):
        raise ModelError("The checkpoint did not materialize every model tensor.")
    network.eval()
    network.requires_grad_(False)
    return network


class DDColorizer:
    """Single-worker, frame-based inference preserving original image dimensions.

    A model instance has mutable encoder hooks; do not share it concurrently
    between processing workers. This model predicts plausible chrominance, not
    historically verified colors. Temporal stabilization belongs to the video
    processing layer rather than this independent-frame model.
    """

    def __init__(
        self,
        network: Any,
        *,
        torch_module: Any,
        model: str,
        device: str,
        input_size: int,
    ) -> None:
        self._network = network
        self._torch = torch_module
        self.device = device
        self.input_size = input_size
        spec = MODEL_SPECS[model]
        self.model_info: dict[str, Any] = {
            "name": f"DDColor {model}",
            "model": model,
            "repository": spec.repository,
            "revision": spec.revision,
            "sha256": spec.sha256,
            "license": "Apache-2.0",
            "architecture_revision": ARCHITECTURE_REVISION,
            "device": device,
            "input_size": input_size,
        }

    def colorize(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Colorize one BGR uint8 frame while retaining its luminance and size."""
        _validate_frame(frame_bgr)
        height, width = frame_bgr.shape[:2]
        image = frame_bgr.astype(np.float32) / 255.0

        # DDColor's published preprocessing uses neutral Lab chrominance even
        # for a sepia or partially colored input. Never feed original colors in.
        resized = cv2.resize(image, (self.input_size, self.input_size))
        lightness = cv2.cvtColor(resized, cv2.COLOR_BGR2Lab)[:, :, :1]
        neutral_lab = np.concatenate(
            (lightness, np.zeros_like(lightness), np.zeros_like(lightness)), axis=2
        )
        gray_rgb = cv2.cvtColor(neutral_lab, cv2.COLOR_LAB2RGB)
        tensor = self._torch.from_numpy(
            np.ascontiguousarray(gray_rgb.transpose(2, 0, 1))
        ).unsqueeze(0)

        try:
            with self._torch.inference_mode():
                prediction = self._network(tensor.to(self.device))
                if prediction.ndim != 4 or prediction.shape[:2] != (1, 2):
                    raise ModelError("The model returned an invalid color-channel shape.")
                chroma = prediction[0].float().cpu().numpy().transpose(1, 2, 0)
        except self._torch.cuda.OutOfMemoryError as exc:
            raise ModelError(
                "Not enough GPU memory. Try the tiny model, input size 256, "
                "or CPU processing."
            ) from exc
        if not np.isfinite(chroma).all():
            raise ModelError("The model returned non-finite colors; processing was stopped.")

        # Bilinear interpolation avoids visible color blocks at full video size.
        chroma = cv2.resize(chroma, (width, height), interpolation=cv2.INTER_LINEAR)
        # Compress raw predicted chroma before its first RGB conversion. Once
        # clipped to RGB, the original a:b hue direction cannot be recovered.
        return render_gamut_safe(frame_bgr, chroma + 128.0)


def create_colorizer(
    model: str = "tiny",
    device: str = "auto",
    input_size: int = 256,
    cache_dir: str | Path | None = None,
) -> DDColorizer:
    """Download a verified official checkpoint if needed and initialize DDColor.

    Hugging Face's normal cache is used unless ``cache_dir`` is supplied. A
    previously downloaded model can be loaded offline (``HF_HUB_OFFLINE=1``).
    Downloads contain tensor weights only; no remote Python code is executed.
    """
    if model not in MODEL_SPECS:
        raise ValueError("model must be 'tiny' or 'large'.")
    if device not in ("auto", "cpu", "cuda"):
        raise ValueError("device must be 'auto', 'cpu', or 'cuda'.")
    if type(input_size) is not int or input_size not in INPUT_SIZES:
        raise ValueError("input_size must be 256 or 512.")

    try:
        import torch
        from huggingface_hub import hf_hub_download
    except (ImportError, OSError) as exc:
        raise ModelError(
            "AI dependencies are missing or could not load. Follow the README's "
            "PyTorch installation step, then reinstall VideoColor using the "
            "installation guide."
        ) from exc
    version = tuple(int(part) for part in torch.__version__.split("+")[0].split(".")[:2])
    if version < (2, 6):
        raise ModelError("PyTorch 2.6 or newer is required for safe checkpoint loading.")
    resolved_device = _resolve_device(device, torch)
    spec = MODEL_SPECS[model]
    LOGGER.info(
        "Loading DDColor %s on %s (first download: %.0f MB).",
        model, resolved_device, spec.size_bytes / 1_000_000,
    )
    try:
        checkpoint = Path(hf_hub_download(
            repo_id=spec.repository,
            filename="pytorch_model.bin",
            revision=spec.revision,
            cache_dir=str(cache_dir) if cache_dir is not None else None,
        ))
    except Exception as exc:
        raise ModelError(
            f"Could not download or find cached DDColor {model} weights "
            f"({spec.size_bytes / 1_000_000:.0f} MB). Check internet access, "
            "available disk space, and your Hugging Face cache settings. "
            "No video is sent to Hugging Face."
        ) from exc
    _verify_checkpoint(checkpoint, spec)

    try:
        network = _load_network(checkpoint, spec, input_size, torch)
        network.to(resolved_device)
    except torch.cuda.OutOfMemoryError as exc:
        raise ModelError(
            "Not enough GPU memory to load DDColor. Select the tiny model or CPU."
        ) from exc
    except (RuntimeError, ValueError, KeyError, MemoryError) as exc:
        raise ModelError(
            "DDColor could not be initialized. Check free RAM, select the tiny "
            "model, and verify that the supported PyTorch version is installed."
        ) from exc
    return DDColorizer(
        network, torch_module=torch, model=model,
        device=resolved_device, input_size=input_size,
    )
