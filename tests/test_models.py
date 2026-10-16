"""Offline model-boundary tests; no checkpoints or network access required."""

import hashlib
import json
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from videocolor.models import (
    DDColorizer, MODEL_SPECS, ModelError, ModelSpec, _resolve_device,
    _load_network, _validate_frame, _verify_checkpoint, create_colorizer,
)


@pytest.mark.parametrize("kwargs", [
    {"model": "untrusted/repository"},
    {"device": "cuda:1"},
    {"input_size": 0},
    {"input_size": 513},
    {"input_size": True},
    {"input_size": 256.0},
])
def test_invalid_options_fail_before_dependency_import_or_download(kwargs):
    with pytest.raises(ValueError):
        create_colorizer(**kwargs)


@pytest.mark.parametrize("frame", [
    None,
    np.zeros((10, 10), dtype=np.uint8),
    np.zeros((10, 10, 4), dtype=np.uint8),
    np.zeros((10, 10, 3), dtype=np.float32),
    np.zeros((0, 10, 3), dtype=np.uint8),
])
def test_bad_frames_rejected(frame):
    with pytest.raises(ValueError):
        _validate_frame(frame)


def test_device_selection_does_not_silently_downgrade_requested_cuda():
    cpu_only = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    gpu = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True))
    assert _resolve_device("auto", cpu_only) == "cpu"
    assert _resolve_device("auto", gpu) == "cuda"
    assert _resolve_device("cpu", gpu) == "cpu"
    with pytest.raises(ModelError, match="CUDA is unavailable"):
        _resolve_device("cuda", cpu_only)


def test_checkpoint_integrity_rejects_same_size_tampering(tmp_path):
    original = b"known checkpoint"
    checkpoint = tmp_path / "pytorch_model.bin"
    spec = ModelSpec("official/model", "pinned", hashlib.sha256(original).hexdigest(),
                     len(original), "convnext-t")
    checkpoint.write_bytes(original)
    _verify_checkpoint(checkpoint, spec)
    checkpoint.write_bytes(b"x" * len(original))
    with pytest.raises(ModelError, match="SHA-256"):
        _verify_checkpoint(checkpoint, spec)
    checkpoint.write_bytes(b"truncated")
    with pytest.raises(ModelError, match="unexpected size"):
        _verify_checkpoint(checkpoint, spec)


def test_model_provenance_is_immutable_pinned_and_serializable():
    for spec in MODEL_SPECS.values():
        assert len(spec.revision) == 40
        assert len(spec.sha256) == 64
        assert spec.repository.startswith("piddnad/")


@pytest.fixture
def torch_module():
    return pytest.importorskip("torch")


def _colorizer(torch_module, output_chroma=(0.0, 0.0)):
    class Network:
        def __call__(self, image):
            assert not torch_module.is_grad_enabled()
            assert image.shape == (1, 3, 256, 256)
            # Input must be neutral RGB regardless of incoming sepia/color.
            assert float((image[:, 0] - image[:, 1]).abs().max()) < 0.002
            assert float((image[:, 1] - image[:, 2]).abs().max()) < 0.002
            output = torch_module.zeros((1, 2, 256, 256))
            output[:, 0] = output_chroma[0]
            output[:, 1] = output_chroma[1]
            return output

    return DDColorizer(Network(), torch_module=torch_module, model="tiny",
                       device="cpu", input_size=256)


@pytest.mark.parametrize("shape", [(1, 1), (31, 57), (57, 31)])
def test_inference_retains_dimensions_dtype_and_luminance(torch_module, shape):
    rng = np.random.default_rng(7)
    gray = rng.integers(10, 240, shape, dtype=np.uint8)
    frame = np.repeat(gray[:, :, None], 3, axis=2)
    original = frame.copy()
    colorizer = _colorizer(torch_module)
    result = colorizer.colorize(frame[:, ::-1])  # non-contiguous input is valid
    assert result.shape == frame.shape
    assert result.dtype == np.uint8
    np.testing.assert_array_equal(frame, original)
    np.testing.assert_allclose(result.astype(float), frame[:, ::-1], atol=1)
    json.dumps(colorizer.model_info)


def test_inference_retains_input_lightness_and_correct_bgr_order(torch_module):
    frame = np.full((23, 47, 3), (60, 90, 130), dtype=np.uint8)
    result = _colorizer(torch_module, (20.0, 15.0)).colorize(frame)
    original_l = cv2.cvtColor(frame.astype(np.float32) / 255, cv2.COLOR_BGR2Lab)[:, :, 0]
    result_l = cv2.cvtColor(result.astype(np.float32) / 255, cv2.COLOR_BGR2Lab)[:, :, 0]
    np.testing.assert_allclose(result_l, original_l, atol=0.5)
    assert result[0, 0, 2] > result[0, 0, 0]  # positive Lab a/b is reddish, not blue


def test_nonfinite_model_prediction_stops_processing(torch_module):
    with pytest.raises(ModelError, match="non-finite"):
        _colorizer(torch_module, (float("nan"), 0)).colorize(
            np.full((8, 8, 3), 128, dtype=np.uint8)
        )


@pytest.mark.parametrize("output_chroma", [(60.0, 50.0), (-60.0, 50.0), (45.0, -60.0)])
def test_extreme_model_colors_preserve_lightness_and_raw_prediction_hue(torch_module, output_chroma):
    levels = np.array([0, 3, 16, 32, 64, 128, 192, 224, 240, 252, 255], np.uint8)
    frame = np.repeat(np.repeat(levels[:, None, None], 3, axis=2), 8, axis=1)
    result = _colorizer(torch_module, output_chroma).colorize(frame)
    source_lab = cv2.cvtColor(frame.astype(np.float32) / 255, cv2.COLOR_BGR2LAB)
    result_lab = cv2.cvtColor(result.astype(np.float32) / 255, cv2.COLOR_BGR2LAB)
    assert np.max(abs(result_lab[:, :, 0] - source_lab[:, :, 0])) < 0.5
    assert np.all(result[0] == 0) and np.all(result[-1] == 255)
    magnitude = np.linalg.norm(result_lab[:, :, 1:], axis=2)
    actual_hue = np.arctan2(result_lab[:, :, 2], result_lab[:, :, 1])
    predicted_hue = np.arctan2(output_chroma[1], output_chroma[0])
    difference = abs((actual_hue - predicted_hue + np.pi) % (2 * np.pi) - np.pi) * 180 / np.pi
    assert np.max(difference[magnitude > 10]) < 3
    # Confirm that the same raw prediction would cause a meaningful error if
    # clipped before temporal processing had access to its original hue.
    naive_lab = source_lab.copy()
    naive_lab[:, :, 1:] = output_chroma
    naive_rgb = cv2.cvtColor(naive_lab, cv2.COLOR_LAB2BGR)
    naive_roundtrip = cv2.cvtColor(naive_rgb, cv2.COLOR_BGR2LAB)
    assert np.max(abs(naive_roundtrip[:, :, 0] - source_lab[:, :, 0])) > 5


def test_ddcolor_can_construct_on_meta_without_allocating_parameters(torch_module):
    from videocolor._vendor.ddcolor.model import DDColor

    with torch_module.device("meta"):
        model = DDColor(encoder_name="convnext-t", input_size=(256, 256),
                        num_output_channels=2, last_norm="Spectral", num_queries=100)
    assert all(parameter.is_meta for parameter in model.parameters())
    # Legacy torch.Tensor constants for mean/std can be real, but large
    # parameter/buffer allocations must stay metadata-only.
    actual_bytes = sum(buffer.numel() * buffer.element_size()
                       for buffer in model.buffers() if not buffer.is_meta)
    assert actual_bytes <= 64


@pytest.mark.parametrize("wrapped", [False, True])
def test_mapped_assignment_matches_normal_loading_with_spectral_hooks(
    torch_module, tmp_path, monkeypatch, wrapped,
):
    from videocolor._vendor.ddcolor import model as architecture

    class SmallNetwork(torch_module.nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.conv = torch_module.nn.utils.spectral_norm(torch_module.nn.Conv2d(3, 2, 3, padding=1))
            self.norm = torch_module.nn.BatchNorm2d(2)
            self.register_buffer("offset", torch_module.tensor(0.25))

        def forward(self, image):
            return self.norm(self.conv(image)) + self.offset

    reference = SmallNetwork().eval()
    sample = torch_module.randn(1, 3, 7, 9)
    with torch_module.inference_mode():
        expected = reference(sample)
    checkpoint = tmp_path / "model.bin"
    state = reference.state_dict()
    torch_module.save({"params": state} if wrapped else state, checkpoint)
    before = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    monkeypatch.setattr(architecture, "DDColor", SmallNetwork)
    loaded = _load_network(checkpoint, MODEL_SPECS["tiny"], 256, torch_module).to("cpu")
    with torch_module.inference_mode():
        actual = loaded(sample)
    torch_module.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert not loaded.training
    assert not any(parameter.requires_grad for parameter in loaded.parameters())
    assert before == hashlib.sha256(checkpoint.read_bytes()).hexdigest()


def test_mapped_loader_rejects_missing_checkpoint_parameters(torch_module, tmp_path, monkeypatch):
    from videocolor._vendor.ddcolor import model as architecture

    class SmallNetwork(torch_module.nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.linear = torch_module.nn.Linear(2, 2)

    checkpoint = tmp_path / "missing-parameters.bin"
    torch_module.save({}, checkpoint)
    monkeypatch.setattr(architecture, "DDColor", SmallNetwork)
    with pytest.raises(RuntimeError, match="Missing key"):
        _load_network(checkpoint, MODEL_SPECS["tiny"], 256, torch_module)
