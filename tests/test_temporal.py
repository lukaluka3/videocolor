import cv2
import numpy as np
import pytest

from videocolor.temporal import TemporalChroma, _render_gamut_safe


def gray_and_color(light=128, a=132, b=145, size=(64, 96)):
    lab = np.empty((*size, 3), np.uint8)
    lab[:, :, 0] = light
    lab[:, :, 1:] = (128, 128)
    gray = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    lab[:, :, 1:] = (a, b)
    return gray, cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def test_smoothing_reduces_static_chroma_flicker_and_preserves_luminance():
    source, first = gray_and_color(a=130, b=140)
    _, second = gray_and_color(a=135, b=147)
    smoother = TemporalChroma(0.8)
    prior = smoother.apply(source, first)
    result = smoother.apply(source, second)
    raw = TemporalChroma(0).apply(source, second)
    previous_ab = cv2.cvtColor(prior, cv2.COLOR_BGR2LAB)[:, :, 1:].astype(float)
    result_lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
    raw_lab = cv2.cvtColor(raw, cv2.COLOR_BGR2LAB)
    assert np.mean(abs(result_lab[:, :, 1:] - previous_ab)) < np.mean(abs(raw_lab[:, :, 1:] - previous_ab))
    source_l = cv2.cvtColor(source, cv2.COLOR_BGR2LAB)[:, :, 0].astype(float)
    assert np.max(abs(result_lab[:, :, 0] - source_l)) <= 1


def test_scene_cut_does_not_import_previous_color():
    source, first = gray_and_color(light=50, a=145, b=140)
    next_source, second = gray_and_color(light=180, a=135, b=130)
    smoother = TemporalChroma(1)
    smoother.apply(source, first)
    result = smoother.apply(next_source, second)
    assert np.array_equal(result, TemporalChroma(0).apply(next_source, second))


def test_large_color_disagreement_is_not_smeared():
    source, first = gray_and_color(a=100, b=110)
    _, second = gray_and_color(a=150, b=150)
    smoother = TemporalChroma(1)
    smoother.apply(source, first)
    assert np.array_equal(smoother.apply(source, second), TemporalChroma(0).apply(source, second))


def test_resume_seed_matches_uninterrupted_temporal_state():
    source, first = gray_and_color(a=135, b=140)
    _, second = gray_and_color(a=138, b=144)
    uninterrupted = TemporalChroma(0.35)
    saved = uninterrupted.apply(source, first)
    expected = uninterrupted.apply(source, second)
    resumed = TemporalChroma(0.35)
    resumed.seed(source, saved)
    assert np.array_equal(expected, resumed.apply(source, second))


def test_motion_warp_aligns_previous_chroma_and_rejects_exposed_pixels(monkeypatch):
    """Known flow isolates alignment/occlusion logic from flow-estimator accuracy."""
    source, first = gray_and_color(a=129, b=129)
    first_lab = cv2.cvtColor(first, cv2.COLOR_BGR2LAB)
    first_lab[:, 30:50, 1:] = (139, 139)
    first = cv2.cvtColor(first_lab, cv2.COLOR_LAB2BGR)
    _, second = gray_and_color(a=132, b=132)
    smoother = TemporalChroma(0.8)
    smoother.apply(source, first)
    calls = []

    def known_flow(a, b):
        flow = np.zeros((*a.shape, 2), np.float32)
        flow[:, :, 0] = -10 if not calls else 10
        calls.append(True)
        return flow

    monkeypatch.setattr(smoother, "_flow", known_flow)
    result = cv2.cvtColor(smoother.apply(source, second), cv2.COLOR_BGR2LAB)
    raw = cv2.cvtColor(TemporalChroma(0).apply(source, second), cv2.COLOR_BGR2LAB)
    assert result[20, 55, 1] > raw[20, 55, 1]  # old patch moved right
    assert result[20, 35, 1] < raw[20, 35, 1]  # no stale patch at its old position
    assert np.array_equal(result[:, :9], raw[:, :9])  # outside old frame, use new color


@pytest.mark.parametrize("strength", [-0.1, 1.1, float("nan")])
def test_invalid_strength(strength):
    with pytest.raises(ValueError):
        TemporalChroma(strength)


def test_colorizer_shape_mismatch_fails():
    source, color = gray_and_color()
    with pytest.raises(ValueError, match="same"):
        TemporalChroma().apply(source, color[:10])


def test_zero_color_strength_is_achromatic_and_preserves_source_lightness():
    source, predicted = gray_and_color(light=160, a=146, b=150)
    result = TemporalChroma(color_strength=0).apply(source, predicted)
    assert np.max(np.ptp(result.astype(np.int16), axis=2)) <= 1
    source_l = cv2.cvtColor(source, cv2.COLOR_BGR2LAB)[:, :, 0].astype(float)
    result_l = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)[:, :, 0]
    assert np.max(abs(source_l - result_l)) <= 1


def test_default_colors_are_gentler_than_full_prediction_without_exposure_change():
    source, predicted = gray_and_color(light=128, a=146, b=152)
    mild = cv2.cvtColor(TemporalChroma(0).apply(source, predicted), cv2.COLOR_BGR2LAB)
    full = cv2.cvtColor(TemporalChroma(0, color_strength=1).apply(source, predicted), cv2.COLOR_BGR2LAB)
    mild_chroma = np.linalg.norm(mild[:, :, 1:].astype(float) - 128, axis=2).mean()
    full_chroma = np.linalg.norm(full[:, :, 1:].astype(float) - 128, axis=2).mean()
    assert 0.78 < mild_chroma / full_chroma < 0.92
    source_l = cv2.cvtColor(source, cv2.COLOR_BGR2LAB)[:, :, 0].astype(float)
    assert np.max(abs(mild[:, :, 0] - source_l)) <= 1
    assert np.max(abs(full[:, :, 0] - source_l)) <= 1


@pytest.mark.parametrize("color_strength", [0.25, 0.85, 1.0])
def test_color_strength_does_not_progressively_fade_static_frames(color_strength):
    source, predicted = gray_and_color(light=128, a=146, b=152)
    smoother = TemporalChroma(0.9, color_strength=color_strength)
    first = smoother.apply(source, predicted)
    result = first
    for _ in range(25):
        result = smoother.apply(source, predicted)
    first_ab = cv2.cvtColor(first, cv2.COLOR_BGR2LAB)[:, :, 1:].astype(float)
    last_ab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)[:, :, 1:].astype(float)
    assert np.max(abs(last_ab - first_ab)) <= 1
    resumed = TemporalChroma(0.9, color_strength=color_strength)
    resumed.seed(source, result)
    assert np.array_equal(smoother.apply(source, predicted), resumed.apply(source, predicted))


@pytest.mark.parametrize("color_strength", [-0.01, 1.01, float("nan"), float("inf")])
def test_invalid_color_strength(color_strength):
    with pytest.raises(ValueError, match="Color strength"):
        TemporalChroma(color_strength=color_strength)


def _gamut_fixture():
    levels = np.array([0, 1, 3, 8, 16, 32, 64, 128, 192, 224, 240, 248, 252, 254, 255], np.uint8)
    source = np.repeat(np.repeat(levels[:, None, None], 3, axis=2), 72, axis=1)
    angles = np.arange(72, dtype=np.float32) * np.pi / 36
    ab = np.stack((55 * np.cos(angles), 55 * np.sin(angles)), axis=-1)
    chroma = np.broadcast_to(ab + 128, (*source.shape[:2], 2)).copy()
    return source, chroma, angles


def test_gamut_compression_preserves_lightness_across_shadows_highlights_and_hues():
    source, chroma, _ = _gamut_fixture()
    source_lab = cv2.cvtColor(source.astype(np.float32) / 255, cv2.COLOR_BGR2LAB)
    unconstrained_lab = source_lab.copy()
    unconstrained_lab[:, :, 1:] = chroma - 128
    clipped = np.rint(cv2.cvtColor(unconstrained_lab, cv2.COLOR_LAB2BGR) * 255).astype(np.uint8)
    protected = _render_gamut_safe(source, chroma)
    clipped_lab = cv2.cvtColor(clipped.astype(np.float32) / 255, cv2.COLOR_BGR2LAB)
    protected_lab = cv2.cvtColor(protected.astype(np.float32) / 255, cv2.COLOR_BGR2LAB)
    before_error = abs(clipped_lab[:, :, 0] - source_lab[:, :, 0])
    after_error = abs(protected_lab[:, :, 0] - source_lab[:, :, 0])
    assert before_error.mean() > 4  # The fixture exercises actual out-of-gamut clipping.
    assert after_error.mean() < 0.12
    assert after_error.max() < 0.5  # Tolerance includes final RGB uint8 quantization.


def test_gamut_compression_preserves_hue_direction_instead_of_white_balancing():
    source, chroma, angles = _gamut_fixture()
    protected_lab = cv2.cvtColor(_render_gamut_safe(source, chroma).astype(np.float32) / 255, cv2.COLOR_BGR2LAB)
    magnitude = np.linalg.norm(protected_lab[:, :, 1:], axis=2)
    hue = np.arctan2(protected_lab[:, :, 2], protected_lab[:, :, 1])
    degrees = abs((hue - angles + np.pi) % (2 * np.pi) - np.pi) * 180 / np.pi
    # Near-neutral colors have no stable hue after 8-bit quantization.
    assert np.max(degrees[magnitude > 10]) < 3


def test_gamut_renderer_leaves_representable_colors_unchanged():
    source, _ = gray_and_color(light=128)
    chroma = np.empty((*source.shape[:2], 2), np.float32)
    chroma[:, :, :] = (138, 143)
    expected_lab = cv2.cvtColor(source.astype(np.float32) / 255, cv2.COLOR_BGR2LAB)
    expected_lab[:, :, 1:] = chroma - 128
    expected = cv2.cvtColor(expected_lab, cv2.COLOR_LAB2BGR)
    assert expected.min() > 0 and expected.max() < 1
    expected = np.rint(expected * 255).astype(np.uint8)
    assert np.array_equal(_render_gamut_safe(source, chroma), expected)


def test_exact_source_black_white_stay_exact_through_temporal_processing_and_resume():
    source, prediction = gray_and_color(light=128, a=150, b=150)
    source[:, :24] = 0
    source[:, 72:] = 255
    smoother = TemporalChroma(0.8, color_strength=1)
    saved = smoother.apply(source, prediction)
    assert np.all(saved[:, :24] == 0)
    assert np.all(saved[:, 72:] == 255)
    next_prediction = np.flip(prediction, axis=1).copy()
    expected = smoother.apply(source, next_prediction)
    resumed = TemporalChroma(0.8, color_strength=1)
    resumed.seed(source, saved)
    assert np.array_equal(expected, resumed.apply(source, next_prediction))
    assert np.all(expected[:, :24] == 0)
    assert np.all(expected[:, 72:] == 255)


def test_near_black_protection_does_not_raise_source_exposure():
    source = np.full((64, 96, 3), 3, np.uint8)
    _, prediction = gray_and_color(light=128, a=150, b=155)
    result = TemporalChroma(color_strength=1).apply(source, prediction)
    source_l = cv2.cvtColor(source.astype(np.float32) / 255, cv2.COLOR_BGR2LAB)[:, :, 0]
    result_l = cv2.cvtColor(result.astype(np.float32) / 255, cv2.COLOR_BGR2LAB)[:, :, 0]
    assert np.max(abs(result_l - source_l)) < 0.5


def test_manual_balance_changes_hue_once_and_remains_resumable():
    source, predicted = gray_and_color(light=128, a=144, b=138)
    neutral = cv2.cvtColor(TemporalChroma(0).apply(source, predicted), cv2.COLOR_BGR2LAB)
    smoother = TemporalChroma(0.8, tint=-6, warmth=3)
    first = smoother.apply(source, predicted)
    balanced = cv2.cvtColor(first, cv2.COLOR_BGR2LAB)
    assert balanced[0, 0, 1] < neutral[0, 0, 1] - 3
    assert balanced[0, 0, 2] > neutral[0, 0, 2] + 1
    last = first
    for _ in range(20):
        last = smoother.apply(source, predicted)
    last_ab = cv2.cvtColor(last, cv2.COLOR_BGR2LAB)[:, :, 1:].astype(float)
    assert np.max(abs(last_ab - balanced[:, :, 1:].astype(float))) <= 1
    resumed = TemporalChroma(0.8, tint=-6, warmth=3)
    resumed.seed(source, last)
    assert np.array_equal(smoother.apply(source, predicted), resumed.apply(source, predicted))


def test_balance_with_zero_strength_is_gray_and_extreme_source_pixels_stay_exact():
    source, predicted = gray_and_color(light=128, a=150, b=150)
    source[:, :24] = 0
    source[:, 72:] = 255
    output = TemporalChroma(color_strength=0, tint=-20, warmth=20).apply(source, predicted)
    assert np.max(np.ptp(output.astype(int), axis=2)) <= 1
    assert np.all(output[:, :24] == 0)
    assert np.all(output[:, 72:] == 255)


@pytest.mark.parametrize("options", [{"tint": float("nan")}, {"tint": 20.1}, {"warmth": -20.1}])
def test_invalid_manual_balance(options):
    with pytest.raises(ValueError, match="between -20 and 20"):
        TemporalChroma(**options)
