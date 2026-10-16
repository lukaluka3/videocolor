"""Shared source-lightness-preserving color processing for AI and temporal color."""

from __future__ import annotations

import cv2
import numpy as np

# OpenCV uint8 Lab stores L* scaled to 0-255; these limits are written in L*.
_CAST_SAMPLE_L = (2.0, 18.0)  # Dark picture content, excluding near-black letterbox bars.
_CAST_FADE_L = (18.0, 55.0)  # The cast correction fades out through the mid-tones.
_MAX_CAST = 15.0  # Never remove more than this; dark scenes may truly be colored.
_SHADOW_L = (3.0, 26.0)  # Color fades in from neutral black.
_HIGHLIGHT_L = (82.0, 99.0)  # Near-white loses most of its color, like film.
_HIGHLIGHT_KEEP = 0.35
_CHROMA_CEILING = 34.0  # Soft limit on a*/b* distance from neutral.
_MIN_CAST_PIXELS = 500


def _smoothstep(values: np.ndarray, low: float, high: float) -> np.ndarray:
    t = np.clip((values - low) / (high - low), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _lightness(luminance: np.ndarray) -> np.ndarray:
    return luminance.astype(np.float32) * (100.0 / 255.0)


def remove_shadow_cast(luminance: np.ndarray, ab: np.ndarray) -> np.ndarray:
    """Prepare a raw prediction: soften color edges and remove a dark-area cast.

    ``luminance`` is OpenCV uint8-Lab L and ``ab`` is a*/b* offset from neutral.
    DDColor often fills dark areas with one uniform green/teal tint; its median
    over dark content is removed, fading out through the mid-tones. Lightness is
    not used for anything else and remains the source's.
    """
    height = ab.shape[0]
    sigma = 1.6 * height / 720
    if sigma >= 0.5:
        ab = cv2.GaussianBlur(ab, (0, 0), sigma)
    lightness = _lightness(luminance)
    dark = (lightness > _CAST_SAMPLE_L[0]) & (lightness < _CAST_SAMPLE_L[1])
    if np.count_nonzero(dark) < _MIN_CAST_PIXELS:
        return ab
    cast = np.median(ab[dark], axis=0)
    size = float(np.linalg.norm(cast))
    if size > _MAX_CAST:
        cast *= _MAX_CAST / size
    fade = 1.0 - _smoothstep(lightness, *_CAST_FADE_L)
    return ab - cast[None, None, :] * fade[:, :, None]


def film_chroma_response(luminance: np.ndarray, ab: np.ndarray) -> np.ndarray:
    """Fade color toward black and white, then softly cap saturation.

    Real color film loses chroma in deep shadows and near-white highlights, and
    rarely shows neon colors. The cap is smooth (tanh), so ordering of colors by
    saturation is retained and moderate colors are barely changed.
    """
    lightness = _lightness(luminance)
    weight = _smoothstep(lightness, *_SHADOW_L)
    weight *= 1.0 - (1.0 - _HIGHLIGHT_KEEP) * _smoothstep(lightness, *_HIGHLIGHT_L)
    ab = ab * weight[:, :, None]
    chroma = np.linalg.norm(ab, axis=2)
    scale = np.ones_like(chroma)
    moving = chroma > 1e-6
    scale[moving] = _CHROMA_CEILING * np.tanh(chroma[moving] / _CHROMA_CEILING) / chroma[moving]
    return ab * scale[:, :, None]


def render_gamut_safe(source_bgr: np.ndarray, chroma: np.ndarray) -> np.ndarray:
    """Reduce only out-of-gamut chroma, retaining Lab hue and source lightness.

    OpenCV clips Lab-to-sRGB conversions to [0, 1]. A clipped channel identifies
    a gamut boundary: binary search approaches it from neutral chroma while
    keeping L and the a:b ratio fixed. Only affected pixels require the search.
    No white balance, exposure change, or stylistic shadow desaturation occurs
    here; the optional natural grade is applied earlier, to the chroma.
    ``chroma`` uses OpenCV's uint8-Lab convention (neutral is 128), but remains
    floating point so temporal/color-strength arithmetic is not quantized early.
    """
    lab = cv2.cvtColor(source_bgr.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)
    lab[:, :, 1:] = chroma - 128.0
    rendered = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    black = np.all(source_bgr == 0, axis=2)
    white = np.all(source_bgr == 255, axis=2)
    endpoints = black | white
    clipped = np.any((rendered <= 0.0) | (rendered >= 1.0), axis=2) & ~endpoints
    if np.any(clipped):
        pixels = lab[clipped].reshape(-1, 1, 3)
        original_ab = pixels[:, :, 1:].copy()
        low = np.zeros((len(pixels), 1, 1), dtype=np.float32)
        high = np.ones_like(low)
        for _ in range(10):
            midpoint = (low + high) * 0.5
            pixels[:, :, 1:] = original_ab * midpoint
            candidate = cv2.cvtColor(pixels, cv2.COLOR_LAB2BGR)
            fits = np.all((candidate > 0.0) & (candidate < 1.0), axis=2, keepdims=True)
            low = np.where(fits, midpoint, low)
            high = np.where(fits, high, midpoint)
        pixels[:, :, 1:] = original_ab * low
        rendered[clipped] = cv2.cvtColor(pixels, cv2.COLOR_LAB2BGR).reshape(-1, 3)
    result = np.clip(np.rint(rendered * 255.0), 0, 255).astype(np.uint8)
    # Exact digital black/white carry no color information and must stay exact.
    result[endpoints] = source_bgr[endpoints]
    return result
