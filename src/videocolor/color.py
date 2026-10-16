"""Shared source-lightness-preserving rendering for AI and temporal color."""

from __future__ import annotations

import cv2
import numpy as np


def render_gamut_safe(source_bgr: np.ndarray, chroma: np.ndarray) -> np.ndarray:
    """Reduce only out-of-gamut chroma, retaining Lab hue and source lightness.

    OpenCV clips Lab-to-sRGB conversions to [0, 1]. A clipped channel identifies
    a gamut boundary: binary search approaches it from neutral chroma while
    keeping L and the a:b ratio fixed. Only affected pixels require the search.
    No white balance, exposure change, or stylistic shadow desaturation occurs.
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
