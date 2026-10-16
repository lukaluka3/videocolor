"""Motion-compensated chroma smoothing, with conservative reset/rejection rules."""

from __future__ import annotations

import cv2
import numpy as np

from .color import render_gamut_safe


COLOR_PIPELINE_REVISION = 2


# Keep the private helper import usable for existing isolated rendering checks.
_render_gamut_safe = render_gamut_safe


class TemporalChroma:
    """Blend only aligned, consistent chroma; keep the source luminance.

    Optical flow is an estimate, not a guarantee of flicker-free footage. Scene
    cuts, occlusions, large color disagreements, and brightness changes reduce
    or disable blending. A strength of zero disables temporal blending.
    ``color_strength`` gently reduces model chroma: zero is grayscale, one is
    the full prediction. It does not change exposure or establish true colors.
    """

    def __init__(self, strength: float = 0.35, color_strength: float = 0.85,
                 tint: float = 0.0, warmth: float = 0.0):
        if not np.isfinite(strength) or not 0 <= strength <= 1:
            raise ValueError("Smoothing strength must be between 0 and 1.")
        self.strength = float(strength)
        if not np.isfinite(color_strength) or not 0 <= color_strength <= 1:
            raise ValueError("Color strength must be between 0 and 1.")
        self.color_strength = float(color_strength)
        for name, value in (("Tint", tint), ("Warmth", warmth)):
            if not np.isfinite(value) or not -20 <= value <= 20:
                raise ValueError(f"{name} must be a finite number between -20 and 20.")
        self.balance = np.array([tint, warmth], dtype=np.float32)
        self.previous_l: np.ndarray | None = None
        self.previous_ab: np.ndarray | None = None

    def seed(self, source_bgr: np.ndarray, color_bgr: np.ndarray) -> None:
        """Restore state from an already completed frame when resuming."""
        self._validate(source_bgr, color_bgr)
        self.previous_l = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2LAB)[:, :, 0]
        self.previous_ab = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2LAB)[:, :, 1:].astype(np.float32)

    @staticmethod
    def _validate(source: np.ndarray, color: np.ndarray) -> None:
        if source.shape != color.shape or source.ndim != 3 or source.shape[2] != 3:
            raise ValueError("Source and color frames must have the same H x W x 3 shape.")
        if source.dtype != np.uint8 or color.dtype != np.uint8:
            raise ValueError("Source and color frames must use uint8 pixels.")

    @staticmethod
    def _scene_cut(previous: np.ndarray, current: np.ndarray) -> bool:
        previous_hist = cv2.calcHist([previous], [0], None, [32], [0, 256]).ravel()
        current_hist = cv2.calcHist([current], [0], None, [32], [0, 256]).ravel()
        difference = np.abs(previous_hist - current_hist).sum() / (2 * current.size)
        brightness_change = np.mean(np.abs(previous.astype(np.float32) - current))
        return bool(difference > 0.5 or brightness_change > 45)

    @staticmethod
    def _flow(first: np.ndarray, second: np.ndarray) -> np.ndarray:
        height, width = first.shape
        ratio = min(1.0, 512 / max(height, width))
        if ratio < 1:
            small_size = (max(2, round(width * ratio)), max(2, round(height * ratio)))
            a = cv2.resize(first, small_size, interpolation=cv2.INTER_AREA)
            b = cv2.resize(second, small_size, interpolation=cv2.INTER_AREA)
        else:
            a, b = first, second
        flow = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 21, 3, 5, 1.2, 0)
        if ratio < 1:
            flow = cv2.resize(flow, (width, height), interpolation=cv2.INTER_LINEAR)
            flow[:, :, 0] *= width / a.shape[1]
            flow[:, :, 1] *= height / a.shape[0]
        return flow

    def apply(self, source_bgr: np.ndarray, color_bgr: np.ndarray) -> np.ndarray:
        self._validate(source_bgr, color_bgr)
        luminance = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2LAB)[:, :, 0]
        color_lab = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2LAB)
        chroma = color_lab[:, :, 1:].astype(np.float32)
        # Scale each fresh prediction once. Previous saved chroma is already
        # adjusted; rescaling it after blending would fade colors over time.
        # Manual Lab balance is applied once to each fresh prediction. Neutral
        # defaults do nothing; strength zero must remain grayscale even with
        # balance offsets. Never adjust the saved previous frame a second time.
        chroma = 128.0 + self.color_strength * (chroma - 128.0 + self.balance)
        previous = self.previous_l
        if (
            self.strength > 0 and previous is not None and self.previous_ab is not None
            and previous.shape == luminance.shape and not self._scene_cut(previous, luminance)
        ):
            backwards = self._flow(luminance, previous)
            forwards = self._flow(previous, luminance)
            height, width = luminance.shape
            grid_x, grid_y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
            map_x, map_y = grid_x + backwards[:, :, 0], grid_y + backwards[:, :, 1]
            warped_l = cv2.remap(previous, map_x, map_y, cv2.INTER_LINEAR)
            warped_ab = cv2.remap(self.previous_ab, map_x, map_y, cv2.INTER_LINEAR)
            warped_forward = cv2.remap(forwards, map_x, map_y, cv2.INTER_LINEAR)
            consistency_error = np.linalg.norm(backwards + warped_forward, axis=2)
            tolerance = 1.0 + 0.05 * (np.linalg.norm(backwards, axis=2) + np.linalg.norm(warped_forward, axis=2))
            valid = (map_x >= 0) & (map_x <= width - 1) & (map_y >= 0) & (map_y <= height - 1)
            valid &= consistency_error < tolerance
            light_error = np.abs(warped_l.astype(np.float32) - luminance)
            chroma_error = np.linalg.norm(warped_ab - chroma, axis=2)
            # Newly revealed areas and implausible color jumps should use the new prediction.
            valid &= (light_error < 20) & (chroma_error < 35)
            confidence = np.exp(-np.square(light_error / 10)) * np.exp(-np.square(chroma_error / 25))
            weight = (self.strength * confidence * valid)[:, :, None]
            chroma = chroma * (1 - weight) + warped_ab * weight
        result = render_gamut_safe(source_bgr, chroma)
        # Store the quantized image actually saved on disk, so resume is reproducible.
        self.seed(source_bgr, result)
        return result
