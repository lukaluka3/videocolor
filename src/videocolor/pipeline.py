"""Recoverable local-video processing with explicit ownership of job outputs."""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import shutil
import warnings
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Iterator
from uuid import uuid4

import cv2
import numpy as np

from .media import MediaError, encode_video, extract_frames, inspect_video, output_dimensions, verify_output
from .temporal import COLOR_PIPELINE_REVISION, TemporalChroma

JOB_FILE = "videocolor-job.json"
JOB_VERSION = 1


class JobError(RuntimeError):
    """A job cannot be created or resumed without risking inconsistent results."""


def _save_json(path: Path, content: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    with temporary.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(content, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _fingerprint(source: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return {"name": source.name, "size": source.stat().st_size, "sha256": digest.hexdigest()}


@contextlib.contextmanager
def _job_lock(directory: Path) -> Iterator[None]:
    path = directory / ".videocolor.lock"
    if path.is_symlink():
        raise JobError("The job lock must not be a symbolic link.")
    handle = path.open("a+b")
    try:
        if path.stat().st_size == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise JobError("Another process is already using this output directory.") from exc
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        handle.close()


def _read_frame(path: Path, width: int, height: int) -> np.ndarray:
    if path.is_symlink() or not path.is_file():
        raise JobError(f"Missing or unsafe frame: {path.name}")
    try:
        frame = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    except (OSError, cv2.error, ValueError) as exc:
        raise JobError(f"Cannot decode frame: {path.name}") from exc
    if frame is None or frame.shape != (height, width, 3):
        raise JobError(f"Invalid frame or unexpected dimensions: {path.name}")
    return frame


def _write_frame(path: Path, frame: np.ndarray) -> None:
    if path.exists() or path.is_symlink():
        raise JobError(f"Refusing to replace an existing completed frame: {path.name}")
    success, encoded = cv2.imencode(".png", frame)
    if not success:
        raise JobError("OpenCV could not encode the completed frame.")
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    with temporary.open("xb") as handle:
        handle.write(encoded.tobytes())
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _frame_list(directory: Path, width: int, height: int, expected: int | None = None) -> list[Path]:
    if directory.is_symlink() or not directory.is_dir():
        raise JobError("The job frame directory is missing or unsafe.")
    frames = sorted(directory.glob("*.png"))
    if expected is not None and len(frames) != expected:
        raise JobError("Saved source frame count does not match the job manifest.")
    for number, frame in enumerate(frames, 1):
        if frame.name != f"{number:08d}.png":
            raise JobError("Saved frames are not a complete consecutive sequence; use a new output directory.")
        _read_frame(frame, width, height)
    return frames


def _validate_options(
    start: float, duration: float | None, max_height: int | None,
    input_size: int, smoothing: float, color_strength: float,
    tint: float = 0.0, warmth: float = 0.0,
) -> None:
    if not math.isfinite(start) or start < 0:
        raise ValueError("Start must be a finite number greater than or equal to zero.")
    if duration is not None and (not math.isfinite(duration) or duration <= 0):
        raise ValueError("Duration must be a positive finite number, or None for the full remainder.")
    if max_height is not None and (not isinstance(max_height, int) or max_height < 2):
        raise ValueError("Maximum height must be an integer of at least 2 pixels, or None.")
    if type(input_size) is not int or input_size not in (256, 512):
        raise ValueError("Model input size must be 256 or 512.")
    if not math.isfinite(smoothing) or not 0 <= smoothing <= 1:
        raise ValueError("Smoothing must be between 0 and 1.")
    if not math.isfinite(color_strength) or not 0 <= color_strength <= 1:
        raise ValueError("Color strength must be between 0 and 1.")
    for name, value in (("Tint", tint), ("Warmth", warmth)):
        if not math.isfinite(value) or not -20 <= value <= 20:
            raise ValueError(f"{name} must be a finite number between -20 and 20.")


def _disk_preflight(directory: Path, job: dict[str, Any], info: dict[str, Any]) -> None:
    """Reserve a conservative raw-PNG budget instead of filling a user's disk.

    Actual PNG compression varies with the footage. A smaller preview/resolution
    is the supported way to reduce the estimate. Already saved source and color
    frames are excluded from the remaining budget when resuming.
    """
    if (directory / "result.mp4").exists():
        return
    config = job["config"]
    seconds = config["duration"]
    if info["duration"] is not None:
        remaining = max(0, info["duration"] - config["start"])
        seconds = min(seconds, remaining) if seconds is not None else remaining
    if "frame_count" in job:
        count = int(job["frame_count"])
    elif seconds is not None:
        count = math.ceil(seconds * float(Fraction(info["fps"])))
    else:
        warnings.warn("Video duration is unknown; disk needs cannot be estimated. Monitor available disk space.", RuntimeWarning)
        count = 0
    source_frames = 0 if "source_directory" in job else count
    color_frames = max(0, count - int(job.get("completed_frames", 0)))
    pixels = config["width"] * config["height"]
    # PNG can be slightly larger than raw RGB for incompressible images.
    per_png = pixels * 3 + config["height"] + 2048
    encode_budget = max(64 * 1024 * 1024, pixels * count // 2)
    estimated = int((source_frames + color_frames) * per_png * 1.01 + encode_budget)
    free = shutil.disk_usage(directory).free
    job["estimated_additional_disk_bytes"] = estimated
    if free < estimated:
        gib = 1024 ** 3
        raise JobError(
            f"Insufficient disk space for the conservative PNG working-file estimate: "
            f"{estimated / gib:.2f} GiB needed, {free / gib:.2f} GiB available. "
            "Use a shorter duration, a lower maximum height, or another drive. "
            "Actual compressed files may be smaller; completed job frames are retained."
        )


def process_video(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    model: str = "tiny",
    device: str = "auto",
    input_size: int = 256,
    cache_dir: str | Path | None = None,
    start: float = 0.0,
    duration: float | None = 10.0,
    max_height: int | None = 720,
    smoothing: float = 0.35,
    color_strength: float = 0.85,
    tint: float = 0.0,
    warmth: float = 0.0,
    resume: bool = False,
    progress: Callable[[int, int], None] | None = None,
    _colorizer: Any = None,
) -> Path:
    """Colorize a preview (default: first 10 seconds, at most 720p), or full clip.

    Output is always ``output_dir/result.mp4``. A directory with an existing job
    requires ``resume=True`` and identical source bytes and options. Interrupted
    frames remain available, and completed frames are never overwritten. ``None``
    duration processes the remainder; ``None`` max_height preserves source size
    apart from the even dimensions required by the output codec.
    ``color_strength`` controls saturation of predicted colors, from grayscale
    (0) to full model chroma (1), without changing source luminance.

    The progress callback receives ``(completed_frames, total_frames)`` and may
    raise ``InterruptedError`` to cancel between frames. The private colorizer
    hook is intended for isolated tests only; the normal path uses real weights.
    """
    _validate_options(start, duration, max_height, input_size, smoothing, color_strength, tint, warmth)
    if model not in ("tiny", "large") or device not in ("auto", "cpu", "cuda"):
        raise ValueError("Use model tiny/large and device auto/cpu/cuda.")
    source = Path(input_path).expanduser().resolve()
    info = inspect_video(source)
    if info["duration"] is not None and start >= info["duration"]:
        raise ValueError("Start time is at or beyond the end of the video.")
    width, height = output_dimensions(info, max_height)
    raw_directory = Path(output_dir).expanduser()
    if raw_directory.is_symlink():
        raise JobError("Output directory must not be a symbolic link.")
    directory = raw_directory.resolve()
    if directory in source.parents or directory == source:
        raise JobError("Choose a dedicated output directory that does not contain the input video.")
    marker = directory / JOB_FILE
    if resume and not marker.is_file():
        raise JobError("There is no saved job to resume in this directory.")
    if directory.exists():
        if not directory.is_dir():
            raise JobError("Output path is already a file.")
        if any(directory.iterdir()) and not marker.is_file():
            raise JobError("Output directory is not an owned VideoColor job; choose a new or empty directory.")
    else:
        directory.mkdir(parents=True)
    fingerprint = _fingerprint(source)
    config = {
        "model": model, "device": device, "input_size": input_size,
        "start": float(start), "duration": float(duration) if duration is not None else None,
        "max_height": max_height, "smoothing": float(smoothing),
        "color_strength": float(color_strength),
        "tint": float(tint), "warmth": float(warmth),
        "width": width, "height": height, "fps": info["fps"],
    }
    with _job_lock(directory):
        if marker.is_symlink():
            raise JobError("Job manifest must not be a symbolic link.")
        if marker.exists():
            if not resume:
                raise JobError("This output directory already contains a job. Use --resume or a new directory.")
            try:
                job = json.loads(marker.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise JobError("The saved job manifest cannot be read.") from exc
            if not isinstance(job, dict) or job.get("format") != "videocolor-job" or job.get("version") != JOB_VERSION:
                raise JobError("The job manifest format is not supported.")
            saved_config = job.get("config")
            if isinstance(saved_config, dict):
                saved_config = {"tint": 0.0, "warmth": 0.0, **saved_config}
            if job.get("input") != fingerprint or saved_config != config:
                raise JobError("Resume requires the same input bytes/name and processing options. Choose a new output directory.")
        else:
            if resume:
                raise JobError("There is no saved job to resume in this directory.")
            job = {
                "format": "videocolor-job", "version": JOB_VERSION, "input": fingerprint,
                "config": config, "source_info": info, "status": "created",
                "completed_frames": 0,
                "notes": [
                    "AI colors are estimates, not evidence of historical colors.",
                    "Variable frame rates are normalized to the source average frame rate.",
                    "Only the first audio stream is included, transcoded to AAC; source metadata is removed.",
                ],
            }
            _save_json(marker, job)
        try:
            _disk_preflight(directory, job, info)
            if "source_directory" not in job:
                source_frames = directory / f"source-{uuid4().hex}"
                source_frames.mkdir()
                job["status"] = "extracting"
                _save_json(marker, job)
                extract_frames(source, source_frames, info, start=start, duration=duration, width=width, height=height)
                source_list = _frame_list(source_frames, width, height)
                if not source_list:
                    raise MediaError("No frames were decoded in the requested interval. Try an earlier start or longer duration.")
                job["source_directory"] = source_frames.name
                job["frame_count"] = len(source_list)
                _save_json(marker, job)
            else:
                source_name = job["source_directory"]
                if not isinstance(source_name, str) or Path(source_name).name != source_name or not source_name.startswith("source-"):
                    raise JobError("Unsafe source frame directory in the job manifest.")
                source_frames = directory / source_name
                source_list = _frame_list(source_frames, width, height, expected=job["frame_count"])
                if not source_list:
                    raise JobError("The saved source frame list is empty.")
            total = len(source_list)
            colored = directory / "colored"
            if not colored.exists():
                colored.mkdir()
            completed = _frame_list(colored, width, height)
            if len(completed) > total:
                raise JobError("There are more completed frames than source frames.")
            job["completed_frames"] = len(completed)
            if progress:
                progress(len(completed), total)
            final = directory / "result.mp4"
            if final.is_symlink():
                raise JobError("Output video must not be a symbolic link.")
            if final.exists():
                if len(completed) != total:
                    raise JobError("An output video exists but its frame sequence is incomplete. Choose a new directory.")
                verify_output(final, frame_count=total, width=width, height=height, has_audio=info["has_audio"])
                job["status"] = "complete"
                _save_json(marker, job)
                return final
            if len(completed) < total:
                if completed and job.get("color_pipeline_revision", 1) != COLOR_PIPELINE_REVISION:
                    raise JobError(
                        "These frames were created with a different color-processing version. "
                        "Start a new output directory, or finish this job with the original application version."
                    )
                if _colorizer is None:
                    from .models import create_colorizer
                    colorizer = create_colorizer(model=model, device=device, input_size=input_size, cache_dir=cache_dir)
                else:
                    colorizer = _colorizer
                model_info = getattr(colorizer, "model_info", {})
                # Keep public model facts; never serialize local cache paths.
                new_model_info = {
                    key: value for key, value in model_info.items()
                    if key in ("name", "model", "device", "input_size", "sha256", "license", "architecture",
                               "repository", "revision", "architecture_revision")
                    and isinstance(value, (str, int, float, bool, type(None)))
                } if isinstance(model_info, dict) else {}
                if completed:
                    saved_model_info = job.get("model_info", {})
                    identity_keys = ("model", "repository", "revision", "sha256", "architecture_revision", "device", "input_size")
                    if any(saved_model_info.get(key) != new_model_info.get(key) for key in identity_keys):
                        raise JobError(
                            "The loaded model revision or resolved device differs from the saved frames. "
                            "Use the original model/runtime or a new output directory."
                        )
                job["model_info"] = new_model_info
                job["color_pipeline_revision"] = COLOR_PIPELINE_REVISION
                job["status"] = "colorizing"
                _save_json(marker, job)
                temporal = TemporalChroma(smoothing, color_strength=color_strength, tint=tint, warmth=warmth)
                if completed:
                    previous_source = _read_frame(source_list[len(completed) - 1], width, height)
                    temporal.seed(previous_source, _read_frame(completed[-1], width, height))
                for index in range(len(completed), total):
                    original = _read_frame(source_list[index], width, height)
                    result = colorizer.colorize(original)
                    if not isinstance(result, np.ndarray):
                        raise JobError("The colorizer did not return an image array.")
                    result = temporal.apply(original, result)
                    _write_frame(colored / source_list[index].name, result)
                    job["completed_frames"] = index + 1
                    _save_json(marker, job)
                    if progress:
                        progress(index + 1, total)
            job["status"] = "encoding"
            _save_json(marker, job)
            temporary_video = directory / f"result-{uuid4().hex}.partial.mp4"
            encode_video(source, colored, temporary_video, info, start=start, frame_count=total)
            verify_output(temporary_video, frame_count=total, width=width, height=height, has_audio=info["has_audio"])
            if final.exists():
                raise JobError("An output video appeared during encoding; refusing to overwrite it.")
            os.replace(temporary_video, final)
            job["status"] = "complete"
            job.pop("last_error_type", None)
            _save_json(marker, job)
            return final
        except BaseException as exc:
            job["status"] = "interrupted" if isinstance(exc, (KeyboardInterrupt, InterruptedError)) else "failed"
            job["last_error_type"] = type(exc).__name__
            try:
                _save_json(marker, job)
            except OSError:
                pass  # A full disk must not hide the original error.
            raise
