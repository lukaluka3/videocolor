"""Small, checked FFmpeg helpers. Commands are always argument lists, never shells."""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Any


class MediaError(RuntimeError):
    """A media file or FFmpeg operation could not be processed safely."""


def executable(name: str) -> str:
    found = shutil.which(name)
    if not found:
        raise MediaError(
            f"{name} was not found on PATH. Install FFmpeg (including ffprobe), "
            "then reopen your terminal. See the installation guide."
        )
    return found


def run_ffmpeg(arguments: list[str]) -> None:
    command = [executable("ffmpeg"), "-nostdin", "-hide_banner", "-loglevel", "error"]
    result = subprocess.run(
        command + arguments, capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    if result.returncode:
        details = result.stderr.strip()[-4000:] or "No diagnostic output."
        raise MediaError(f"FFmpeg failed (exit {result.returncode}): {details}")


def _probe(path: Path, *, count_frames: bool = False) -> dict[str, Any]:
    args = [executable("ffprobe"), "-v", "error", "-protocol_whitelist", "file,pipe", "-show_format", "-show_streams"]
    if count_frames:
        args.append("-count_frames")
    args += ["-of", "json", str(path)]
    try:
        result = subprocess.run(
            args, capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=300 if count_frames else 30,
        )
    except subprocess.TimeoutExpired as exc:
        raise MediaError("FFprobe timed out while reading the video.") from exc
    if result.returncode:
        raise MediaError(f"FFprobe could not read the video: {result.stderr.strip()[-2000:]}")
    try:
        return json.loads(result.stdout)
    except (ValueError, TypeError) as exc:
        raise MediaError("FFprobe returned an invalid response.") from exc


def _positive_float(value: object) -> float | None:
    try:
        number = float(value)  # type: ignore[arg-type]
        return number if math.isfinite(number) and number > 0 else None
    except (TypeError, ValueError):
        return None


def inspect_video(path: str | Path) -> dict[str, Any]:
    """Inspect the first moving-picture stream and first audio stream.

    ``width``/``height`` are display dimensions, including quarter-turn rotation;
    ``fps`` is a rational string. Variable-frame-rate inputs are normalized to
    this average rate by the processing pipeline. Absolute file paths and source
    metadata are intentionally omitted from the returned information.
    """
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise MediaError(f"Input video does not exist or is not a file: {source.name}")
    probe = _probe(source)
    videos = [
        stream for stream in probe.get("streams", [])
        if stream.get("codec_type") == "video"
        and not stream.get("disposition", {}).get("attached_pic")
    ]
    if not videos:
        raise MediaError("The input does not contain a moving-picture video stream.")
    video = videos[0]
    fps: Fraction | None = None
    for key in ("avg_frame_rate", "r_frame_rate"):
        try:
            candidate = Fraction(str(video.get(key, "0/1")))
            if candidate > 0:
                fps = candidate
                break
        except (ValueError, ZeroDivisionError):
            pass
    if fps is None:
        raise MediaError("The video has no usable frame rate.")
    coded_width, coded_height = int(video.get("width", 0)), int(video.get("height", 0))
    if min(coded_width, coded_height) < 2:
        raise MediaError("Video dimensions must be at least 2 by 2 pixels.")
    rotation = 0.0
    for side_data in video.get("side_data_list", []):
        if "rotation" in side_data:
            rotation = float(side_data["rotation"])
            break
    else:
        try:
            rotation = float(video.get("tags", {}).get("rotate", 0))
        except (ValueError, TypeError):
            pass
    sar = Fraction(1, 1)
    try:
        candidate_sar = Fraction(str(video.get("sample_aspect_ratio", "1:1")).replace(":", "/"))
        if candidate_sar > 0:
            sar = candidate_sar
    except (ValueError, ZeroDivisionError):
        pass
    width, height = max(2, round(coded_width * sar)), coded_height
    if round(rotation) % 180 in (90,):
        width, height = height, width
    audio = next((s for s in probe.get("streams", []) if s.get("codec_type") == "audio"), None)
    try:
        video_start = float(video.get("start_time", 0))
        format_start = float(probe.get("format", {}).get("start_time", 0))
        start_offset = max(0.0, video_start - format_start)
        if not math.isfinite(start_offset):
            start_offset = 0.0
    except (ValueError, TypeError):
        start_offset = 0.0
    duration = _positive_float(video.get("duration"))
    if duration is None:
        container_duration = _positive_float(probe.get("format", {}).get("duration"))
        duration = max(0.0, container_duration - start_offset) if container_duration is not None else None
    return {
        "width": width, "height": height,
        "coded_width": coded_width, "coded_height": coded_height,
        "rotation": rotation, "fps": f"{fps.numerator}/{fps.denominator}",
        "sample_aspect_ratio": str(sar), "video_start_offset": start_offset,
        "duration": duration, "has_audio": audio is not None,
        "video_stream_index": int(video["index"]),
        "audio_stream_index": int(audio["index"]) if audio else None,
        "codec": video.get("codec_name", "unknown"),
    }


def output_dimensions(info: dict[str, Any], max_height: int | None) -> tuple[int, int]:
    width, height = int(info["width"]), int(info["height"])
    scale = min(1.0, max_height / height) if max_height else 1.0
    # H.264/yuv420p needs even dimensions. Never enlarge the source.
    return max(2, int(width * scale) // 2 * 2), max(2, int(height * scale) // 2 * 2)


def extract_frames(
    source: Path, directory: Path, info: dict[str, Any], *,
    start: float, duration: float | None, width: int, height: int,
) -> None:
    seek_time = start + info.get("video_start_offset", 0.0)
    args = ["-n", "-ss", format(seek_time, ".12g"), "-protocol_whitelist", "file,pipe", "-i", str(source)]
    if duration is not None:
        args += ["-t", format(duration, ".12g")]
    args += [
        "-map", f"0:{info['video_stream_index']}", "-an", "-sn", "-dn",
        "-vf", f"fps={info['fps']},scale={width}:{height}:flags=lanczos,setsar=1",
        "-fps_mode", "cfr", "-start_number", "1", str(directory / "%08d.png"),
    ]
    run_ffmpeg(args)


def encode_video(
    source: Path, frames: Path, destination: Path, info: dict[str, Any], *,
    start: float, frame_count: int,
) -> None:
    seconds = float(Fraction(frame_count, 1) / Fraction(info["fps"]))
    args = ["-n", "-framerate", info["fps"], "-start_number", "1", "-protocol_whitelist", "file,pipe", "-i", str(frames / "%08d.png")]
    if info["has_audio"]:
        seek_time = start + info.get("video_start_offset", 0.0)
        args += ["-ss", format(seek_time, ".12g"), "-protocol_whitelist", "file,pipe", "-i", str(source)]
    args += ["-map", "0:v:0"]
    if info["has_audio"]:
        args += [
            "-map", f"1:{info['audio_stream_index']}", "-c:a", "aac", "-b:a", "192k",
            # Add silence for a late/short audio track; never shorten the video.
            "-af", "aresample=async=1:first_pts=0,apad",
        ]
    else:
        args += ["-an"]
    args += [
        "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
        "-t", format(seconds, ".12g"), "-map_metadata", "-1", "-map_metadata:s", "-1", "-map_chapters", "-1",
        "-metadata", "creation_time=", "-movflags", "+faststart", str(destination),
    ]
    run_ffmpeg(args)


def verify_output(path: Path, *, frame_count: int, width: int, height: int, has_audio: bool) -> None:
    probe = _probe(path, count_frames=True)
    video = next((s for s in probe.get("streams", []) if s.get("codec_type") == "video"), None)
    if not video or int(video.get("nb_read_frames", 0)) != frame_count:
        raise MediaError("Encoded output frame count did not match the completed frames.")
    if (video.get("width"), video.get("height")) != (width, height):
        raise MediaError("Encoded output dimensions did not match the completed frames.")
    audio = any(s.get("codec_type") == "audio" for s in probe.get("streams", []))
    if has_audio != audio:
        raise MediaError("Encoded output audio did not match the expected stream selection.")
