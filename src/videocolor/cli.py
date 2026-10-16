"""Command-line interface; heavy dependencies load only when needed."""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import re
import shutil
import subprocess
import sys
import time

from . import __version__


def doctor() -> int:
    """Check dependencies without contacting model hosts or uploading files."""
    print(f"VideoColor {__version__} | Python {sys.version.split()[0]}")
    okay = True
    for name in ("ffmpeg", "ffprobe"):
        executable = shutil.which(name)
        if executable:
            try:
                result = subprocess.run([executable, "-version"], capture_output=True, text=True,
                                        encoding="utf-8", errors="replace", timeout=15)
                lines = result.stdout.splitlines()
                print(lines[0] if result.returncode == 0 and lines else f"FAIL: {name} cannot run")
                okay &= result.returncode == 0 and bool(lines)
                if result.returncode == 0 and lines:
                    version = re.search(rf"^{name} version n?(\d+)(?:[.\s-]|$)", lines[0])
                    if version and int(version.group(1)) < 6:
                        print(f"FAIL: {name} 6 or newer is required. Update FFmpeg and reopen this terminal.")
                        okay = False
                if name == "ffmpeg":
                    codecs = subprocess.run([executable, "-hide_banner", "-encoders"], capture_output=True,
                                            text=True, encoding="utf-8", errors="replace", timeout=15)
                    encoder_ok = codecs.returncode == 0 and "libx264" in codecs.stdout and " aac " in codecs.stdout
                    print("H.264/AAC encoders: " + ("available" if encoder_ok else "MISSING"))
                    okay &= encoder_ok
            except (OSError, subprocess.TimeoutExpired) as exc:
                print(f"FAIL: {name}: {exc}")
                okay = False
        else:
            print(f"MISSING: {name}. Install FFmpeg and reopen this terminal.")
            okay = False
    for name in ("numpy", "cv2", "huggingface_hub", "torch"):
        try:
            module = __import__(name)
            print(f"{name}: {getattr(module, '__version__', 'installed')}")
            if name == "torch":
                version = tuple(int(part) for part in module.__version__.split("+")[0].split(".")[:2])
                if version < (2, 6):
                    print("FAIL: PyTorch 2.6 or newer is required for safe model loading.")
                    okay = False
                cuda = module.cuda.is_available()
                print("Device: " + (module.cuda.get_device_name(0) if cuda else "CPU (start with a 1-second preview)"))
        except (ImportError, OSError, RuntimeError) as exc:
            print(f"FAIL: {name}: {exc}")
            okay = False
    print("Desktop GUI: " + ("tkinter installed" if importlib.util.find_spec("tkinter") else "tkinter missing (optional)"))
    print("Model weights are downloaded on first use; doctor does not download them.")
    return 0 if okay else 1


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Colorize black-and-white video locally. AI colors are estimates.")
    result.add_argument("--version", action="version", version=f"VideoColor {__version__}")
    commands = result.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="Check FFmpeg, Python dependencies, and device")
    commands.add_parser("gui", help="Open the desktop interface")
    info = commands.add_parser("info", help="Inspect a video without downloading a model")
    info.add_argument("input", help="Input video path")
    process = commands.add_parser("process", help="Create a short preview or process a full video")
    process.add_argument("input", help="Input video path")
    process.add_argument("--output", required=True, help="New/empty job directory, or matching job with --resume")
    process.add_argument("--model", choices=("tiny", "large"), default="tiny")
    process.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    process.add_argument("--input-size", type=int, choices=(256, 512), default=256)
    process.add_argument("--start", type=float, default=0.0, help="Start time in seconds")
    duration = process.add_mutually_exclusive_group()
    duration.add_argument("--duration", type=float, default=10.0, help="Preview duration in seconds (default: 10)")
    duration.add_argument("--full-video", action="store_true", help="Process from start time to end")
    size = process.add_mutually_exclusive_group()
    size.add_argument("--max-height", type=int, default=720, help="Output height limit; never upscale (default: 720)")
    size.add_argument("--no-resize", action="store_true", help="Preserve source size, rounded to even dimensions")
    process.add_argument("--smoothing", type=float, default=0.35, help="Motion-aware chroma smoothing, 0 to 1")
    process.add_argument("--color-strength", type=float, default=0.85,
                         help="Color intensity, 0 to 1 (default: 0.85; lower values reduce oversaturation)")
    process.add_argument("--resume", action="store_true", help="Continue a job with identical input/settings")
    process.add_argument("--tint", type=float, default=0.0,
                         help="Manual color balance, -20 to 20: negative green, positive magenta (default: 0)")
    process.add_argument("--warmth", type=float, default=0.0,
                         help="Manual color balance, -20 to 20: negative blue, positive yellow (default: 0)")
    process.add_argument("--no-natural-grade", action="store_true",
                         help="Turn off the film-like grade (neutral shadows, gentle highlights, no neon colors)")
    process.add_argument("--cache-dir", help="Optional model cache directory")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        if args.command == "doctor":
            return doctor()
        if args.command == "gui":
            from .gui import launch
            launch()
            return 0
        if args.command == "info":
            from .media import inspect_video
            print(json.dumps(inspect_video(args.input), indent=2))
            return 0
        from .pipeline import process_video
        print("AI colors are estimates. Review a short preview before processing a full video.", flush=True)
        print("Reading/extracting video. The first run also downloads model weights.", flush=True)
        started = time.monotonic()
        last_update = 0.0

        def progress(done: int, total: int) -> None:
            nonlocal last_update
            now = time.monotonic()
            if done in (0, total) or now - last_update >= 1:
                print(f"Frames: {done}/{total} ({done / total:.0%}) | elapsed {now - started:.0f}s", flush=True)
                last_update = now

        output = process_video(
            args.input, args.output, model=args.model, device=args.device, input_size=args.input_size,
            start=args.start, duration=None if args.full_video else args.duration,
            max_height=None if args.no_resize else args.max_height, smoothing=args.smoothing,
            color_strength=args.color_strength,
            tint=args.tint, warmth=args.warmth, natural_grade=not args.no_natural_grade,
            resume=args.resume, cache_dir=args.cache_dir, progress=progress,
        )
        print(f"Saved: {output}")
        return 0
    except (KeyboardInterrupt, InterruptedError):
        print("\nStopped. Completed frames are saved; use the same settings with --resume.", file=sys.stderr)
        return 130
    except (ValueError, OSError, RuntimeError, ImportError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
