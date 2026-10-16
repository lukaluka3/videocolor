"""Build a self-contained Colab notebook from audited public source (no media)."""
from __future__ import annotations

import base64
import hashlib
import io
import json
from pathlib import Path
import textwrap
import zipfile

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "VideoGaferadeba_Colab.ipynb"


def payload_bytes() -> bytes:
    files = [ROOT / name for name in ("pyproject.toml", "README.md", "LICENSE", "THIRD_PARTY_NOTICES.md", "DISCLAIMER.md")]
    files += [p for p in (ROOT / "src" / "videocolor").rglob("*") if p.is_file()
              and (p.suffix == ".py" or p.name == "NOTICE.md"
                   or (p.name.startswith("LICENSE-") and p.suffix == ".txt"))
              and "__pycache__" not in p.parts]
    memory = io.BytesIO()
    with zipfile.ZipFile(memory, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(files, key=lambda p: p.relative_to(ROOT).as_posix()):
            item = zipfile.ZipInfo(path.relative_to(ROOT).as_posix(), date_time=(2026, 1, 1, 0, 0, 0))
            item.compress_type = zipfile.ZIP_DEFLATED
            item.create_system = 3
            item.external_attr = 0o100644 << 16
            archive.writestr(item, path.read_text(encoding="utf-8").encode("utf-8"))
    return memory.getvalue()


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": textwrap.dedent(source).strip().splitlines(keepends=True)}


def code(source: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": textwrap.dedent(source).strip().splitlines(keepends=True)}


def build() -> dict:
    payload = payload_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    encoded = base64.b64encode(payload).decode("ascii")
    cells = [markdown('''
        # VideoColor · colorize black-and-white video

        **Try a short preview first.** DDColor predicts plausible colors; it cannot recover true historical colors.
        Moving objects, skin, uniforms, and scene changes may have incorrect or unstable colors.

        1. In Colab, choose **Runtime → Change runtime type → T4 GPU** if available.
        2. Run **Setup**, then choose/upload a video.
        3. Run a 10-second preview. On CPU, use **1–3 seconds** and a low output height first.
        4. Review the result before processing a longer clip.

        This notebook contains its own source package. No access to the author's account, private repository,
        or Drive link is required. Sign in to your own Google account to use Colab.
        Files uploaded here are processed in **your Google Colab runtime**. Local CLI/desktop processing stays on your computer.
        The first model download is about **220 MB (tiny)** or **912 MB (large)** from Hugging Face; your video is not sent there.

        Original software is MIT licensed; bundled DDColor code and selected weights have separate licenses described
        in the included third-party notices. The software is provided **AS IS**, without warranties. You are responsible
        for rights to your footage, consent, backups, and how you represent generated colors. See DISCLAIMER.md.
        Colab availability, limits, GPU access, and processing speed are not guaranteed.
    '''), markdown('''
        ## 1 · Setup
        Run this once per runtime. It verifies and installs the embedded source, preserving bundled license notices.
        No Google Drive mount is performed here. Setup needs internet for Python dependencies.
    '''), code(f'''
        import base64
        import hashlib
        import io
        from pathlib import Path
        import shutil
        import subprocess
        import sys
        import zipfile

        EMBEDDED_SOURCE_B64 = """{encoded}"""
        EXPECTED_SHA256 = "{digest}"
        package_zip = base64.b64decode(EMBEDDED_SOURCE_B64, validate=True)
        if hashlib.sha256(package_zip).hexdigest() != EXPECTED_SHA256:
            raise RuntimeError("Embedded source failed its integrity check. Download a fresh notebook.")
        source_dir = Path("/content") / ("videocolor-source-" + EXPECTED_SHA256[:16])
        source_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(package_zip)) as archive:
            for member in archive.infolist():
                target = (source_dir / member.filename).resolve()
                if not target.is_relative_to(source_dir.resolve()) or member.is_dir():
                    raise RuntimeError("Unexpected path in the embedded package.")
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.read(member))
        subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", str(source_dir)], check=True)
        for executable in ("ffmpeg", "ffprobe"):
            if not shutil.which(executable):
                raise RuntimeError("FFmpeg is missing from this runtime. Install FFmpeg before continuing.")
        subprocess.run([sys.executable, "-m", "videocolor", "doctor"], check=True)
        print("Ready. Source and license notices:", source_dir)
    '''), markdown('''
        ## 2 · Choose your video
        Upload a file, or explicitly enable Drive below and enter **your own** path.
        Mounting Drive lets this notebook access your Drive files. Mount only if you want to use it.
        Large files are usually easier to access through Drive. No account details are saved in the distributed notebook.
    '''), code('''
        # @title Video input
        use_drive = False # @param {type:"boolean"}
        drive_video_path = "" # @param {type:"string"}

        from google.colab import files
        from videocolor.media import inspect_video
        import json
        import os
        import uuid

        input_video = None
        output_video = None
        if use_drive:
            if not drive_video_path.strip():
                raise ValueError("Enter the path to your own Drive video first.")
            from google.colab import drive
            drive.mount("/content/drive")
            selected = Path(drive_video_path).expanduser().resolve()
            if not selected.is_file():
                raise ValueError("The selected Drive video does not exist.")
            input_video = selected
        else:
            upload_dir = Path("/content/videocolor-uploads") / uuid.uuid4().hex
            upload_dir.mkdir(parents=True)
            previous_directory = Path.cwd()
            try:
                os.chdir(upload_dir)
                uploaded = files.upload()
            finally:
                os.chdir(previous_directory)
            if len(uploaded) != 1:
                raise ValueError("Upload exactly one video, then rerun this cell.")
            selected_name = next(iter(uploaded))
            candidate = (upload_dir / selected_name).resolve()
            if not candidate.is_relative_to(upload_dir.resolve()):
                raise ValueError("Unexpected uploaded filename.")
            input_video = candidate
            del uploaded
        print(json.dumps(inspect_video(input_video), indent=2))
    '''), markdown('''
        ## 3 · Settings and processing
        Output defaults to at most 720p. The model analyzes each frame at 256×256, then transfers predicted colors
        to the output dimensions. A large output alone does not add model detail.

        Jobs keep source and colored PNG frames to support recovery. At 720p/30fps, a conservative raw estimate
        is **about 166 MB per second** for both sets before PNG compression, plus the model and encoded output.
        A long video can need tens of gigabytes. Use a short preview first.

        `resume` requires the same video, settings, and job folder. Give a **new job name** when changing settings.
        `/content` is temporary: Colab may delete it on disconnect/reset. For persistent jobs, first mount Drive and
        set `work_directory` to a folder you choose there. Drive processing can be slower.
    '''), code('''
        # @title Colorization settings
        model = "tiny" # @param ["tiny", "large"]
        device = "auto" # @param ["auto", "cuda", "cpu"]
        model_input_size = 256 # @param [256, 512] {type:"raw"}
        color_strength = 0.85 # @param {type:"slider", min:0, max:1, step:0.05}
        tint = 0.0 # @param {type:"slider", min:-20, max:20, step:1}
        warmth = 0.0 # @param {type:"slider", min:-20, max:20, step:1}
        natural_grade = True # @param {type:"boolean"}
        start_seconds = 0.0 # @param {type:"number"}
        preview_seconds = 10.0 # @param {type:"number"}
        full_video = False # @param {type:"boolean"}
        max_height = 720 # @param {type:"integer"}
        smoothing = 0.35 # @param {type:"slider", min:0, max:1, step:0.05}
        work_directory = "/content/videocolor-jobs" # @param {type:"string"}
        job_name = "preview" # @param {type:"string"}
        resume = False # @param {type:"boolean"}

        import re
        import time
        from videocolor.pipeline import process_video
        from IPython.display import clear_output

        output_video = None
        if input_video is None or not input_video.is_file():
            raise ValueError("Choose a video in the previous cell first.")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", job_name):
            raise ValueError("Job name must use 1–64 letters, numbers, underscores or hyphens.")
        if not work_directory.strip():
            raise ValueError("Choose a work directory.")
        job_directory = Path(work_directory).expanduser() / job_name
        started = time.monotonic()
        last_update = 0.0

        def progress(done, total):
            global last_update
            now = time.monotonic()
            if done in (0, total) or now - last_update > 2:
                clear_output(wait=True)
                print(f"Frames: {done}/{total} ({done / total:.0%}) · elapsed {now - started:.0f}s")
                print("Job:", job_directory)
                if done == total:
                    print("Encoding and checking video/audio…")
                last_update = now

        print("Preparing frames and model. First download can take several minutes…")
        output_video = process_video(
            input_video, job_directory, model=model, device=device, input_size=int(model_input_size),
            start=start_seconds, duration=None if full_video else preview_seconds,
            max_height=max_height, smoothing=smoothing, color_strength=color_strength,
            tint=tint, warmth=warmth, natural_grade=natural_grade, resume=resume, progress=progress,
        )
        print("Saved:", output_video)
    '''), markdown('''
        ## 4 · Review and download
        Inspect motion and scene changes; colors are estimates. The result keeps the first audio stream, re-encoded
        to AAC, if present. Variable frame rate is normalized to the source average rate. Subtitles, chapters, extra
        audio tracks, and source metadata are not included. Large videos are offered for download without embedding.
    '''), code('''
        from IPython.display import Video, display

        if output_video is None or not Path(output_video).is_file():
            raise RuntimeError("No completed video from the latest run. Finish processing first.")
        if Path(output_video).stat().st_size <= 25 * 1024 * 1024:
            display(Video(str(output_video), embed=True, width=720))
        else:
            print("Video is larger than 25 MB. Download it to review in your local player.")
    '''), code('''
        # @title Download the completed result
        if output_video is None or not Path(output_video).is_file():
            raise RuntimeError("No completed video from the latest run.")
        files.download(str(output_video))
    '''), markdown('''
        ## Troubleshooting
        - **CUDA unavailable:** select CPU, or select a GPU runtime and rerun setup. CPU can be slow.
        - **Out of memory:** choose `tiny`, model input 256, and a lower output height. Close other GPU jobs.
        - **Job already exists:** enable `resume` with identical input/settings, or use a new job name.
        - **Disk full:** choose a shorter interval/lower height or a larger work location. Keep original footage backed up.
        - **Download failed:** check connectivity and free space, then retry. Completed frames remain in the job folder.
        - **Color flicker or trails:** compare a short preview with smoothing 0 and 0.35. Some footage will remain difficult.
        - **Colors look too vivid:** reduce `color_strength` to 0.6–0.8. The default 0.85 gently reduces predicted saturation;
          1 uses the model's full color and 0 is grayscale. Less saturation cannot correct a wrongly predicted hue.
        - **Unwanted color cast:** compare a small manual `tint` adjustment (negative green, positive magenta) or
          `warmth` adjustment (negative blue, positive yellow). Both default to 0. These affect the whole image;
          review skin, clothing, and sky, and use a new job name for each comparison. They do not recover true colors.
        - **Shadows or highlights look gray:** `natural_grade` (on by default) keeps deep shadows and near-white
          areas neutral and softly limits neon colors, like real color film. Turn it off to see the model's raw colors.

        Before sharing your executed notebook, clear outputs and remove your own file paths/account details.
        Do not include private footage or downloaded model weights in a public repository.
    ''')]
    return {"nbformat": 4, "nbformat_minor": 5, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"}, "colab": {"provenance": []},
        "videocolor_payload_sha256": digest,
    }, "cells": [dict(cell, id=f"videocolor-{index:02d}") for index, cell in enumerate(cells)]}


if __name__ == "__main__":
    NOTEBOOK.write_text(json.dumps(build(), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Built {NOTEBOOK.name} from current public source.")
