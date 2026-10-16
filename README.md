# VideoColor

**Turn a black-and-white video into an AI-colorized viewing copy. Preview a
short clip, compare the result, and keep the original audio.**

VideoColor wraps the [DDColor research model](https://github.com/piddnad/DDColor)
in a practical video workflow for family recordings, archival footage, and
creative projects. Use a desktop window or command line locally, or upload
the included notebook to Google Colab.

AI invents plausible colors; it cannot identify the true original colors.
Always keep your source and label shared results as AI-colorized.

## What you get

- A short preview by default, so you can judge the result before a full job.
- Two pretrained models: a smaller `tiny` model and a larger `large` model.
- Motion-based color smoothing with scene-change detection to reduce flicker.
- Adjustable color strength, with a restrained default to reduce oversaturation.
- Manual tint and warmth controls for comparing and correcting a whole-image cast.
- Protection for black, white, and colors that would clip and alter brightness.
- An MP4 viewing copy with audio retained through AAC encoding.
- Resumable work when a job is interrupted and its files are still available.
- Local processing after installation and the first model download.

This is an early release, with [documented limitations](docs/limitations.md)
and a [record of completed validation](docs/validation.md).
Color flicker and incorrect colors remain possible. Neural upscaling, face
reconstruction, and manual color guidance are not included.

## Start in Google Colab

This is the simplest starting point if you do not already have Python and a
compatible GPU set up locally.

1. Download [VideoGaferadeba_Colab.ipynb](VideoGaferadeba_Colab.ipynb) from this
   repository. On GitHub, use the notebook's **Download raw file** action.
2. Open [Google Colab](https://colab.research.google.com/) and choose
   **File > Upload notebook**.
3. Under **Runtime > Change runtime type**, select a GPU if one is available.
4. Run the notebook cells in order. Upload a short video that you have the
   right to process, then generate a preview.
5. Inspect the result before choosing a full-video job. Download your output
   before leaving the runtime.

The notebook includes the application source. You do not need a separate
repository URL or access to anyone else's Drive. It installs dependencies
and downloads the selected model on its first run. Colab runs on Google's
infrastructure; GPU access, session length, and quotas are not guaranteed.
Keep private recordings local if you do not want to upload them.

## Run on your computer

Use Python 3.11 or 3.12, FFmpeg 6 or newer with ffprobe on your `PATH`, and enough space
for the model cache and working frames. An NVIDIA GPU is recommended;
CPU processing can be much slower. The model downloads are approximately
220 MB (`tiny`) or 912 MB (`large`), in addition to Python dependencies.

As a conservative working-space budget, allow about 1.7 GiB for a ten-second
720p preview at 30 fps, plus the model and installed dependencies. Full
recordings can need tens of GiB because resumable jobs retain frame files.

Download the source ZIP and extract it. Open a terminal in the extracted
folder. Install [Python](https://www.python.org/downloads/) and an FFmpeg build
from the options on [FFmpeg's download page](https://ffmpeg.org/download.html).
Both `ffmpeg -version` and `ffprobe -version` should work in a new terminal.

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m videocolor doctor
.\.venv\Scripts\python.exe -m videocolor gui
```

The commands use the virtual environment directly, so PowerShell script
activation is not required. Choose your video in the window, start with a
preview, and open the exported result when processing finishes.

For an NVIDIA GPU, install a matching PyTorch build using the
[official PyTorch installation selector](https://pytorch.org/get-started/locally/)
in that same virtual environment. Run `doctor` again to check whether CUDA is
available. Installing the application alone does not guarantee GPU support.

### Linux / Apple Silicon macOS

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m videocolor doctor
python -m videocolor gui
```

The desktop window requires Tk; some Linux distributions package it
separately as `python3-tk`. The command line works without Tk. This release
supports CPU and NVIDIA CUDA execution; Apple Metal acceleration is not
implemented. Intel macOS is not supported by this installation path because
the required PyTorch versions no longer provide official Intel Mac binaries;
see [PyTorch's announcement](https://dev-discuss.pytorch.org/t/pytorch-macos-x86-builds-deprecation-starting-january-2024/1690).

## Process your first clip

In the commands below, `python` means the Python interpreter from your
virtual environment. On Windows, use `.\.venv\Scripts\python.exe` if you
have not activated it.

Inspect a video:

```console
python -m videocolor info "clip.mp4"
```

Make a ten-second preview:

```console
python -m videocolor process "clip.mp4" --output outputs/preview
```

Open `outputs/preview/result.mp4`. For a quicker CPU trial, begin with a
smaller, one-second preview:

```console
python -m videocolor process "clip.mp4" --output outputs/cpu-trial --model tiny --device cpu --duration 1 --max-height 240
```

Once you are happy with a preview, process the whole recording into a
different folder:

```console
python -m videocolor process "clip.mp4" --output outputs/full --full-video
```

For a more restrained color treatment, compare a short preview at model
resolution 512 and color strength 0.8. Resolution 512 costs substantially
more compute and can change predicted hues; it is not always better:

```console
python -m videocolor process "clip.mp4" --output outputs/color-comparison --duration 2 --input-size 512 --color-strength 0.8
```

The desktop window and notebook expose the same controls. Review faces,
clothing, sky, motion, and scene changes. Reducing saturation can soften
excessive color, but cannot identify the real hue of an object.

If interrupted, repeat the same command and append `--resume`. Keep the input
and job settings unchanged. For all options, run
`python -m videocolor process --help` or read the [usage guide](docs/usage.md).

The default color strength is `0.85`. For a more restrained preview, add
`--color-strength 0.65`; `1` uses the full predicted color and `0` gives
grayscale. This adjusts saturation, not the historical accuracy of the colors.

For an unwanted magenta or blue cast, try small manual balance changes on a
short preview. `--tint` moves green (negative) to magenta (positive);
`--warmth` moves blue (negative) to yellow (positive). Both default to `0`
and accept `-20` to `20`. This example was useful on one local clip; it is
not a universal preset:

```console
python -m videocolor process "clip.mp4" --output outputs/balance-trial --duration 2 --model large --input-size 256 --tint -6 --warmth 3
```

Review skin, sky, and clothing together: a global adjustment can help one
area and harm another. These controls are also in the desktop window and
notebook. Use a new job folder when changing them.

## How it works

FFmpeg decodes the video. DDColor predicts color for each frame while keeping
the source's luminance information. Motion-based smoothing blends neighboring
chroma estimates and resets on detected cuts. FFmpeg encodes the processed
frames and source audio into an MP4.

When a predicted color cannot fit in the display color range at the source
brightness, VideoColor reduces that pixel's chroma along the same Lab hue
direction. Exact black and white stay unchanged. This avoids some colored
shadow artifacts; it cannot correct an incorrectly predicted object color.

The smoothing step reduces some instability; it does not guarantee consistent
color throughout a scene. A preview is the best way to judge your recording.

## Contributing and publishing

[Contributing](CONTRIBUTING.md) explains how to report problems and run tests.
[Publishing](docs/publishing.md) explains how to build a source release that
excludes videos, working frames, model caches, and private notebook outputs.
Do not upload your entire working directory.

## License and credits

Original VideoColor code is [MIT-licensed](LICENSE). DDColor's implementation
and the selected official model cards use Apache-2.0; retained upstream files
include their own notices. Model weights and FFmpeg binaries are downloaded
or installed separately. See [third-party notices](THIRD_PARTY_NOTICES.md).

Read the [disclaimer](DISCLAIMER.md) before use. The software is provided
without warranty. You remain responsible for the rights to your video and
audio and for reviewing generated results.
