# Using VideoColor

Start with the installation instructions in the [README](../README.md).
Run every Python command with the interpreter from the environment where
you installed VideoColor.

## Check the installation

```console
python -m videocolor doctor
python -m videocolor info "clip.mp4"
```

`doctor` reports the tools and compute support available in your environment.
`info` reads video information without running the colorization model. A
working installation still needs to download the selected weights before its
first colorization job.

For a desktop workflow, run `python -m videocolor gui`, select a local video,
and choose a new output folder. The preview controls let you try settings
before committing to the full recording.

## Preview the part that matters

The default command processes a ten-second preview. Include a face, moving
objects, and a scene cut where possible: one still frame does not reveal
temporal color problems.

```console
python -m videocolor process "clip.mp4" --output outputs/preview
```

To inspect a different interval, use seconds:

```console
python -m videocolor process "clip.mp4" --output outputs/preview-30s --start 30 --duration 5
```

Open `result.mp4` inside the selected output folder. A new output folder keeps
each experiment separate. The job also retains working artifacts to support
resuming; do not share that directory without reviewing its contents.

## Choose a model and device

| Option | Purpose |
| --- | --- |
| `--model tiny` | Smaller DDColor model; a useful first trial, especially on CPU. |
| `--model large` | Larger DDColor ModelScope model; more memory and download space. Compare a preview before choosing it. |
| `--device auto` | Use CUDA when available, otherwise CPU. |
| `--device cuda` | Require a compatible NVIDIA GPU and PyTorch installation. |
| `--device cpu` | Use CPU explicitly; long recordings may take impractically long. |

The larger model is not guaranteed to look better on every source. Use
separate output folders to compare models on the same interval.

```console
python -m videocolor process "clip.mp4" --output outputs/large-trial --model large --device cuda --duration 3
```

## Control size and color smoothing

`--color-strength` controls how much predicted chroma appears in the output.
The default `0.85` gives a more restrained treatment than the model's full
color prediction. Use values from `0` (grayscale) to `1` (full predicted
color). For example:

```console
python -m videocolor process "clip.mp4" --output outputs/soft-color --duration 5 --color-strength 0.65
```

Lower strength can make exaggerated colors less distracting, but it cannot
correct a wrongly predicted hue or establish an object's historical color.
Inspect faces, clothing, and backgrounds on your own preview. Use a new job
folder when changing color strength.

The default output-height cap is 720 pixels. `--max-height` limits output
size; it is not AI upscaling. Use a smaller cap to reduce downstream frame
storage and processing costs. `--no-resize` disables that cap.

```console
python -m videocolor process "clip.mp4" --output outputs/small-trial --duration 2 --max-height 240
```

`--smoothing` controls how strongly motion-aligned color from the preceding
frame influences the current one. The default is `0.35`. Setting it to `0`
disables this blending. Stronger smoothing may reduce flicker but can leave
colored trails when motion estimation is wrong. Scene-change detection
attempts to avoid transferring color across cuts, but cannot detect every
transition perfectly.

`--input-size 256` is the default neural input size. `--input-size 512` can
change color quality at a higher compute cost; it does not set the exported
video dimensions. Compare both on a short interval before choosing 512.

`--tint` and `--warmth` provide manual whole-image Lab color balance, both
defaulting to `0` and limited to `-20` through `20`. Negative tint adds green;
positive tint adds magenta. Negative warmth adds blue; positive warmth adds
yellow. Start with changes of 2–4 and compare a short interval. They do not
detect a neutral object or recover historical colors automatically. Review
skin, sky, and clothing together before applying a correction to a full clip.
Offsets affect fresh predictions once, before color strength and smoothing;
they do not accumulate on every frame. Strength `0` remains grayscale.

The natural grade is on by default. It gives the colors a film-like response:
a uniform green/teal cast in dark picture content is removed (near-black
letterbox bars are ignored when measuring it), deep shadows fade to neutral
black, near-white areas keep only a little color, and very saturated colors
are softly limited. Color edges are slightly softened; detail and brightness
always come from the source. Add `--no-natural-grade` to see the model's
unmodified colors, for example when a scene is meant to be strongly colored.

```console
python -m videocolor process "clip.mp4" --output outputs/raw-colors --duration 2 --no-natural-grade
```

```console
python -m videocolor process "clip.mp4" --output outputs/balance-trial --duration 2 --model large --tint -6 --warmth 3
```

```console
python -m videocolor process "clip.mp4" --output outputs/no-smoothing --duration 5 --smoothing 0
```

## Process the complete recording

After inspecting your preview:

```console
python -m videocolor process "clip.mp4" --output outputs/full --model tiny --full-video
```

The first moving-picture video stream is processed, along with the first
audio stream when present. Audio is encoded as AAC. Variable frame rates are
normalized to a constant average rate. Subtitles, chapters, and original
metadata are not included. Keep the source for archival use.

## Resume an interrupted job

Repeat the exact original command with `--resume`:

```console
python -m videocolor process "clip.mp4" --output outputs/full --model tiny --full-video --resume
```

Resume needs the same input, processing settings, and intact work files.
Changing the model, time interval, size, color strength, tint, warmth, natural grade, or smoothing means starting a new
job in another output folder. Resume cannot restore data lost when a
temporary Colab runtime is reset. Keep enough free space for the intermediate
frames while the job is in progress.

Version 0.2 changes the color-rendering step. Completed older videos can
still be opened and verified, but an unfinished older job cannot mix its
existing frames with the new renderer. Finish it with the original version,
or use a new job folder to render the entire interval with this version.

## Cache and offline use

The selected model is downloaded from its official Hugging Face repository
at a pinned revision. VideoColor verifies the expected file hash before
loading it. Model files are stored in the Hugging Face cache unless a
different location is selected:

```console
python -m videocolor process "clip.mp4" --output outputs/cached-preview --cache-dir model-cache
```

Use the same cache location for later jobs. Complete installation and a
successful model download before disconnecting from the internet. Do not
publish this cache with the project's source, or replace its contents with
untrusted checkpoints.

To explicitly disable Hugging Face network requests after the selected model
has been cached, set `HF_HUB_OFFLINE=1` before starting Python. In PowerShell:

```powershell
$env:HF_HUB_OFFLINE = "1"
.\.venv\Scripts\python.exe -m videocolor process "clip.mp4" --output outputs/offline-preview
```

In a Linux or macOS shell with the environment activated:

```sh
HF_HUB_OFFLINE=1 python -m videocolor process "clip.mp4" --output outputs/offline-preview
```

Use the same `--cache-dir` if you selected a custom cache previously. A model
that has not been downloaded will fail to load in offline mode.

## Troubleshooting

| Symptom | What to try |
| --- | --- |
| `ffmpeg` or `ffprobe` is missing | Install both tools, add their directory to `PATH`, reopen the terminal, and rerun `doctor`. |
| CUDA is unavailable | Check the driver and install a compatible GPU build using PyTorch's official selector. Use `--device cpu` for a small trial. |
| GPU runs out of memory | Close other GPU applications, try `--model tiny`, and start a short preview in a new output folder. |
| CPU processing is too slow | Test a one-second clip with `tiny`; use a compatible GPU for longer work. A smaller output cap reduces frame-handling costs but does not proportionally shrink the model's neural workload. |
| First run appears busy before frames are processed | The model may be downloading or initializing. Check the download/progress output and connectivity. |
| Download fails or hash verification fails | Check network and disk space, then retry with a fresh dedicated cache directory. Do not bypass verification or substitute an unknown weight file. |
| Output colors flicker or smear | Preview weaker smoothing. Try the other model on the same interval. Some footage will need manual correction outside VideoColor. |
| Resume rejects a job | Reuse the original input and settings or choose a fresh output folder. Do not mix files from different jobs. |
| GUI cannot import Tk | Use the command line or install Tk for the Python interpreter you are using. On some Linux distributions, the package is named `python3-tk`. |
| Colab session disconnects | Reconnect and resume only if your input and work folder survived. Temporary runtime storage can be lost entirely. |

For a bug report, follow [Contributing](../CONTRIBUTING.md). Remove personal
paths from logs and use footage you are allowed to share.
