# Changelog

## 0.2.0 — Unreleased

- Add neutral-by-default manual tint and warmth controls to the command line,
  desktop window, and notebook, with saved settings and resume validation.
- Preserve exact source black and white. Compress only colors that would
  clip outside the display gamut, keeping source lightness and predicted
  Lab hue direction instead of introducing colored shadows.
- Load verified checkpoints with mapped tensors and metadata-only model
  construction, avoiding duplicate initialized model weights during loading.
- Record the color-processing revision. An interrupted job from an older
  renderer must be finished with that version or restarted in a new folder;
  completed older results remain accessible without changing their video.

## 0.1.0 — Initial source snapshot

First public-source release of VideoColor, focused on processing a user's
own black-and-white recording with a short preview before a full job.

- Local command-line workflow and a desktop file-selection window.
- A self-contained English Colab notebook.
- DDColor tiny and large model choices with pinned weight revisions and
  integrity checks.
- Motion-based chroma smoothing and scene-change resets.
- Adjustable color strength, defaulting to restrained `0.85` chroma.
- MP4 export with AAC audio, preview intervals, and an output-height cap.
- Resumable processing with input and setting checks.
- English installation, usage, limitations, contribution, and publishing
  documentation; MIT license for original code and separate upstream notices.
- Source packaging designed to exclude private videos, notebook outputs,
  cached models, working frames, and local environments.

The earlier personal notebook's neural upscaling, face reconstruction, and
frame interpolation experiments are not part of this release. Generated
colors remain approximate and can vary between frames. See
[limitations](docs/limitations.md) before choosing a production workflow.
