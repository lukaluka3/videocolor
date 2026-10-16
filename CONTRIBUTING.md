# Contributing to VideoColor

Useful contributions make the path from a user's own video to a usable
colorized result easier to understand and more reliable.

## Report a problem

Include your operating system, Python version, VideoColor version, the
command or notebook step, and the complete error message. Run
`python -m videocolor doctor` and include its output after removing personal
paths. For video-specific failures, include the output of
`python -m videocolor info "clip.mp4"` after the same review.

A short recording that reproduces the issue is helpful only if you can
publicly share it. Do not upload family recordings, copyrighted footage,
credentials, entire model caches, or private notebook outputs. A synthetic
clip is preferable when it reproduces the failure.

## Work on the code

Use a Python virtual environment. From the repository root:

```console
python -m pip install -e ".[dev]"
python -m pytest
```

Install FFmpeg and ffprobe for media integration tests. The automated test
suite should not require downloading pretrained models. Tests with a stub
colorizer check the video pipeline; they do not demonstrate neural model
quality or GPU compatibility.

For a change to frame ordering, timing, resume, or audio, include a small
synthetic regression test. For a model or preprocessing change, record the
model revision, dependency versions, device, and preview settings. Explain
what you tested and what you could not test.

Keep changes focused and update the relevant documentation. Avoid adding
mandatory cloud accounts, silently replacing a user's files, or downloading
unreviewed executable code at runtime. Preserve third-party notices when
modifying model code. New dependencies need a clear purpose and compatible
licensing.

## Share visual results responsibly

Describe the source rights and whether the example is synthetic or archival.
Use the same source interval for before/after comparisons. Do not present
plausible generated colors as verified original colors, or conventional
resizing as recovery of missing detail.

Contributions to original VideoColor code are made under the project's MIT
license. Changes to third-party files retain their existing licenses. Submit
only material that you have permission to contribute.
