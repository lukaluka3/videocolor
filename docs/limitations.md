# What to expect

## Generated color is an interpretation

DDColor predicts plausible color from visual patterns. Black-and-white input
does not encode the original hue of an object. Different colors can have the
same brightness, so even a convincing result can be wrong. Faces, uniforms,
flags, skies, and small objects need careful review.

Use VideoColor to make a viewing copy, explore a creative treatment, or
prepare a draft for manual correction. Keep the source and label the result
as AI-colorized. Do not use generated colors as historical evidence.

The default color strength of `0.85` reduces predicted chroma for a more
restrained appearance. A lower setting can soften oversaturation; it cannot
make an incorrect hue historically accurate. Natural-looking output and
verified original color are different things.

Manual tint and warmth adjustments affect the entire frame. They can reduce
a visible cast but may make another region less plausible. Their defaults
are neutral; no automatic white balance or historical color inference is
claimed. The gamut protection keeps exact black/white and reduces chroma
that would otherwise clip and change brightness. It does not fix a model's
incorrect object classification or predicted hue.

The default natural grade removes one estimated cast from dark content and
reduces color near black, near white, and at very high saturation. It makes
typical archive footage look less artificial; it is a look, not a
restoration of true colors. A genuinely colored dark scene (for example
deep green foliage at night) can lose some of its color; compare with
`--no-natural-grade` when that matters.

## Motion and scene changes

The model colorizes individual frames. VideoColor's motion-based chroma
smoothing can reduce changes between neighboring frames, but it does not
turn the model into a temporally trained video model. Fast motion,
occlusions, film damage, flashes, fades, and uncertain scene boundaries can
still produce flicker, color trails, or sudden color changes. Try weaker
smoothing when moving objects leave colored trails.

This release does not offer reference-image guidance, hand-painted color
hints, model training, neural super-resolution, face reconstruction, or
frame interpolation. Increasing a video's pixel dimensions would not, by
itself, recover missing photographic detail.

## Sources and output

Start with a short, progressive, standard-dynamic-range recording. Heavily
compressed, scratched, very dark, tinted, or interlaced material may need
separate preparation. HDR, unusual rotations, multiple video tracks,
subtitles, and unusual frame timing require additional review; this is not
an archival preservation transcoder.

The exported MP4 is a newly encoded viewing copy. The first moving-picture
video stream and first audio stream are selected. Audio is re-encoded to
AAC, so it is not bit-for-bit identical to the source. Silent inputs stay
silent. Variable frame rates are normalized to a constant average rate, which
may duplicate or drop frames. Subtitles, chapters, other tracks, and source
metadata are not preserved. Keep the original container for archival use.

## Hardware, storage, and time

An NVIDIA GPU with a compatible PyTorch installation is recommended. CPU
processing is available but may be impractical for long recordings. There
is no universal processing-time estimate: model choice, frame count, source
size, device, and storage all matter.

Processing frames and retaining resumable work can require far more space
than the compressed input. As a conservative working-space budget, allow
about 1.7 GiB for a ten-second 720p preview at 30 fps, plus model files and
installed dependencies. Full recordings can need tens of GiB. Actual PNG
compression and frame dimensions affect usage; the estimate is not a fixed
output size. Check the job folder's growth after a preview.
Changing settings requires a separate output folder so that results from
different jobs are not mixed.

The first model download needs a network connection and additional disk
space. A fully populated model cache can be used locally without downloading
the model again. A local virtual environment and downloaded weights are not
part of a source release.

Colab sessions can disconnect or reset, and accelerator availability and
usage limits vary. Work kept only on a temporary Colab runtime can disappear.
Download useful outputs promptly or choose persistent storage deliberately.

## Validation boundaries

Automated tests exercise validation, image processing, and video assembly
with controlled inputs. They cannot prove subjective color quality or that
every codec, device, driver, and dependency combination works. Check the
[validation record](validation.md) for the environments actually tested, and
run a short preview on your own footage before a full job.

See the [disclaimer](../DISCLAIMER.md) and [usage guide](usage.md).
