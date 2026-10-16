# Validation record

This record describes checks actually performed during preparation of
VideoColor 0.1.0 and 0.2.0 on 2026-10-16. It is not a guarantee that every source,
machine, driver, or dependency version will behave identically.

## Local environment

| Component | Version / device |
| --- | --- |
| Operating system | Windows |
| Python | 3.12.9 |
| PyTorch | 2.11.0+cpu |
| FFmpeg | 8.1 |
| NumPy | 2.4.6 |
| OpenCV | 4.14.0 |
| huggingface-hub | 1.18.0 |
| Inference device | CPU; CUDA was unavailable |

The official pinned `tiny` checkpoint was downloaded, checked against its
expected SHA-256 digest, loaded with strict tensor-name matching, and used
for real neural inference on synthetic image input. Both 256 and 512 model
input sizes produced valid images. Cached loading with network access
disabled and repeatability for the same input were also checked.

These are execution checks. A synthetic image does not establish accurate
skin color, good performance on damaged film, or stable color in complex
motion. No representative visual-quality benchmark is claimed.

A 0.4-second interval from a local recording also passed the complete
decode, neural inference, smoothing, encode, and output-verification path.
Its output was a silent 256 × 144 MP4 at 30 fps with all 12 expected frames.
A second run processed a three-second interval at a 360-pixel output height:
90 frames, 638 × 360, 30 fps, no audio, and a verified three-second MP4.
The width reflects the input's non-square pixel aspect ratio. This run took
about 86 seconds on the tested CPU with the tiny model at input size 256.
That is a measured example, not a general speed promise.

Those recordings and outputs are private working material and are not
included in the public source release. These short checks do not establish
long-video performance.

## Application and publication checks

The desktop interface was created, its event loop exercised, and it was
closed in a local smoke check. Automated tests cover settings validation and
the image/video pipeline with controlled inputs. Separate release tests
cover private-data rejection, notebook output removal requirements, embedded
archive path checks, and refusal to overwrite an existing release archive.

The initial 0.1.0 local suite passed **124 tests**, and the Ruff source checks
passed. Video assembly tests use generated footage and explicit test
colorizers; the real neural checks above are separate. Source and wheel
builds also completed with all retained third-party license files included.

A second virtual environment was created without access to system Python
packages. It installed the built wheel and resolved PyTorch 2.14.1+cpu,
NumPy 2.5.4, OpenCV 4.14.0, and huggingface-hub 1.33.0. `pip check` and the
application's `doctor` command passed. With cached weights and offline mode,
it processed the three-second, 90-frame clip at model input 256 and color
strength 0.85 in about 73 seconds. It also processed a real frame at input
512. The result was visually reviewed as an example, not scored against
known original colors.

A further two-second video preview used input size 512 and color strength
0.8: all 60 frames were preserved at 638 × 360 and 30 fps. Processing took
about 150 seconds on the clean CPU environment. This comparison illustrates
the cost of the higher model resolution; it is not a promise of better
color on other footage.

In that clean environment, a hidden desktop window also completed an
actual three-frame job: settings entry, worker thread, model inference,
progress updates, encoding, and result-button enablement were exercised.

The source-release checker also decodes the notebook's embedded source,
scans its contents, checks its hash, and compares it with a fresh build from
the current source tree. Merely searching the visible notebook text would
not check the encoded contents.

## Version 0.2 color and loading checks

Mapped checkpoint loading was compared with the previous loader in separate
processes using the pinned tiny model at input size 512. The raw network
output was exactly equal, before any changes to the color-rendering step.
Immediately after loading, process working sets were approximately 282 MiB
with mapped assignment and 661 MiB with the previous loader. Later inference
peaks were not lower in that comparison; this is a loading improvement, not
a promise of lower peak RAM throughout a video job.

The new color renderer reduces chroma only where conversion would clip outside
sRGB, preserving the source's Lab lightness and predicted hue direction.
The protection runs before the first model-to-RGB conversion and after
temporal blending. Tests exercise extreme chroma at multiple brightness
levels, hue preservation, unchanged representable colors, exact black/white,
and resumed processing. They do not establish the true colors of any object.

An isolated real-frame test of the final rendering step reduced mean Lab
lightness error from about 0.125 to 0.069 units, with maximum error dropping
from about 1.715 to 0.336 units. That test reused an existing predicted frame;
it measures brightness preservation, not semantic color accuracy. The step
took about 20 ms at 638 x 360 and 87 ms at 1280 x 720 in a local CPU sample;
images with many out-of-gamut pixels can cost substantially more.

The official pinned large checkpoint was then downloaded, size/hash verified,
strictly loaded with mapped assignment, and exercised offline on the same
real frame at input sizes 256 and 512. Peak process working sets in these
separate frame checks were about 1.5 GiB and 2.1 GiB respectively. In this
example 512 produced a stronger purple cast; higher resolution was not
automatically better. Cropping the black side borders only for model input
was also compared, without a consistent visual benefit; automatic cropping
was not added to the application.

The current pipeline processed the same two-second, 60-frame interval twice:
tiny/512 with color strength 0.8, and large/256 with strength 0.85, tint -6,
warmth +3, both at smoothing 0.35 and 638 x 360 output. The large run completed
in about 99 seconds on the tested CPU, including decoding and encoding.
The latter looked less orange/magenta in a local visual comparison, but
clothing still changed hue during movement. These manual settings are an
example, not a general recommendation or proof of historical colors.

The complete version 0.2 suite passed **152 tests**, with Ruff checks and the
54-file source privacy/notebook audit passing. Balance tests cover finite
range validation, its direction, grayscale at zero strength, non-accumulation,
and byte-identical resumed frames. Older completed jobs and interrupted
encoding remain usable; partial older color renders cannot mix revisions.

A hidden desktop smoke check entered tint/warmth in the actual controls,
validated them, passed them through the worker thread, and reached the
completed state with Open result enabled. That UI check used a stub process;
the separate command-line video checks above used real neural inference.

## Not established by these checks

- Real CUDA or Google Colab execution and performance.
- Apple Silicon or Linux desktop behavior.
- Quality on a representative collection of faces, archival film, camera
  motion, scene changes, and damaged recordings.
- Long-running jobs across every supported input format.

Run a short preview of your own source and inspect color, motion, audio, and
timing before committing to a long job. Contributions that expand this
record should identify the exact environment, model revision, source rights,
and settings used.

The CI workflow is configured for Python 3.11 and 3.12 on Windows and Ubuntu.
A configured workflow is not evidence that hosted CI has already passed;
review its run results after publication.
