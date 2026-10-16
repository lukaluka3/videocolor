# Third-party notices

VideoColor's original application code and documentation use the
[MIT License](LICENSE). That license does not replace the licenses for model
implementations, pretrained weights, dependencies, or input media.

## DDColor implementation and pretrained models

DDColor is the colorization research and model architecture by Xiaoyang Kang,
Tao Yang, Wenqi Ouyang, Peiran Ren, Lingzhi Li, and Xuansong Xie. VideoColor
does not claim to have trained DDColor.

- [Official DDColor source](https://github.com/piddnad/DDColor).
- [Code license: Apache-2.0](https://github.com/piddnad/DDColor/blob/master/LICENSE).
- [Official model selection notes](https://github.com/piddnad/DDColor/blob/master/MODEL_ZOO.md).
- [Tiny model card: Apache-2.0](https://huggingface.co/piddnad/ddcolor_paper_tiny/blob/main/README.md).
- [ModelScope model card: Apache-2.0](https://huggingface.co/piddnad/ddcolor_modelscope/blob/main/README.md).

The code license and model-card declarations above were checked separately.
The bundled inference subset derives from DDColor revision
`2adb63f2656ac41cbdf7b894cddd94121a3faf13`; it adapts imports and PyTorch
helpers for a standalone inference runtime. See the retained license texts
and change notices in [`src/videocolor/_vendor/ddcolor`](src/videocolor/_vendor/ddcolor).

| CLI model | Upstream model | Pinned revision |
| --- | --- | --- |
| `tiny` | `piddnad/ddcolor_paper_tiny` | `cf9fd99c1d7472689ec7413441c1b799a51866a3` |
| `large` | `piddnad/ddcolor_modelscope` | `060f67494e31883a4b13cb27f889f3154847ada4` |

Model weights are downloaded from the upstream publisher and are not
included in VideoColor's source release. Preserve the applicable notices
and license when redistributing upstream code or weights. The model license
does not establish rights in arbitrary input footage or guarantee that
generated color is accurate.

The upstream project acknowledges code from BasicSR, ColorFormer, BigColor,
ConvNeXt, Mask2Former, and DETR. The retained U-Net helpers also share code
ancestry with FastAI v1 and DeOldify; their Apache-2.0 and MIT license texts
are included alongside the other notices. Retained upstream files remain
governed by their respective terms.

Research citation:

```bibtex
@inproceedings{kang2023ddcolor,
  title={DDColor: Towards Photo-Realistic Image Colorization via Dual Decoders},
  author={Kang, Xiaoyang and Yang, Tao and Ouyang, Wenqi and Ren, Peiran and Li, Lingzhi and Xie, Xuansong},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={328--338},
  year={2023}
}
```

## FFmpeg and ffprobe

[FFmpeg](https://ffmpeg.org/) supplies the external media tools used for
decoding and encoding. This source release does not include FFmpeg binaries.
Install a build appropriate for your system. FFmpeg's license depends on its
build configuration: optional GPL components affect the resulting binary's
terms. See [FFmpeg's licensing information](https://ffmpeg.org/legal.html).

If you distribute a packaged executable or container containing FFmpeg,
review the actual build and fulfill its distribution requirements. A link
to FFmpeg or this application's MIT license is not a substitute for those
requirements.

## Python dependencies

Python packages installed for VideoColor retain their own licenses and
notices. Dependency declarations are in `pyproject.toml`; the licenses of
the actual installed versions are available in their package metadata and
upstream repositories. They are not relicensed by VideoColor.

## Features from the earlier prototype

The earlier personal notebook experimented with Real-ESRGAN and GFPGAN.
They are not included in this release's processing pipeline. Their code
licenses, checked for reference, are
[BSD-3-Clause for Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN/blob/master/LICENSE)
and [Apache-2.0 with additional notices for GFPGAN](https://github.com/TencentARC/GFPGAN/blob/master/LICENSE).
Adding them later requires reviewing the exact code, weights, dependencies,
and redistribution terms used in that integration.
