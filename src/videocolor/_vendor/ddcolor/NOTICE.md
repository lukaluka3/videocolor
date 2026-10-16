# Vendored DDColor inference architecture

These five inference files originate from the official
[piddnad/DDColor](https://github.com/piddnad/DDColor) repository, commit
`2adb63f2656ac41cbdf7b894cddd94121a3faf13`:

| Local file | Original path |
| --- | --- |
| `model.py` | `ddcolor/model.py` |
| `convnext.py` | `basicsr/archs/ddcolor_arch_utils/convnext.py` |
| `unet.py` | `basicsr/archs/ddcolor_arch_utils/unet.py` |
| `transformer_utils.py` | `basicsr/archs/ddcolor_arch_utils/transformer_utils.py` |
| `position_encoding.py` | `basicsr/archs/ddcolor_arch_utils/position_encoding.py` |

DDColor is by Xiaoyang Kang, Tao Yang, Wenqi Ouyang, Peiran Ren, Lingzhi Li,
and Xuansong Xie (DAMO Academy, Alibaba Group). The DDColor project distributes
these files under Apache License 2.0, retained as `LICENSE-DDColor.txt`.

Modifications by VideoColor contributors:

- Added provenance comments and changed architecture imports to local relative
  imports; the BasicSR training framework is not installed or imported.
- Replaced ConvNeXt's optional `timm` import/fallback with PyTorch's built-in
  truncated-normal initializer and a standalone stochastic-depth helper. No
  `timm` or `torchvision` dependency is needed.
- No model architecture or checkpoint tensor names were changed.
- Construct the small ConvNeXt stochastic-depth schedule explicitly on CPU so
  the network can initialize on PyTorch's metadata-only device. Checkpoint
  tensors can then be assigned directly without a second parameter allocation.

The existing upstream copyright comments are retained. This source includes
components adapted upstream from the following projects; their licenses are
also retained here:

- [ConvNeXt](https://github.com/facebookresearch/ConvNeXt), Copyright (c) Meta
  Platforms, Inc. and affiliates; MIT (`LICENSE-ConvNeXt.txt`). License retrieved
  at commit `048efcea897d999aed302f2639b6270aedf8d4c8`.
- [Mask2Former](https://github.com/facebookresearch/Mask2Former), Copyright (c)
  2022 Meta, Inc.; MIT (`LICENSE-Mask2Former.txt`). License retrieved at commit
  `9b0651c6c1d5b3af2e6da0589b719c514ec0d69a`.
- [DETR](https://github.com/facebookresearch/detr), Copyright (c) Facebook, Inc.
  and its affiliates; Apache 2.0 (`LICENSE-DETR.txt`). License retrieved at commit
  `29901c51d7fe8712168b8d0d64351170bc0f83e0`.
- The `unet.py` helpers share source ancestry with [FastAI v1](https://github.com/fastai/fastai1)
  (Apache 2.0; `LICENSE-FastAI.txt`, retrieved at commit
  `15bc02c61673400178fbdaf3417485755c299561`) and
  [DeOldify](https://github.com/jantic/DeOldify) (MIT, Copyright (c) 2018 Jason
  Antic; `LICENSE-DeOldify.txt`, retrieved at commit
  `5f86c28923799036ad4b1bf7af8c629ac65efd75`). Their notices are retained for
  the shared normalization, convolution, pixel-shuffle, and UNet helpers. No
  DeOldify model or weights are used.

DDColor also acknowledges ColorFormer, BigColor, and BasicSR in its upstream
README. This vendored subset contains inference architecture only; no training
data, pretrained weights, private media, or third-party demonstration images
are redistributed with this package.

The separately downloaded official Hugging Face checkpoints are identified,
revision-pinned, and SHA-256 verified in `videocolor.models.MODEL_SPECS`. Their
model cards declare Apache 2.0. Colorization produces a plausible interpretation,
not evidence of a subject's original colors.
