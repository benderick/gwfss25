# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Newer timm versions access torch.fx during module import, while PyTorch 1.9
# does not expose it as an attribute until the submodule is imported.
import torch.fx  # noqa: F401

from .backbone.swin import D2SwinTransformer
from .pixel_decoder.fpn import BasePixelDecoder
from .pixel_decoder.msdeformattn import MSDeformAttnPixelDecoder
from .meta_arch.mask_former_head import MaskFormerHead
from .meta_arch.per_pixel_baseline import PerPixelBaselineHead, PerPixelBaselinePlusHead

from .backbone.dinov2 import DinoV2BaseBackbone, DinoV2LargeBackbone
from .backbone.vit import ViTBaseBackbone, ViTLargeBackbone
from .backbone.deit import DeiTBaseBackbone
from .backbone.beit_adapter import BEiTAdapterLargeBackbone
