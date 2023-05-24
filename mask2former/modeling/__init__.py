# Copyright (c) Facebook, Inc. and its affiliates.
# Adapted for AutoFocusFormer by Ziwen 2023

from .backbone.aff import AutoFocusFormer

from .pixel_decoder.msdeformattn_pc import MSDeformAttnPixelDecoder
from .meta_arch.mask_former_head import MaskFormerHead
