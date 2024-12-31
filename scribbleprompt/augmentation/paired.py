# This code was originally written by Jose Javier Gonzalez Ortiz 
# for use in UniverSeg (https://github.com/JJGO/UniverSeg).
# It is included here with their permission, without modifications.
from typing import Any, Optional, Literal

import torch
from torch import Tensor
import kornia.augmentation as KA


class PairAugmentationBase(KA.AugmentationBase2D):

    """Base class for augmentations that work on pairs of 
    (image, label), useful for segmentation augmentations
    """

    def transform_input(self, image: Tensor, params: dict[str, Any]):
        return image

    def transform_mask(self, mask: Tensor, params: dict[str, Any]):
        return mask

    def forward(self, image, mask, params=None):
        undo_resize = False
        if len(image.shape) != len(mask.shape):
            # add a dummy dimension for categorical encoded labels lacking channels dim
            mask = mask[:, None]
            undo_resize = True
        params = params or self.forward_parameters(image.shape, mask.shape)
        image = self.transform_input(image, params)
        mask = self.transform_mask(mask, params)
        self._params = params
        if undo_resize:
            mask = mask[:, 0]
        return image, mask


def _from_individual_aug(module, mode: Literal["both", "image", "mask"]):

    """Hack to be able to easily convert augmentations that operate on a individual
    input to work on pairs of (image, label) pairs. The mode dictates to which
    elements the augmentation is applied
    - both - makes sense for geometrical operations like flips/crops/affine/grid-distortion
    - image - makes sense for noise operations like gaussian-noise/blur/intensity-shift
    - mask - intended for label augmentations (less common) such as label erosion/dilation
    """

    assert mode in ("both", "image", "mask")

    class Wrapper(PairAugmentationBase):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.aug = module(*args, **kwargs)

        def generate_parameters(self, input_shape: torch.Size, mask_shape: torch.Size):
            return self.aug.generate_parameters(input_shape)

        def forward_parameters(self, input_shape: torch.Size, mask_shape: torch.Size):
            return self.aug.forward_parameters(input_shape)

        def transform_input(self, image: Tensor, params: dict[str, Any]):
            if mode in ("both", "image"):
                return self.aug(image, params)
            return image

        def transform_mask(self, mask: Tensor, params: dict[str, Any]):
            if mode in ("both", "mask"):
                return self.aug(mask, params)
            return mask

        def __repr__(self):
            return self.aug.__repr__()

    return type(module.__name__, (Wrapper,), {})


# Deterministic Image only
Normalize = _from_individual_aug(KA.Normalize, mode="image")
Denormalize = _from_individual_aug(KA.Denormalize, mode="image")

# Random Image only
ColorJitter = _from_individual_aug(KA.ColorJitter, mode="image")
RandomInvert = _from_individual_aug(KA.RandomInvert, mode="image")
RandomPosterize = _from_individual_aug(KA.RandomPosterize, mode="image")
RandomSharpness = _from_individual_aug(KA.RandomSharpness, mode="image")
RandomSolarize = _from_individual_aug(KA.RandomSolarize, mode="image")
#     Fixed params
RandomBoxBlur = _from_individual_aug(KA.RandomBoxBlur, mode="image")
RandomGaussianBlur = _from_individual_aug(KA.RandomGaussianBlur, mode="image")
RandomGaussianNoise = _from_individual_aug(KA.RandomGaussianNoise, mode="image")

# Deterministic Both
CenterCrop = _from_individual_aug(KA.CenterCrop, mode="both")
LongestMaxSize = _from_individual_aug(KA.LongestMaxSize, mode="both")
PadTo = _from_individual_aug(KA.PadTo, mode="both")
Resize = _from_individual_aug(KA.Resize, mode="both")
SmallestMaxSize = _from_individual_aug(KA.SmallestMaxSize, mode="both")

# Random Both
RandomAffine = _from_individual_aug(KA.RandomAffine, mode="both")
RandomCrop = _from_individual_aug(KA.RandomCrop, mode="both")
RandomErasing = _from_individual_aug(KA.RandomErasing, mode="both")
RandomFisheye = _from_individual_aug(KA.RandomFisheye, mode="both")
RandomHorizontalFlip = _from_individual_aug(KA.RandomHorizontalFlip, mode="both")
RandomMotionBlur = _from_individual_aug(KA.RandomMotionBlur, mode="both")
RandomPerspective = _from_individual_aug(KA.RandomPerspective, mode="both")
RandomResizedCrop = _from_individual_aug(KA.RandomResizedCrop, mode="both")
RandomRotation = _from_individual_aug(KA.RandomRotation, mode="both")
RandomThinPlateSpline = _from_individual_aug(KA.RandomThinPlateSpline, mode="both")
RandomVerticalFlip = _from_individual_aug(KA.RandomVerticalFlip, mode="both")
#     Fixed params
RandomElasticTransform = _from_individual_aug(KA.RandomElasticTransform, mode="both")


from .variable import (
    RandomVariableGaussianBlur,
    RandomVariableBoxBlur,
    RandomVariableGaussianNoise,
    RandomVariableElasticTransform,
    RandomBrightnessContrast,
)


# fmt: off
RandomBrightnessContrast = _from_individual_aug(RandomBrightnessContrast, mode="image")
RandomVariableBoxBlur = _from_individual_aug(RandomVariableBoxBlur, mode="image")
RandomVariableGaussianBlur = _from_individual_aug(RandomVariableGaussianBlur, mode="image")
RandomVariableGaussianNoise = _from_individual_aug(RandomVariableGaussianNoise, mode="image")
RandomVariableElasticTransform = _from_individual_aug(RandomVariableElasticTransform, mode="both")
# fmt: on

from .geometry import (
    RandomScale,
    RandomTranslate,
    RandomShear,
)

RandomScale = _from_individual_aug(RandomScale, mode="both")
RandomTranslate = _from_individual_aug(RandomTranslate, mode="both")
RandomShear = _from_individual_aug(RandomShear, mode="both")


from .label import (
    RandomCannyEdges,
    RandomDilation,
    RandomErosion,
    RandomMorphGradient,
    RandomVariableDilation,
    RandomVariableErosion,
    RandomSobelEdges,
    RandomFlipIntensities,
)

# Label only
RandomCannyEdgesLabel = _from_individual_aug(RandomCannyEdges, mode="mask")
RandomSobelEdgesLabel = _from_individual_aug(RandomSobelEdges, mode="mask")
RandomDilationLabel = _from_individual_aug(RandomDilation, mode="mask")
RandomErosionLabel = _from_individual_aug(RandomErosion, mode="mask")
RandomMorphGradientLabel = _from_individual_aug(RandomMorphGradient, mode="mask")
RandomVariableDilationLabel = _from_individual_aug(RandomVariableDilation, mode="mask")
RandomVariableErosionLabel = _from_individual_aug(RandomVariableErosion, mode="mask")
RandomFlipLabel = _from_individual_aug(RandomFlipIntensities, mode="mask")
RandomFlipIntensities = _from_individual_aug(RandomFlipIntensities, mode="image")


from .channel import (
    ChannelwiseColorJitter,
    ChannelwiseRandomInvert,
    ChannelwiseRandomPosterize,
    ChannelwiseRandomSharpness,
    ChannelwiseRandomSolarize,
    ChannelwiseRandomBoxBlur,
    ChannelwiseRandomGaussianBlur,
    ChannelwiseRandomGaussianNoise,
    ChannelwiseRandomAffine,
    ChannelwiseRandomCrop,
    ChannelwiseRandomErasing,
    ChannelwiseRandomFisheye,
    ChannelwiseRandomHorizontalFlip,
    ChannelwiseRandomMotionBlur,
    ChannelwiseRandomPerspective,
    ChannelwiseRandomResizedCrop,
    ChannelwiseRandomRotation,
    ChannelwiseRandomThinPlateSpline,
    ChannelwiseRandomVerticalFlip,
    ChannelwiseRandomElasticTransform,
    ChannelwiseRandomBrightnessContrast,
    ChannelwiseRandomVariableBoxBlur,
    ChannelwiseRandomVariableGaussianBlur,
    ChannelwiseRandomVariableGaussianNoise,
    ChannelwiseRandomVariableElasticTransform,
)

# fmt: off
ChannelwiseColorJitter = _from_individual_aug(ChannelwiseColorJitter, mode="image")
ChannelwiseRandomInvert = _from_individual_aug(ChannelwiseRandomInvert, mode="image")
ChannelwiseRandomPosterize = _from_individual_aug(ChannelwiseRandomPosterize, mode="image")
ChannelwiseRandomSharpness = _from_individual_aug(ChannelwiseRandomSharpness, mode="image")
ChannelwiseRandomSolarize = _from_individual_aug(ChannelwiseRandomSolarize, mode="image")
ChannelwiseRandomBoxBlur = _from_individual_aug(ChannelwiseRandomBoxBlur, mode="image")
ChannelwiseRandomGaussianBlur = _from_individual_aug(ChannelwiseRandomGaussianBlur, mode="image")
ChannelwiseRandomGaussianNoise = _from_individual_aug(ChannelwiseRandomGaussianNoise, mode="image")
ChannelwiseRandomAffine = _from_individual_aug(ChannelwiseRandomAffine, mode="both")
ChannelwiseRandomCrop = _from_individual_aug(ChannelwiseRandomCrop, mode="both")
ChannelwiseRandomErasing = _from_individual_aug(ChannelwiseRandomErasing, mode="both")
ChannelwiseRandomFisheye = _from_individual_aug(ChannelwiseRandomFisheye, mode="both")
ChannelwiseRandomHorizontalFlip = _from_individual_aug(ChannelwiseRandomHorizontalFlip, mode="both")
ChannelwiseRandomMotionBlur = _from_individual_aug(ChannelwiseRandomMotionBlur, mode="both")
ChannelwiseRandomPerspective = _from_individual_aug(ChannelwiseRandomPerspective, mode="both")
ChannelwiseRandomResizedCrop = _from_individual_aug(ChannelwiseRandomResizedCrop, mode="both")
ChannelwiseRandomRotation = _from_individual_aug(ChannelwiseRandomRotation, mode="both")
ChannelwiseRandomThinPlateSpline = _from_individual_aug(ChannelwiseRandomThinPlateSpline, mode="both")
ChannelwiseRandomVerticalFlip = _from_individual_aug(ChannelwiseRandomVerticalFlip, mode="both")
ChannelwiseRandomElasticTransform = _from_individual_aug(ChannelwiseRandomElasticTransform, mode="both")
ChannelwiseRandomBrightnessContrast = _from_individual_aug(ChannelwiseRandomBrightnessContrast, mode="image")
ChannelwiseRandomVariableBoxBlur = _from_individual_aug(ChannelwiseRandomVariableBoxBlur, mode="image")
ChannelwiseRandomVariableGaussianBlur = _from_individual_aug(ChannelwiseRandomVariableGaussianBlur, mode="image")
ChannelwiseRandomVariableGaussianNoise = _from_individual_aug(ChannelwiseRandomVariableGaussianNoise, mode="image")
ChannelwiseRandomVariableElasticTransform = _from_individual_aug(ChannelwiseRandomVariableElasticTransform, mode="both")
# ChannelwiseRandomScale = _from_individual_aug(ChannelwiseRandomScale, mode="both")
# ChannelwiseRandomTranslate = _from_individual_aug(ChannelwiseRandomTranslate, mode="both")
# ChannelwiseRandomShear = _from_individual_aug(ChannelwiseRandomShear, mode="both")
# fmt: on
