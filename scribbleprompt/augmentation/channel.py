# This code was originally written by Jose Javier Gonzalez Ortiz 
# for use in UniverSeg (https://github.com/JJGO/UniverSeg).
# It is included here with their permission, without modifications.
import kornia.augmentation as KA
import einops as E


def _channelwise_aug(module):
    class Wrapper(module):
        def forward(self, input, params=None):
            B = input.shape[0]
            input = E.rearrange(input, "B C H W -> (B C) 1 H W")
            output = super().forward(input, params)
            output = E.rearrange(output, "(B C) 1 H W -> B C H W", B=B)
            return output
        def forward_parameters(self, input_shape):
            B, C, H, W = input_shape
            return super().forward_parameters((B*C, 1, H, W))

    return type(f"Channelwise{module.__name__}", (Wrapper,), {})


from .variable import (
    RandomVariableGaussianBlur,
    RandomVariableBoxBlur,
    RandomVariableGaussianNoise,
    RandomVariableElasticTransform,
    RandomBrightnessContrast,
)

from .geometry import (
    RandomScale,
    RandomTranslate,
    RandomShear,
)

KA.ColorJitter.forward
ChannelwiseColorJitter = _channelwise_aug(KA.ColorJitter)
ChannelwiseRandomInvert = _channelwise_aug(KA.RandomInvert)
ChannelwiseRandomPosterize = _channelwise_aug(KA.RandomPosterize)
ChannelwiseRandomSharpness = _channelwise_aug(KA.RandomSharpness)
ChannelwiseRandomSolarize = _channelwise_aug(KA.RandomSolarize)
ChannelwiseRandomBoxBlur = _channelwise_aug(KA.RandomBoxBlur)
ChannelwiseRandomGaussianBlur = _channelwise_aug(KA.RandomGaussianBlur)
ChannelwiseRandomGaussianNoise = _channelwise_aug(KA.RandomGaussianNoise)
ChannelwiseRandomAffine = _channelwise_aug(KA.RandomAffine)
ChannelwiseRandomCrop = _channelwise_aug(KA.RandomCrop)
ChannelwiseRandomErasing = _channelwise_aug(KA.RandomErasing)
ChannelwiseRandomFisheye = _channelwise_aug(KA.RandomFisheye)
ChannelwiseRandomHorizontalFlip = _channelwise_aug(KA.RandomHorizontalFlip)
ChannelwiseRandomMotionBlur = _channelwise_aug(KA.RandomMotionBlur)
ChannelwiseRandomPerspective = _channelwise_aug(KA.RandomPerspective)
ChannelwiseRandomResizedCrop = _channelwise_aug(KA.RandomResizedCrop)
ChannelwiseRandomRotation = _channelwise_aug(KA.RandomRotation)
ChannelwiseRandomThinPlateSpline = _channelwise_aug(KA.RandomThinPlateSpline)
ChannelwiseRandomVerticalFlip = _channelwise_aug(KA.RandomVerticalFlip)
ChannelwiseRandomElasticTransform = _channelwise_aug(KA.RandomElasticTransform)
ChannelwiseRandomBrightnessContrast = _channelwise_aug(RandomBrightnessContrast)
ChannelwiseRandomVariableBoxBlur = _channelwise_aug(RandomVariableBoxBlur)
ChannelwiseRandomVariableGaussianBlur = _channelwise_aug(RandomVariableGaussianBlur)
ChannelwiseRandomVariableGaussianNoise = _channelwise_aug(RandomVariableGaussianNoise)
ChannelwiseRandomVariableElasticTransform = _channelwise_aug(RandomVariableElasticTransform)
