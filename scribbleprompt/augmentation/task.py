# This code was originally written by Jose Javier Gonzalez Ortiz 
# for use in UniverSeg (https://github.com/JJGO/UniverSeg).
# It is included here with their permission, without modifications.
from typing import Union

from kornia.augmentation import AugmentationBase2D


def is_task_aug(aug: Union[str, AugmentationBase2D]) -> bool:
    if isinstance(aug, AugmentationBase2D):
        aug = aug.__class__.__name__

    _is_task_aug = {
        "ChannelwiseColorJitter": False,
        "ChannelwiseRandomAffine": False,
        "ChannelwiseRandomBoxBlur": False,
        "ChannelwiseRandomBrightnessContrast": False,
        "ChannelwiseRandomCrop": False,
        "ChannelwiseRandomElasticTransform": False,
        "ChannelwiseRandomErasing": False,
        "ChannelwiseRandomFisheye": False,
        "ChannelwiseRandomGaussianBlur": False,
        "ChannelwiseRandomGaussianNoise": False,
        "ChannelwiseRandomHorizontalFlip": False,
        "ChannelwiseRandomInvert": False,
        "ChannelwiseRandomMotionBlur": False,
        "ChannelwiseRandomPerspective": False,
        "ChannelwiseRandomPosterize": False,
        "ChannelwiseRandomResizedCrop": False,
        "ChannelwiseRandomRotation": False,
        "ChannelwiseRandomSharpness": False,
        "ChannelwiseRandomSolarize": False,
        "ChannelwiseRandomThinPlateSpline": False,
        "ChannelwiseRandomVariableBoxBlur": False,
        "ChannelwiseRandomVariableElasticTransform": False,
        "TorchvisionChannelwiseRandomVariableElasticTransform": False,
        "ChannelwiseRandomVariableGaussianBlur": False,
        "ChannelwiseRandomVariableGaussianNoise": False,
        "ChannelwiseRandomVerticalFlip": False,
        # Task Augs
        "RandomAffine": True,
        "RandomBrightnessContrast": True,
        "RandomDilation": True,
        "RandomErosion": True,
        "RandomFlipIntensities": True,
        "RandomFlipLabel": True,
        "RandomHorizontalFlip": True,
        "RandomMorphGradient": True,
        "RandomScale": True,
        "RandomSharpness": True,
        "RandomShear": True,
        "RandomSobelEdgesLabel": True,
        "RandomTranslate": True,
        "RandomVariableBoxBlur": True,
        "RandomVariableDilation": True,
        "RandomVariableElasticTransform": True,
        "RandomVariableErosion": True,
        "RandomVariableGaussianBlur": True,
        "RandomVariableGaussianNoise": True,
        "RandomVerticalFlip": True,
    }

    return _is_task_aug[aug]
