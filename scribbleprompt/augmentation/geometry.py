# This code was originally written by Jose Javier Gonzalez Ortiz 
# for use in UniverSeg (https://github.com/JJGO/UniverSeg).
# It is included here with their permission, without modifications.
from typing import Optional, Union

import torch
import kornia.augmentation as KA
from kornia.constants import Resample, SamplePadding


def RandomScale(scale, **kwargs):
    return KA.RandomAffine(degrees=0.0, translate=0.0, scale=scale, shear=0.0, **kwargs)


def RandomTranslate(translate, **kwargs):
    return KA.RandomAffine(
        degrees=0.0, translate=translate, scale=0.0, shear=0.0, **kwargs
    )


def RandomShear(shear, **kwargs):
    return KA.RandomAffine(degrees=0.0, translate=0.0, scale=0.0, shear=shear, **kwargs)
