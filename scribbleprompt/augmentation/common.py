# This code was originally written by Jose Javier Gonzalez Ortiz 
# for use in UniverSeg (https://github.com/JJGO/UniverSeg).
# It is included here with their permission, without modifications.
from typing import Union

import numpy as np
import kornia.augmentation as KA


def _as2tuple(value: Union[int, float, tuple[int, int]]) -> tuple[int, int]:
    # because kornia.morphology works only with two-tuples
    if isinstance(value, (int, float)):
        return (value, value)
    if isinstance(value, list):
        value = tuple(value)
    assert isinstance(value, tuple) and len(value) == 2, f"Invalid 2-tuple {value}"
    return value


def _as_single_val(value):
    if isinstance(value, (int, float)):
        return value
    assert (
        isinstance(value, (tuple, list)) and len(value) == 2
    ), f"Invalid 2-tuple {value}"
    if value[0] == value[1]:
        return value[0]
    if any(isinstance(i, float) for i in value):
        value = (float(value[0]), float(value[1]))
    if isinstance(value[0], int):
        return np.random.randint(*value)
    else:
        return np.random.uniform(*value)


class AugmentationBase2D(KA.AugmentationBase2D):

    """ Dummy class because Kornia really wants me to overload
    the .compute_transformation method
    """

    def compute_transformation(self, input, params):
        return self.identity_matrix(input)
