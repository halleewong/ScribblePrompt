# This code was originally written by Jose Javier Gonzalez Ortiz 
# for use in UniverSeg (https://github.com/JJGO/UniverSeg).
# It is included here with their permission, without modifications.
from typing import Any, Dict, List, Literal, Optional, Union

import einops as E
import numpy as np
from torch import nn

from pylot.util.meta import delegates

from . import paired


class SegmentationSequential(nn.Sequential):
    """Given a list of augmentation modules with segmentation API
    f(x, y) -> x, y
    it applies them one after the other depending on the value of random_apply
    - False -> applies them sequentially
    - True -> applies all of them in a random order
    - n: int -> applies a random subset of N augmentations
    - (n: int, b: int) -> applies a random subset of randint(n,m)
    """

    def __init__(
        self,
        *augmentations: list[nn.Module],
        random_apply: Union[int, bool, tuple[int, int]] = False,
    ):
        super().__init__()

        self.random_apply = random_apply

        for i, augmentation in enumerate(augmentations):
            self.add_module(f"{augmentation.__class__.__name__}_{i}", augmentation)

    def _get_idxs(self):

        N = len(self)

        if self.random_apply is False:
            return np.arange(N)
        elif self.random_apply is True:
            return np.random.permutation(N)
        elif isinstance(self.random_apply, int):
            assert 1 <= self.random_apply <= len(self)
            return np.random.permutation(N)[: self.random_apply]
        elif isinstance(self.random_apply, tuple):
            n = np.random.randint(*self.random_apply)
            return np.random.permutation(N)[:n]
        else:
            raise TypeError(f"Invalid type {type(self.random_apply)}")

    def forward(self, image, segmentation):
        for i in self._get_idxs():
            image, segmentation = self[i](image, segmentation)
        return image, segmentation

    def support_forward(self, images, segmentations):
        x, y = images, segmentations
        support_size = x.shape[1]
        x = E.rearrange(x, "B S C H W -> B (S C) H W")
        y = E.rearrange(y, "B S C H W -> B (S C) H W")
        x, y = self.forward(x, y)
        x = E.rearrange(x, "B (S C) H W -> B S C H W", S=support_size)
        y = E.rearrange(y, "B (S C) H W -> B S C H W", S=support_size)
        return x, y


def augmentations_from_config(config: List[Dict[str, Any]]) -> SegmentationSequential:

    augmentations = []

    random_apply = False
    for aug in config:
        assert len(aug) == 1 and isinstance(aug, dict)
        for name, params in aug.items():
            if name == "random_apply":
                random_apply = params
            else:
                augmentations.append(getattr(paired, name)(**params))

    return SegmentationSequential(*augmentations, random_apply=random_apply)
