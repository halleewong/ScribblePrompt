# This code was originally written by Jose Javier Gonzalez Ortiz 
# for use in UniverSeg (https://github.com/JJGO/UniverSeg).
# It is included here with their permission, without modifications.
"""Augmentations intended for segmentation labels
"""

from typing import Any, Optional

import einops as E
import kornia as K
import torch
from pydantic import validate_arguments

from .common import AugmentationBase2D
from .variable import FilterBase, VariableFilterBase, _as2tuple


class RandomCannyEdges(AugmentationBase2D):
    @validate_arguments
    def __init__(
        self,
        same_on_batch: bool = False,
        p: float = 0.5,
        keepdim: bool = False,
    ) -> None:
        super().__init__(p=p, same_on_batch=same_on_batch, p_batch=1.0, keepdim=keepdim)

    def apply_transform(
        self,
        input: torch.Tensor,
        params: dict[str, torch.Tensor],
        flags: dict[str, Any],
        transform: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B = input.shape[0]
        input = E.rearrange(input, "B C H W -> (B C) 1 H W")
        edges = K.filters.canny(input)[1].abs()  # [1] is after hysteresis
        edges = E.rearrange(edges, "(B C) 1 H W -> B C H W", B=B)
        return (edges - edges.min()) / (edges.max() - edges.min())


class RandomSobelEdges(AugmentationBase2D):
    @validate_arguments
    def __init__(
        self,
        same_on_batch: bool = False,
        p: float = 0.5,
        keepdim: bool = False,
        eps=1e-7,
    ) -> None:
        super().__init__(p=p, same_on_batch=same_on_batch, p_batch=1.0, keepdim=keepdim)
        self.eps = eps

    def apply_transform(
        self,
        input: torch.Tensor,
        params: dict[str, torch.Tensor],
        flags: dict[str, Any],
        transform: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B = input.shape[0]
        input = E.rearrange(input, "B C H W -> (B C) 1 H W")
        edges = K.filters.sobel(input).abs()
        edges = E.rearrange(edges, "(B C) 1 H W -> B C H W", B=B)
        return (edges - edges.min()) / (edges.max() - edges.min()).clamp_min(self.eps)


class RandomDilation(FilterBase):
    def apply_transform(
        self,
        input: torch.Tensor,
        params: dict[str, torch.Tensor],
        flags: dict[str, Any],
        transform: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        kernel_size = _as2tuple(self.kernel_size)
        sigma = _as2tuple(self.sigma)

        kernel = K.filters.get_gaussian_kernel2d(kernel_size, sigma)
        return K.morphology.dilation(
            input, kernel.to(input.device), engine="convolution"
        )


class RandomErosion(FilterBase):
    def apply_transform(
        self,
        input: torch.Tensor,
        params: dict[str, torch.Tensor],
        flags: dict[str, Any],
        transform: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        kernel_size = _as2tuple(self.kernel_size)
        sigma = _as2tuple(self.sigma)

        kernel = K.filters.get_gaussian_kernel2d(kernel_size, sigma)
        return K.morphology.erosion(
            input, kernel.to(input.device), engine="convolution"
        )


class RandomMorphGradient(FilterBase):
    def apply_transform(
        self,
        input: torch.Tensor,
        params: dict[str, torch.Tensor],
        flags: dict[str, Any],
        transform: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        kernel_size = _as2tuple(self.kernel_size)
        sigma = _as2tuple(self.sigma)

        kernel = K.filters.get_gaussian_kernel2d(kernel_size, sigma)
        return K.morphology.gradient(
            input, kernel.to(input.device), engine="convolution"
        )


class RandomVariableDilation(VariableFilterBase):
    def apply_transform(
        self,
        input: torch.Tensor,
        params: dict[str, torch.Tensor],
        flags: dict[str, Any],
        transform: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        kernel_size = _as2tuple(params["kernel_size"])
        sigma = _as2tuple(params["sigma"])

        kernel = K.filters.get_gaussian_kernel2d(kernel_size, sigma)
        return K.morphology.dilation(
            input, kernel.to(input.device), engine="convolution"
        )


class RandomVariableErosion(VariableFilterBase):
    def apply_transform(
        self,
        input: torch.Tensor,
        params: dict[str, torch.Tensor],
        flags: dict[str, Any],
        transform: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        kernel_size = _as2tuple(params["kernel_size"])
        sigma = _as2tuple(params["sigma"])

        kernel = K.filters.get_gaussian_kernel2d(kernel_size, sigma)
        return K.morphology.erosion(
            input, kernel.to(input.device), engine="convolution"
        )


class RandomFlipIntensities(AugmentationBase2D):
    def apply_transform(
        self,
        input: torch.Tensor,
        params: dict[str, torch.Tensor],
        flags: dict[str, Any],
        transform: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return 1 - input
