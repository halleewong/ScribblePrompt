# This code was originally written by Jose Javier Gonzalez Ortiz 
# for use in UniverSeg (https://github.com/JJGO/UniverSeg).
# It is included here with their permission, without modifications.
import random
from typing import Any, Optional, Union

import kornia as K
import kornia.augmentation as KA
import numpy as np
import torch
from kornia.constants import BorderType
from pydantic import validate_arguments

from .common import AugmentationBase2D, _as2tuple, _as_single_val


class RandomBrightnessContrast(AugmentationBase2D):
    def __init__(
        self,
        brightness: Union[float, tuple[float, float]] = 0.0,
        contrast: Union[float, tuple[float, float]] = 1.0,
        same_on_batch: bool = False,
        p: float = 0.5,
        keepdim: bool = False,
    ) -> None:
        super().__init__(
            p=p,
            same_on_batch=same_on_batch,
            p_batch=1.0,
            keepdim=keepdim,
        )
        self.brightness = brightness
        self.contrast = contrast

    def generate_parameters(self, input_shape: torch.Size):
        brightness = _as_single_val(self.brightness)
        contrast = _as_single_val(self.contrast)

        order = np.random.permutation(2)

        return dict(brightness=brightness, contrast=contrast, order=order)

    def apply_transform(
        self,
        input: torch.Tensor,
        params: dict[str, torch.Tensor],
        flags: dict[str, Any],
        transform: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        transforms = [
            lambda img: K.enhance.adjust_brightness(img, params["brightness"]),
            lambda img: K.enhance.adjust_contrast(img, params["contrast"]),
        ]

        jittered = input
        for idx in params["order"].tolist():
            t = transforms[idx]
            jittered = t(jittered)

        return jittered


class FilterBase(AugmentationBase2D):
    @validate_arguments
    def __init__(
        self,
        kernel_size: Union[int, tuple[int, int]],
        sigma: Union[float, tuple[float, float]],
        same_on_batch: bool = False,
        p: float = 0.5,
        keepdim: bool = False,
    ) -> None:
        super().__init__(
            p=p,
            same_on_batch=same_on_batch,
            p_batch=1.0,
            keepdim=keepdim,
        )
        self.kernel_size = kernel_size
        self.sigma = sigma


class VariableFilterBase(FilterBase):

    """Helper class for tasks that involve a random filter"""

    def generate_parameters(self, input_shape: torch.Size):
        kernel_size = _as_single_val(self.kernel_size)
        sigma = _as_single_val(self.sigma)
        return dict(kernel_size=kernel_size, sigma=sigma)


class RandomVariableGaussianBlur(VariableFilterBase):
    def __init__(
        self,
        kernel_size: Union[int, tuple[int, int]],
        sigma: Union[float, tuple[float, float]],
        border_type: str = "reflect",
        same_on_batch: bool = False,
        p: float = 0.5,
        keepdim: bool = False,
    ) -> None:
        super().__init__(
            kernel_size=kernel_size,
            sigma=sigma,
            p=p,
            same_on_batch=same_on_batch,
            keepdim=keepdim,
        )
        self.flags = dict(border_type=BorderType.get(border_type))

    def apply_transform(
        self,
        input: torch.Tensor,
        params: dict[str, torch.Tensor],
        flags: dict[str, Any],
        transform: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        kernel_size = _as2tuple(self.kernel_size)
        sigma = _as2tuple(self.sigma)

        return K.filters.gaussian_blur2d(
            input, kernel_size, sigma, flags["border_type"].name.lower()
        )


class RandomVariableBoxBlur(AugmentationBase2D):
    def __init__(
        self,
        kernel_size: Union[int, tuple[int, int]] = 3,
        border_type: str = "reflect",
        normalized: bool = True,
        same_on_batch: bool = False,
        p: float = 0.5,
        keepdim: bool = False,
    ) -> None:
        super().__init__(
            p=p,
            same_on_batch=same_on_batch,
            p_batch=1.0,
            keepdim=keepdim,
        )
        self.flags = dict(border_type=border_type, normalized=normalized)

    def generate_parameters(self, input_shape: torch.Size):
        kernel_size = _as_single_val(self.kernel_size)
        return dict(kernel_size=kernel_size)

    def apply_transform(
        self,
        input: torch.Tensor,
        params: dict[str, torch.Tensor],
        flags: dict[str, Any],
        transform: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        kernel_size = _as2tuple(params["kernel_size"])
        return K.filters.box_blur(
            input, kernel_size, flags["border_type"], flags["normalized"]
        )


class RandomVariableGaussianNoise(AugmentationBase2D):
    def __init__(
        self,
        mean: Union[float, tuple[float, float]] = 0.0,
        std: Union[float, tuple[float, float]] = 1.0,
        same_on_batch: bool = False,
        p: float = 0.5,
        keepdim: bool = False,
    ) -> None:
        super().__init__(
            p=p,
            same_on_batch=same_on_batch,
            p_batch=1.0,
            keepdim=keepdim,
        )
        self.mean = mean
        self.std = std

    def generate_parameters(self, input_shape: torch.Size):
        mean = _as_single_val(self.mean)
        std = _as_single_val(self.std)

        if torch.cuda.is_available():
            noise = torch.empty(input_shape, dtype=torch.float32, device='cuda').normal_(mean, std)
        else:
            noise = torch.empty(noise, dtype=torch.float32, device='cpu').normal_(mean, std)

        return dict(noise=noise)

    def apply_transform(
        self,
        input: torch.Tensor,
        params: dict[str, torch.Tensor],
        flags: dict[str, Any],
        transform: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return input + params["noise"].to(input)


def validate_elastic_sigma_alpha(sigma, alpha):
    if isinstance(alpha, (tuple, list)):
        alpha = max(alpha)
    if isinstance(sigma, (tuple, list)):
        sigma = max(sigma)
    if sigma / alpha < 1:
        raise ValueError("Alpha and Sigma seem to be swapped")


class RandomVariableElasticTransform(AugmentationBase2D):
    def __init__(
        self,
        kernel_size: Union[int, tuple[int, int]] = 63,
        sigma: Union[float, tuple[float, float]] = 32,
        alpha: Union[float, tuple[float, float]] = 1.0,
        align_corners: bool = False,
        mode: str = "bilinear",
        padding_mode: str = "zeros",
        same_on_batch: bool = False,
        p: float = 0.5,
        keepdim: bool = False,
    ) -> None:
        super().__init__(
            p=p,
            same_on_batch=same_on_batch,
            p_batch=1.0,
            keepdim=keepdim,
        )
        validate_elastic_sigma_alpha(sigma, alpha)
        self.flags = dict(
            kernel_size=kernel_size,
            sigma=sigma,
            alpha=alpha,
            align_corners=align_corners,
            mode=mode,
            padding_mode=padding_mode,
        )

    def generate_parameters(self, shape: torch.Size) -> dict[str, torch.Tensor]:
        B, _, H, W = shape

        # By default self.device (which is what kornia prefers, it default to cpu) so
        # the conv2d's to lowpass filter the noise happen on the cpu regardless of
        # input.device value. To bypass this, we force the noise device to 'cuda'
        # whenever possible

        device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.same_on_batch:
            noise = torch.rand(1, 2, H, W, device=device, dtype=self.dtype).repeat(
                B, 1, 1, 1
            )
        else:
            noise = torch.rand(B, 2, H, W, device=device, dtype=self.dtype)

        kernel_size = _as_single_val(self.flags["kernel_size"])
        sigma = _as_single_val(self.flags["sigma"])
        alpha = _as_single_val(self.flags["alpha"])

        return dict(
            noise=noise * 2 - 1, kernel_size=kernel_size, sigma=sigma, alpha=alpha
        )

    def apply_transform(
        self,
        input: torch.Tensor,
        params: dict[str, torch.Tensor],
        flags: dict[str, Any],
        transform: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        assert (
            input.device == params["noise"].device
        ), f"Input/Noise with different devices {input.device} & {params['noise'].device}"
        return K.geometry.transform.elastic_transform2d(
            input,
            params["noise"],  # .to(input),
            _as2tuple(params["kernel_size"]),
            _as2tuple(params["sigma"]),
            _as2tuple(params["alpha"]),
            flags["align_corners"],
            flags["mode"],
            flags["padding_mode"],
        )

