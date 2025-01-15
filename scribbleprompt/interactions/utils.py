from typing import Union
import warnings
import numpy as np
import torch
import cv2
import skimage.measure

# -----------------------------------------------------------------------------
# Random sampling
# -----------------------------------------------------------------------------

def _as_single_val(value, high: bool = True) -> Union[int,float]:
    """
    Args:
        high: if True, include the upper bound for the range (for integer ranges)
    """
    if isinstance(value, (int, float)):
        return value
    
    if isinstance(value, (tuple, list)):
        if len(value) == 1:
            return value[0]
        else:
            assert len(value) == 2, f"Invalid 2-tuple {value}"

        if any(isinstance(i, float) for i in value):
            value = (float(value[0]), float(value[1]))

        if isinstance(value[0], int):
            if high:
                return np.random.randint(value[0], value[1]+1)
            else:
                return np.random.randint(*value)
        else:
            return np.random.uniform(*value)

def chance(x: Union[float,int,bool]) -> bool:
    """
    Args:
        x: probability of returning True
    """
    if x == 0:
        return False
    elif x == 1:
        return True
    else:
        return np.random.rand() < x

# -----------------------------------------------------------------------------
# Connected Components
# -----------------------------------------------------------------------------

def get_components(seg: torch.Tensor, background: bool = False, show: bool = False, return_area: bool = False):
    """
    Get a map of all the components in an image
    Args:
        seg: must be binary 1 x H x W or H x W
        background: if False, only get the components where seg != 0, otherwise get components for both the label and background
        show: if True, plot
        return_area: if True, return list of areas of each components 
    """
    assert seg.dtype == torch.int8 or seg.dtype == torch.int32, "seg must be integer"
    if len(seg.shape)==3:
        seg = seg.squeeze()

    assert len(seg.shape)==2, "seg must be 2D"

    comps, n = skimage.measure.label(seg, return_num=True)
    
    if background:
        # Get connected components of the background
        back_comps = skimage.measure.label(1-seg) 
        back_comps[ np.nonzero(back_comps) ] = back_comps[ np.nonzero(back_comps) ] + n
        combined_map = comps + back_comps
    else:
        combined_map = comps
    
    if show:
        import os
        os.environ['NEURITE_BACKEND'] = 'pytorch'
        import neurite as ne
        if background:
            ne.plot.slices([seg, comps, back_comps, combined_map], 
                           titles=['Seg', 'Seg Components', '1-Seg Components', 'Combined Components'], 
                           cmaps=['viridis'], do_colorbars=True)
        else:
            ne.plot.slices([seg, comps], titles=['Seg', "Components"], cmaps=['viridis'], do_colorbars=True, width=10)
    
    if return_area:
        n_components = combined_map.max()
        areas = [(combined_map==i).sum() for i in range(1, n_components+1)]
        return combined_map, areas
    else:
        return combined_map

# -----------------------------------------------------------------------------
# Distance Transform
# -----------------------------------------------------------------------------

def get_combined_dt(error_region: torch.Tensor, background: bool = False):
    """
    Get a combined distance transform of the false positives and false negatives 
    Args:
        error_region (torch.Tensor): (1, height, width) on [-1,0,1] where +1 is false positive and -1 is false negative
    """
    fp_mask = torch.abs(torch.clamp(error_region, min=0.0)).cpu()
    fn_mask = torch.abs(torch.clamp(error_region, max=0.0)).cpu()

    # Note: distanceTransform expects a binary image
    fp_mask_dt = cv2.distanceTransform(fp_mask[0,...,None].numpy().astype(np.uint8), cv2.DIST_L2, 0)
    fn_mask_dt = cv2.distanceTransform(fn_mask[0,...,None].numpy().astype(np.uint8), cv2.DIST_L2, 0)

    mask_dt = fp_mask_dt + fn_mask_dt # shape: (height, width)

    if background:
        background_mask = (error_region==0).float().cpu()
        background_mask_dt = cv2.distanceTransform(background_mask[0,...,None].numpy().astype(np.uint8), cv2.DIST_L2, 0)
        mask_dt += background_mask_dt

    return mask_dt 

# -----------------------------------------------------------------------------
# Debugging
# -----------------------------------------------------------------------------

def warn_in_range(tensor, range_to_check=None, name='tensor'):
    """
    Check if tensor contains NaN/Inf and (optional) is in range
    """
    if tensor.isnan().any():
        warnings.warn(f'{name} contains NaN')
    if tensor.isinf().any():
        warnings.warn(f'{name} contains inf')
    if range_to_check is not None:
        assert len(range_to_check) == 2, f'range should be in form [min, max] {range_to_check}'
        if tensor.min() < range_to_check[0]:
            warnings.warn(f'{name} should be in {range_to_check}, found: {tensor.min()}')
        if tensor.max() > range_to_check[1]:
            warnings.warn(f'{name} should be in {range_to_check}, found: {tensor.max()}')