from typing import Union
import warnings
import numpy as np

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