import numpy as np

def _as_single_val(value, high: bool = True):
    """
    Args:
        value: int, float, or range to sample from
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