# This code was originally written by Jose Javier Gonzalez Ortiz 
# for use in UniverSeg (https://github.com/JJGO/UniverSeg).
# It is included here with their permission, without modifications.
from .containers import SegmentationSequential, augmentations_from_config

from .geometry import (
    RandomScale,
    RandomTranslate,
    RandomShear,
)

from .label import (
    RandomCannyEdges,
    RandomDilation,
    RandomErosion,
    RandomVariableDilation,
    RandomVariableErosion,
)

from .label import (
    RandomCannyEdges,
    RandomDilation,
    RandomErosion,
    RandomVariableDilation,
    RandomVariableErosion,
)

from .task import is_task_aug
