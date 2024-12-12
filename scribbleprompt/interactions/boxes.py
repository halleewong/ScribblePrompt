import torch
import numpy as np
from typing import Literal

# -----------------------------------------------------------------------------
# Bounding Box Simulation
# -----------------------------------------------------------------------------

class UniformBBox:
    """
    Sample bounding boxes with jitter randomly sampled from a uniform 
     distribution [0, max_jitter)
    """
    def __init__(self, max_jitter: int = 20, train: bool = True):
        self.max_jitter = max_jitter
        self.train = train

    @property
    def attrs(self):
        return {
            "jitter": 'uniform',
            "max_jitter": self.max_jitter,
        }

    def sample_bbox(self, seg: torch.Tensor) -> torch.Tensor:
        """
        Sample bounding boxes for a batch of segmentations
        """
        H,W = seg.shape[-2:]
        device = seg.device
        in_ndim = len(seg.shape)

        if in_ndim==3:
            seg = seg.unsqueeze(0)

        bs = seg.shape[0]
    
        x,y = torch.meshgrid(
            torch.arange(H, device=device), 
            torch.arange(W, device=device), 
            indexing='xy'
            )
        x = x.repeat(bs,1,1)
        y = y.repeat(bs,1,1)

        if seg.sum() == 0:
            # If no segmentation
            if self.train:
                return torch.zeros((bs,1,4), device=device)
            else:
                return None
        
        x_idx = torch.where(seg > 0, x, 0).reshape(bs,-1)
        y_idx = torch.where(seg > 0, y, 0).reshape(bs,-1)

        x_min, _ = x_idx.min(-1)
        x_max, _ = x_idx.max(-1)
        y_min, _ = y_idx.min(-1)
        y_max, _ = y_idx.max(-1)

        if self.max_jitter == 0:
            x_jitter = torch.zeros((2,bs), device=device)
            y_jitter = torch.zeros((2,bs), device=device)
        else:
            x_jitter = torch.randint(0, self.max_jitter, size=(2,bs), device=device)
            y_jitter = torch.randint(0, self.max_jitter, size=(2,bs), device=device)

        x_min = torch.clamp(x_min - x_jitter[0], min=0, max=W)
        x_max = torch.clamp(x_max + x_jitter[1], min=0, max=W)
        y_min = torch.clamp(y_min - y_jitter[0], min=0, max=H)
        y_max = torch.clamp(y_max + y_jitter[1], min=0, max=H)
        
        box = torch.stack([x_min, y_min, x_max, y_max], dim=1)
        if in_ndim == 4:
            box = box.unsqueeze(1)

        # shape: (b,1,4) or (1,4)
        return box 


    def __call__(self, seg: torch.Tensor) -> np.ndarray:
        """
        Args:
            seg: (b,1,H,W) or (1,H,W) mask in [0,1] to fit a bounding box to

        Returns:
            bbox (torch.Tensor): bounding box coordinates 
                [x_min, y_min, x_max, y_max] with shape (b,1,4) or (1,4)

        Note: if the given mask is empty a box [0, 0, 0, 0] is returned
        """
        assert len(seg.shape) in [3,4], \
            f"mask must be Bx1xHxW or 1xHxW. currently {seg.shape}"
        assert seg.shape[-3] == 1, \
            f"mask must have 1 channel. currently {seg.shape[-3]}"
        
        return self.sample_bbox(seg)
