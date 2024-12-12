import torch
from typing import Tuple, Literal

def bbox_shaded(boxes: torch.Tensor, 
                shape: Tuple[int,int] = (128,128), 
                device='cuda') -> torch.Tensor:
    """
    Represent a bounding box as a binary image with 1 inside the bbox and 0 outside
    Args:
        boxes Bx1x4 [x_min, y_min, x_max, y_max]
        shape (tuple): (H,W)
        device (str): 'cuda' or 'cpu'
    Returns:
        bbox_embed (torch.Tensor): Bx1xHxW according to shape
    """
    assert len(shape)==2, "shape must be 2D"
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.int().cpu().numpy()

    batch_size = boxes.shape[0]
    bbox_embed = torch.zeros((batch_size,1)+tuple(shape), device=device, dtype=torch.float32)

    if boxes is not None:
        for i in range(batch_size):
            x_min, y_min, x_max, y_max = boxes[i,0,:]
            bbox_embed[ i, 0, y_min:y_max, x_min:x_max ] = 1.0

    return bbox_embed


def click_onehot(point_coords: torch.Tensor, 
                 point_labels: torch.Tensor, 
                 shape: Tuple[int,int] = (128,128), 
                 indexing: Literal['xy','uv'] = 'xy') -> torch.Tensor:
    """
    Represent clicks a masks of zeros with 1s at the click locations
    Args:
        point_coords (torch.Tensor): BxNx2 tensor of xy oordinates
        point_labels (torch.Tensor): BxN tensor of click labels
        shape (tuple): (H,W)
        indexing (str): 'xy' or 'uv' indexing
    Returns:
        embed (torch.Tensor): Bx2xHxW tensor of clicks
    """
    assert len(point_coords.shape) == 3, "point_coords must be BxNx2"
    assert point_coords.shape[-1] == 2, "point_coords must be BxNx2"
    assert point_labels.shape[-1] == point_coords.shape[1], "point_labels must be BxN"
    assert len(shape)==2, f"shape must be 2D: {shape}"

    device = point_coords.device
    batch_size = point_coords.shape[0]
    n_points = point_coords.shape[1]

    embed = torch.zeros((batch_size,2)+shape, device=device)
    labels = point_labels.flatten().float()

    idx_coords = torch.cat(
        (torch.arange(batch_size, device=device).reshape(-1,1).repeat(1,n_points)[...,None], point_coords), 
        axis=2).reshape(-1,3)

    if indexing=='xy':
        embed[ idx_coords[:,0], 0, idx_coords[:,2], idx_coords[:,1] ] = labels
        embed[ idx_coords[:,0], 1, idx_coords[:,2], idx_coords[:,1] ] = 1.0-labels
    else:
        embed[ idx_coords[:,0], 0, idx_coords[:,1], idx_coords[:,2] ] = labels
        embed[ idx_coords[:,0], 1, idx_coords[:,1], idx_coords[:,2] ] = 1.0-labels

    return embed