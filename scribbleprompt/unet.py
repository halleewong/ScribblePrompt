from typing import Literal, Tuple, Optional, Dict
import torch

from network import UNet


class ScribblePromptUNet:

    weights = {
        "v1": "../checkpoints/scribbleprompt_unet_v1_res128.pt"
    }
    
    def __init__(self, version: Literal["v1"] = "v1", pretrained: bool = True, device = None) -> None:
        
        self.version = version
        self.device = device
        self.build_model(pretrained=pretrained)
    
    def build_model(self, pretrained: bool):
        """
        Build model
        """
        self.model = UNet(
            in_channels = 5,
            out_channels = 1,
            features = [192, 192, 192, 192],
        )
        if pretrained:
            #state = torch.hub.load_state_dict_from_url(self.weights[self.version])

            checkpoint_file = self.weights[self.version]
            with open(checkpoint_file, "rb") as f:
                state = torch.load(f, map_location=self.device)
            
            self.model.load_state_dict(state)

    def to(self, device):
        self.model.to(device)
        self.device = device

    @torch.no_grad()
    def predict(img: torch.Tensor, 
                point_coords: Optional[torch.Tensor] = None, # B x n x 2
                point_labels: Optional[torch.Tensor] = None, # B x n 
                scribbles = None, # B x 2 x H x W
                box = None, # B x 1 x 4
                mask_input = None, # B x 1 x H x W
                ):
        
        pass


# -----------------------------------------------------------------------------
# Prepare inputs
# -----------------------------------------------------------------------------

def prepare_inputs(self, inputs: Dict[str,torch.Tensor], device = None) -> torch.Tensor:
    """
    Prepare inputs for network

    Returns: 
        x (torch.Tensor): B x 5 x H x W
    """
    img = inputs['img']
    if device is None:
        device = img.device

    img = img.to(device)
    shape = tuple(img.shape[-2:])
    
    if inputs.get("box") is not None:
        # Embed bounding box
        # Input: B x 1 x 4 
        # Output: B x 1 x H x W
        box_embed = bbox_shaded(inputs['box'], shape=shape, device=device)
    else:
        box_embed = torch.zeros(img.shape, device=device)

    if inputs.get("point_coords") is not None:
        # Embed points
        # B x 2 x H x W
        scribble_click_embed = click_onehot(inputs['point_coords'], inputs['point_labels'], shape=shape)
    else:
        scribble_click_embed = torch.zeros((img.shape[0], 2) + shape, device=device)

    if inputs.get("scribble") is not None:
        # Combine scribbles with click embedding
        # B x 2 x H x W
        scribble_click_embed = torch.clamp(scribble_click_embed + inputs.get('scribble'), min=0.0, max=1.0)

    if inputs.get('mask_input') is not None:
        # Previous prediction
        mask_input = inputs['mask_input']
    else:
        # Initialize empty channel for mask input
        mask_input = torch.zeros(img.shape, device=img.device)

    x = torch.cat((img, box_embed, scribble_click_embed, mask_input), dim=-3)
    # B x 5 x H x W

    return x
    
# -----------------------------------------------------------------------------
# Encode clicks and bounding boxes
# -----------------------------------------------------------------------------

def click_onehot(point_coords, point_labels, shape: Tuple[int,int] = (128,128), indexing='xy'):
    """
    Represent clicks as two HxW binary masks (one for positive clicks and one for negative) 
    with 1 at the click locations and 0 otherwise

    Args:
        point_coords (torch.Tensor): BxNx2 tensor of xy coordinates
        point_labels (torch.Tensor): BxN tensor of labels (0 or 1)
        shape (tuple): output shape     
    Returns:
        embed (torch.Tensor): Bx2xHxW tensor 
    """
    assert indexing in ['xy','uv'], f"Invalid indexing: {indexing}"
    assert len(point_coords.shape) == 3, "point_coords must be BxNx2"
    assert point_coords.shape[-1] == 2, "point_coords must be BxNx2"
    assert point_labels.shape[-1] == point_coords.shape[1], "point_labels must be BxN"
    assert len(shape)==2, f"shape must be 2D: {shape}"

    device = point_coords.device
    batch_size = point_coords.shape[0]
    n_points = point_coords.shape[1]

    embed = torch.zeros((batch_size,2)+shape, device=device)
    labels = point_labels.flatten().float()

    idx_coords = torch.cat((
        torch.arange(batch_size, device=device).reshape(-1,1).repeat(1,n_points)[...,None], 
        point_coords
    ), axis=2).reshape(-1,3)

    if indexing=='xy':
        embed[ idx_coords[:,0], 0, idx_coords[:,2], idx_coords[:,1] ] = labels
        embed[ idx_coords[:,0], 1, idx_coords[:,2], idx_coords[:,1] ] = 1.0-labels
    else:
        embed[ idx_coords[:,0], 0, idx_coords[:,1], idx_coords[:,2] ] = labels
        embed[ idx_coords[:,0], 1, idx_coords[:,1], idx_coords[:,2] ] = 1.0-labels

    return embed


def bbox_shaded(boxes, shape: Tuple[int,int] = (128,128), device='cpu'):
    """
    Represent a bounding box as a binary mask with 1 inside the box and 0 outside

    Args:
        boxes (torch.Tensor): Bx1x4 [x1, y1, x2, y2]
    Returns:
        bbox_embed (torch.Tesor): Bx1xHxW according to shape
    """
    assert len(shape)==2, "shape must be 2D"
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.int().cpu().numpy()

    batch_size = boxes.shape[0]
    n_boxes = boxes.shape[1]
    bbox_embed = torch.zeros((batch_size,1)+tuple(shape), device=device, dtype=torch.float32)

    if boxes is not None:
        for i in range(batch_size):
            for j in range(n_boxes):
                x1, y1, x2, y2 = boxes[i,j,:]
                x_min = min(x1,x2)
                x_max = max(x1,x2)
                y_min = min(y1,y2)
                y_max = max(y1,y2)
                bbox_embed[ i, 0, y_min:y_max, x_min:x_max ] = 1.0

    return bbox_embed