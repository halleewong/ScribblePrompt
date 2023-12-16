import os
from typing import Literal, Tuple, Optional, Dict
import torch
import torch.nn.functional as F

from .network import UNet


class ScribblePromptUNet:

    weights = {
        "v1": "./checkpoints/ScribblePrompt_unet_v1_nf192_res128.pt"
    }
    
    def __init__(self, version: Literal["v1"] = "v1", device = None) -> None:
        
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.version = version
        self.device = device
        self.build_model(pretrained=True)
        self.input_size = (128,128)
    
    def build_model(self, pretrained: bool = True):
        """
        Build model
        """
        self.model = UNet(
            in_channels = 5,
            out_channels = 1,
            features = [192, 192, 192, 192],
        ).to(self.device)
        if pretrained:
            checkpoint_file = self.weights[self.version]
            assert os.path.exists(checkpoint_file), f"Checkpoint file not found: {checkpoint_file}. Please download from Dropbox"
            
            with open(checkpoint_file, "rb") as f:
                state = torch.load(f, map_location=self.device)
            
            self.model.load_state_dict(state)

    def to(self, device):
        self.model.to(device)
        self.device = device

    @torch.no_grad()
    def predict(self,
                img: torch.Tensor, # B x 1 x H x W
                point_coords: Optional[torch.Tensor] = None, # B x n x 2
                point_labels: Optional[torch.Tensor] = None, # B x n 
                scribbles: Optional[torch.Tensor] = None, # B x 2 x H x W
                box: Optional[torch.Tensor] = None, # B x 1 x 4
                mask_input = None, # B x 1 x H x W
                return_logits: bool = False,
                ):
        """
        Make predictions from pytorch tensor inputs with grayscale images
        If the tensors are on the GPU it will prepare the inputs on GPU and retun the mask on the GPU

        Note: if batch size > 1, the number of clicks/boxes must be the same for each image in the batch

        Args:
            img: torch.Tensor (B x 1 x H x W) image to segment on [0,1] scale
            point_coords: torch.Tensor (B x n x 2) coordinates of pos/neg clicks in [x,y] format
            point_labels: torch.Tensor (B x n) labels of clicks (0 or 1)
            scribbles: torch.Tensor (B x 2 x H x W) pos/neg scribble inputs
            box: torch.Tensor (B x 1 x 4) bounding box inputs in [x1, y1, x2, y2] format
            mask_input: torch.Tensor (B x 1 x 128 x 128) logits of previous prediction
            return_logits: bool, if True return logits instead of mask on [0,1] scale
        
        Returns:
            mask (torch.Tensor): B x 1 x H x W prediction for each image in batch
        
        """
        assert (len(img.shape)==4) and (img.shape[1]==1), f"img shape should be B x 1 x H x W. current shape: {img.shape}"
        assert img.min() >= 0 and img.max() <= 1, f"img should be on [0,1] scale. current range: {img.min()} {img.max()}"

        prompts = {
            'img': img,
            'point_coords': point_coords,
            'point_labels': point_labels,
            'scribbles': scribbles,
            'box': box,
            'mask_input': mask_input,
        }
        # Prepare inputs for ScribblePrompt unet (B x 5 x H x W)
        x = prepare_inputs(prompts).float().to(self.device)

        yhat = self.model(x)

        # B x 1 x H x W
        if return_logits:
            return yhat
        else:
            return torch.sigmoid(yhat)

    
# -----------------------------------------------------------------------------
# Prepare inputs
# -----------------------------------------------------------------------------

def rescale_inputs(inputs: Dict[str,any], input_size: Tuple[int] = (128,128)):
    """
    Rescale the inputs 
    """ 
    h,w = inputs['img'].shape[-2:]
    if [h,w] != input_size:
        
        inputs.update(dict(
            img = F.interpolate(inputs['img'], size=input_size, mode='bilinear')
        ))

        if inputs.get('scribbles') is not None:
            inputs.update({
                'scribbles': F.interpolate(inputs['scribbles'], size=input_size, mode='bilinear') 
            })
        
        if inputs.get("box") is not None:
            boxes = inputs.get("box").clone()
            coords = boxes.reshape(-1, 2, 2)
            coords[..., 0] = coords[..., 0] * (input_size[1] / w)
            coords[..., 1] = coords[..., 1] * (input_size[0] / h)
            inputs.update({'box': coords.reshape(1, -1, 4).int()})
        
        if inputs.get("point_coords") is not None:
            coords = inputs.get("point_coords").clone()
            coords[..., 0] = coords[..., 0] * (input_size[1] / w)
            coords[..., 1] = coords[..., 1] * (input_size[0] / h)
            inputs.update({'point_coords': coords.int()})

    return inputs

def prepare_inputs(inputs: Dict[str,torch.Tensor], device = None) -> torch.Tensor:
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

    if inputs.get("scribbles") is not None:
        # Combine scribbles with click embedding
        # B x 2 x H x W
        scribble_click_embed = torch.clamp(scribble_click_embed + inputs.get('scribbles'), min=0.0, max=1.0)

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