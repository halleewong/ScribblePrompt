from typing import Optional, Literal, Tuple
import torch
import torch.nn.functional as F
import numpy as np
import pathlib

from segment_anything.predictor import SamPredictor
from segment_anything.build_sam import sam_model_registry

checkpoint_dir = pathlib.Path(os.path.abspath(__file__)).parent.parent / "checkpoints"

class ScribblePromptSAM(SamPredictor):

    weights = {
        "v1": checkpoint_dir / "ScribblePrompt_sam_v1_vit_b_res128.pt"
    }

    def __init__(self, version: Literal["v1"] = "v1", device = None) -> None:

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.version = version

        checkpoint = self.weights[self.version]
        sam_model = sam_model_registry["vit_b"](checkpoint=checkpoint).to(device)

        super().__init__(sam_model=sam_model)
        self.input_size = (self.model.image_encoder.img_size, self.model.image_encoder.img_size) # 1024 x 1024

    def prepare_box(self, box):
        """
        Rescale bounding box to the input size of the image encoder
        Args:
            box: (B,4) torch.Tensor
        """
        boxes = self.prepare_coords(box.reshape(-1,2,2))
        return boxes.reshape(-1,4)

    def prepare_coords(self, point_coords):
        """
        Rescale coordinates to the input size of the image encoder
        Args:
            point_coords: (B,2) torch.Tensor
        """
        old_h, old_w = self.original_size
        new_h, new_w = self.input_size
        # Otherwise it will change the original tensor and we'll resize multiple times
        coords = point_coords.clone()
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords

    @torch.no_grad()
    def prompt_forward(self, points_coords, points_labels, boxes, masks=None):
        """
        Prompt Encoder
        """
        # Prepare prompt inputs (resizing)
        if points_coords is not None:
            assert (points_labels is not None), "point_labels must be supplied if point_coords is supplied."
            points_coords = self.prepare_coords(points_coords)
            points = (points_coords, points_labels)
        else:
            points = None

        if boxes is not None:
            boxes = self.prepare_box(boxes)

        sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
            points=points,
            boxes=boxes,
            masks=masks,
        )
    
        return sparse_embeddings, dense_embeddings
    
    @torch.no_grad()
    def prepare_image(self, img: torch.Tensor, resize=True, normalize=True):
        """
        Resize and normalize image for input to the image encoder
        Args:
            resize: (bool) whether to resize the image to 1024 x 1024 before applying SAM's resizing procedure
            normalize: (bool) whether to normalize the image following SAM's pixel normalization procedure
        Returns:
            transformed_image: (torch.Tensor) B x 3 x 1024 x 1024
        """
        if resize:
            resized_image = F.interpolate(img, size=self.input_size, mode="bilinear").repeat(1,3,1,1)
        else:
            resized_image = img.repeat(1,3,1,1)

        assert (
            len(resized_image.shape) == 4
            and resized_image.shape[1] == 3
            and max(*resized_image.shape[2:]) == self.model.image_encoder.img_size
        ), f"set_torch_image input must be B x C x H x W with long side {self.model.image_encoder.img_size}."
        
        # Normalize the pixel values (also pads if not correct size)
        if normalize:
            transformed_image = self.model.preprocess(255*resized_image)
        else:
            transformed_image = resized_image

        return transformed_image

    @torch.no_grad()
    def encoder_forward(self, img: torch.Tensor, reisze=True, normalize=True):
        """
        Image Encoder
        Args:
            img: (torch.Tensor) 1 x 1 x H x W on [0,1]
            resize: (bool) whether to resize the image to 1024 x 1024 before applying SAM's resizing procedure
            normalize: (bool) whether to normalize the image following SAM's pixel normalization procedure
        """
        transformed_image = self.prepare_image(img, resize=reisze, normalize=normalize)
        # Return the features instead of using set_image so we can use for training or multiple iterations with batch size > 1
        features = self.model.image_encoder(transformed_image)
        return features

    def decoder_forward(self,
                        img_features,
                        sparse_embeddings,
                        dense_embeddings,
                        multimask_output: bool = True
                        ):
        """
        Mask Decoder

        Returns:
          (torch.Tensor): The output masks in BxCxHxW format, where C is the
            number of masks, and (H, W) is the original image size.
          (torch.Tensor): An array of shape BxCxHxW, where C is the number
            of masks and H=W=256. These low res logits can be passed to
            a subsequent iteration as mask input.
          (torch.Tensor): An array of shape BxC containing the model's
            predictions for the quality of each mask.
        """
        # Predict masks
        low_res_masks, iou_predictions = self.model.mask_decoder(
            image_embeddings=img_features, # B x 256 x 64 x 64
            image_pe=self.model.prompt_encoder.get_dense_pe(), # 1 x 256 x 64 x 64
            sparse_prompt_embeddings=sparse_embeddings, # B x n x 256 
            dense_prompt_embeddings=dense_embeddings, # B x 256 x 64 x 64
            multimask_output=multimask_output,
        )
        # Upscale the masks to the original image resolution
        masks = self.model.postprocess_masks(low_res_masks, self.input_size, self.original_size)

        return masks, low_res_masks, iou_predictions
    
    @torch.no_grad()
    def predict(self, 
                img: torch.Tensor, 
                point_coords: Optional[torch.Tensor] = None, # B x n x 2
                point_labels: Optional[torch.Tensor] = None, # B x n 
                scribbles: Optional[torch.Tensor] = None, # B x 2 x H x W
                box: Optional[torch.Tensor] = None, # B x 1 x 4
                mask_input: Optional[torch.Tensor] = None, # B x 1 x 256 x 256
                img_features: Optional[torch.Tensor] = None, # B x 16 x 256 x 256
                return_logits: bool = False, 
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Make predictions from pytorch tensor inputs 

        Args:
            img: torch.Tensor (B x 1 x H x W) image to segment on [0,1] scale
            point_coords: torch.Tensor (B x n x 2) coordinates of pos/neg clicks in [x,y] format
            point_labels: torch.Tensor (B x n) labels of clicks (0 or 1)
            scribbles: torch.Tensor (B x 2 x H x W) pos/neg scribble inputs
            box: torch.Tensor (B x 1 x 4) bounding box inputs in [x1, y1, x2, y2] format
            mask_input: torch.Tensor (B x 1 x 256 x 256) low res logits of previous prediction
            img_features: torch.Tensor (B x 16 x 256 x 256) image features from the image encoder 
                (to avoid re-running the image encoder unecessarily)
            return_logits: bool, if True return logits instead of masks on [0,1] scale

        Returns:
            masks: torch.Tensor (B x 1 x H x W) prediction for each image in batch
            img_features: torch.Tensor (B x 16 x 256 x 256) image features from the image encoder for each image in batch
            low_res_masks: torch.Tensor (B x 1 x H x W) low res mask for each image in batch

        Note: if batch size > 1, the number of clicks/boxes and pixels covered by the scribbles 
        must be the same for each image in the batch

        """
        assert (len(img.shape)==4) and (img.shape[1]==1), f"img shape should be B x 1 x H x W. current shape: {img.shape}"
        assert img.min() >= 0 and img.max() <= 1, f"img should be on [0,1] scale. current range: {img.min()} {img.max()}"

        self.original_size = img.shape[-2:]
        
        if scribbles is not None:
            # Convert scribble to clicks
            if len(scribbles.shape) == 4:
                scribbles = scribbles.squeeze(0)

            assert len(scribbles.shape)==3, f"scribble shape: {scribbles.shape}"
            scribble_points, scribble_labels = scribbles_to_clicks(scribbles)

            if point_coords is not None:
                point_coords = torch.cat([point_coords, scribble_points], dim=-2)
                point_labels = torch.cat([point_labels, scribble_labels], dim=-1)
            else:
                point_coords = scribble_points
                point_labels = scribble_labels

        if img_features is None:
            # Converts image to RGB and normalizes the pixel values
            img_features = self.encoder_forward(img, normalize=True) 

        sparse_embeddings, dense_embeddings = self.prompt_forward(
            point_coords, point_labels, box, mask_input
        )

        masks, low_res_masks, iou_predictions = self.decoder_forward(
            img_features, sparse_embeddings, dense_embeddings, multimask_output=True # ScribblePrompt SAM was trained in multi-mask mode
        )

        if not return_logits:
            masks = torch.sigmoid(masks)

        # Get the best mask based on predicted IoU
        bs = masks.shape[0]
        iou_predictions, best_idx = torch.max(iou_predictions, dim=1)
        low_res_masks = low_res_masks[range(bs), best_idx, None, ...] # bs x 1 x H x W
        masks = masks[range(bs), best_idx, None, ...] # bs x 1 x H x W
    
        return masks, img_features, low_res_masks
    

# -----------------------------------------------------------------------------
# Encode scribbles
# -----------------------------------------------------------------------------

def scribbles_to_clicks(scribble: torch.Tensor):
    """
    Converts scribbles represented as a mask to individual clicks for every non-zero pixel

    Args:
        scribble: torch.Tensor (2 x H x W) positive scribble mask and negative scribble mask
    Returns:
        points_coords: torch.Tensor (1 x n x 2)
        points_labels: torch.Tensor (1 x n)
    """
    assert len(scribble.shape) == 3, "scribble mask must be 3D tensor 2 x H x W, {scribbles.shape}"
    assert scribble.shape[0] == 2, "scribble mask must have 2 channels, {scribbles.shape}"

    device = scribble.device

    pos_points = torch.nonzero(scribble[0,...])
    neg_points = torch.nonzero(scribble[1,...])
    
    scribble_points = torch.cat([pos_points, neg_points], dim=0)
    
    # Switch order of coordinates so they are xy instead of uv
    scribble_points = torch.stack([scribble_points[:,1],scribble_points[:,0]], dim=-1).unsqueeze(0)
    
    scribble_labels = torch.cat([
        torch.ones((1,pos_points.shape[0]), device=device), 
        torch.zeros((1,neg_points.shape[0]), device=device)
    ], dim=1)
    
    return scribble_points, scribble_labels

