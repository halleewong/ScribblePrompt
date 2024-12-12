import os
from typing import Optional, List, Tuple, Dict, Any, Union, Literal
import numpy as np
import torch
from itertools import chain,combinations

from .utils import _as_single_val, chance, warn_in_range

# -----------------------------------------------------------------------------
# Base class for sampling prompts
# -----------------------------------------------------------------------------

class SuperFlexiblePrompt:
    """
    Flexible prompt generator class
    """
    def __init__(self,
                # Balance of different prompts
                prob_bbox: float = 0.5,
                prob_scribble: float = 0.0,
                prob_click: float = 0.0,
                single_correction_type: bool = True,
                partition_correction_scribbles: bool = True,
                single_init_prompt_type: bool = False,
                atleast_one_init_prompt: bool = True,
                # First prompt round parameters
                init_pos_click: Union[int,List[int]] = 1, 
                init_neg_click: Union[int,List[int]] = 1,
                init_pos_scribble: Union[int,List[int]] = 1, 
                init_neg_scribble: Union[int,List[int]] = 1,
                # Click parameters for subsequent prompts
                cutoff: float = 0.5,
                correction_clicks: Union[int,List[int]] = 1,
                correction_scribbles: Union[int,List[int]] = 1,
                # Click parameters
                init_pos_click_generators: Optional[Union[callable, List[callable]]] = None, 
                init_neg_click_generators: Optional[Union[callable, List[callable]]] = None, 
                correction_click_generators: Optional[Union[callable, List[callable]]] = None, 
                # Scribble parameters
                init_pos_scribble_generators: Optional[Union[callable, List[callable]]] = None,
                init_neg_scribble_generators: Optional[Union[callable, List[callable]]] = None,
                correction_scribble_generators: Optional[Union[callable, List[callable]]] = None,
                # BBox parameters
                box_generators: Optional[Union[callable, List[callable]]] = None,
                # Previous mask as input
                prev_mask: Union[float,int,bool] = True, 
                # Debug
                debug: bool = False,
                train: bool = True,
                ):
        self.prob_bbox = prob_bbox
        self.prob_scribble = prob_scribble
        self.prob_click = prob_click
        self.cutoff = cutoff

        self.single_correction_type = single_correction_type
        if self.single_correction_type:
            assert (self.prob_scribble + self.prob_click) > 0,\
                 "single_correction_type requires prob_scribble or prob_click > 0"
            total_prob = self.prob_scribble + self.prob_click
            self.correction_scribble_prob = self.prob_scribble/total_prob
        self.partition_correction_scribbles = partition_correction_scribbles
        self.train = train

        # Initial prompts
        self.single_init_prompt_type = single_init_prompt_type
        self.atleast_one_init_prompt = atleast_one_init_prompt

        # Subsequent prompts
        self.correction_clicks = correction_clicks
        self.correction_scribbles = correction_scribbles
    
        # Number of prompts to generate initially
        self.init_pos_click = init_pos_click
        self.init_neg_click = init_neg_click
        self.init_pos_scribble = init_pos_scribble
        self.init_neg_scribble = init_neg_scribble

        # Generators
        self.init_pos_click_generators = init_pos_click_generators
        self.init_neg_click_generators = init_neg_click_generators
        self.correction_click_generators = correction_click_generators

        self.init_pos_scribble_generators = init_pos_scribble_generators
        self.init_neg_scribble_generators = init_neg_scribble_generators
        self.correction_scribble_generators = correction_scribble_generators

        self.box_generators = box_generators

        # For downstream compatability
        self.num_entries = 1

        # Conditioning on previous mask
        if isinstance(prev_mask, bool):
            # If True -> 1, False -> 0 (for backwards compatibility)
            self.prev_mask = float(prev_mask)
        else:
            assert prev_mask>=0 and prev_mask<=1, "prev_mask probability must be between 0 and 1"
            self.prev_mask = prev_mask
            
        self.debug = debug

        if self.atleast_one_init_prompt:
            # Sample all non-empty combinations equally
            prompt_types = []
            if prob_bbox > 0:
                prompt_types.append(0)
            if prob_click > 0:
                prompt_types.append(1)
            if prob_scribble > 0:
                prompt_types.append(2)
            self.all_combos = list(
                chain.from_iterable([combinations(prompt_types, r) for r in range(1,len(prompt_types)+1)])
            )

    def subsequent_prompt(self,
                        mask_pred: torch.Tensor, # to use as an input to the model in the next iteration
                        binary_mask_pred: torch.Tensor, # to use for calculating error region & sampling interactions
                        prev_input: dict, # previous prompt dictionary
                        new_prompt : bool = True, # whether to sample new (corrective) interactions
                        show: bool = False,
                        ):
        """
        Sample clicks or scribbles from the error region
        """
        if new_prompt:
            assert binary_mask_pred.dtype == torch.int32, "binary_mask_pred must be in torch.int32"

            seg = prev_input.get("seg") # 1xHxW
            # Make sure mask is binary
            binary_seg = (seg > self.cutoff).int()
            error_region = (binary_seg - binary_mask_pred).int() # 1xHxW in [-1,+1]

            if self.single_correction_type:
                do_scribble = chance(self.correction_scribble_prob)
                do_click = (not do_scribble)
            else:
                # Sample whether to do clicking and scribbling independently
                do_scribble = chance(self.prob_scribble)
                do_click = chance(self.prob_click)

            if do_scribble:

                n_scribble = _as_single_val(self.correction_scribbles)

                # error regions
                false_neg = torch.clamp(error_region, min=0)
                false_pos = -torch.clamp(error_region, max=0)

                if self.partition_correction_scribbles:
                    if self.train:
                        # Randomly divide the scribbles between FP and FN region
                        pos_n_scribble = np.random.randint(0,n_scribble+1)
                        neg_n_scribble = n_scribble - pos_n_scribble
                    else:
                        if false_neg.sum() == 0:
                            neg_n_scribble = 0
                            pos_n_scribble = n_scribble
                        elif false_pos.sum() == 0:
                            pos_n_scribble = 0
                            neg_n_scribble = n_scribble
                        else:
                            # Allocate scribbles proportional to areas of FP/FN regions
                            p = (false_pos.sum() / (false_neg.sum() + false_pos.sum())).cpu().item()
                            # print("p",p)
                            pos_n_scribble = sum([chance(p) for _ in range(n_scribble)])
                            neg_n_scribble = n_scribble - pos_n_scribble
                        
                else:
                    pos_n_scribble = n_scribble
                    neg_n_scribble = n_scribble

                # FN scribble
                pos_scribble = self.sample_scribble(false_neg, n_scribbles=neg_n_scribble, type="correction")         

                # FP scribble
                neg_scribble = self.sample_scribble(false_pos, n_scribbles=pos_n_scribble, type="correction")

                # Combine with previous scribbles
                new_scribble = torch.cat((pos_scribble, neg_scribble), axis=1)
                prev_input["scribble"] = prev_input.get("scribble",0) + new_scribble

            if do_click:

                n_click = _as_single_val(self.correction_clicks)
                # Sample clicks from the error region
                click_coord, click_label = self.sample_click(binary_seg, mask=error_region, n_clicks=n_click, type="correction")

                if click_coord is not None:
                    if "point_coords" not in prev_input:
                        # If the first prompt was a bounding box
                        prev_input["point_coords"] = click_coord
                        prev_input["point_labels"] = click_label
                    else:
                        old_click_coord = prev_input.get("point_coords")
                        old_click_label = prev_input.get("point_labels")

                        prev_input.update(
                            {
                                "point_coords": torch.cat((old_click_coord, click_coord), axis=-2),
                                "point_labels": torch.cat((old_click_label, click_label), axis=-1),
                            }
                        )

        # Update the mask
        if chance(self.prev_mask):
            prev_input["mask_input"] = mask_pred
        
        if self.debug:
            warn_in_range(prev_input.get("point_coords",torch.zeros(1)), name='point_coords in subsequent_prompt')
            warn_in_range(prev_input.get("point_labels",torch.zeros(1)), name='point_labels in subsequent_prompt')
            warn_in_range(prev_input.get("scribble",torch.zeros(1)), name='mask_input in subsequent_prompt')
            warn_in_range(prev_input.get("mask_input",torch.zeros(1)), name='mask_input in subsequent_prompt')

        return prev_input
      

    def sample_box(self, seg):
        """
        Sample a bounding box generator and apply to the given args
        """
        if isinstance(self.box_generators, list):
            return self.box_generators[np.random.randint(0,len(self.box_generators))](seg)
        else:
            return self.box_generators(seg)
        

    def sample_click(self, seg, mask, n_clicks, type: Literal["init_pos","init_neg","correction"]):
        """
        Sample a click generator and apply to the given args
        """
        if n_clicks == 0:
            return None, None

        if type == "init_pos":
            click_generators = self.init_pos_click_generators
        elif type == "init_neg":
            click_generators = self.init_neg_click_generators
        else:
            click_generators = self.correction_click_generators

        if isinstance(click_generators, list):
            click_fn = click_generators[np.random.randint(0,len(click_generators))]
        else:
            click_fn = click_generators

        return click_fn(seg, mask, n_clicks)
        

    def sample_scribble(self, mask, n_scribbles, type: Literal["init_pos","init_neg","correction"]):
        """
        Sample a scribble generator and apply to the given args
        """
        if (n_scribbles == 0) or mask.sum()==0:
            return torch.zeros(mask.shape, dtype=torch.float32, device=mask.device)
        
        if type == "init_pos":
            scribble_generators = self.init_pos_scribble_generators
        elif type == "init_neg":
            scribble_generators = self.init_neg_scribble_generators
        else:
            scribble_generators = self.correction_scribble_generators

        if isinstance(scribble_generators, list):
            scribble_fn = scribble_generators[np.random.randint(0,len(scribble_generators))]
        else:
            scribble_fn = scribble_generators

        return scribble_fn(mask.float(), n_scribbles)

    def __call__(self, 
                 img: torch.Tensor, 
                 seg: torch.Tensor, 
                 prob_bbox: Optional[float] = None, 
                 prob_scribble: Optional[float] = None, 
                 prob_click: Optional[float] = None
                 ) -> dict:
        """
        Sample first iteration input interactions (SAM compatible format)
        Args:
            img (torch.Tensor): image tensor 3xHxW
            seg (torch.Tensor): segmentation mask 1xHxW
            prob_bbox (float): overide probability of sampling a bounding box
            prob_scribble (float): overide probability of sampling a scribble
            prob_click (float): overide probability of sampling a click
        """
        inputs = {
            "img": img,
            "seg": seg,
        }
        prob_bbox = self.prob_bbox if prob_bbox is None else prob_bbox
        prob_scribble = self.prob_scribble if prob_scribble is None else prob_scribble
        prob_click = self.prob_click if prob_click is None else prob_click
        if prob_bbox + prob_scribble + prob_click == 0:
            return inputs

        binary_seg = (seg > self.cutoff).int()

        if self.single_init_prompt_type:
            # Sample only one prompt type
            probs = np.array([prob_bbox, prob_click, prob_scribble])
            normalized_prob = probs / probs.sum()
            prompt_type = np.random.choice([0,1,2], p=normalized_prob, size=1)
            if prompt_type == 0:
                do_bbox, do_click, do_scribble = True, False, False
            elif prompt_type == 1:  
                do_bbox, do_click, do_scribble = False, True, False
            else:
                do_bbox, do_click, do_scribble = False, False, True
        elif self.atleast_one_init_prompt:
            # Sample all non-empty combinations equally
            combo = self.all_combos[np.random.randint(0,len(self.all_combos))]
            do_bbox = (0 in combo)
            do_click = (1 in combo)
            do_scribble = (2 in combo)
        else:
            # Flip coins independently for each prompt type
            do_bbox = chance(prob_bbox)
            do_scribble = chance(prob_scribble)
            do_click = chance(prob_click)

        if do_bbox:
            # Bounding box
            inputs["box"] = self.sample_box(binary_seg)

        if do_scribble:

            n_pos_scribble = _as_single_val(self.init_pos_scribble)
            n_neg_scribble = _as_single_val(self.init_neg_scribble)

            # Scribbles
            pos_scribble = self.sample_scribble(binary_seg, n_scribbles=n_pos_scribble, type="init_pos")
            neg_scribble = self.sample_scribble(1-binary_seg, n_scribbles=n_neg_scribble, type="init_neg")
            inputs["scribble"] = torch.cat((pos_scribble, neg_scribble), axis=1)
        
        if do_click:

            n_pos_click = _as_single_val(self.init_pos_click)
            n_neg_click = _as_single_val(self.init_neg_click)

            # Clicks
            click_coord_lst = []
            click_label_lst = []

            # Sample click from label
            click_coord, click_label = self.sample_click(seg=binary_seg, mask=binary_seg, n_clicks=n_pos_click, type="init_pos")
            if click_coord is not None:
                click_coord_lst.append(click_coord)
                click_label_lst.append(click_label)

            # Sample additional negative clicks from background
            click_coord, click_label = self.sample_click(seg=binary_seg, mask=1-binary_seg, n_clicks=n_neg_click, type="init_neg")
            if click_coord is not None:
                click_coord_lst.append(click_coord)
                click_label_lst.append(click_label)

            if len(click_coord_lst) > 0:
                inputs["point_coords"] = torch.cat(click_coord_lst, axis=-2) 
                inputs["point_labels"] = torch.cat(click_label_lst, axis=-1) 
        
        return inputs

# -----------------------------------------------------------------------------
# Child class for geneating prompts and embedding in scribbleprompt-unet input format
# -----------------------------------------------------------------------------

class FlexiblePromptEmbed(SuperFlexiblePrompt):
    """
    Args:
        prev_mask: whether to update the previous mask in (channel 5)
    """
    def __init__(self,
                # Functions for embedding clicks and bounding boxes
                click_embed: Any = None,
                bbox_embed: Any = None,
                # For calculating error regions
                from_logits: float = True,
                sam: bool = False,
                **kwargs
     ):
        """
        Args:
            prev_mask: can be a bool or the probability of including the previous mask 
            (i.e. best mask chosen by user) as a model input
        """
        if "click_generators" in kwargs.keys():
            click_generators = kwargs.pop("click_generators")
            kwargs.update({
                "init_pos_click_generators": click_generators,
                "init_neg_click_generators": click_generators,
                "correction_click_generators": click_generators,
                })
        if "scribble_generators" in kwargs.keys():
            scribble_generators = kwargs.pop("scribble_generators")
            kwargs.update({
                "init_pos_scribble_generators": scribble_generators,
                "init_neg_scribble_generators": scribble_generators,
                "correction_scribble_generators": scribble_generators,
                })
        SuperFlexiblePrompt.__init__(self, **kwargs)
        self.click_embed = click_embed
        self.bbox_embed = bbox_embed
        self.from_logits = from_logits
        self.sam = sam
    
    def update_embed(self, prev_input: Dict, new_input: Dict = None, mask_pred: Optional[torch.Tensor] = None, new_prompt=True):
        """
        Args:
            mask_pred: a previous mask prediction to condition on

        Assumes x is (B x n x C x H x W)
        """
        if not self.sam:

            if new_prompt:
                # Update click embedding
                if "point_coords" in new_input:
                    click_embed = self.click_embed(
                        point_coords=new_input['point_coords'],
                        point_labels=new_input['point_labels'],
                        shape=tuple(prev_input['img'].shape[-2:])
                        )
                else:
                    click_embed = torch.zeros(
                        (prev_input['img'].shape[0], 2) + tuple(prev_input['img'].shape[-2:]), device=prev_input['img'].device)
                
                # Get scribble embedding
                scribble_embed = prev_input.get("scribble", 
                    torch.zeros((prev_input['img'].shape[0], 2) + tuple(prev_input['img'].shape[-2:]), device=prev_input['img'].device)
                )

                if self.debug:
                    warn_in_range(click_embed, name="click_embed in update_embed")
                    warn_in_range(scribble_embed, name="scribble_embed in update_embed")

                scribble_click_embed = torch.clamp(scribble_embed + click_embed, min=0.0, max=1.0)
            else: 
                scribble_click_embed = torch.cat([
                    prev_input['x'].select(-3, 2), # pos clicks
                    prev_input['x'].select(-3, 2) # neg clicks
                ], dim=-3)

            img = prev_input['img']
            box_embed = prev_input.get('x').select(-3, 1).unsqueeze(1)

            mask_input = prev_input.get("mask_input", torch.zeros(img.shape, device=img.device))

            # Re-assemble input channels with new click embedding
            x = torch.cat((img, box_embed, scribble_click_embed, mask_input), dim=-3)         

            if self.debug:
                warn_in_range(x, name="x in update_embed")

            new_input['x'] = x

        return new_input

    def subsequent_prompt(self,
                          mask_pred: torch.Tensor, # to use as an input to the model in the next iteration
                          binary_mask_pred: Optional[torch.Tensor] = None, # to use for calculating error region & sampling click
                          prev_input: dict = dict(),
                          new_prompt: bool = True
                          ):

        if binary_mask_pred is None:
            # Optional for compatibility with sam
            if self.from_logits:
                binary_mask_pred = (torch.sigmoid(mask_pred) > self.cutoff).int()
            else:
                binary_mask_pred = (mask_pred > self.cutoff).int()

        # Sample new prompts (click or scribble)
        new_input = super().subsequent_prompt(mask_pred, binary_mask_pred, prev_input, new_prompt=new_prompt)
        
        # Update model input
        new_input_with_embed = self.update_embed(prev_input, new_input, mask_pred, new_prompt=new_prompt)

        return new_input_with_embed
        
    def embed(self, inputs):
        """
        inputs are in SAM format except for scribble
        """
        img = inputs['img']
        device = img.device if img is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        shape = tuple(img.shape[-2:])
        # Embed these prompts for the cross convolution model

        if inputs.get("box") is not None:
            # Embed bounding box
            # Input: B x 1 x 4 
            # Output: B x 1 x H x W
            box_embed = self.bbox_embed(inputs['box'], shape=shape, device=device)
        else:
            box_embed = torch.zeros(img.shape, device=device)

        if inputs.get("point_coords") is not None:
            # Embed points
            # B x 2 x H x W
            click_embed = self.click_embed(inputs['point_coords'], inputs['point_labels'], shape=shape)
        else:
            click_embed = torch.zeros((img.shape[0], 2) + shape, device=device)

        if inputs.get("scribble") is not None:
            # Combine scribbles with click embedding
            # B x 2 x H x W
            click_embed = torch.clamp(click_embed + inputs.get('scribble'), min=0.0, max=1.0)

        if inputs.get('mask_input', None) is not None:
            # Used in predictor GUI
            mask_input = inputs['mask_input']
        else:
            # Initialize empty channel for mask input
            mask_input = torch.zeros(img.shape, device=img.device)

        x = torch.cat((img, box_embed, click_embed, mask_input), dim=-3)
        # B x 5 x H x W

        return x
    
    def __call__(self, img: torch.Tensor, seg: torch.Tensor, **kwargs) -> dict:
        """
        Get first prompt
        """
        # Get first prompt
        inputs = super().__call__(img, seg, **kwargs)

        # Embed prompts
        if not self.sam:
            inputs['x'] = self.embed(inputs)
            inputs['y'] = seg[:,None,...].repeat(1, self.num_entries, 1, 1, 1).float()

        return inputs

