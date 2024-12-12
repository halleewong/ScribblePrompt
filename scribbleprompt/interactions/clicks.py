import warnings
from typing import Optional, Union, Tuple
import numpy as np
import torch
import cv2
import kornia.augmentation as KA

import os
os.environ['NEURITE_BACKEND'] = 'pytorch'

class RandomClick:
    """
    Generates a variable number of clicks with (x,y) coordinates by randomly 
     sampling clicks uniformly from a given mask
    """
    def __init__(self, 
                 min_sep: int = 0,
                 return_xy: bool = True, 
                 train = True, 
                 show: bool = False,
                 ):
        assert min_sep >= 0
        self.min_sep = min_sep
        self.return_xy = return_xy
        # If training, need to make sure same number of points are sampled 
        # per image for SAM dataloader 
        self.train = train 
        self.show = show

    @property
    def attrs(self):
        return {
            "min_sep": self.min_sep,
            "return_xy": self.return_xy,
            "train": self.train,
        }

    def _sample_coords(self, seg, coords, n_clicks=1):
        
        device = seg.device

        if len(coords) > 0:
            # Sample click from the mask
            idx = torch.randint(0, coords.shape[0], size=(n_clicks,), device=device)
            click_coord = coords[idx,:]
        else:
            if self.train:
                warnings.warn("No more points to sample from. Sampling uniformly at random")
                # Sample a click uniformly at random from the entire image
                H,W = seg.shape[-2:]
                click_coord = torch.stack([
                    torch.randint(0, W, size=(n_clicks,), device=device),
                    torch.randint(0, H, size=(n_clicks,), device=device)
                    ], dim=-1)
            else:
                return None, None

        # Get the label of the click
        click_label = seg[:, click_coord[:,0], click_coord[:,1]].flatten()
        click_coord = torch.stack((click_coord[:,1], click_coord[:,0] ), axis=-1)

        # shape n x 2, n 
        return click_coord, click_label
    

    def _sample_click(self, seg, mask=None, n_clicks=1):
        """
        Sample a click from a (single) mask if possible, otherwise sample 
         uniformly at random
        Args:
            seg (torch.Tensor): 1xHxW ground truth segmentation in [0,1] 
                used to determine whether the clicks are positive or negative
            mask (torch.Tensor): 1xHxW mask. Clicks will be samples from non-zero 
                pixels in this mask. If None, sample from the seg mask
        """
        assert seg.ndim == 3, f"Seg must be 1xHxW: {seg.shape}"

        # If no error region mask is provided, sample from the label
        if mask is None:
            mask = seg

        assert len(mask.shape) == 3, f"Mask must be 1xHxW: {mask.shape}"

        coords = torch.nonzero(mask.squeeze())

        if self.min_sep == 0:
            return self._sample_coords(seg, coords, n_clicks=n_clicks)
        else:
            click_coord_lst = []
            click_label_lst = []
            for _ in range(n_clicks):
                click_coord, click_label = self._sample_coords(seg, coords, n_clicks=1)
                if click_coord is None:
                    break
                click_coord_lst.append(click_coord)
                click_label_lst.append(click_label)
                if self.return_xy:
                    y,x = click_coord[-1,:]
                else:
                    x,y = click_coord[-1,:]
                # Remove points near the click
                idx = (coords[:,0] - x)**2 + (coords[:,1] - y)**2 >= self.min_sep**2
                coords = coords[idx,:]

            if len(click_coord_lst) == 0:
                return None, None
            
            click_coord = torch.concat(click_coord_lst, dim=0)
            click_label = torch.concat(click_label_lst, dim=0)

            if self.show:
                import neurite as ne
                from ..plot import show_points
                fig,axes = ne.plot.slices([seg.cpu().squeeze(), mask.cpu().squeeze()],
                               titles=["Seg","Mask"], show=False)
                for ax in axes:
                    show_points(click_coord.cpu(), click_label.cpu(), ax)
                plt.show()

            # shape n x 2, n 
            return click_coord, click_label

    def sample_click(self, seg, mask=None, n_clicks=1):
        """
        Sample a click for each example in a batch
        """
        if len(seg.shape) == 3:
            return self._sample_click(seg, mask, n_clicks=n_clicks)

        elif len(seg.shape) == 4:
            batch_size = seg.shape[0]
            
            click_coord_lst = []
            click_label_lst = []
            for i in range(batch_size):
                if mask is not None:
                    click_coord, click_label = self._sample_click(seg[i,...], mask[i,...], n_clicks=n_clicks)
                else:
                    click_coord, click_label = self._sample_click(seg[i,...], mask=None, n_clicks=n_clicks)

                if click_coord is not None:
                    click_coord_lst.append(click_coord)
                    click_label_lst.append(click_label)

            if len(click_coord_lst) == 0:
                return None, None
            else:
                click_coord = torch.stack(click_coord_lst, dim=0)
                click_label = torch.stack(click_label_lst, dim=0)

            # shapes: b x n x 2 and b x n
            return click_coord, click_label

    def __call__(self, seg: torch.Tensor, mask: Optional[torch.Tensor] = None, n_clicks: Optional[int] = None):
        """
        Args:
            seg: (1,H,W) ground truth segmentation in {0,1} (used to label clicks as pos/neg)
            mask: (1,H,W) mask to sample clicks from in {-1,0,1}
        Returns:
            click_coord: (n,2) np.array coordinates of the click
            click_label: (n,) np.array label of the click

        """
        assert seg.dtype == torch.int32, f"Seg must be int32. Currently {seg.dtype}"
        if mask is not None:
            assert mask.dtype == torch.int32, f"Mask must be int32. Currently {mask.dtype}"

        click_coord, click_label = self.sample_click(seg, mask, n_clicks=n_clicks)
        # shapes: b x n x 2 and b x n
        return click_coord, click_label


class ComponentCenterClick(RandomClick):
    """
    Click in the center of each conecting components up to the maximum number 
     of clicks in order from largest to smallest component

    Args:
        deterministic (bool): if True, place clicks deterministically from the 
            largest component to smallest component. Set True during evaluation
        background (bool): whether to include the background when sampling from 
            the label map (only used when mask=None)
    """
    def __init__(self, 
                 background: bool = False,
                 min_sep: int = 0,
                 return_xy: bool = True, 
                 train = True, 
                 show: bool = False,
                 ):
        super().__init__(min_sep=min_sep, return_xy=return_xy, train=train, show=show)
        self.deterministic = (not self.train)
        self.background = background

    @property
    def attrs(self):
        return {
            "min_sep": self.min_sep,
            "return_xy": self.return_xy,
            "train": self.train,
            "background": self.background,
        }

    def get_center_click(self, mask: np.array, dt_map: np.array):
        """
        Sample a center click according to a masked distance map
        Args:
            mask: binary mask
        """
        # Mask off the distance transform map (e.g. get a dt for a single component)
        mask_dt = mask * dt_map

        # Find center points
        coords_y, coords_x = np.where(mask_dt == mask_dt.max())  
            
        # If there are multiple points fartherst from the boundary, sample one at random
        idx = torch.randint(0, len(coords_x), size=(1,1), device='cpu').item()

        return np.array([coords_y[idx], coords_x[idx]])
        
    def sample_single_click(self, seg: torch.Tensor, mask: Optional[torch.Tensor] = None, n_clicks: int = 1):
        """
        Args:
            seg (torch.Tensor): 1xHxW ground truth segmentation in [0,1]
            mask (torch.Tensor): 1xHxW error mask := (pred - seg) in {-1,0,+1} (can also be seg-pred doesn't matter)
            n_clicks: the maximum number of clicks (1 per component)
        """
        assert len(seg.shape) == 3, "Seg must be 1xHxW"

        include_background = False
        if mask is None:
            mask = seg
            if self.background:
                include_background = True

        if len(mask.shape)==4:
            mask = mask.squeeze().unsqueeze(0)

        assert len(mask.shape) == 3, "Mask must be 1xHxW " + str(mask.shape)
        device = seg.device

        if (torch.abs(mask).sum() == 0) and not self.background:
            # If the error region is empty or the label map is empty
            if self.deterministic:
                warnings.warn("Provided error region or label map is empty. No new clicks")
                return None, None
            else:
                warnings.warn("Provided error region or label map is empty. Sampling uniformly at random")
                click_coord = torch.randint(0, mask.shape[-1], size=(n_clicks,2), device=device)
                click_label = seg[:, click_coord[:,0], click_coord[:,1]].flatten()
        else:
            if include_background:
                mask[ mask == 0 ] = -2.0

            # Get all the components in the mask
            components, areas = get_components(mask.cpu(), background=include_background, return_area=True, show=False)
            n_components = len(areas)

            # Calculate distance transform
            mask_dt = get_combined_dt(mask, background=include_background)
            
            random_clicks = 0
            if n_components < n_clicks:
                if self.train:
                    random_clicks = n_clicks - n_components
                    warnings.warn(f"Not enough components in the error region. Sampling {random_clicks} random clicks")
                else:
                    warnings.warn(f"Not enough components in the error region. Only sampling {n_components} instead of {n_clicks}")     
                
            n_clicks = min(n_clicks, n_components)

            coords_lst = []

            if self.deterministic or n_clicks == n_components:
                # Do clicks from largest to smallest component 
                sampled_labels = np.argsort(-np.array(areas))[:n_clicks]
            else:
                # Sample clicks from components proportional to area
                sampled_labels = np.random.choice(len(areas), size=n_clicks, p=areas/sum(areas), replace=False)

            for label in sampled_labels:
                # Get the point with maximum dt for that component
                binary_mask = (components==label+1).astype(np.int32)
                coords = self.get_center_click(binary_mask, mask_dt)
                coords_lst.append(coords)

            coords_y, coords_x = np.array(coords_lst).T # join list of tuples
            # Get labels for clicks
            click_label = seg[:, coords_y, coords_x].flatten()

            # Handle when n_clicks = 1 vs. > 1
            if len(coords_x.shape) <= 1:
                coords_x = coords_x[:,None]
                coords_y = coords_y[:,None]

            if self.return_xy:
                click_coord = torch.tensor(np.concatenate([coords_x, coords_y], 1), device=device)
            else:
                click_coord = torch.tensor(np.concatenate([coords_y, coords_x], 1), device=device)

            if random_clicks > 0:
                # shapes: n x 2 and n
                rand_coords, rand_labels = super().sample_single_click(seg=seg, mask=mask, n_clicks=random_clicks)
                click_coord = torch.cat((click_coord, rand_coords), dim=0)
                click_label = torch.cat((click_label, rand_labels), dim=0)

            if self.show:
                
                import neurite as ne
                from ..plot import show_points
                import matplotlib.pyplot as plt
                fig,axes = ne.plot.slices(
                    [seg.cpu().squeeze(), mask.cpu().squeeze(), components.squeeze(), mask_dt.squeeze()],
                    titles=["Seg", "Mask", "Components", "Distance Transform"], 
                    cmaps=["gray","gray","viridis","gray"],
                    show=False, do_colorbars=True
                )
                for ax in axes:
                    show_points(click_coord.cpu(), click_label.cpu(), ax)
                plt.show()

        # shapes: n x 2 and n
        return click_coord, click_label


class RandBorderClick(RandomClick):
    """
    Sample negative clicks from extreme points in a random width boundary region
    Note: can provided mask := 1 - seg to sample from the label boundary region
    """
    def __init__(self, 
                 blur_kernel_size: int = 33, 
                 blur_sigma: Union[float,Tuple[float]] = (5.0, 20.0),
                 min_sep: int = 8,
                 return_xy: bool = True, 
                 train = True, 
                 show: bool = False,
                 ):
        super().__init__(min_sep=min_sep, return_xy=return_xy, train=train, show=show)
        # Blur settings
        self.blur_fn = KA.RandomGaussianBlur(
            kernel_size=(blur_kernel_size, blur_kernel_size), sigma=blur_sigma, p=1., keepdim=True
        )
        self._show = show
        self.blur_kernel_size = blur_kernel_size
        self.blur_sigma = blur_sigma
    
    @property
    def attrs(self):
        return super().attrs() + {
            "blur_kernel_size": self.blur_kernel_size,
            "blur_sigma": self.blur_sigma,  
        } 
    
    def _sample_click(self, seg, mask=None, n_clicks=1, img=None):
        """
        Sample clicks for single segmentation map
        """
        assert len(seg.shape)==3, f"seg must be 1 x h x w. currently {seg.shape}"

        if mask is None:
            mask = seg

        assert len(mask.shape)==3, f"mask must be 1 x h x w. currently {mask.shape}"
        rev_mask = (1 - mask).float()

        # Get a random width boundary region
        blur_mask = self.blur_fn(rev_mask)
        corrected_blur_mask = torch.maximum(blur_mask, rev_mask) # Set pixels in the area we don't want to be 1

        # Sample two cutoffs 
        min_bs = corrected_blur_mask.min().cpu()
        max_bs = (mask*corrected_blur_mask).max().cpu()
        thresh1,thresh2 = np.random.uniform(min_bs, max_bs, size=2)

        if thresh1 > thresh2:
            boundary = ((thresh2 <= corrected_blur_mask)&(corrected_blur_mask < thresh1)).float()
        else:
            boundary = ((thresh1 <= corrected_blur_mask)&(corrected_blur_mask < thresh2)).float()

        click_coord, click_label = super()._sample_click(seg=seg, mask=boundary, n_clicks=n_clicks)

        if self._show:
            import neurite as ne
            import matplotlib.pyplot as plt
            from ..plot import show_points, show_mask

            if img is not None:
                fig,axes = ne.plot.slices(
                    [img.cpu().numpy()]+[x.cpu().numpy() for x in [seg, 1-rev_mask, 1-blur_mask, 1-corrected_blur_mask, boundary, seg]], 
                    ["Image", "GT Segmentation", "Input Mask", "Blurred Mask", "Corrected Blurred Mask", 
                    f"Boundary Region", "Boundary Region (overlay)"], 
                    show=False, width=20)
            else:
                fig,axes = ne.plot.slices(
                    [x.cpu().numpy() for x in [seg, rev_mask, blur_mask, corrected_blur_mask, boundary, seg]], 
                    ["Binary Segmentation", "1 - Input Mask", "Blurred Mask", "Corrected Blurred Mask", 
                    f"Boundary Region", "Boundary Region (overlay)"], 
                    show=False)
            for ax in axes[-2:]:
                show_points(click_coord.cpu(), click_label.cpu(), ax)
            show_mask(boundary.cpu(), axes[-1])
            plt.show()
    
        # shapes: b x n x 2 and b x n
        return click_coord, click_label