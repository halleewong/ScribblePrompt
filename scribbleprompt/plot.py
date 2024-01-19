import numpy as np
import matplotlib.pyplot as plt

def show_scribbles(mask, ax, alpha=0.5):
    """
    Overlay positive scribbles (green) and negative (red) scribbles 
    Args:
        mask: 1 x (C) x H x W or 2 x (C) x H x W
        ax: matplotlib axis
        alpha: transparency of the overlay
    """
    mask = mask.squeeze() # 2 x H x W
    if len(mask.shape)==2:
        # If there's only channel of scribbles, overlay the scribbles in blue
        h, w = mask.shape
        color = np.array([30/255, 144/255, 255/255, alpha])
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)
    elif len(mask.shape)==3:
        # If there are 2 channels, take the first channel as positive scribbles (green) and the second channel as negaitve scribbles (red)
        c, h, w = mask.shape
        color = np.array([0, 1, 0, alpha])
        mask_image = mask[0,...].reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)
        color = np.array([1, 0, 0, alpha])
        mask_image = mask[1,...].reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)
    else:
        raise ValueError("mask must be 2 or 3 dimensional")