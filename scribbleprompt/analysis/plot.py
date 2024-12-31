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


# -----------------------------------------------------------------------------
# Functions adapted from https://github.com/facebookresearch/segment-anything/blob/main/notebooks/predictor_example.ipynb
# -----------------------------------------------------------------------------

def show_mask(mask, ax, random_color=False, alpha=0.5):
    """
    Overlay mask
    """
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([alpha])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, alpha])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=150):
    """
    Overlay positive and negative clicks. Points should be xy coordinates
    Args:
        coords: (N, 2) array of xy coordinates
        labels: (N,) array of labels (0 or 1)
        ax: matplotlib axis
        marker_size: size of the markers
    """
    pos_points = coords[labels==1]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='.',
                s=marker_size, edgecolor='white', linewidth=1.25)
    neg_points = coords[labels==0]
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='.',
               s=marker_size, edgecolor='white', linewidth=1.25)
    
def show_boxes(box, ax, lw=3):
    """
    Overlay bounding boxes in yellow
    """
    box = box.squeeze()
    if len(box.shape)==2:
        for i in range(box.shape[0]):
            show_boxes(box[i,...], ax, lw)
    else:
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='gold', facecolor=(0,0,0,0), lw=lw))