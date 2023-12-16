# ScribblePrompt

### [Project Page](https://scribbleprompt.csail.mit.edu) | [Paper](https://arxiv.org/abs/2312.07381) | [Demo](https://huggingface.co/spaces/halleewong/ScribblePrompt) | [Video](https://youtu.be/L8CiAoHzPUE)

Official implementation of "ScribblePrompt: Fast and Flexible Interactive Segmentation for any Medical Image" 

[Hallee E. Wong](https://halleewong.github.io/), [Marianne Rakic](https://mariannerakic.github.io/), [John Guttag](https://people.csail.mit.edu/guttag/), [Adrian V. Dalca](http://www.mit.edu/~adalca/)

## Updates

🚨 **This repo is under construction!** 🚨 Please check back for updates

## Overview

**ScribblePrompt** is an interactive segmentation tool that enables users to segment *unseen* structures in medical images using clicks, bounding boxes *and scribbles*.

## Try ScribblePrompt

* Online Gradio Demo: https://huggingface.co/spaces/halleewong/ScribblePrompt

## Models

We provide [checkpoints](https://www.dropbox.com/scl/fo/zl12obhnsqc2mq7ulviq9/h?rlkey=suaj632fd9aqd6c2gtajz1ywc&dl=0) for two versions of ScribblePrompt: 

* **ScribblePrompt-UNet** with an efficient fully-convolutional architecture  

* **ScribblePrompt-SAM** based on the [Segment Anything Model](https://github.com/facebookresearch/segment-anything)

Both models have been trained with iterative **click, bounding box and scribble interactions** on a diverse collection of 65 medical imaging datasets with both real and synthetic labels. 

## Installation

You can install `scribbleprompt` in two ways:

* **With pip**:

```
pip install git+https://github.com/halleewong/ScribblePrompt.git
```

* **Manually**: cloning it and installing dependencies
```
git clone https://github.com/halleewong/ScribblePrompt
python -m pip install -r ./ScribblePrompt/requirements.txt
export PYTHONPATH="$PYTHONPATH:$(realpath ./ScribblePrompt)"
```

## Getting Started

First, download the model [checkpoints](https://www.dropbox.com/scl/fo/zl12obhnsqc2mq7ulviq9/h?rlkey=suaj632fd9aqd6c2gtajz1ywc&dl=0) to `./checkpoints`.

To instantiate ScribblePrompt-UNet and make a prediction:
```
from scribbleprompt import ScribblePromptUNet

sp_unet = ScribblePromptUNet()

mask = sp_unet.predict(
    image,        # (B, 1, H, W) 
    point_coords, # (B, n, 2)
    point_labels, # (B, n)
    scribbles,    # (B, 2, H, W)
    box,          # (B, n, 4)
    mask_input,   # (B, 1, H, W)
) # -> (B, 1, H, W) 
```

To instantiate ScribblePrompt-SAM and make a prediction:
```
from scribbleprompt import ScribblePromptSAM

sp_sam = ScribblePromptSAM()

mask, img_features, low_res_logits = sp_sam.predict(
    image,        # (B, 1, H, W) 
    point_coords, # (B, n, 2)
    point_labels, # (B, n)
    scribbles,    # (B, 2, H, W)
    box,          # (B, n, 4)
    mask_input,   # (B, 1, 256, 256)
) # -> (B, 1, H, W), (B, 16, 256, 256), (B, 1, 256, 256)

```
`image` should have spatial dimensions $(H,W) = (128,128)$ and pixel values min-max normalized to the $[0,1]$ range. 

For ScribblePrompt-UNet, `mask_input` should be the logits from the previous prediction. For ScribblePrompt-SAM, `mask_input` should be `low_res_logits` from the previous prediction. 

## To Do

- [x] Release Gradio demo 
- [x] Release model code and weights
- [ ] Release jupyter notebook tutorial
- [ ] Release scribble simulation code
- [ ] Release segmentation labels collected using ScribblePrompt

## Acknowledgements

Code for ScribblePrompt SAM builds on [Segment Anything](https://github.com/facebookresearch/segment-anything) 

## Citation

If you find our work or any of our materials useful, please cite our paper:
```
@article{wong2023scribbleprompt,
  title={ScribblePrompt: Fast and Flexible Interactive Segmentation for Any Medical Image},
  author={Hallee E. Wong and Marianne Rakic and John Guttag and Adrian V. Dalca},
  journal={arXiv:2312.07381},
  year={2023},
}
```

## License

This project is released under the [Apache 2.0 License](https://github.com/halleewong/ScribblePrompt/blob/main/LICENSE) 




