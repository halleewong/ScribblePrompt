<a href=https://arxiv.org/abs/2312.07381><img src="https://img.shields.io/badge/arxiv-2312.07381-orange?logo=arxiv&logoColor=white"/></a>
<a href="https://huggingface.co/spaces/halleewong/ScribblePrompt"><img alt="Spaces" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue"></a>
<a href="https://colab.research.google.com/drive/14ExpVy3PjCCp4VzgTo27Yh_aLBafK8cX?usp=sharing"><img alt="Colab" src="https://colab.research.google.com/assets/colab-badge.svg"></a>

# ScribblePrompt

### [Project Page](https://scribbleprompt.csail.mit.edu) | [Paper](https://arxiv.org/abs/2312.07381) | [Online Demo](https://huggingface.co/spaces/halleewong/ScribblePrompt) | [Video](https://youtu.be/L8CiAoHzPUE)

Official implementation of [ScribblePrompt: Fast and Flexible Interactive Segmentation for any Biomedical Image](https://arxiv.org/abs/2312.07381) accepted at ECCV 2024

[Hallee E. Wong](https://halleewong.github.io/), [Marianne Rakic](https://mariannerakic.github.io/), [John Guttag](https://people.csail.mit.edu/guttag/), [Adrian V. Dalca](http://www.mit.edu/~adalca/)

## Updates

* (2024-12-31) Released example training code
* (2024-12-12) Released full prompt simulation code
* (2024-07-01) ScribblePrompt has been accepted to ECCV 2024!
* (2024-06-17) ScribblePrompt won the [Bench-to-Bedside Award](https://dca-in-mi.github.io/awards.html) at the CVPR 2024 DCAMI Workshop!
* (2024-04-16) Released [MedScribble](https://github.com/halleewong/ScribblePrompt/tree/main/MedScribble) -- a diverse dataset of segmentation tasks with scribble annotations
* (2024-04-15) An updated version of the paper is on arXiv!
* (2024-04-14) Added [Google Colab Tutorial](https://colab.research.google.com/drive/14ExpVy3PjCCp4VzgTo27Yh_aLBafK8cX?usp=sharing)
* (2024-01-19) Released scribble simulation code
* (2023-12-15) Released model code and weights 
* (2023-12-12) Paper and online demo released

## Overview

**ScribblePrompt** is an interactive segmentation tool that enables users to segment *unseen* structures in medical images using scribbles, clicks, *and* bounding boxes. 

![](https://github.com/halleewong/ScribblePrompt/blob/website/assets/gifs/total_segmentator.gif)
![](https://github.com/halleewong/ScribblePrompt/blob/website/assets/gifs/wbc.gif)
![](https://github.com/halleewong/ScribblePrompt/blob/website/assets/gifs/drive.gif)
![](https://github.com/halleewong/ScribblePrompt/blob/website/assets/gifs/buid.gif)
![](https://github.com/halleewong/ScribblePrompt/blob/website/assets/gifs/hipxray.gif)
![](https://github.com/halleewong/ScribblePrompt/blob/website/assets/gifs/acdc.gif)

## Try ScribblePrompt

* Interactive [online demo](https://huggingface.co/spaces/halleewong/ScribblePrompt) on Hugging Face Spaces
* See [Installation](https://github.com/halleewong/ScribblePrompt?tab=readme-ov-file#installation) and [Getting Started](https://github.com/halleewong/ScribblePrompt?tab=readme-ov-file#getting-started) for how to run the Gradio demo locally
* Jupyter notebook [colab tutorial](https://colab.research.google.com/drive/14ExpVy3PjCCp4VzgTo27Yh_aLBafK8cX?usp=sharing) using pre-trained models
* Jupyter notebook [tutorials](https://github.com/halleewong/ScribblePrompt/tree/main/notebooks) on training and the prompt generator code

## Models

We provide [checkpoints](https://www.dropbox.com/scl/fo/zl12obhnsqc2mq7ulviq9/h?rlkey=suaj632fd9aqd6c2gtajz1ywc&dl=0) for two versions of ScribblePrompt: 

* **ScribblePrompt-UNet** with an efficient fully-convolutional architecture  

* **ScribblePrompt-SAM** based on the [Segment Anything Model](https://github.com/facebookresearch/segment-anything)

Both models have been trained with iterative **scribbles, click, and bounding box interactions** on a diverse collection of 65 medical imaging datasets with both real and synthetic labels. 

## MedScribble Dataset

We release MedScribble, a dataset of multi-annotator scribble annotations for diverse biomedical image segmentation tasks, under [`./MedScribble`](https://github.com/halleewong/ScribblePrompt/tree/main/MedScribble). See [the readme](https://github.com/halleewong/ScribblePrompt/tree/main/MedScribble/README.md) for more info and [`./MedScribble/tutorial.ipynb`](https://github.com/halleewong/ScribblePrompt/tree/main/MedScribble/tutorial.ipynb) for a preview of the data.  

## Installation

You can install `scribbleprompt` in two ways:

* **With pip**:

```
# For basic inference
pip install "scribbleprompt @ git+https://github.com/halleewong/ScribblePrompt.git"

# For prompt simulation and training (additional dependencies)
pip install "scribbleprompt[training] @ git+https://github.com/halleewong/ScribblePrompt.git"
```

* **Manually**: cloning it and installing dependencies
```
git clone https://github.com/halleewong/ScribblePrompt
python -m pip install -r ./ScribblePrompt/requirements.txt
python -m pip install -r ./ScribblePrompt/requirements_training.txt
export PYTHONPATH="$PYTHONPATH:$(realpath ./ScribblePrompt)"
```

The following optional dependencies are necessary for the local Gradio app demo:
```
pip install gradio==3.40.1
```

## Getting Started

First, download the model [checkpoints](https://www.dropbox.com/scl/fo/zl12obhnsqc2mq7ulviq9/h?rlkey=suaj632fd9aqd6c2gtajz1ywc&dl=0) to `./checkpoints`.

To run an interactive demo locally: 
```
python demos/app.py
```

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
For best results, `image` should have spatial dimensions $(H,W) = (128,128)$ and pixel values min-max normalized to the $[0,1]$ range. 

For ScribblePrompt-UNet, `mask_input` should be the logits from the previous prediction. For ScribblePrompt-SAM, `mask_input` should be `low_res_logits` from the previous prediction. 

## Training

>Note: our training code requires the [pylot](https://github.com/JJGO/pylot) library. The inference code above does not.  We recommend installing via pip:
>```
>pip install git+https://github.com/JJGO/pylot.git@87191921033c4391546fd88c5f963ccab7597995
>```


The configuration settings for training are controlled by yaml config files. We provide two example configs in [`./configs`](https://github.com/halleewong/ScribblePrompt/tree/main/configs) for fine-tuning from the pre-trained ScribblePrompt-UNet weights as well as training from scratch on an example dataset.

To fine-tune ScribblePrompt-UNet from the pre-trained weights:
```
python scribbleprompt/experiment/unet.py -config finetune_unet.yaml 
```

To train a model from scratch:
```
python scribbleprompt/experiment/unet.py -config train_unet.yaml 
```

For a more in-depth tutorial see [`./notebooks/training.ipynb`](https://github.com/halleewong/ScribblePrompt/tree/main/notebooks/training.ipynb).

## To Do

- [x] Release Gradio demo 
- [x] Release model code and weights
- [x] Release jupyter notebook tutorial
- [x] Release scribble simulation code
- [x] Release MedScribble dataset
- [x] Release training code
- [ ] Release segmentation labels collected using ScribblePrompt

## Acknowledgements

* Our training code builds on the [`pylot`](https://github.com/JJGO/pylot) library for deep learning experiment management. We also make use of data augmentation code originally developed for [UniverSeg](https://github.com/JJGO/UniverSeg). Thanks to [@JJGO](https://github.com/JJGO) for sharing this code! 

* We use functions from [voxsynth](https://github.com/dalcalab/voxynth) for applying random deformations during scribble simulation 

* Code for ScribblePrompt-SAM builds on [Segment Anything](https://github.com/facebookresearch/segment-anything). Thanks to Meta AI for open-sourcing the model. 


## Citation

If you find our work or any of our materials useful, please cite our paper:
```
@article{wong2024scribbleprompt,
  title={ScribblePrompt: Fast and Flexible Interactive Segmentation for Any Biomedical Image},
  author={Hallee E. Wong and Marianne Rakic and John Guttag and Adrian V. Dalca},
  journal={European Conference on Computer Vision (ECCV)},
  year={2024},
}
```

## License

Code for this project is released under the [Apache 2.0 License](https://github.com/halleewong/ScribblePrompt/blob/main/LICENSE) 




