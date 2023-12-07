# ScribblePrompt

### [Project Page](https://scribbleprompt.csail.mit.edu) | [Demo](https://huggingface.co/spaces/halleewong/ScribblePrompt) | [Video](https://youtu.be/L8CiAoHzPUE)

Official implementation of "ScribblePrompt: Fast and Flexible Interactive Segmentation for any Medical Image" 

[Hallee E. Wong](https://halleewong.github.io/), [Marianne Rakic](https://mariannerakic.github.io/), [John V. Guttag](https://people.csail.mit.edu/guttag/), [Adrian V. Dalca](http://www.mit.edu/~adalca/)

## Overview

**ScribblePrompt** is an interactive segmentation tool that enables users to segment *unseen* structures in medical images using clicks, bounding boxes *and scribbles*.

## Try ScribblePrompt

* Online Gradio Demo: https://huggingface.co/spaces/halleewong/ScribblePrompt

## Models

We provide model weights for two versions of ScribblePrompt: 

* **ScribblePrompt Unet** with an efficient fully-convolutional architecture  

* **ScribblePrompt SAM** based on the [Segment Anything Model](https://github.com/facebookresearch/segment-anything)

Both models have been trained with iterative **click, bounding box and scribble interactions** on a diverse collection of 65 medical imaging datasets. 

## Installation

You can install `scribbleprompt` in two ways:

* With pip

```
pip install git+https://github.com/halleewong/ScribblePrompt.git
```

* Manually: cloning it and installing dependencies
```
git clone https://github.com/halleewong/ScribblePrompt
python -m pip install -r ./ScribblePrompt/requirements.txt
export PYTHONPATH="$PYTHONPATH:$(relpath ./ScribblePrompt)"
```
See [here](https://github.com/facebookresearch/segment-anything#installation) for Segment Anything's dependencies.

## Ongoing

[] Release jupyter notebook tutorial
[] Release scribble simulation code
[] Release segmentations labels collected using ScribblePrompt

## Acknowledgements

Code for ScribblePrompt SAM builds on [Segment Anything](https://github.com/facebookresearch/segment-anything) 

## License

This project is released under 

## Citation


