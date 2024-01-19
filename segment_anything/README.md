# Segment Anything

Code from: https://github.com/facebookresearch/segment-anything

## Modifications

* `./modeling/mask_decoder.py`: fixed error with batch size > 1
* `./build_sam.py`: set device when loading weights to avoid errors in cpu-only environments