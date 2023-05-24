#!/usr/bin/env bash

# path to config file
CONFIG="../configs/cityscapes/panoptic-segmentation/aff/maskformer2_aff_small_bs32_45k.yaml"

# path to pre-trained checkpoint
CKPT="../checkpoints/city_pan/aff_small.pth"

# path to images for prediction
INPUTS="../imgs/*.jpg"

# path to blurred version of input images (optional)
BLUR="../imgs_blur/"

# output folder to store results
OUTPUT="demo_res"

# create output folder
mkdir $OUTPUT

# run visualization code
cd demo/
python demo.py --config-file $CONFIG \
  --input $INPUTS \
  --output ../$OUTPUT \
  --blur $BLUR \
  --opts MODEL.WEIGHTS $CKPT \

# The --opts flag should always be the last one
# Remove --blur flag to visualize predictions on original images
