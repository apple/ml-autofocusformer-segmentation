This directory contains a few tools.

* `convert-pretrained-model-to-d2.py`

Tool to convert ImageNet pre-trained weights for D2.

* `analyze_model.py`

Tool to analyze model parameters and flops.

Usage for semantic segmentation (ADE20K only, use with caution!):

```
python tools/analyze_model.py --num-inputs 1 --tasks flop --use-fixed-input-size --config-file CONFIG_FILE
```

Note that, for semantic segmentation (ADE20K only), we use a dummy image with fixed size that equals to `cfg.INPUT.CROP.SIZE[0] x cfg.INPUT.CROP.SIZE[0]`.
Please do not use `--use-fixed-input-size` for calculating FLOPs on other datasets like COCO!

Usage for panoptic and instance segmentation:

```
python tools/analyze_model.py --num-inputs 100 --tasks flop --config-file CONFIG_FILE
```

Note that, for panoptic and instance segmentation, we compute the average flops over 100 real validation images.
