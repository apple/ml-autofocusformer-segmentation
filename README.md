# AutoFocusFormer

[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](CODE_OF_CONDUCT.md)
[![CLUSTEN](https://img.shields.io/badge/CUDA%20Extension-CLUSTEN-red)](clusten/)

This software project accompanies the research paper, *AutoFocusFormer: Image Segmentation off the Grid* (CVPR 2023).

[Chen Ziwen](https://www.chenziwe.com), Kaushik Patnaik, [Shuangfei Zhai](https://scholar.google.com/citations?user=G6vdBYsAAAAJ&hl=en), [Alvin Wan](http://alvinwan.com), [Zhile Ren](https://jrenzhile.com), [Alex Schwing](https://alexander-schwing.de/), [Alex Colburn](https://www.colburn.org), [Li Fuxin](https://web.engr.oregonstate.edu/~lif/)

[arXiv](https://arxiv.org/abs/2304.12406) | [AFF-Classification](https://github.com/apple/ml-autofocusformer) | [AFF-Segmentation (this repo)](https://github.com/apple/ml-autofocusformer-segmentation)

## Introduction

AutoFocusFormer (AFF) the first **adaptive**-downsampling network capable of **dense** prediction tasks such as semantic/instance segmentation.

AFF abandons the traditional grid structure of image feature maps, and automatically learns to retain the most important pixels with respect to the task goal.

<div align="center">
  <img src="aff.png" width="100%" height="100%"/>
</div><br/>

AFF consists of a local-attention transformer backbone and a task-specific head. The backbone consists of four stages, each stage containing three modules: balanced clustering, local-attention transformer blocks, and adaptive downsampling.

<div align="center">
  <img src="architecture.png" width="100%" height="100%"/>
</div><br/>

AFF demonstrates significant savings on FLOPs (see our models with 1/5 downsampling rate), and significant improvement on recognition of small objects.

Notably, AFF-Small achieves **44.0** instance segmentation AP and **66.9** panoptic segmentation PQ on Cityscapes val with a backbone of only **42.6M** parameters, a performance on par with Swin-Large, a backbone with **197M** params (saving **78%**!).

<div align="center">
  <img src="demo1.png" width="100%" height="100%"/>
</div><br/>

<div align="center">
  <img src="demo2.png" width="100%" height="100%"/>
</div><br/>

This repository contains the AFF backbone and the point cloud-version of the Mask2Former segmentation head.

We also add a few convenient functionalities, such as visualizing prediction results on blurred version of the images, and evaluating on cocofied lvis v1 annotations.

## Main Results with Pretrained Models 

**ADE20K Semantic Segmentation (val)**
| backbone | method | pretrain | crop size | mIoU | FLOPs | checkpoint |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| AFF-Mini | Mask2Former | ImageNet-1K | 512x512 | 46.5 | 48.3G | [Box](https://apple.box.com/s/nvgp3fiw867qn0st2ackzr5qgkv1g5dt) |
| AFF-Mini-1/5 | Mask2Former | ImageNet-1K | 512x512 | 46.0 | 39.9G | [Box](https://apple.box.com/s/unlt9onup1drzuy0ll8rdr82lw85k739) | 
| AFF-Tiny | Mask2Former | ImageNet-1K | 512x512 | 50.2 | 64.6G | [Box](https://apple.box.com/s/w914lhy6rnljz8rzwqov3y2f2ualrane) | 
| AFF-Tiny-1/5 | Mask2Former | ImageNet-1K | 512x512 | 50.0 | 51.1G | [Box](https://apple.box.com/s/hlbfzgupgq147lm2fpbclrxvbf9br4fd) | 
| AFF-Small | Mask2Former | ImageNet-1K | 512x512 | 51.2 | 87G | [Box](https://apple.box.com/s/zsjgd1f537bgiif58jomsoxqsgjs198a) | 
| AFF-Small-1/5 | Mask2Former | ImageNet-1K | 512x512 | 51.9 | 67.2G | [Box](https://apple.box.com/s/knbssdds7fhps7hs7wanbdy1h47ol767) | 

**Cityscapes Instance Segmentation (val)**
| backbone | method | pretrain | AP | checkpoint |
| :---: | :---: | :---: | :---: | :---: |
| AFF-Mini | Mask2Former | ImageNet-1K | 40.0 | [Box](https://apple.box.com/s/pmsx5quow1zuj8f04077dr32fqajnq9s) | 
| AFF-Tiny | Mask2Former | ImageNet-1K | 42.7 | [Box](https://apple.box.com/s/bcsqhc930wau8d8dw98voy0m0z322hde) | 
| AFF-Small | Mask2Former | ImageNet-1K | 44.0 | [Box](https://apple.box.com/s/s49vsrfwjkx7whaf4tm486az2kes6qj7) | 

**Cityscapes Panoptic Segmentation (val)**
| backbone | method | pretrain | PQ(s.s.) | checkpoint |
| :---: | :---: | :---: | :---: | :---: |
| AFF-Mini | Mask2Former | ImageNet-1K | 62.7 | [Box](https://apple.box.com/s/0u3ij089j2mfux4tkc0w93guzoc3353d) | 
| AFF-Tiny | Mask2Former | ImageNet-1K | 65.7 | [Box](https://apple.box.com/s/8elap0rver05sq4hh7vv2am8tmnt24cx) | 
| AFF-Small | Mask2Former | ImageNet-1K | 66.9 | [Box](https://apple.box.com/s/kzr33pisibgqbbpz0p8n5t1g1wzescqq) | 

**COCO Instance Segmentation (val)**
| backbone | method | pretrain | epochs | AP | FLOPs | checkpoint |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| AFF-Mini | Mask2Former | ImageNet-1K | 50 | 42.3 | 148G | [Box](https://apple.box.com/s/qt2beqgc28uimauj228ppuza8br1ysje) | 
| AFF-Mini-1/5 | Mask2Former | ImageNet-1K | 50 | 42.3 | 120G | [Box](https://apple.box.com/s/ktb25um52x5n6quhw6s4tpobe2ysx0l3) | 
| AFF-Tiny | Mask2Former | ImageNet-1K | 50 | 45.3 | 204G | [Box](https://apple.box.com/s/lfta4x12bxf63l2x5iwpl2dmvamvvbde) | 
| AFF-Tiny-1/5 | Mask2Former | ImageNet-1K | 50 | 44.5 | 152G | [Box](https://apple.box.com/s/34wn40yyhzyp83mtyqaj1xpd1v9vodex) | 
| AFF-Small | Mask2Former | ImageNet-1K | 50 | 46.4 | 281G | [Box](https://apple.box.com/s/fsn3blcjnsej0evrwuyig0wcvg07v6dx) | 
| AFF-Small-1/5 | Mask2Former | ImageNet-1K | 50 | 45.7 | 206G | [Box](https://apple.box.com/s/g6vte0c9vhyt3kbga8lt46e5xyertre7) | 

## Getting Started

### Clone this repo

```bash
git clone git@github.com:apple/ml-autofocusformer-segmentation.git
cd ml-autofocusformer-segmentation
```
One can download the pre-trained checkpoints through the links in the tables above.

### Create environment and install requirements

```bash
sh create_env.sh
```

See further documentation inside the script file.

Our experiments are run with `CUDA==11.6` and `pytorch==1.12`.

### Prepare data

Please refer to [dataset README](datasets/README.md).

### Prepare pre-trained backbone checkpoint

Use `tools/convert-pretrained-model-to-d2.py` to convert any torch checkpoint `.pth` file trained on ImageNet into a Detectron2 model zoo format `.pkl` file.
```
python tools/convert-pretrained-model-to-d2.py aff_mini.pth aff_mini.pkl
```
Otherwise, d2 will assume the checkpoint is for the entire segmentation model and will not add `backbone.` to the parameter names, and thus the checkpoint will not be properly loaded. 

### Train and evaluate

Modify the arguments in script `run_aff_segmentation.sh` and run
```bash
sh run_aff_segmentation.sh
```
for training or evaluation.

One can also directly modify the config files in `configs/`.

### Visualize predictions for pre-trained models

See script `run_demo.sh`. More details can be found in [Mask2Former GETTING_STARTED.md](https://github.com/facebookresearch/Mask2Former/blob/main/GETTING_STARTED.md).

### Analyze model FLOPs

See [tools README](tools/README.md).

## Citing AutoFocusFormer

```BibTeX
@inproceedings{autofocusformer,
    title = {AutoFocusFormer: Image Segmentation off the Grid},
    booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    author = {Ziwen, Chen and Patnaik, Kaushik and Zhai, Shuangfei and Wan, Alvin and Ren, Zhile and Schwing, Alex and Colburn, Alex and Fuxin, Li},
    year = {2023},
}
```
