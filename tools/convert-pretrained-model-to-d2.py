#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Adapted for AutoFocusFormer by Ziwen 2023

import pickle as pkl
import sys

import torch

"""
Usage:
  # run the conversion
  python ./convert-pretrained-model-to-d2.py aff.pth aff.pkl
  # Then, use aff.pkl in config:
MODEL:
  WEIGHTS: "/path/to/aff.pkl"
INPUT:
  FORMAT: "RGB"
"""

if __name__ == "__main__":
    input = sys.argv[1]

    obj = torch.load(input, map_location="cpu")["model"]

    res = {"model": obj, "__author__": "third_party", "matching_heuristics": True}

    with open(sys.argv[2], "wb") as f:
        pkl.dump(res, f)
