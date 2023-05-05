#!/bin/bash

CUDA_VISIBLE_DEVICES=${1} python ./tools/parsing_mask_to_fl.py --input_path ${2}  --parsing_type ATR --output_path  ${3}

