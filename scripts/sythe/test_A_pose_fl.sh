#!/bin/bash
SUBJ=${2}
ROOT='.'
CUDA_VISIBLE_DEVICES=${1} python infer_fl.py --gpu-ids 0 --rec-root ./selfrecon_sythe/${2}/${3} --data-type synthe --a_pose --C
