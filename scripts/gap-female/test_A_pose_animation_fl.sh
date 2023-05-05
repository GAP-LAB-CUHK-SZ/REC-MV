#!/bin/bash
SUBJ=${2}
ROOT='.'
CUDA_VISIBLE_DEVICES=${1} python infer_fl_animation.py --gpu-ids 0 --rec-root $ROOT/a_pose_female_process/${2}/${3} --data-type snug  --C  
