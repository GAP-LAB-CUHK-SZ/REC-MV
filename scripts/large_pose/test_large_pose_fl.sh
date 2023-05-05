#!/bin/bash
SUBJ=${2}
ROOT='.'
CUDA_VISIBLE_DEVICES=${1} python infer_fl.py --gpu-ids 0 --rec-root ./gap-female-largepose/${2}/${3} --data-type large_pose --C
# CUDA_VISIBLE_DEVICES=${1} python infer_fl_curve.py --gpu-ids 0 --rec-root ./female_large_pose_process_new/${2}/${3} --data-type large_pose --a_pose --C
