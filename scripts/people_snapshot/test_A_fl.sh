#!/bin/bash
SUBJ=${2}
ROOT='.'
CUDA_VISIBLE_DEVICES=${1} python infer_fl.py --gpu-ids 0 --rec-root ./people_snapshot_public_proprecess/${2}/${3} --data-type people_snap --a_pose --C
# CUDA_VISIBLE_DEVICES=${1} python infer_fl_curve.py --gpu-ids 0 --rec-root ./people_snapshot_public_proprecess/${2}/${3} --data-type people_snap --a_pose --C
