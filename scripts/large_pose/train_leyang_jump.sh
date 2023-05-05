#!/bin/bash
if [ $# -eq 5 ]; then
    resume=${5}
else
    resume=None;
fi
# CUDA_VISIBLE_DEVICES=${1} python train.py --gpu-ids 0 --conf ./configs/female_large_pose/leyang_jump.conf --data ./gap-female-largepose/leyang_jump/ --save-folder ${3} --project_name ${4} --exp_name leyang_jump_fl --resume $resume --data_type large_pose  --a_pose
CUDA_VISIBLE_DEVICES=${1} python train_large_pose.py --gpu-ids 0 --conf ./configs/female_large_pose/leyang_jump_large_pose.conf --data ./gap-female-largepose/leyang_jump/ --save-folder ${2} --project_name ${3} --exp_name leyang_jump_fl  --data_type large_pose  
