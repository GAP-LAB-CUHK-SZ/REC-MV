#!/bin/bash
if [ $# -eq 5 ]; then
    resume=${5}
else
    resume=None;
fi
# train a-pose
#CUDA_VISIBLE_DEVICES=${1} python train.py --gpu-ids 0 --conf ./configs/female_large_pose/lingteng_dance.conf --data ./gap-female-largepose/lingteng_dance/ --save-folder ${2} --project_name ${3} --exp_name anran_run_fl --resume $resume --data_type large_pose  --a_pose

# train large-pose
# CUDA_VISIBLE_DEVICES=${1} python train_large_pose.py --gpu-ids 0 --conf ${2} --data ./female_large_pose_process_new/anran_run/ --save-folder ${3} --project_name ${4} --exp_name anran_run_large_fl --data_type large_pose  
CUDA_VISIBLE_DEVICES=${1} python train_large_pose.py --gpu-ids 0 --conf ./configs/female_large_pose/lingteng_dance_large_pose.conf --data ./gap-female-largepose/lingteng_dance/ --save-folder ${2} --project_name ${3} --exp_name lingteng_dance_fl  --data_type large_pose  
