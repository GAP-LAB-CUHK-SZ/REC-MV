#!/bin/bash
if [ $# -eq 4 ]; then
    resume=${4}
else
    resume=None;
fi

# traing self-rotated video
CUDA_VISIBLE_DEVICES=${1} python train.py --gpu-ids 0 --conf ${2} --data ./female_large_pose_process_new/anran_tic/ --save-folder ${3} --project_name ${4} --exp_name anran_tic_fl --resume $resume --data_type large_pose  --a_pose

# cp
# cp -rd ./gap-female-largepose/anran-tic/${3} ./gap-female-largepose/anran-tic/xxxx_large_pose 

# training large pose video
# CUDA_VISIBLE_DEVICES=${1} python train_large_pose.py --gpu-ids 0 --conf ./configs/female_large_pose/anran_tic_large_pose.conf --data ./gap-female-largepose/anran_tic/ --save-folder ${2} --project_name ${3} --exp_name anran_tic_fl  --data_type large_pose  
