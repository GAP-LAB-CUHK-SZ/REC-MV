#!/bin/bash
if [ $# -eq 5 ]; then
    resume=${4}
else
    resume=None;
fi
# train a-pose
CUDA_VISIBLE_DEVICES=${1} python train.py --gpu-ids 0 --conf ./configs/people_snapshot/female-3-casual.conf --data ./people_snapshot_public_proprecess/female-3-casual/ --save-folder ${2} --project_name ${3} \
    --exp_name female_3_casual_fl --resume $resume --data_type people_snapshot  --a_pose

# train large-pose
# CUDA_VISIBLE_DEVICES=${1} python train_large_pose.py --gpu-ids 0 --conf ${2} --data ./female_large_pose_process_new/anran_run/ --save-folder ${3} --project_name ${4} --exp_name anran_run_large_fl --data_type large_pose  
