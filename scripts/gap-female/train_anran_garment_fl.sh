#!/bin/bash
if [ $# -eq 4 ]; then
    resume=${4}
else
    resume=None;
fi


# echo ${resume}

CUDA_VISIBLE_DEVICES=${1} python train.py --gpu-ids 0 --conf ./configs/gap-female/config_anran_garment_10-5-1.conf --data ./a_pose_female_process/anran/ --save-folder ${2} --project_name ${3} \
    --exp_name anran_fl --resume $resume --data_type  scene
