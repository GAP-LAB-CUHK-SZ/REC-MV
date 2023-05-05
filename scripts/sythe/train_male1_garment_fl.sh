#!/bin/bash
if [ $# -eq 5 ]; then
    resume=${5}
else
    resume=None;
fi
CUDA_VISIBLE_DEVICES=${1} python train.py --gpu-ids 0 --conf ./configs/sythe/male1_10_5_1.conf --data ./selfrecon_sythe/male_outfit1/ --save-folder ${2} --project_name ${3} --exp_name male_outfit1_fl --resume $resume --data_type synthe 
