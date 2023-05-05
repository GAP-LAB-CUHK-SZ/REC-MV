#!/bin/bash
if [ $# -eq 5 ]; then
    resume=${5}
else
    resume=None;
fi
CUDA_VISIBLE_DEVICES=${1} python train.py --gpu-ids 0 --conf ./configs/sythe/female1_10_5_1_wolcc.conf --data ./selfrecon_sythe/female_outfit1/ --save-folder ${2} --project_name ${3} --exp_name female_outfit1_fl --resume $resume --data_type synthe 
#CUDA_VISIBLE_DEVICES=${1} python train.py --gpu-ids 0 --conf ./configs/sythe/female1_10_5_1_wolcc.conf --data ./selfrecon_sythe/female_outfit1/ --save-folder ${2} --project_name ${3} --exp_name female_outfit1_fl --resume $resume --data_type synthe 
