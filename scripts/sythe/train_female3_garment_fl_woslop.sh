#!/bin/bash
if [ $# -eq 4 ]; then
    resume=${4}
else
    resume=None;
fi



CUDA_VISIBLE_DEVICES=${1} python train.py --gpu-ids 0 --conf ./configs/sythe/female3_10_5_1_woslop.conf --data ./selfrecon_sythe/female_outfit3/ --save-folder ${2} --project_name ${3} --exp_name female_outfit3_fl --resume $resume --data_type synthe 
# CUDA_VISIBLE_DEVICES=${1} python train.py --gpu-ids 0 --conf ./configs/sythe/female3_10_5_1.conf --data ./selfrecon_sythe/female_outfit3/ --save-folder ${2} --project_name ${3} --exp_name female_outfit3_fl --resume $resume --data_type synthe 
