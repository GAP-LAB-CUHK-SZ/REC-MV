 # CUDA_VISIBLE_DEVICES=${1} python ./proprecess/mask2parsing_mask.py --gpu-ids 0 --conf config_garment.conf --data ./a_pose_female_process/${2} --save-folder result
 CUDA_VISIBLE_DEVICES=${1} python ./proprecess/mask2parsing_mask.py --gpu-ids 0 --conf ${2} --data ${3} --save-folder result
