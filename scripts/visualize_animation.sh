#!/bin/bash

path=${1}
abspath=`readlink -f $path`

echo ${abspath}
color_path=${abspath}/colors/
meshs_path=${abspath}/meshs/
result_path=${abspath}/results

mkdir -p $result_path

color_result_file=${abspath}/results/sanim_colors.mp4
meshs_result_file=${abspath}/results/anim_meshes.mp4


cd ${color_path}

ffmpeg -r 30 -pattern_type glob -i '*.png' -vcodec libx264 -crf 18 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -pix_fmt yuv420p $color_result_file

cd ${meshs_path}
ffmpeg -r 30 -pattern_type glob -i '*.png' -vcodec libx264 -crf 18 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -pix_fmt yuv420p $meshs_result_file


cd ${result_path}

