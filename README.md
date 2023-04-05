# REC-MV: REconstructing 3D Dynamic Cloth from Monucular Videos (CVPR2023)

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) 
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>

## [Project Page](https://lingtengqiu.github.io/2023/REC-MV/) | [Paper](https://lingtengqiu.github.io/2023/REC-MV/) 

This is the official PyTorch implementation of [REC-MV](https://lingtengqiu.github.io/2023/REC-MV/).

we will release the code soon.

## TODO:triangular_flag_on_post:

- [x] Preprocess datasets
- [ ] Pretrained weights
- [ ] Demo
- [ ] Training Code

## Requirements

Python 3
Pytorch3d (0.4.0, some compatibility issues may occur in higher versions of pytorch3d)

Note: A GTX 3090 is recommended to run REC-MV, make sure enough GPU memory if using other cards.

## Install
```bash
conda env create REC-MV
conda activate REC-MV
pip install -r requirements.txt
bash install.sh
```

It is recommended to install pytorch3d 0.4.0 from source.

```bash
wget -O pytorch3d-0.4.0.zip https://github.com/facebookresearch/pytorch3d/archive/refs/tags/v0.4.0.zip
unzip pytorch3d-0.4.0.zip
cd pytorch3d-0.4.0 && python setup.py install && cd ..
```

To download the [SMPL](https://smpl.is.tue.mpg.de/) models from [here](https://mailustceducn-my.sharepoint.com/:f:/g/personal/jby1993_mail_ustc_edu_cn/EqosuuD2slZCuZeVI2h4RiABguiaB4HkUBusnn_0qEhWjQ?e=c6r4KS) and move pkls to smpl_pytorch/model.



## Preprocess Datasets

#### SMPL Prediction

The preprocessing of  dataset is described here. If you want to optimize your own data, you can run [VideoAvatar](https://graphics.tu-bs.de/people-snapshot) or [TCMR ](https://github.com/hongsukchoi/TCMR_RELEASE)to get the initial SMPL estimation. Surely,  you can use your own SMPL initialization and normal prediction method then use REC-MV to reconstruct.

#### Normal map Prediction

To enable our normal optimization, you have to install [PIFuHD](https://shunsukesaito.github.io/PIFuHD/) and [Lightweight Openpose](https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch) in your $ROOT1 and $ROOT2 first. Then copy generate_normals.py and generate_boxs.py to $ROOT1 and $ROOT2 seperately, and run the following code to extract normals before running REC-MV:

```bash
cd $ROOT2
python generate_boxs.py --data $ROOT/video-category/imgs
cd $ROOT1
python generate_normals.py --imgpath $ROOT/video-category/imgs
```

#### Parsing foreground  Mask

We utilize a awesome [RobustVideoMatting](https://github.com/PeterL1n/RobustVideoMatting) to parsing human mask from monocular Videos.

#### Parsing Garment Semantic label.

[Self-Correction-Human-Parsing](https://github.com/GoGoDuck912/Self-Correction-Human-Parsing) is employed to segment garment labels. Note that we find **ATR** pretrained weight is better than other checkpoints, so we suggest you to load the **ATR** checkpoint.

#### Initialized Voxel Skinning Weights.

To better model skirts or dresses skinning weights, we apply [fite](https://github.com/jsnln/fite) to diffuse skinning weights in whole voxel space. Specifically, we initialized skinning weights as the step1 said([Link](https://github.com/jsnln/fite))

The following commands given you an example to obtain PeopleSnapshot diffused skinning weights.

```bash
#!/bin/bash
#! example for processing people snapshot
# name_list=( female-3-casual female-3-sport female-4-casual female-6-plaza female-7-plaza )
name_list=( male-2-casual )
for name in ${name_list[@]}; do
    python -m step1_diffused_skinning.compute_diffused_skinning --config configs/step1/${name}.yaml
done

# clean tmp files
rm -rf ./data_tmp_constraints/*
rm -rf ./data_tmp_skinning_grid/*
```

#### Preprocess Datasets from Ours.

We provide links to the datas we have already processed

##### OneDrive

- [PeopleSnapshot](https://cuhko365-my.sharepoint.com/:u:/g/personal/220019047_link_cuhk_edu_cn/EYS0ivryIX1MnZtBbs8u_ccBHFFUjZQQpsO9WMWy665R1A?e=LWbXTD)
- [CUHKszCap-A](https://cuhko365-my.sharepoint.com/:u:/g/personal/220019047_link_cuhk_edu_cn/EaDhqIkcY5lEhIi5U9f-yqEB_MGv78TWtFycWxc_uSPL6g?e=6NQntH)
- [CUHKszCap-L](https://cuhko365-my.sharepoint.com/:u:/g/personal/220019047_link_cuhk_edu_cn/EaVVeJlkwmVPlRLAgb3-_KQBQviHrTAp9txR-HBgynxZIQ?e=48v5eQ)

##### Baidu Drive

- [PeopleSnapshot](https://pan.baidu.com/s/1QqBPWok-RDmQ_ZbJpqnJMQ?pwd=whdm)(PWD: wdhm)
- [CUHKszCap-A](https://pan.baidu.com/s/1XX0bZyPG2Hci-ynA31mcKw?pwd=grx5) (PWD: grx5)
- [CUHKszCap-L](https://pan.baidu.com/s/1V3u9QN6X45Q5SAVIhRI5TQ?pwd=9pne)(PWD: 9pne)

## A Gentle Introduction

![](./figs/teaser.png)

Reconstructing dynamic 3D garment surfaces with open boundaries from monocular videos is an important problem as it provides a practical and low-cost solution for clothes digitization. Recent neural rendering methods achieve high-quality dynamic clothed human reconstruction results from monocular video, but these methods cannot separate the garment surface from the body. To address the above limitations, in this paper, we formulate this task as an optimization problem of 3D garment feature curves and surface reconstruction from monocular video. We introduce a novel approach, called REC-MV, to jointly optimize the explicit feature curves and the implicit signed distance field (SDF) of the garments. Then the open garment meshes can be extracted via garment template registration in the canonical space. 


## Citation

If you use REC-MV in your research, please consider the following BibTeX entry and giving us a star🌟!

```BibTeX
@inproceedings{qiu2023recmv
  title={REC-MV: REconstructing 3D Dynamic Cloth from Monucular Videos},
  author={Qiu, Lingteng and Chen, Guanying and Zhou, Jiapeng and Xu, Mutian and Wang, Junle, and Han, Xiaoguang},
  booktitle={CVPR},
  year={2023}
}
```

## Acknowledgements

Here are some great resources we benefit or utilize from:

- [SelfRecon](https://github.com/jby1993/SelfReconCode) and [Open-PIFuhd](https://github.com/lingtengqiu/Open-PIFuhd) for Our code base.

- [VideoAvatar](https://graphics.tu-bs.de/people-snapshot) and [TCMR ](https://github.com/hongsukchoi/TCMR_RELEASE) for SMPL initialization.
- [SMPL](https://smpl.is.tue.mpg.de/) for Parametric Body Representation
- [Self-Correction-Human-Parsing](https://github.com/GoGoDuck912/Self-Correction-Human-Parsing) for Garment Parsing
- [RobustVideoMatting](https://github.com/PeterL1n/RobustVideoMatting) for Foreground Parsing
- [PyTorch3D](https://github.com/facebookresearch/pytorch3d) for Differential Explicit Rendering
- [Fite](https://github.com/jsnln/fite) for Skinning weights initialization

 
