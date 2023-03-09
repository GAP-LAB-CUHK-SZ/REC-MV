# REC-MV: REconstructing 3D Dynamic Cloth from Monucular Videos (CVPR2023)

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) 
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>

## [Project Page](https://lingtengqiu.github.io/2023/REC-MV/) | [Paper](https://lingtengqiu.github.io/2023/REC-MV/) 

This is the official PyTorch implementation of [REC-MV]().


we will release the code soon.

## Requirements
Python 3
Pytorch3d (0.4.0, some compatibility issues may occur in higher versions of pytorch3d)
Note: A GTX 3090 is recommended to run SelfRecon, make sure enough GPU memory if using other cards.

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



## A Gentle Introduction

![](/figs/teaser.png)

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


