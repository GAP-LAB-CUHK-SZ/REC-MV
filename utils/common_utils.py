"""
@File: format_transfer.py
@Author: Lingteng Qiu
@Email: qiulingteng@link.cuhk.edu.cn
@Date: 2022-07-06
@Desc: common settings
"""


import numpy as np
import torchvision.utils as vutils
import torch, random
import torch.nn.functional as F
import trimesh
from pytorch3d.structures import Meshes
from engineer.utils.garment_structure import Garment_Polygons, Garment_Mesh
import wandb
import cv2


def upper_bound(arr,left,right, target):
    while left <right:
        mid = (left+right) >> 1
        if arr[mid] <= target:
            left = mid+1
        else:
            right = mid
    return left

def lower_bound(arr,left,right, target):
    while left <right:
        mid = (left+right) >> 1


        if arr[mid] < target:
            left = mid+1
        else:
            right = mid
    return left
# print arguments
def print_args(args):
    print("################################  args  ################################")
    for k, v in args.__dict__.items():
        print("{0: <10}\t{1: <30}\t{2: <20}".format(k, str(v), str(type(v))))
    print("########################################################################")


# torch.no_grad warpper for functions
def make_nograd_func(func):
    def wrapper(*f_args, **f_kwargs):
        with torch.no_grad():
            ret = func(*f_args, **f_kwargs)
        return ret

    return wrapper


# convert a function into recursive style to handle nested dict/list/tuple variables
def make_recursive_func(func):
    def wrapper(vars):
        if isinstance(vars, list):
            return [wrapper(x) for x in vars]
        elif isinstance(vars, tuple):
            return tuple([wrapper(x) for x in vars])
        elif isinstance(vars, dict):
            return {k: wrapper(v) for k, v in vars.items()}
        else:
            return func(vars)

    return wrapper


def make_recursive_meta_func(meta, parents =''):
    meta_infos = {}
    if isinstance(meta, dict):
        for k, v in meta.items():
            meta_infos.update(make_recursive_meta_func(v, parents+k+'/'))
    elif isinstance(meta, list) or isinstance(meta, tuple):
        # only support the length of list is lq 3
        for iter_name, v in enumerate(meta[:3]):
            meta_infos.update(make_recursive_meta_func(v, parents+"{:03d}".format(iter_name)+'/'))
    else:
        return {parents[:-1]: meta}
    return meta_infos






@make_recursive_func
def unsqueeze_tensor(vars):
    if isinstance(vars, torch.Tensor):
        return vars.unsqueeze(0)
    elif isinstance(vars, Garment_Polygons):
        return vars
    elif isinstance(vars, Garment_Mesh):
        return vars
    else:
        raise NotImplementedError("invalid input type {} for unsqueeze_tensor".format(type(vars)))
@make_recursive_func
def wandb_img(vars):
    if isinstance(vars, torch.Tensor):
        return wandb.Image(vars.float())
    elif isinstance(vars, np.ndarray):
        return wandb.Image(vars.astype(np.float))
    elif isinstance(vars, str):
        return vars
    else:
        raise NotImplementedError



@make_recursive_func
def inverse_image_normalized(vars):
    if isinstance(vars, torch.Tensor):
        vars = (vars /2 + 0.5) * 255
        return vars
    elif isinstance(vars, np.ndarray):
        vars = (vars /2 + 0.5) * 255
        return vars
    elif isinstance(vars, str):
        return vars
    else:
        raise NotImplementedError("invalid input type {} for squeeze_tensor".format(type(vars)))

@make_recursive_func
def bgr2rgb(vars):
    if isinstance(vars, np.ndarray):
        #NOTE that no consider batch
        if len(vars.shape) == 3:
            vars = vars[..., ::-1]
        return vars
    elif isinstance(vars, str):
        return vars
    else:
        raise NotImplementedError("invalid input type {} for squeeze_tensor".format(type(vars)))

@make_recursive_func
def resize256(vars):
    if isinstance(vars, np.ndarray):
        if len(vars.shape) == 3:
            h, w, _ = vars.shape
        if len(vars.shape) == 2:
            h, w = vars.shape
        ratio = 256. / min(h,w)
        return cv2.resize(vars,(int(ratio * w),int(ratio * h)))
    elif isinstance(vars, str):
        return vars
    else:
        raise NotImplementedError("invalid input type {} for squeeze_tensor".format(type(vars)))

@make_recursive_func
def resize512(vars):
    if isinstance(vars, np.ndarray):
        if len(vars.shape) == 3:
            h, w, _ = vars.shape
        if len(vars.shape) == 2:
            h, w = vars.shape
        ratio = 512. / min(h,w)
        return cv2.resize(vars,(int(ratio * w),int(ratio * h)))
    elif isinstance(vars, str):
        return vars
    else:
        raise NotImplementedError("invalid input type {} for squeeze_tensor".format(type(vars)))

@make_recursive_func
def squeeze_tensor(vars):
    if isinstance(vars, torch.Tensor):
        return vars.squeeze(0)
    elif isinstance(vars, Garment_Polygons):
        return vars
    elif isinstance(vars, Garment_Mesh):
        return vars
    else:
        raise NotImplementedError("invalid input type {} for squeeze_tensor".format(type(vars)))

@make_recursive_func
def tensor2float(vars):
    if isinstance(vars, float):
        return vars
    elif isinstance(vars, torch.Tensor):
        return vars.data.item()
    else:
        raise NotImplementedError("invalid input type {} for tensor2float".format(type(vars)))


@make_recursive_func
def tensor2numpy(vars):
    if isinstance(vars, np.ndarray):
        return vars
    elif isinstance(vars, torch.Tensor):
        return vars.detach().cpu().numpy().copy()
    elif isinstance(vars, str):
        return vars
    else:
        raise NotImplementedError("invalid input type {} for tensor2numpy".format(type(vars)))

@make_recursive_func
def numpy2tensor(vars):
    if isinstance(vars, torch.Tensor):
        return vars
    elif isinstance(vars, np.ndarray):
        return torch.from_numpy(vars)
    else:
        raise NotImplementedError("invalid input type {} for numpy2tensor".format(type(vars)))

@make_recursive_func
def trimesh2torch3d(vars):
    if isinstance(vars, Meshes):
        return vars
    elif isinstance(vars, trimesh.Trimesh):
        # trimesh.Trimesh.vertices, trimesh.Trimesh.faces
        return Meshes(verts = [numpy2tensor(vars.vertices.astype(np.float32))], faces = [numpy2tensor(vars.faces)])
    else:
        raise NotImplementedError("invalid input type {} for trimesh2torch3d".format(type(vars)))

@make_recursive_func
def torch3dverts(vars):
    if isinstance(vars, torch.Tensor):
        return vars
    elif isinstance(vars, Meshes):
        # trimesh.Trimesh.vertices, trimesh.Trimesh.faces
        return  vars.verts_packed()[None]
    else:
        raise NotImplementedError("invalid input type {} for torch3dverts".format(type(vars)))
@make_recursive_func
def tocuda(vars):
    if isinstance(vars, torch.Tensor):
        return vars.to(torch.device("cuda"))
    elif isinstance(vars, str):
        return vars
    elif isinstance(vars, Meshes):
        return vars.to(torch.device("cuda"))
    else:
        raise NotImplementedError("invalid input type {} for tocuda".format(type(vars)))

@make_recursive_func
def tocpu(vars):
    if isinstance(vars, torch.Tensor):
        return vars.to(torch.device("cpu"))
    elif isinstance(vars, str):
        return vars
    elif isinstance(vars, Meshes):
        return vars.to(torch.device("cpu"))
    else:
        raise NotImplementedError("invalid input type {} for tocpu".format(type(vars)))

def save_scalars(logger, mode, scalar_dict, global_step):
    scalar_dict = tensor2float(scalar_dict)
    for key, value in scalar_dict.items():
        if not isinstance(value, (list, tuple)):
            name = '{}/{}'.format(mode, key)
            logger.add_scalar(name, value, global_step)
        else:
            for idx in range(len(value)):
                name = '{}/{}_{}'.format(mode, key, idx)
                logger.add_scalar(name, value[idx], global_step)


def save_images(logger, mode, images_dict, global_step):
    images_dict = tensor2numpy(images_dict)

    def preprocess(name, img):
        if not (len(img.shape) == 3 or len(img.shape) == 4):
            raise NotImplementedError("invalid img shape {}:{} in save_images".format(name, img.shape))
        if len(img.shape) == 3:
            img = img[:, np.newaxis, :, :]
        img = torch.from_numpy(img[:1])
        return vutils.make_grid(img, padding=0, nrow=1, normalize=True, scale_each=True)

    for key, value in images_dict.items():
        if not isinstance(value, (list, tuple)):
            name = '{}/{}'.format(mode, key)
            logger.add_image(name, preprocess(name, value), global_step)
        else:
            for idx in range(len(value)):
                name = '{}/{}_{}'.format(mode, key, idx)
                logger.add_image(name, preprocess(name, value[idx]), global_step)
