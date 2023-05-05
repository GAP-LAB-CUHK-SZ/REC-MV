import sys
sys.path.append('./')
import torch
import numpy as np
from dataset.dataset import getDatasetAndLoader
from model import getOptNet
from pyhocon import ConfigFactory,HOCONConverter
import argparse
import trimesh
import os
import os.path as osp
from MCAcc import Seg3dLossless
import utils
import pdb
parser = argparse.ArgumentParser(description='neu video body rec')
parser.add_argument('--gpu-ids',nargs='+',type=int,metavar='IDs',
                    help='gpu ids')
parser.add_argument('--conf',default=None,metavar='M',
                    help='config file')
parser.add_argument('--data',default=None,metavar='M',
                    help='data root')
parser.add_argument('--model',default=None,metavar='M',
                    help='pretrained scene model')
parser.add_argument('--model-rm-prefix',nargs='+',type=str,metavar='rm prefix', help='rm model prefix')
parser.add_argument('--sdf-model',default=None,metavar='M',
                    help='substitute sdf model')
parser.add_argument('--save-folder',default=None,metavar='M',help='save folder')
args = parser.parse_args()


#point render
resolutions={'coarse':
[
    (14+1, 20+1, 8+1),
    (28+1, 40+1, 16+1),
    (56+1, 80+1, 32+1),
    (112+1, 160+1, 64+1),
    (224+1, 320+1, 128+1),
],
'medium':
[
    (18+1, 24+1, 12+1),
    (36+1, 48+1, 24+1),
    (72+1, 96+1, 48+1),
    (144+1, 192+1, 96+1),
    (288+1, 384+1, 192+1),
],
'fine':
[
    (20+1, 26+1, 14+1),
    (40+1, 52+1, 28+1),
    (80+1, 104+1, 56+1),
    (160+1, 208+1, 112+1),
    (320+1, 416+1, 224+1),
]
}

resolutions_higher = [
    (32+1, 32+1, 32+1),
    (64+1, 64+1, 64+1),
    (128+1, 128+1, 128+1),
    (256+1, 256+1, 256+1),
    (512+1, 512+1, 512+1),
]



config=ConfigFactory.parse_file(args.conf)

if len(args.gpu_ids):
    device=torch.device(args.gpu_ids[0])
else:
    device=torch.device(0)
data_root=args.data
if args.save_folder is None:
    print('please set save-folder...')
    assert(False)

save_root=osp.join(data_root,args.save_folder)
debug_root=osp.join(save_root,'debug')
os.makedirs(save_root,exist_ok=True)
os.makedirs(debug_root,exist_ok=True)
# save the config file
with open(osp.join(save_root,'config.conf'),'w') as ff:
    ff.write(HOCONConverter.convert(config,'hocon'))


#[deform: 128, render: 256]
condlen={'deformer':config.get_int('mlp_deformer.condlen'),'renderer':config.get_int('render_net.condlen')}
# batch_size:3
batch_size=config.get_int('train.coarse.point_render.batch_size')

dataset,dataloader=getDatasetAndLoader(data_root,condlen,batch_size,
                        config.get_bool('train.shuffle'),config.get_int('train.num_workers'),
                        config.get_bool('train.opt_pose'),config.get_bool('train.opt_trans'),config.get_config('train.opt_camera'), garment_type=config.train.garment_type,
                        data_type = config.train.data_type)

for i in range(len(dataset)):
    save_path = dataset.parsing_mask(i)
    print('saveing: {}'.format(save_path))
