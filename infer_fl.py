import torch
import numpy as np
from dataset.dataset import getDatasetAndLoader
from model import getOptNet
from pyhocon import ConfigFactory,HOCONConverter
import argparse
import openmesh as om
import os
import os.path as osp
from MCAcc import Seg3dLossless
import utils
import cv2
from tqdm import tqdm
from pytorch3d.renderer import (
    RasterizationSettings,
    HardPhongShader,
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    SoftSilhouetteShader,
    AlphaCompositor
)
from pytorch3d.io import save_ply, save_obj
import trimesh
import pdb
from utils.constant import TEMPLATE_GARMENT

parser = argparse.ArgumentParser(description='neu video body infer')
parser.add_argument('--gpu-ids',nargs='+',type=int,metavar='IDs',
                    help='gpu ids')
parser.add_argument('--batch-size',default=1,type=int,metavar='IDs',
                    help='batch size')
parser.add_argument('--rec-root',default=None,metavar='M',
                    help='data root')
parser.add_argument('--frames',default=-1,type=int,metavar='frames',
                    help='render frame nums')
parser.add_argument('--nV',action='store_true',help='not save video')
parser.add_argument('--data-type',default= 'synthe',help='the type of inference dataset')
parser.add_argument('--nI',action='store_true',help='not save image')
parser.add_argument('--C',action='store_true',help='overlay on gtimg')
parser.add_argument('--nColor',action='store_true',help='not render images')
parser.add_argument('--a_pose', action='store_true', help='using a-pose images to extract garment_meshes')
args = parser.parse_args()

assert(not(args.nV and args.nI))
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

# resolutions = [
#     (32+1, 32+1, 32+1),
#     (64+1, 64+1, 64+1),
#     (128+1, 128+1, 128+1),
#     (256+1, 256+1, 256+1),
#     (512+1, 512+1, 512+1),
# ]
# resolutions = [
#     (14+1, 20+1, 8+1),
#     (28+1, 40+1, 16+1),
#     (56+1, 80+1, 32+1),
#     (112+1, 160+1, 64+1),
#     (224+1, 320+1, 128+1),
# ]
# resolutions = [
#     (18+1, 24+1, 12+1),
#     (36+1, 48+1, 24+1),
#     (72+1, 96+1, 48+1),
#     (144+1, 192+1, 96+1),
#     (288+1, 384+1, 192+1),
# ]

config=ConfigFactory.parse_file(osp.join(args.rec_root,'config.conf'))
device=args.gpu_ids[0]
deformer_condlen=config.get_int('mlp_deformer.condlen')
renderer_condlen=config.get_int('render_net.condlen')
# batch_size=config.get_int('train.coarse.batch_size')
batch_size=args.batch_size
shuffle=False

condlen={'deformer': int(config.get_int('mlp_deformer.condlen') * (1+len(TEMPLATE_GARMENT[config.get_string('train.garment_type')]))),'renderer':config.get_int('render_net.condlen')}


dataset,dataloader = getDatasetAndLoader(osp.normpath(osp.join(args.rec_root,osp.pardir)),condlen,batch_size,
                        shuffle,config.get_int('train.num_workers'),
                        config.get_bool('train.opt_pose'),config.get_bool('train.opt_trans'),config.get_config('train.opt_camera'), garment_type =config.get_string('train.garment_type'), data_type = args.data_type, a_pose = args.a_pose)


# optNet,sdf_initialized=getOptNet(dataset, args.save_folder, batch_size,bmins,bmaxs,resolutions['coarse'],device,config,use_initial_sdf ,visualizer= wv)
optNet,sdf_initialized=getOptNet(dataset, args.rec_root.split('/')[-1], batch_size,None,None,resolutions['fine'],device,config)
optNet,dataloader=utils.set_hierarchical_config(config,'fine',optNet,dataloader,resolutions['fine'])

# load_alpha_curve
optNet.align_fl(os.path.join(args.rec_root, 'fl_init', 'init_trans_matrix.pth'))


# optNet.eval()
# raster_settings_silhouette = RasterizationSettings(
# image_size=(dataset.H,dataset.W),
# blur_radius=0.,
# bin_size=int(2 ** max(np.ceil(np.log2(max(dataset.H,dataset.W))) - 4, 4)),
# faces_per_pixel=1,
# perspective_correct=True,
# clip_barycentric_coords=False,
# cull_backfaces=False
#        renderer = MeshRendererWithFragments(
#        rasterizer=MeshRasterizer(
#            cameras=cameras,
#            raster_settings=raster_settings_silhouette
#        ),
#        shader=SoftSilhouetteShader()
#        )

# original camera setting
# raster_settings = RasterizationSettings(
#     image_size=(dataset.H,dataset.W),
#     bin_size=int(2 ** max(np.ceil(np.log2(max(dataset.H,dataset.W))) - 4, 4)),
#     blur_radius=0,
#     faces_per_pixel=1,
#     perspective_correct=True,
#     clip_barycentric_coords=False,
#     cull_backfaces=False
#     )

# binsize is very important in test
raster_settings=RasterizationSettings(
        image_size=(dataset.H,dataset.W),
        blur_radius=0.,
        # blur_radius=np.log(1. / 1e-4 - 1.)*3.e-6,
        bin_size=(92 if max(dataset.H,dataset.W)>1024 and max(dataset.H,dataset.W)<=2048 else None),
        faces_per_pixel=1,
        perspective_correct=True,
        clip_barycentric_coords=False,
        cull_backfaces= False
    )
optNet.maskRender.rasterizer.raster_settings=raster_settings
# optNet.maskRender.shader=SoftSilhouetteShader()
optNet.maskRender.shader=HardPhongShader(device,optNet.maskRender.rasterizer.cameras)
optNet.pcRender=None

H=dataset.H
W=dataset.W
print('load model: '+osp.join(args.rec_root,'latest.pth'))
optNet, dataset, ___=utils.load_model(osp.join(args.rec_root,'latest.pth'),optNet,dataset,device)
optNet.dataset=dataset
optNet.eval()


if 'train.fine.point_render' in config:
    raster_settings_silhouette = PointsRasterizationSettings(
        image_size=(H,W),
        radius=config.get_float('train.fine.point_render.radius'),
        # radius=0.002,
        bin_size=64,
        points_per_pixel=50,
        )
    optNet.pcRender=PointsRenderer(
        rasterizer=PointsRasterizer(
            cameras=optNet.maskRender.rasterizer.cameras,
            raster_settings=raster_settings_silhouette
        ),
            compositor=AlphaCompositor(background_color=(1,1,1,1))
        ).to(device)



# ratio={'sdfRatio':1.,'deformerRatio':0.5,'renderRatio':1.}
ratio={'sdfRatio':1.,'deformerRatio':1.,'renderRatio':1.}
TmpVs_list,Tmpfs_list=optNet.discretizeSDF(ratio,None,0.)





# writing body vertices
body_TmpVs, bodyTmpfs = TmpVs_list[0], Tmpfs_list[0]
mesh = om.TriMesh(body_TmpVs.detach().cpu().numpy(), bodyTmpfs.cpu().numpy())
om.write_mesh(osp.join(args.rec_root,'tmp_body.ply'),mesh)



garment_TmpVs, garment_Tmpfs = TmpVs_list[1:], Tmpfs_list[1:]






# for TmpVs, Tmpfs, garment_name in zip(garment_TmpVs, garment_Tmpfs, optNet.garment_names):
#     # to show demo for leader!
#     # mesh = trimesh.load('./tmp_no_sleeve_upper_open.ply', process = False)
#     # Tmpfs = torch.from_numpy(mesh.faces).long().to(Tmpfs)
#     # TmpVs = torch.from_numpy(mesh.vertices).float().to(TmpVs)
#     mesh = om.TriMesh(TmpVs.detach().cpu().numpy(), Tmpfs.cpu().numpy())
#     om.write_mesh(osp.join(args.rec_root,'tmp_{}.ply'.format(garment_name)),mesh)


os.makedirs(osp.join(args.rec_root,'colors'),exist_ok=True)
os.makedirs(osp.join(args.rec_root,'meshs'),exist_ok=True)
os.makedirs(osp.join(args.rec_root,'smpl_meshs'),exist_ok=True)
os.makedirs(osp.join(args.rec_root,'def1meshs'),exist_ok=True)
os.makedirs(osp.join(args.rec_root,'render'),exist_ok=True)

errors={}
errors['maskE']=-1.*np.ones((len(dataset)))
gts={}


optNet.registry_meshes = None
for data_index, (frame_ids, outs) in enumerate(dataloader):




    if data_index*batch_size > args.frames if args.frames>=0 else False:
        break

    imgs=outs['img']
    masks=outs['mask']


    if args.nColor:
        print(data_index*batch_size)
    else:
        print(data_index*batch_size,end='    ')
    frame_ids=frame_ids.long().to(device)
    gts['mask']=masks.to(device)
    if args.C:
        gts['image']=(imgs.to(device)+1.)/2.

    colors_list,imgs_list,def1imgs_list,defVs_list, smpl_list, merge_imgs=optNet.infer_garment(garment_TmpVs,garment_Tmpfs, dataset.H, dataset.W,ratio,frame_ids,args.nColor,gts, args.rec_root)
    # colors_list,imgs_list,def1imgs_list,defVs_list, smpl_list=optNet.infer_garment_fl(garment_TmpVs,garment_Tmpfs, dataset.H, dataset.W,ratio,frame_ids,args.nColor,gts, args.rec_root)
    cv2.imwrite(osp.join(args.rec_root,'render/{:06d}.png'.format(frame_ids[0])),merge_imgs[:,:,[2,1,0]])

    for colors, imgs, def1imgs, def_meshes, garment_name, garment_fs in zip(colors_list, imgs_list, def1imgs_list, defVs_list, optNet.garment_names, garment_Tmpfs):
        for fid,img,def1img,def_mesh in zip(frame_ids.cpu().numpy().reshape(-1),imgs,def1imgs,def_meshes):

            save_obj(osp.join(args.rec_root,'meshs/{}_{:06d}.obj'.format(garment_name, fid)), verts = def_mesh.verts_packed().detach().cpu().float(), faces = def_mesh.faces_packed().detach().cpu().long())
            # np.save(osp.join(args.rec_root,'meshs/{}_{:06d}.npy'.format(garment_name, fid)),defV.reshape(-1,3))
            if not args.nI:
                cv2.imwrite(osp.join(args.rec_root,'meshs/{}_{:06d}.png'.format(garment_name, fid)),img[:,:,[2,1,0]])
                cv2.imwrite(osp.join(args.rec_root,'def1meshs/{}_{:06d}.png'.format(garment_name, fid)),def1img[:,:,[2,1,0]])
        if colors is not None:
            os.makedirs(osp.join(args.rec_root,'colors'),exist_ok=True)
            if not args.nI:
                for fid,color in zip(frame_ids.cpu().numpy().reshape(-1),colors):
                    cv2.imwrite(osp.join(args.rec_root,'colors/{}_{:06d}.png'.format(garment_name, fid)),color)

    # save_smpl
    smpl_obj = smpl_list[0]
    save_obj(osp.join(args.rec_root,'smpl_meshs/smpl_{:06d}.obj'.format(fid)), verts = smpl_obj.verts_packed().detach().cpu(), faces =smpl_obj.faces_packed().detach().cpu())
    # errors['maskE'][frame_ids.cpu().numpy()]=gts['maskE']


