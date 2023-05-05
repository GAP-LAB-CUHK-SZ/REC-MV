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
import cv2
from utils.constant import TEMPLATE_GARMENT, FL_INFOS, FL_COLOR
from engineer.visualizer.wandb_visualizer import wandb_visualizer
import random
import math
import pdb
import wandb
from pytorch3d.io import save_ply, save_obj
debug =  False
parser = argparse.ArgumentParser(description='neu video body rec')
parser.add_argument('--gpu-ids',nargs='+',type=int,metavar='IDs',
                    help='gpu ids')
parser.add_argument('--conf',default=None,metavar='M',
                    help='config file')
parser.add_argument('--data',default=None,metavar='M',
                    help='data root')
parser.add_argument('--model-rm-prefix',nargs='+',type=str,metavar='rm prefix', help='rm model prefix')
parser.add_argument('--sdf-model',default=None,metavar='M',
                    help='substitute sdf model')
parser.add_argument('--save-folder',default=None,metavar='M',help='save folder')
parser.add_argument('--project_name', type = str, required = True, help='exp name show by wandb')
parser.add_argument('--exp_name', type = str, required = True, help='exp name show by wandb')
parser.add_argument('--data_type', type = str, required = True, help='the type of dataset')
parser.add_argument('--a_pose', action= 'store_true', help='the type of dataset')
parser.add_argument('--curve_sampling', type = int, default = 1, help='the type of dataset')
parser.add_argument('--resume',default=None, metavar='M',
                    help='pretrained scene model')
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
# registry visualized tool
wv_resume = True if args.resume is not None and osp.isfile(args.resume) else False

wv = wandb_visualizer(args.project_name, args.exp_name, resume = wv_resume) if not debug else None
# wv = None

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
condlen={'deformer': int(config.get_int('mlp_deformer.condlen') * (1+len(TEMPLATE_GARMENT[config.get_string('train.garment_type')]))),'renderer':config.get_int('render_net.condlen')}
# batch_size:3
batch_size=config.get_int('train.coarse.point_render.batch_size')
dataset,dataloader=getDatasetAndLoader(data_root,condlen,batch_size,
                        config.get_bool('train.shuffle'),config.get_int('train.num_workers'),
                        config.get_bool('train.opt_pose'),config.get_bool('train.opt_trans'),config.get_config('train.opt_camera'), config.get_string('train.garment_type'), data_type = config.train.data_type,
                        curve_sampling = args.curve_sampling, a_pose = args.a_pose)

garment_type = config.get_string('train.garment_type')

# if debug:
#     for ind in range(len(dataset)):
#         __, datas = dataset[ind]
#         gt_joints2d = datas['gt_joints2d']
#         img = datas['img']
#         draw_board =((img / 2. + 0.5) *255.).cpu().numpy().astype(np.uint8)
#
#         for joint in gt_joints2d.astype(np.int):
#             draw_board = cv2.circle(draw_board, (int(joint[0]), int(joint[1])), 2, (0, 0, 255), 1)
#
#         cv2.imwrite('./debug/joints2d/{:04d}.png'.format(ind), draw_board)
#         print('./debug/joints2d/{:04d}.png'.format(ind))

# os.makedirs('debug/{}/masks/'.format(config.train.garment_type), exist_ok = True)
# garment_type = config.train.garment_type
# for ind in range(len(dataset)):
#     __, datas = dataset[ind]
#
#     upper_mask =datas['upper']
#     bottom_mask =datas['bottom']
#
#
#     upper_bottom_mask =datas['upper_bottom']
#
#     # NOTE the order of body mask and garment_mask
#
#     upper_mask = upper_mask.detach().cpu().numpy().astype(np.uint8) * 255
#     bottom_mask = bottom_mask.detach().cpu().numpy().astype(np.uint8) * 255
#     upper_bottom_mask = upper_bottom_mask.detach().cpu().numpy().astype(np.uint8) * 255
#
#     cv2.imwrite('debug/{}/masks/upper_{:03d}.png'.format(garment_type,ind), upper_mask)
#     cv2.imwrite('debug/{}/masks/bottom_{:03d}.png'.format(garment_type,ind), bottom_mask)
#     cv2.imwrite('debug/{}/masks/upper_bottom_{:03d}.png'.format(garment_type, ind), upper_bottom_mask)
#
#     print('frame {:3d}'.format(ind))
# xxxx


bmins=None
bmaxs=None

#-1200
if config.get_int('train.initial_iters')<=0:
    use_initial_sdf=True
else:
    use_initial_sdf=False

# tmpsdf网络->输出为257, sdf 值和sdf 的feature embedding(256)
# deformer_network (128 + 39(xyz+embeding)) 输出为256值
# perspective camera
# seg3d loss 的约束engine
# silluhute render 器
# rendernet work
optNet,sdf_initialized=getOptNet(dataset, args.save_folder, batch_size,bmins,bmaxs,resolutions['coarse'],device,config,use_initial_sdf ,visualizer= wv, opt_large = False)
optNet,dataloader=utils.set_hierarchical_config(config,'coarse',optNet,dataloader,resolutions['coarse'])

print('box:')
print(optNet.engine.b_min.view(-1).tolist())
print(optNet.engine.b_max.view(-1).tolist())
optNet.train()

# initialized the sdf_field
if sdf_initialized>0:
    optNet.initializeTmpSDF(sdf_initialized,osp.join(data_root,args.save_folder, 'initial_sdf_idr'+'_%d_%d.pth'%(config.get_int('sdf_net.multires'),config.get_int('train.skinner_pose_type'))),True, dataloader = dataloader)
    engine = Seg3dLossless(
            query_func=None,
            b_min = optNet.engine.b_min[0],
            b_max = optNet.engine.b_max[0],
            resolutions=resolutions['coarse'],
            align_corners=False,
            balance_value=0.0, # be careful
            visualize=False,
            debug=False,
            use_cuda_impl=False,
            faster=False
        ).to(device)
    verts_list,faces_list=optNet.discretizeSDF(-1,engine)
    # save_body_mesh
    body_verts, body_faces = verts_list[0], faces_list[0]
    mesh = trimesh.Trimesh(body_verts.cpu().numpy(), body_faces.cpu().numpy())
    optNet.load_init_sdf_vertices(mesh)
    mesh.export(osp.join(data_root,args.save_folder,'initial_sdf_idr'+'_%d_%d.ply'%(config.get_int('sdf_net.multires'),config.get_int('train.skinner_pose_type'))))



    # save_garment_mesh
    garment_verts, garment_faces = verts_list[1:], faces_list[1:]
    for garment_idx, garment_name in enumerate(TEMPLATE_GARMENT[optNet.garment_type]):
        mesh = trimesh.Trimesh(garment_verts[garment_idx].cpu().numpy(), garment_faces[garment_idx].cpu().numpy())
        mesh.export(osp.join(data_root,args.save_folder, 'initial_sdf_%s_idr'%(garment_name)+'_%d_%d.ply'%(config.get_int('sdf_net.multires'),config.get_int('train.skinner_pose_type'))))

# using initial curve align
optNet.align_fl(os.path.join(data_root, args.save_folder, 'fl_init', 'init_trans_matrix.pth'))
start_epoch = 0

learnable_ws=dataset.learnable_weights()
optimizer = torch.optim.Adam([{'params':learnable_ws},{'params':[p for p in optNet.parameters() if p.requires_grad]}], lr=config.get_float('train.learning_rate'))
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, config.get_list('train.scheduler.milestones'), gamma=config.get_float('train.scheduler.factor'))
ratio={'sdfRatio':None,'deformerRatio':None,'renderRatio':None}
optNet.opt_times=0.
in_fine_hie=False
optNet.isfine= False






if args.resume is not None and osp.isfile(args.resume):
    print('load model: '+args.resume,end='')
    if args.sdf_model is not None:
        print(' and substitute sdf model with: '+args.sdf_model,end='')
        sdf_initialized=-1
    optNet,dataset, start_epoch=utils.load_model(args.resume,optNet,dataset,device,args.sdf_model,args.model_rm_prefix)

    if start_epoch >= config.get_int('train.fine.start_epoch'):
        optNet,dataloader=utils.set_hierarchical_config(config,'fine',optNet,dataloader,resolutions['fine'])
        print('enable fine hierarchical')
        torch.cuda.empty_cache()
        in_fine_hie=True
        optNet.draw=True
        optNet.isfine = True
    elif start_epoch >= config.get_int('train.medium.start_epoch'):
        optNet,dataloader=utils.set_hierarchical_config(config,'medium',optNet,dataloader,resolutions['medium'])
        print('enable fine hierarchical')
        torch.cuda.empty_cache()

    learnable_ws=dataset.learnable_weights()
    optimizer = torch.optim.Adam([{'params':learnable_ws},{'params':[p for p in optNet.parameters() if p.requires_grad]}], lr=config.get_float('train.learning_rate'))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, config.get_list('train.scheduler.milestones'), gamma=config.get_float('train.scheduler.factor'))
    for __ in range(start_epoch+1):
        scheduler.step()

    # compute opt_time
    coarse_epoch = config.get_int('train.coarse.start_epoch')
    medium_epoch = config.get_int('train.medium.start_epoch')
    fine_epoch = config.get_int('train.fine.start_epoch')
    coarse_batch_size=config.get_int('train.coarse.point_render.batch_size')
    medium_batch_size=config.get_int('train.medium.point_render.batch_size')
    fine_batch_size=config.get_int('train.fine.point_render.batch_size')
    coarse_time = np.ceil(len(dataset) /coarse_batch_size) * (medium_epoch - coarse_epoch)
    medium_time = np.ceil(len(dataset) /medium_batch_size) * (fine_epoch - medium_epoch)
    fine_time = np.ceil(len(dataset) /fine_batch_size) * (start_epoch - medium_epoch + 1)
    optNet.opt_times+= (coarse_time+medium_time+fine_time)
    start_epoch+=1

    fl_curve_meshes = optNet.inter_free_curve.curve_to_mesh()

    for fl_curve, fl_name in zip(fl_curve_meshes, optNet.fl_names):
        save_obj('debug/cano_fl/{}.obj'.format(fl_name), fl_curve.verts_packed().detach().cpu(),fl_curve.faces_packed().detach().cpu() )




# 200
nepochs=config.get_int('train.nepoch')
# 2048
sample_pix_num=config.get_int('train.sample_pix_num')
# coarse {
#   start_epoch = 0
#   point_render {
#     radius = 0.006
#     remesh_intersect = 30
#     batch_size = 3
#   }
# }
# medium {
#   start_epoch = 6
#   point_render {
#     radius = 0.00465
#     remesh_intersect = 60
#     batch_size = 2
#   }
# }
# fine {
#   start_epoch = 12
#   point_render {
#     radius = 0.0041
#     remesh_intersect = 120
#     batch_size = 1
#   }
# }
#
for epoch in range(start_epoch,nepochs):

    if config.get_int('train.medium.start_epoch')>=0 and epoch==config.get_int('train.medium.start_epoch'):
        optNet,dataloader=utils.set_hierarchical_config(config,'medium',optNet,dataloader,resolutions['medium'])
        torch.cuda.empty_cache()
        print('enable medium hierarchical')
        utils.save_model(osp.join(save_root,"coarse.pth"),epoch,optNet,dataset)
    if config.get_int('train.fine.start_epoch')>=0 and epoch==config.get_int('train.fine.start_epoch'):
        optNet,dataloader=utils.set_hierarchical_config(config,'fine',optNet,dataloader,resolutions['fine'])
        print('enable fine hierarchical')
        torch.cuda.empty_cache()
        utils.save_model(osp.join(save_root,"medium.pth"),epoch,optNet,dataset)
        in_fine_hie=True
        optNet.draw=True
        optNet.isfine = True


    for data_index, (frame_ids, outs) in enumerate(dataloader):
        frame_ids=frame_ids.long().to(device)
        optimizer.zero_grad()
        ratio['sdfRatio']=1.
        ratio['deformerRatio']=optNet.opt_times/2500.+0.5
        ratio['renderRatio']=1.
        # core
        loss=optNet(outs,sample_pix_num,ratio,frame_ids,debug_root, global_optimizer = optimizer)
        loss.backward()
        # here also having backward
        optNet.propagateTmpPsGrad(frame_ids,ratio)
        optimizer.step()
        cur_lr = optimizer.state_dict()['param_groups'][0]['lr']
        optNet.draw_loss(optNet.opt_times, total_loss = loss.item(), learning_rate = cur_lr, ratio = ratio)

        if data_index%1==0:
            for garment_idx in range(optNet.garment_size):
                garment_name = optNet.garment_names[garment_idx]
                outinfo='(%s)(%d/%d) (%s): loss = %.5f; color_loss: %.5f, eikonal_loss: %.5f'%(garment_type, epoch,data_index,garment_name, loss.item(),optNet.info['{}_color_loss'.format(garment_name)],optNet.info['{}_grad_loss'.format(garment_name)])+ \
                        (' normal_loss: %.5f,'%optNet.info['{}_normal_loss'.format(garment_name)] if '{}_normal_loss'.format(garment_name) in optNet.info else '')+ \
                        (' def_loss: %.5f,'%optNet.info['def_{}_loss'.format(garment_name)] if 'def_{}_loss'.format(garment_name) in optNet.info else '')+ \
                        (' offset_loss: %.5f,'%optNet.info['{}_offset_loss'.format(garment_name)] if '{}_offset_loss'.format(garment_name) in optNet.info else '')+ \
                        (' dct_loss: %.5f,'%optNet.info['dct_loss'] if 'dct_loss' in optNet.info else '')
                outinfo+='\n'
                outinfo+='\tpc_sdf_l: %.5f'%(optNet.info['pc_{}_loss_sdf'.format(garment_name)])

                outinfo+=';\tpc_norm_l: %.5f; '%(optNet.info['pc_loss_norm']) if 'pc_loss_norm' in optNet.info else '; '
                for k,v in optNet.info['pc_loss'].items():
                    if garment_name in k:
                        outinfo+=k+': %.5f\t'%v

                outinfo+='\n\trayInfo(%d,%d)\tinvInfo(%d,%d)\tratio: (%.2f,%.2f,%.2f)\tremesh: %.3f'%(*optNet.info['{}_rayInfo'.format(garment_name)],*optNet.info['{}_invInfo'.format(garment_name)],ratio['sdfRatio'],ratio['deformerRatio'],ratio['renderRatio'],optNet.info['remesh'])
                print(outinfo)

        optNet.opt_times+=1.
    if in_fine_hie:
        optNet.draw=True
        optNet.isfine = True
    utils.save_model(osp.join(save_root,"latest.pth"),epoch,optNet,dataset)
    scheduler.step()
