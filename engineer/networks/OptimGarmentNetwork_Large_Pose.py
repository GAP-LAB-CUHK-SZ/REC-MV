"""
@File: OptimGarmentNetwork_Large_Pose.
@Author: Lingteng Qiu
@Email: qiulingteng@link.cuhk.edu.cn
@Date: 2023-05-02
@Desc: Fitting Large pose REC-MV, after self-rotated optimization.
"""
import numpy as np
import torch.nn as nn
import torch
from torch.nn.parameter import Parameter
import torch.autograd.functional as F
from FastMinv import Fast3x3Minv
from model.Embedder import get_embedder
import utils
import os
import cv2
from pytorch3d.io import save_obj, save_ply
from pytorch3d.renderer.points.rasterizer import PointsRasterizationSettings
import pdb
from pytorch3d.transforms.transform3d import Transform3d
from utils.constant import TEMPLATE_GARMENT, GARMENT_COLOR_MAP, FL_EXTRACT, FL_COLOR
import MCGpu
from pytorch3d.structures import Meshes,Pointclouds,join_meshes_as_batch
from pytorch3d.loss import (
    chamfer_distance,
    mesh_edge_loss,
    mesh_laplacian_smoothing,
    mesh_normal_consistency,
)
from collections import defaultdict
from engineer.core.fl_optimizer import fl_proj_loss
from model.CameraMine import PointsRendererWithFrags,RectifiedPerspectiveCameras
from pytorch3d.io import load_objs_as_meshes
from torch_scatter import scatter
import model.RenderNet as RenderNet
from pytorch3d.renderer import (
    RasterizationSettings,
    MeshRasterizer,
    SoftSilhouetteShader,
    TexturesVertex,
    BlendParams,
    PointsRasterizationSettings,
    # PointsRenderer,
    PointsRasterizer,
    PointLights,
    HardPhongShader,
    AlphaCompositor
)
import cv2
import trimesh
import openmesh as om
import os.path as osp
from MCAcc import Seg3dLossless
from model.Deformer import initialLBSkinner,getTranslatorNet,CompositeDeformer,LBSkinner
from pytorch3d.renderer.mesh.renderer import MeshRendererWithFragments
from engineer.networks.OptimNetwork import OptimNetwork
from pathlib import Path
from utils.common_utils import tocuda, trimesh2torch3d, tocpu, tensor2numpy, make_nograd_func, numpy2tensor, squeeze_tensor, unsqueeze_tensor, torch3dverts, make_recursive_func, upper_bound, make_recursive_meta_func
from utils.constant import TEMPLATE_GARMENT, GARMENT_FL_MATCH, FL_INFOS, ZBUF_THRESHOLD
from  pytorch3d.ops.knn import knn_points
from engineer.utils.mesh_utils import slice_garment_mesh, merge_meshes
from engineer.core.fl_optimizer import rigid_optimizer, scale_rigid_optimizer
from engineer.optimizer.lap_deform_optimizer import Laplacian_Optimizer
from engineer.optimizer.nricp_optimizer import NRICP_Optimizer_AdamW
from engineer.optimizer.surface_intesection import Surface_Intesection
from engineer.utils.matrix_transform import compute_rotation_matrix_from_ortho6d, icp_rotate_transfrom, center_transform, icp_rotate_center_transform, scale_icp_rotate_center_transform
from engineer.utils.transformation import pytorch3d_mesh_transformation, Rotate_Y_axis
from einops import repeat, rearrange
import torch.nn.functional as F
from model.Deformer import Inverse_Fl_Body
from engineer.utils.garment_structure import Intersect_Free_Curve
from engineer.utils.featureline_utils import get_curve_faces
from engineer.utils.smooth_poses import smooth_poses
from engineer.networks.OptimGarmentNetwork import OptimGarmentNetwork


# list dose not need to recursive
def make_recursive_func_except_list(func):
    def wrapper(vars):
        if isinstance(vars, tuple):
            return tuple([wrapper(x) for x in vars])
        elif isinstance(vars, dict):
            return {k: wrapper(v) for k, v in vars.items()}
        else:
            return func(vars)

    return wrapper
def bi_search(arr, left, right, target):
    while(left<=right):
        mid = (left+right)//2
        if arr[mid]==target:
            return mid
        elif arr[mid]>target:
            right-=1
        else:
            left+=1
    return len(arr)


def decode_color(x):
    ret = []
    while(x):
        ret.append(x % 256)
        x = x//256
    return ret
@make_recursive_func_except_list
def encode_color_map(var):
    if isinstance(var, list):
        return var[0] + (var[1]<<8) + (var[2]<<16)
    else:
        raise NotImplementedError

def red_like(tensor):
    red_tensor = torch.zeros_like(tensor)
    red_tensor[...,2] = 1

    return red_tensor

class OptimGarmentNetwork_LargePose(OptimGarmentNetwork):
    def __init__(self,TmpSdf,Deformer,accEngine,maskRender,netRender,conf=None, garment_net = [], garment_type = None, tmpBodyVs = None, tmpBodyFs = None, use_hand = True):
        super(OptimGarmentNetwork_LargePose, self).__init__(TmpSdf,Deformer,accEngine,maskRender,netRender,conf, garment_net , garment_type, tmpBodyVs, tmpBodyFs, use_hand )
        self.freeze_sdf()






    def freeze_sdf(self):
        '''when fitting large pose we do not optimize surface
        '''
        for garment_net in self.garment_nets:
            for para in garment_net.parameters():
                para.requires_grad = False
        for para in self.sdf.parameters():
            para.requires_grad = False

    def project_2d_loss(self, N, H, W, frame_ids, ratio, cameras, fl_masks, gt_fl_pts, device):
        '''This function is used to compute 2d project loss to explictly deform feature line curve meshes
        In order to guarante curvature of template curve meshes, computing laplacian_loss or edge loss is crucial, for feature curve moving

        NOTE that 2d projection loss is not constrain feature curve position due to invisible position
        In order to obtain better canonical space feature-curve we use sdf to constrain the feature curve position

        N: batch_size
        H: the heigth of image
        W: the width of image
        frame_ids: cur frame, using to debug
        ratio: position encoding parameters for sdfnetwork, deformernetwork, idr
        cameras: cameras is utilized to project feature curve meshes
        fl_masks:  flag to control compute project loss
        gt_fl_pts: ground-truth 2d visible feature line points

        ---------------------------------------
        Return:
            garment_fl_vs_list: using to draw debug project curve meshes
            visible_mask_list: using to draw debug project curve by zbuff
            label_mask_list: using to draw debug project curve by gt_mask
            gt_fl_pts_list: groundtruth 2d curve supervision using to draw debug

        return garment_fl_vs_list, visible_mask_list, label_mask_list, gt_fl_pts_list
            def_fl_normal: using to draw visible or invisible pts
        '''
        # TmpVnum
        # the number of vertices after marching cube
        # compute deform feature line meshes
        # compute project loss -> which is used to explict deform feature line meshes
        project_loss = 0.
        fl_garment_sdf_loss = 0.
        self.info['fl_loss']={}
        proj_img_size = repeat(torch.Tensor([W,H]).float(), 'c -> b c', b= N).to(device)
        d_cond_list, poses, trans, rendcond = self.get_grad_parameters(frame_ids, device)

        offset_fl_verts_list = torch.split(self.inter_free_curve(), 1, dim = 0)

        # computing visible or invisible area for feature curve
        # fl_vs_list      -> canonical space
        # def_fl_vs_list  -> camera view space
        # collecting training data
        # fix the bug gt_fl_mask group 9-29
        def_fl_vs_group, fl_fs_group, fl_vs_group, fl_vs_split_group, def_fl_normal_group, gt_fl_pts_group, gt_fl_masks_group = self.deform_feature_line(cameras,
                d_cond_list, fl_masks, [poses, trans], ratio, offset_fl_verts_list, gt_fl_pts, proj_img_size, N)

        def_fl_meshes_group = {}
        visible_mask_list = []
        label_mask_list = []

        for garment_name in self.garment_names:
            def_fl_vs = def_fl_vs_group[garment_name]
            fl_fs = fl_fs_group[garment_name]
            fl_mesh = merge_meshes(def_fl_vs, fl_fs)
            def_fl_meshes_group[garment_name] = fl_mesh
        for garment_idx, (d_cond, garment_name) in enumerate(zip(d_cond_list[1:], self.garment_names)):

            fl_vs = fl_vs_group[garment_name]
            fl_fs = fl_fs_group[garment_name]
            def_fl_normal = def_fl_normal_group[garment_name]
            def_fl_meshes = def_fl_meshes_group[garment_name]
            fl_vs_split = fl_vs_split_group[garment_name]
            gt_fl_pts = gt_fl_pts_group[garment_name]
            gt_fl_masks = gt_fl_masks_group[garment_name]

            garment_proj_loss, visible_masks, label_masks  = self.compute_fl_proj_loss(def_fl_meshes, def_fl_normal, [d_cond, [poses,trans]],
                    gt_fl_masks, gt_fl_pts, ratio, garment_name, fl_vs, fl_fs,
                    fl_vs_split, garment_idx, cameras, proj_img_size)
            project_loss += garment_proj_loss

            # compute garment_sdf_loss
            cano_fl_sdf =self.garment_nets[garment_idx](torch.cat(fl_vs, dim = 0), ratio).view(-1)
            cano_fl_sdf_loss=(cano_fl_sdf+self.sdfShrinkRadius).abs().mean()
            self.info['fl_loss']['pc_{}_loss_sdf'.format(garment_name)]=cano_fl_sdf_loss.item()
            fl_garment_sdf_loss += cano_fl_sdf_loss*(self.conf.get_float('fl_weight.sdf_weight') if 'fl_weight' in self.conf else 60.)
            visible_mask_list.append(visible_masks)
            label_mask_list.append(label_masks)


        self.fl_optimizer.zero_grad()
        loss = 0. * fl_garment_sdf_loss + 0. * project_loss

        loss.backward()
        self.fl_optimizer.step()
        # draw 2d project pts
        with torch.no_grad():
            garment_fl_vs_list = []
            gt_fl_pts_list = []

            for garment_name in self.garment_names:
                def_fl_meshes = def_fl_meshes_group[garment_name]
                def_fl_verts = torch.cat(def_fl_meshes, dim = 1)
                num_v = def_fl_verts.shape[0]
                screen_pts = cameras.transform_points_screen(def_fl_verts,  proj_img_size)
                fl_vs_split = fl_vs_split_group[garment_name]
                gt_fl_pts = gt_fl_pts_group[garment_name]
                fl_screen_pts = torch.split(screen_pts, fl_vs_split, dim =1)
                garment_fl_vs_list.append(fl_screen_pts)
                gt_fl_pts_list.append(gt_fl_pts)

        return garment_fl_vs_list, visible_mask_list, label_mask_list, gt_fl_pts_list


    def forward(self,datas,sample_pix,ratio,frame_ids,root=None,**kwargs):

        device=frame_ids.device
        gtCs=datas['img'].to(device)
        gtMs=datas['mask'].to(device)
        gt_fl_pts = datas['fl_pts'].to(device)
        fl_masks = datas['fl_masks'].to(device)

        if not self.is_upper_bottom:
            upper_mask =datas['upper'].to(gtMs)
            bottom_mask =datas['bottom'].to(gtMs)
            # NOTE the order of body mask and garment_mask
            gt_body_masks = datas['body'].to(gtMs)
            gt_garment_masks_list = [upper_mask, bottom_mask]
        else:
            upper_bottom_mask =datas['upper_bottom'].to(gtMs)
            # NOTE the order of body mask and garment_mask
            gt_garment_masks_list = [upper_bottom_mask]


        # smpl_vertics used to gereate body mask in garment area
        # rebuild the computation graph from dataset cpu to gpu

        N=gtCs.shape[0]
        focals,princeple_ps,Rs,Ts,H,W=self.dataset.get_camera_parameters(frame_ids.numel(),device)
        cameras=RectifiedPerspectiveCameras(focals,princeple_ps,Rs,Ts,image_size=[(W, H)]).to(device)
        self.maskRender.rasterizer.cameras=cameras

        if self.pcRender:
            self.pcRender.rasterizer.cameras=cameras
        self.info={}
        self.root=None

        # every remesh_intersect time it will update marching cube to rebuild mesh

        if self.body_vs is None or self.body_fs is None or self.forward_time%self.remesh_intersect==0:
            self.marching_cube_update(frame_ids, ratio, root, device)

        old_body_vs=self.body_vs.detach().clone() if self.root else None
        old_garment_vs =[garment.detach().clone() if self.root else None for garment in self.garment_vs]
        old_fl_deform_vs =[fl_deform_vert.detach().clone() if self.root else None for fl_deform_vert in self.inter_free_curve.inference()]
        total_loss = 0.

        # NOTE that project 2d loss would produce optnet parameters grad
        # NOTE that use all project loss no matter what visible or not, the final mesh leads some artifacts
        garment_fl_vs_list, fl_visible_mask_list, gt_fl_visible_mask_list, gt_fl_pts_list = self.project_2d_loss(N, H, W, frame_ids, ratio, cameras, fl_masks, gt_fl_pts, device)
        kwargs['global_optimizer'].zero_grad()

        def_garment_meshes, garment_masks_list, mgt_garment_masks_list, pc_sdf_loss, surface_ps_list = self.mask_loss(N, H, W, frame_ids, ratio, gt_garment_masks_list, device)
        batch_garment_inds_list,batch_garment_inds_list, batch_garment_row_list, batch_garment_col_list, batch_garment_init_pts_list, batch_garment_front_face_list = surface_ps_list
        total_loss += pc_sdf_loss

        d_cond_list, poses, trans, rendcond = self.get_grad_parameters(frame_ids, device)



        # TODO that [gt_fl_ps] needs to group
        self.save_debug(old_garment_vs, self.garment_fs, old_fl_deform_vs, def_garment_meshes, self.deformer.defs[0].offset, garment_fl_vs_list, fl_visible_mask_list, gt_fl_visible_mask_list, gt_fl_pts_list,
                garment_masks_list, gt_garment_masks_list, mgt_garment_masks_list, gtCs, batch_garment_inds_list, batch_garment_row_list, batch_garment_col_list, batch_garment_init_pts_list,
                [d_cond.detach() for d_cond in  d_cond_list[1:]], [poses.detach(), trans.detach()],
                rendcond, ratio, N)

        # sampling train rays NOTE idr need mask supervised
        sample_batch_garment_inds_list, sample_batch_garment_row_list, sample_batch_garment_col_list, sample_batch_garment_init_pts_list, sample_rays_list, cameras = self.sample_train_ray(N, sample_pix,
                gt_garment_masks_list, batch_garment_inds_list, batch_garment_row_list, batch_garment_col_list, batch_garment_init_pts_list,frame_ids, device)


        init_garment_ps_list, check_list = self.opt_garment_surface_ps( frame_ids, cameras, ratio, sample_rays_list,sample_batch_garment_init_pts_list, sample_batch_garment_inds_list, device)



        render_loss = self.surface_render_loss(datas, N, cameras, frame_ids, ratio, check_list, gtCs, init_garment_ps_list, sample_batch_garment_row_list,
            sample_batch_garment_col_list, sample_batch_garment_inds_list, sample_rays_list, device)
        total_loss += render_loss
        dct_loss = self.dct_poses_loss(poses, trans, frame_ids, N)
        total_loss += dct_loss

        self.remesh_time=np.floor(self.remesh_time)+float(self.forward_time%self.remesh_intersect)/float(self.remesh_intersect)
        self.info['remesh']=self.remesh_time
        self.forward_time+=1

        return total_loss



    def propagateTmpPsGrad(self,frame_ids,ratio):

        for garment_idx in range(self.garment_size):
            garment_name = self.garment_names[garment_idx]
            if self.TmpPs[garment_idx] == None or self.TmpPs[garment_idx].grad is None:
                self.info['{}_invInfo'.format(garment_name)]=(-1,-1)
                return
            device=self.TmpPs[garment_idx].device

            d_cond_list, poses, trans, _ = self.get_grad_parameters(frame_ids, device)
            defconds=[d_cond_list[1+garment_idx],[poses,trans]]

            focals,princeple_ps,Rs,Ts,H,W=self.dataset.get_camera_parameters(frame_ids.numel(),device)
            self.maskRender.rasterizer.cameras=RectifiedPerspectiveCameras(focals,princeple_ps,Rs,Ts,image_size=[(W, H)]).to(device)
            grad_l_p=self.TmpPs[garment_idx].grad


            # to make v can backward to camera parameters
            # after loss.backward(), the computation graph of self.rays to camera params has been destroyed, need to be recomputed
            if self.rays[garment_idx].requires_grad:
                v=self.maskRender.rasterizer.cameras.view_rays(torch.cat([self.col_inds[garment_idx].view(-1,1),self.row_inds[garment_idx].view(-1,1),torch.ones_like(self.col_inds[garment_idx].view(-1,1))],dim=-1).float())
            else:
                v=self.rays.detach()

            c=self.maskRender.rasterizer.cameras.cam_pos()
            p=self.TmpPs[garment_idx] #[N,3]
            f=self.garment_nets[garment_idx](p,ratio)

            grad_f_p=torch.autograd.grad(f,p,torch.ones_like(f),retain_graph=False)[0] #[N,3]


            if hasattr(self.deformer.defs[0],'enableSdfcond'):
                raise NotImplemented
                self.sdf(p,ratio)
                d=self.deformer(p,[[defconds[0],self.sdf.rendcond],defconds[1]],self.batch_inds,ratio=ratio)
            else:
                d=self.deformer(p,defconds,self.batch_inds[garment_idx],ratio=ratio, offset_type = garment_name) #[N,3]

            inputs=[p]
            grad_d_p=[]
            opt_defconds=[]


            # deform embedding, trans, and pose condition
            for defcond in defconds:
                if type(defcond) is list:
                    for defc in defcond:
                        if defc.requires_grad:
                            opt_defconds.append(defc)
                else:
                    if defcond.requires_grad:
                        opt_defconds.append(defcond)
            grad_outputs=torch.ones_like(d[...,0])


            outx=torch.autograd.grad(d[...,0],inputs,grad_outputs,retain_graph=True)
            grad_d_p.append(outx[0].view(-1,1,3))
            # if len(opt_defconds):
            #     grad_d_z.append([grad.view(-1,1,opt_defcond.shape[-1]) for grad,opt_defcond in zip(outx[1:],opt_defconds)])
            outy=torch.autograd.grad(d[...,1],inputs,grad_outputs,retain_graph=True)
            grad_d_p.append(outy[0].view(-1,1,3))
            # if len(opt_defconds):
            #     grad_d_z.append([grad.view(-1,1,opt_defcond.shape[-1]) for grad,opt_defcond in zip(outy[1:],opt_defconds)])
            outz=torch.autograd.grad(d[...,2],inputs,grad_outputs,retain_graph=False)
            grad_d_p.append(outz[0].view(-1,1,3))
            # if len(opt_defconds):
            #     grad_d_z.append([grad.view(-1,1,opt_defcond.shape[-1]) for grad,opt_defcond in zip(outz[1:],opt_defconds)])
            grad_d_p=torch.cat(grad_d_p,dim=1) #N,3,3
            # if len(opt_defconds):
            #     grad_d_z=[torch.cat([gradx,grady,gradz],dim=1) for gradx,grady,gradz in zip(grad_d_z[0],grad_d_z[1],grad_d_z[2])]
            v_cross=torch.zeros_like(grad_d_p)
            # cross matrix
            # [0 -a3 a2]
            # [a3 0 -a1]
            # [-a2 a1 0]


            v_cross[:,0,1]=-v[:,2]
            v_cross[:,0,2]=v[:,1]
            v_cross[:,1,0]=v[:,2]
            v_cross[:,1,2]=-v[:,0]
            v_cross[:,2,0]=-v[:,1]
            v_cross[:,2,1]=v[:,0]
            v_cross=v_cross.detach()

            # [B 4, 3]
            a1=v_cross.matmul(grad_d_p)
            b=torch.cat([grad_f_p.view(-1,1,3),a1],dim=1)
            # the gradient d_p_n need satisfied:
            # b @ d_p_n = 0

            btb=b.permute(0,2,1).matmul(b)
            # pseudo_inverse
            btb_inv,check=Fast3x3Minv(btb)
            self.info['{}_invInfo'.format(garment_name)]=(check.numel(),check.sum().item())


            # (A^t @ A) ^ -1 @ A^t


            # rhs_1 = (A^t @ A) ^ -1 @ A^t -> [N, 3, 4]
            # grad_l_p -> [N, 3] -> [N, 1, 3]
            # grad_l_p @ pesudo_inverse A^t
            rhs_1=btb_inv.matmul(b.permute(0,2,1)) #N,3,4
            rhs_1=grad_l_p.view(-1,1,3).matmul(rhs_1) #N,1,4

            loss=0.
            #for phi
            params=[param for param in self.deformer.parameters() if param.requires_grad]

            if hasattr(self.deformer.defs[0],'enableSdfcond'):
                self.sdf(p,ratio)
                d=self.deformer(p,[[defconds[0],self.sdf.rendcond],defconds[1]],self.batch_inds,ratio=ratio)
            else:
                d=self.deformer(p,defconds,self.batch_inds[garment_idx],ratio=ratio, offset_type = garment_name) #[N,3]

            temp=(rhs_1[:,:,-3:].matmul(-v_cross)).view(-1,3)

            if len(opt_defconds):
                params_grads=torch.autograd.grad(d,params,temp,retain_graph=True)
            else:
                params_grads=torch.autograd.grad(d,params,temp,retain_graph=False)
            for param,grad in zip(params,params_grads):
                loss+=(param*grad).sum()
            #for z
            if len(opt_defconds):
                params_grads=torch.autograd.grad(d,opt_defconds,temp,retain_graph=False)
                for opt_defcond,grad in zip(opt_defconds,params_grads):
                    loss+=(opt_defcond*grad).sum()

            if v.requires_grad:
                dc=d.detach()-c.detach().view(1,3)
                dc_cross=torch.zeros_like(grad_d_p)
                dc_cross[:,0,1]=-dc[:,2]
                dc_cross[:,0,2]=dc[:,1]
                dc_cross[:,1,0]=dc[:,2]
                dc_cross[:,1,2]=-dc[:,0]
                dc_cross[:,2,0]=-dc[:,1]
                dc_cross[:,2,1]=dc[:,0]
                grad=rhs_1[:,:,-3:].matmul(dc_cross).view(-1,3)
                # to do: how to propagate v and c grads to camera parameters?
                loss+=(v*grad).sum()

            if c.requires_grad:
                grad=-temp.sum(0)
                loss+=(c*grad).sum()
            loss.backward()

