"""
@File: OptimGarmentNetwork.
@Author: Lingteng Qiu
@Email: qiulingteng@link.cuhk.edu.cn
@Date: 2022-10-01
@Desc: REC-MV-core codes

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
from utils.constant import TEMPLATE_GARMENT, GARMENT_COLOR_MAP, FL_EXTRACT, FL_COLOR, CURVE_AWARE, SMOOTH_TRANS, RENDER_COLORS
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
from engineer.utils.polygons import uniformsample3d
from model.Deformer import initialLBSkinner,getTranslatorNet,CompositeDeformer,LBSkinner
from pytorch3d.renderer.mesh.renderer import MeshRendererWithFragments
from engineer.networks.OptimNetwork import OptimNetwork
from pathlib import Path
from utils.common_utils import tocuda, trimesh2torch3d, tocpu, tensor2numpy, make_nograd_func, numpy2tensor, squeeze_tensor, unsqueeze_tensor, torch3dverts, make_recursive_func, upper_bound, make_recursive_meta_func
from utils.constant import TEMPLATE_GARMENT, GARMENT_FL_MATCH, FL_INFOS, ZBUF_THRESHOLD
from  pytorch3d.ops.knn import knn_points
from engineer.utils.mesh_utils import slice_garment_mesh, merge_meshes
from engineer.core.fl_optimizer import rigid_optimizer, scale_rigid_optimizer
from engineer.optimizer.lap_deform_optimizer import Laplacian_Optimizer, Laplacian_Deform_upper_and_domn_Optimzier
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
from  pytorch3d.structures import Meshes
from pytorch3d.renderer.mesh.renderer import MeshRendererWithFragments

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
            right= mid-1
        else:
            left=mid+1
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

class OptimGarmentNetwork(OptimNetwork):
    def __init__(self,TmpSdf,Deformer,accEngine,maskRender,netRender,conf=None, garment_net = [], garment_type = None, tmpBodyVs = None, tmpBodyFs = None, use_hand = True):

        super(OptimGarmentNetwork, self).__init__(TmpSdf,Deformer,accEngine,maskRender,netRender,conf=conf.get_config('loss_coarse'))
        self.garment_nets= nn.ModuleList()

        self.garment_nets.extend(garment_net)
        self.garment_type = garment_type
        self.register_buffer('tmpBodyVs',tmpBodyVs)
        self.register_buffer('tmpBodyFs',tmpBodyFs)
        self.use_hand = True
        # add head infos
        if not self.use_hand:
            head_ply = trimesh.load('../smpl_clothes_template/clothes_template/head.obj', process= False)
            self.static_pts_idx = None
        else:
            head_ply = trimesh.load('../smpl_clothes_template/clothes_template/portrait.obj', process= False)

            garment_names = TEMPLATE_GARMENT[garment_type]

            if not 'long_sleeve_upper' in garment_names:
                self.load_smpl_static_pts()
            else:
                self.static_pts_idx = None

        head_colors = head_ply.visual.vertex_colors[..., 0]
        self.head_pts_idx = np.where(head_colors == 255)[0]


        self.garment_template_path = Path(conf.get_string('train.garment_template_path'))
        try:
            self.is_upper_bottom = conf.get_bool('train.is_upper_bottom')
        except:
            self.is_upper_bottom = False

        self.init_template(self.garment_template_path)

        self.garment_names = TEMPLATE_GARMENT[garment_type]
        self.fl_names =FL_INFOS[garment_type]


        self.garment_size = len(TEMPLATE_GARMENT[garment_type])
        # body vertices and garment vertices define
        self.body_vs =None
        self.body_fs=None
        self.garment_vs =None
        self.garment_fs =None
        self.fl_deform_verts = None

        # garment_engine setting
        self.garment_engine = None
        self.visualizer = None

    def load_init_sdf_vertices(self,tmp_sdf_mesh):
        self.register_buffer('tmp_sdf_body_vs',torch.from_numpy(tmp_sdf_mesh.vertices).float())
        self.register_buffer('tmp_sdf_face_vs',torch.from_numpy(tmp_sdf_mesh.faces).float())



    def load_smpl_static_pts(self):
        static_ply = trimesh.load('../smpl_clothes_template/clothes_template/smpl_static.obj', process= False)

        static_pts = static_ply.visual.vertex_colors[..., 0]
        static_pts_idx = np.where(static_pts == 255)[0]
        self.static_pts_idx = static_pts_idx


    def accumulate_template(self, body_name, need_joints = True):
        body_mesh = []
        for body in body_name:
            if body in self.dp3d_template.keys():
                body_mesh.append(self.dp3d_template[body])
        return join_meshes_as_batch(body_mesh) if need_joints else body_mesh
    def init_template(self, garment_template_path, add_head = True):
        # to compute boudnary_field
        self.__load_smpl_garment_tempalte(garment_template_path, add_head = add_head)
        self.__load_deepfashion3d_template(garment_template_path)
        align_smpl_tmp = trimesh.load('../smpl_clothes_template/aligned_smpl.obj', process=False)

        align_smpl_verts = torch.from_numpy(align_smpl_tmp.vertices).float()
        vertices = tocuda(align_smpl_verts)[None]
        dp3d_verts = torch3dverts(self.dp3d_template)
        smpl_boundary_colors = dict()
        bg_color = encode_color_map([125,125,125])
        # obtain boundary_infos


        for garment_type in TEMPLATE_GARMENT[self.garment_type]:
            add_v_id = self.template_add_v_id[garment_type]
            dp3d_vert = dp3d_verts[garment_type]
            dp3d_color = self.dp3d_template_color[garment_type]

            dp3d_vert=  tocuda(dp3d_vert)
            dist = knn_points(vertices, dp3d_vert)
            knearest = dist.idx
            smpl_v_color = dp3d_color[knearest]
            smpl_v_color[0 ,add_v_id, ...] = encode_color_map([125,125,125])
            smpl_boundary_colors[garment_type] = smpl_v_color

            # NOTE that the following initialize boundary method
            # is hard to converge.

            # add_v_id = self.template_add_v_id[garment_type]
            # dp3d_vert = dp3d_verts[garment_type]
            # dp3d_color = self.dp3d_template_color[garment_type]
            # dp3d_vert=  tocuda(dp3d_vert)
            # # dist = knn_points(vertices, dp3d_vert)
            # # knearest = dist.idx
            # dist = knn_points(dp3d_vert, vertices)
            # knearest = dist.idx[0,...,0]
            # bg_idx = (dp3d_color == bg_color)
            # knearest = knearest[torch.logical_not(bg_idx)]
            # # smpl is 6890 vertices
            # smpl_v_color = scatter(dp3d_color[torch.logical_not(bg_idx)].cuda(), knearest,reduce='mean', dim_size=6890).detach().cpu()
            # smpl_v_color[smpl_v_color ==0.] = bg_color
            # smpl_v_color = smpl_v_color[None, ..., None]
            # smpl_v_color[0 ,add_v_id, ...] = bg_color
            # smpl_boundary_colors[garment_type] = smpl_v_color

        self.smpl_boundary_colors = smpl_boundary_colors
    def __load_smpl_garment_tempalte(self, template_path, add_head = True):
        '''
        load_smpl garment indx
        '''

        def load_template(temp):

            try:
                smpl_id = np.load(temp, allow_pickle= True)['smpl_id']
            except:
                smpl_id = np.load(temp, allow_pickle= True).item()['smpl_id']

            return smpl_id

        def find_face(vid):

            faces = self.tmpBodyFs.detach().cpu().numpy()
            faces = faces.reshape(-1)
            vertex_set = set(vid)

            faces_belongs =np.asarray([face in vertex_set for face in faces])
            faces_belongs = faces_belongs.reshape(-1,3)
            faces_sum = np.sum(faces_belongs, axis=1)
            face_idx = np.where(faces_sum == 3)[0]


            return face_idx

        # NOTE that the deepfashion3d model has two strange pts, we remove it at the beginning
        remove_id = [5898,2568]
        need_remove_strange_pts = ['long_pants','long_skirt','short_pants','skirt']
        self.template_id = sorted(template_path.glob('smpl_clothes_map/*.pkl'))


        self.template_v_id = {}
        self.template_f_id = {}
        self.template_add_v_id ={}
        head_pts_set = set(self.head_pts_idx.tolist())

        for template_id in self.template_id:
            add_v_set = []
            template_name = str(template_id.stem)
            if template_name in need_remove_strange_pts:
                v_id = load_template(template_id)

                for target in remove_id:
                    idx = bi_search(v_id, 0, len(v_id), target)
                    v_id.pop(idx)
            else:
                v_id = load_template(template_id)
                template_set = set(v_id)
                overlap_set = template_set & head_pts_set
                add_v_set = list(head_pts_set - overlap_set)


                if add_head:
                    v_id.extend(self.head_pts_idx.tolist())
                v_id = np.unique(v_id).tolist()


            self.template_v_id[template_name] = v_id
            self.template_f_id[template_name] = find_face(v_id)
            self.template_add_v_id[template_name] = add_v_set



    def __load_deepfashion3d_template(self, template_path):
        # load_deepfashion3d_color infos and also help us to find boudary infos
        dp3d_template = sorted(template_path.glob('clothes_template/*.ply'))
        extra_template = sorted(template_path.glob('clothes_template/tube.obj'))
        extra_template.extend(sorted(template_path.glob('clothes_template/dress.obj')))
        dp3d_template.extend(extra_template)

        self.dp3d_template = {}
        self.dp3d_template_color = {}
        color_map  = lambda x: x[...,0]+ (x[...,1]<<8) + (x[...,2] <<16)
        for template_id in dp3d_template:

            template_name = str(template_id.stem)
            trimesh_ply = trimesh.load_mesh(template_id, process =False)
            # color_map = trimesh_ply.visual.vertex_colors
            self.dp3d_template[template_name] =trimesh_ply
            colors = (trimesh_ply.visual.vertex_colors[..., :3]).astype(np.int32)
            colors = color_map(colors)
            self.dp3d_template_color[template_name] = colors
            uni_colors = np.unique(colors)
            uni_colors = [decode_color(x) for x in uni_colors]

        self.dp3d_template = trimesh2torch3d(self.dp3d_template)
        self.dp3d_template_color = numpy2tensor(self.dp3d_template_color)


    def obtain_body_pts_id(self, body_name, need_joints = True):

        '''
        return body_idx and body face
        '''
        body_id = [ ]
        face_id = []

        for body in body_name:
            if need_joints:
                body_id.extend((self.template_v_id[body]))
                face_id.extend((self.template_f_id[body]))
            else:
                body_id.append(self.template_v_id[body])
                face_id.append(self.template_f_id[body])
        return (np.unique(body_id), np.unique(face_id)) if need_joints else (body_id, face_id)
    def garment_by_init_smpl(self):
        '''
        obtain init smpl given canonical smpl
        '''

        # save_obj('predict_smpl.obj',vertices[0],torch.from_numpy(self.faces.astype(np.int32)))
        # xxx
        #smpl_idx, smpl_face
        body_idx, face_idx = self.obtain_body_pts_id(TEMPLATE_GARMENT[self.garment_type], False)


        slice_mesh_list = []
        garment_color_map = encode_color_map(GARMENT_COLOR_MAP)
        vertices = self.tmpBodyVs.detach().cpu()[None]
        faces = self.tmpBodyFs.detach().cpu().numpy()
        for body_id, face_id, garment_name in zip(body_idx, face_idx, TEMPLATE_GARMENT[self.garment_type]):
            # 6890 -> colors

            smpl_colors = self.smpl_boundary_colors[garment_name]
            vertices_colors= smpl_colors[:, body_id]
            smpl_verts = vertices[:, body_id]
            smpl_faces = faces[face_id, :]



            # vert, face, color
            slice_mesh_list.append(slice_garment_mesh(body_id, smpl_faces, smpl_verts,
                vertices_colors, decode_color, boundary_color_map = garment_color_map[garment_name],
                garment_type= garment_name, static_pts_idx = self.static_pts_idx))
        # for slice_mesh, slice_name in zip(slice_mesh_list,TEMPLATE_GARMENT[self.garment_type]):
        #     slice_mesh.save_obj(os.path.join('./debug/', 'tmp_{}.obj'.format(slice_name)))


        return slice_mesh_list


    def initializeSDF(self, network, optimizer, sche, batch_size, nepochs, device, vs, ns, with_normals, save_name):

        for epoch in range(1, nepochs + 1):
            permute=torch.randperm(vs.shape[0])
            evs=vs[permute]
            ens=ns[permute]

            evs=torch.split(evs,batch_size)
            ens=torch.split(ens,batch_size)

            for data_index,(mnfld_pnts, normals) in enumerate(zip(evs,ens)):
                mnfld_pnts = mnfld_pnts.to(device)

                if with_normals:
                    normals = normals.to(device)
                # sample pts, containing global sampling and local smapling with 0.01 sigma
                nonmnfld_pnts = utils.sample_points(mnfld_pnts, 1.8,0.01)

                # forward pass

                mnfld_pnts.requires_grad_()
                nonmnfld_pnts.requires_grad_()

                mnfld_pred = network(mnfld_pnts,-1)
                nonmnfld_pred = network(nonmnfld_pnts,-1)

                mnfld_grad = network.gradient(mnfld_pnts, mnfld_pred)
                nonmnfld_grad = network.gradient(nonmnfld_pnts, nonmnfld_pred)

                # manifold loss
                mnfld_loss = (mnfld_pred.abs()).mean()
                # eikonal loss
                grad_loss = ((nonmnfld_grad.norm(2, dim=-1) - 1) ** 2).mean()

                loss = mnfld_loss + 0.1 * grad_loss
                # loss=0
                # normals loss
                if with_normals:
                    normals = normals.view(-1, 3)
                    # print(mnfld_grad[:10,:])
                    # print(mnfld_grad.norm(2, dim=1).mean())
                    normals_loss = ((mnfld_grad - normals).abs()).norm(2, dim=1).mean()
                    loss = loss + 1.0 * normals_loss
                else:
                    normals_loss = torch.zeros(1)
                # back propagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # print status
                if data_index == len(evs)-1:
                    print('Train Epoch: {}\tTrain Loss: {:.6f}\tManifold loss: {:.6f}'
                          '\tGrad loss: {:.6f}\tNormals Loss: {:.6f}'.format(
                        epoch, loss.item(), mnfld_loss.item(), grad_loss.item(), normals_loss.item()))
            sche.step()
        torch.save(network.state_dict(),save_name)

    def align_init_temp(self, garment_templates, fl_templates):
        '''
        given fl points supervised by fl
        we registry garment templates according to fl_vertices especially for skirts
        generally, the method we choose laplacian deformation and laplacian smooth to registry the template to meshes
        '''

        # for garment_template, garment_name in zip(garment_templates, self.garment_names):
        #     garment_template.save_obj('./debug/garment_fl/{}.obj'.format(garment_name))
        # fl_templates = tocpu(fl_templates)
        # for fl_idx, fl_template in enumerate(fl_templates):
        #     save_obj('./debug/garment_fl/{}.obj'.format(self.fl_names[fl_idx]), fl_template.verts_packed(), fl_template.faces_packed())
        align_garment_meshes_packed  = self.garment_engine['fl_init_registry'](source_fl_meshes = garment_templates, target_meshes = fl_templates, source_type = self.garment_names, target_fl_type = self.fl_names, outlayer = False)
        garment_templates = align_garment_meshes_packed['source_fl_meshes']

        # for garment_idx, garment_template in enumerate(garment_templates):
        #     garment_template.save_obj('./debug/garment_fl/align_{}_boundary.obj'.format(self.garment_names[garment_idx]))
        # xxx


        return garment_templates


    def initializeFL(self, dataloader, n_epochs, device, save_mesh_name):
        '''
        feature lines initialized according to projection loss,
        and then we has initial feature_line meshes
        '''
        save_path = '/'.join(save_mesh_name.split('/')[:-1])
        save_fl_path = os.path.join(save_path, 'fl_init')
        os.makedirs(save_fl_path, exist_ok = True)
        # only rigid optimize cannot represent hemline
        # init_fl_meshes = rigid_optimizer(self.deformer.defs[1], self.garment_fl_templates, self.dataset, dataloader, save_fl_path, FL_INFOS[self.garment_type])

        smpl_vs = self.tmpBodyVs
        smpl_fs = self.tmpBodyFs

        smpl_mesh = Meshes([smpl_vs], [smpl_fs])

        init_fl_meshes = scale_rigid_optimizer(self.deformer.defs[1], self.garment_fl_templates, smpl_mesh, self.maskRender, self.dataset, dataloader, save_fl_path, FL_INFOS[self.garment_type])

        return init_fl_meshes



    def initializeTmpSDF(self,nepochs,save_name,with_normals=False, dataloader = None):
        '''
        Initialize For garment sdf, body sdf, initalize all garment mesh
        '''

        def merge_fl_meshes(fl_mesh_list):
            merge_fl = fl_mesh_list[0]
            for merge_fl_idx in range(1, len(fl_mesh_list)):
                merge_fl.update(fl_mesh_list[merge_fl_idx])
            return merge_fl

        network=self.sdf
        # change back to train mode
        network.train()
        optimizer = torch.optim.Adam(
            [
                {
                    "params": network.parameters(),
                    "lr": 0.005,
                    "weight_decay": 0
                }
            ]
            )
        sche=torch.optim.lr_scheduler.StepLR(optimizer,500,0.5)
        vs=self.tmpBodyVs
        device=vs.device


        garment_templates = self.garment_by_init_smpl()
        # dense_boundary OK
        dense_garment_templates = []
        for garment_idx, garment_template in enumerate(garment_templates):
            # edge dense pc times
            for __ in range(2):
                garment_template = garment_template.dense_boundary()
            dense_garment_templates.append(garment_template)
            # debug for garment_template
            # garment_template.save_obj('./debug/boundary/{}_dense_template.obj'.format(self.garment_names[garment_idx]))
            # garment_template.save_boudnary('./debug/boundary/{}_boundary.ply'.format(self.garment_names[garment_idx]), *garment_template.get_fields())

        # extract_garment_template_mesh

        garment_fl_templates = [dense_garment_template.extract_featurelines() for dense_garment_template in dense_garment_templates]
        self.garment_fl_templates = merge_fl_meshes(garment_fl_templates)


        # for garment_feature_name in self.garment_features_templates.keys():
        #     garment_features = self.garment_features_templates[garment_feature_name]
        #     save_obj('./debug/boundary/{}_boundary.obj'.format(garment_feature_name), garment_features.verts_packed(), garment_features.faces_packed())

        print('fitting_init_template_featurelines')

        align_fl_meshes = self.initializeFL(dataloader, nepochs, device,save_name)
        garment_templates = self.align_init_temp(dense_garment_templates, align_fl_meshes)

        for garment_idx, garment_name in enumerate(self.garment_names):
            garment_templates[garment_idx].save_obj('./debug/align_{}_garment.obj'.format(garment_name))

        close_garment_templates = [garment_template.close_hole(garment_name) for garment_template, garment_name in zip(garment_templates, self.garment_names)]


        if with_normals and self.tmpBodyNs is None:
            with_normals=False
        if with_normals:
            ns=self.tmpBodyNs
        else:
            ns=torch.ones_like(vs)/np.sqrt(3)

        # initialized_body
        print('Fitting_body_net!')
        self.initializeSDF(network, optimizer, sche, 5000, nepochs, device, vs, ns, with_normals, save_name)

        # initialized_garement
        for garment_name, inputs, garment_network in zip(TEMPLATE_GARMENT[self.garment_type], close_garment_templates, self.garment_nets):
            print('Fitting_garment_net {}!'.format(garment_name))
            garment_save_name = save_name.replace("sdf",'sdf_{}'.format(garment_name))
            garment_network.train()
            optimizer = torch.optim.Adam(
                [
                    {
                        "params": garment_network.parameters(),
                        "lr": 0.005,
                        "weight_decay": 0
                    }
                ]
                )
            sche=torch.optim.lr_scheduler.StepLR(optimizer,500,0.5)
            garment_vs, garment_ns = tocuda(inputs)
            self.initializeSDF(garment_network, optimizer, sche, 5000, nepochs, device, garment_vs, garment_ns, with_normals, garment_save_name)


    def discretizeSDF(self,ratio,engine=None,balance_value=0.):
        def __discretizeSDF(query_func, ratio, engine, balance_value):
            if engine is None:
                engine=self.engine
            engine.balance_value=balance_value
            engine.query_func=query_func
            sdfs=engine.forward()
            # each bins size, in our space
            verts, faces=MCGpu.mc_gpu(sdfs[0,0].permute(2,1,0).contiguous(),engine.spacing_x,engine.spacing_y,engine.spacing_z,engine.bx,engine.by,engine.bz,balance_value)

            return verts, faces

        def body_query_func(points):
            with torch.no_grad():
                return self.sdf.forward(points.reshape(-1,3),ratio).reshape(1,1,-1)

        def nograd_garment_query_func(func):
            def wrapper(points):
                with torch.no_grad():
                    ret = func.forward(points.reshape(-1,3),ratio).reshape(1,1,-1)
                return ret
            return wrapper

        pts_list = []
        faces_list = []
        # body_marching cube
        body_pts, body_face = __discretizeSDF(body_query_func,ratio,engine=None,balance_value=0.)
        pts_list.append(body_pts)
        faces_list.append(body_face)
        for garment_network in self.garment_nets:
            query_func = nograd_garment_query_func(garment_network)
            # query_func = garment_query_func(func)
            # garment_marching cube
            garment_pts, garment_faces = __discretizeSDF(query_func,ratio,engine=None,balance_value=0.)
            pts_list.append(garment_pts)
            faces_list.append(garment_faces)

        return pts_list, faces_list


    def compute_garment_pc_loss(self, defMeshes, defconds, imgs, gtMs, ratio, garment_type, garment_vs, garment_fs, garment_idx):
        # Mask loss
        N=gtMs.shape[0]
        masks=imgs[...,-1]

        mask_loss=(1.-(masks*gtMs).view(N,-1).sum(1)/(masks+gtMs-masks*gtMs).abs().view(N,-1).sum(1)).mean()

        self.info['pc_loss']['{}_mask_loss'.format(garment_type)]=mask_loss.item()
        loss=mask_loss*(self.conf.get_float('pc_weight.mask_weight') if 'pc_weight.mask_weight' in self.conf else 1.)


        garment_mesh = Meshes(verts=[garment_vs],faces=[garment_fs])
        lap_weight=self.conf.get_float('pc_weight.laplacian_weight') if 'pc_weight' in self.conf else -1.


        # template loss : laplacian_loss, edge_loss, norm loss
        if lap_weight>0.:
            lap_loss=lap_weight*mesh_laplacian_smoothing(tmpMesh,method='uniform')
            loss=loss+lap_loss
            self.info['pc_loss']['lap_loss']=lap_loss.item()/lap_weight
        edge_weight=self.conf.get_float('pc_weight.edge_weight') if 'pc_weight' in self.conf else -1.
        if edge_weight>0.:
            edge_loss=edge_weight*mesh_edge_loss(tmpMesh,target_length=0.)
            loss=loss+edge_loss
            self.info['pc_loss']['edge_loss']=edge_loss.item()/edge_weight
        norm_weight=self.conf.get_float('pc_weight.norm_weight') if 'pc_weight' in self.conf else -1.
        if norm_weight>0.:
            norm_loss=norm_weight*mesh_normal_consistency(tmpMesh)
            loss=loss+norm_loss
            self.info['pc_loss']['norm_loss']=norm_loss.item()/norm_weight
        consistent_weight=self.conf.get_float('pc_weight.def_consistent.weight') if 'pc_weight.def_consistent' in self.conf else -1.
        if consistent_weight>0.:
            # defs[1] -> LBSkinner
            # Note that this is consistency the deform template is close to original template
            offset2=(defMeshes.verts_padded()-self.deformer.defs[1](garment_vs.view(1,-1,3).expand(N,-1,3),defconds[1]))
            offset2=(offset2*offset2).sum(-1)
            if self.conf.get_float('pc_weight.def_consistent.c')>0.:
                consistent_loss=utils.GMRobustError(offset2,self.conf.get_float('pc_weight.def_consistent.c'),True).mean()
            else:
                consistent_loss=torch.sqrt(offset2).mean()
            self.info['pc_loss']['garment_{}_defconst_loss'.format(garment_type)]=consistent_loss.item()
            loss=loss+consistent_loss*consistent_weight

        # self.TmpOptimzier is optimal the vertices of template_field


        return loss


    def get_grad_parameters(self, frame_ids, device):
        poses,trans,d_cond,rendcond=self.dataset.get_grad_parameters(frame_ids,device)

        deform_cond_split_size = split_size = [d_cond.shape[-1] // (self.garment_size +1) for i in range(self.garment_size + 1)]
        d_cond_list = torch.split(d_cond, deform_cond_split_size, dim =-1)

        return d_cond_list, poses, trans, rendcond

    def marching_cube_update(self, frame_ids, ratio, root, device):
        ''' marching cube to obtain explict garment mesh
        which is employed to following explict deform by point clound mask loss

        ratio: position encoding parameters for sdfnetwork, deformernetwork, idr
        device: cur_gpu device
        return:
            None
        '''
        save_fl_mesh_path = os.path.join(root, 'fl_mesh_show')
        os.makedirs(save_fl_mesh_path, exist_ok = True)

        print('tmp sdf generate')
        # marching cube ->16w,  8w 7 w
        vs_list, fs_list=self.discretizeSDF(ratio,None,-self.sdfShrinkRadius)
        self.body_vs, self.body_fs = vs_list[0], fs_list[0]
        self.garment_vs, self.garment_fs = vs_list[1:], fs_list[1:]
        max_garment_vs = max([garment_vs.shape[0] for garment_vs in self.garment_vs])
        max_garment_fs = max([garment_fs.shape[0] for garment_fs in self.garment_fs])
        self.update_hierarchical_config(device, max_garment_vs, max_garment_fs)


        if self.body_vs.shape[0]==0:
            print('tmp sdf vanished...')
            assert(False)
        self.remesh_time=1.+np.floor(self.remesh_time)
        self.body_vs.requires_grad=True
        for garment_vs in self.garment_vs:
            garment_vs.requires_grad=True

        # optimizer the sdf_vertices
        self.TmpOptimizer = torch.optim.SGD([self.body_vs, *self.garment_vs], lr=0.05, momentum=0.9)
        self.garment_optimizer = torch.optim.SGD(self.garment_vs, lr=0.05, momentum=0.9)
        # self.fl_optimizer = torch.optim.AdamW(self.inter_free_curve.parameters(), lr=1e-1)
        self.fl_optimizer = torch.optim.AdamW(self.inter_free_curve.parameters(), lr=1e-4)


        tm=om.TriMesh(self.body_vs.detach().cpu().numpy(),self.body_fs.detach().cpu().numpy())
        # this vertices belongs to which faces
        TmpFid=torch.from_numpy(tm.vertex_face_indices().astype(np.int64)).to(device)
        TmpVid=torch.arange(0,TmpFid.shape[0]).view(-1,1).repeat(1,TmpFid.shape[1]).to(device)
        sel=TmpFid>=0
        # this is indices -> Fid, belongs to which face, Vid belongs to which vertices_id
        self.TmpFid=TmpFid[sel].view(-1)
        self.TmpVid=TmpVid[sel].view(-1)

        self.garment_tmpfid = []
        self.garment_tmpvid = []

        for garment_vs, garment_fs in zip(self.garment_vs, self.garment_fs):
            tm=om.TriMesh(garment_vs.detach().cpu().numpy(),garment_fs.detach().cpu().numpy())
            # this vertices belongs to which faces
            garment_tmpfid=torch.from_numpy(tm.vertex_face_indices().astype(np.int64)).to(device)
            garment_tmpvid=torch.arange(0,garment_tmpfid.shape[0]).view(-1,1).repeat(1,garment_tmpfid.shape[1]).to(device)
            sel=garment_tmpfid>=0
            # this is indices -> Fid, belongs to which face, Vid belongs to which vertices_id
            self.garment_tmpfid.append(garment_tmpfid[sel].view(-1))
            self.garment_tmpvid.append(garment_tmpvid[sel].view(-1))

        # draw garment_vs and garment_fs
        self.root=root
        if self.visualizer is not None:
            self.visualize_curve_mesh(save_fl_mesh_path, int(self.opt_times), device)

    def find_surface_ps(self,def_garment_meshes):
        ''' find visible surface ps along camera view
        NOTE that the maskRender could return frags, which contains barycentric pts position
        def_garment_meshes: view-posed mesh

        ------------------------
        return:
        batch_garment_inds_list:          batch query idx
        batch_garment_row_list:           pix-row idx
        batch_garment_col_list:           pix-col idx
        batch_garment_init_pts_list:      surface pts in canonical space if it can be viewed in current camera position.
        batch_garment_front_face_list:    surface faces id in canonical space
        '''
        batch_garment_inds_list = []
        batch_garment_row_list = []
        batch_garment_col_list = []
        batch_garment_init_pts_list = []
        batch_garment_front_face_list = []

        with torch.no_grad():
            # body find surface
            # __, frags=self.maskRender(def_body_meshes)
            # batch_body_inds,row_body_inds,col_body_inds,init_body_pts,front_body_face_ids = utils.FindSurfacePs(self.body_vs.detach(),self.body_fs,frags)
            # garment find surface
            for def_garment_mesh, garment_vs, garment_fs in zip(def_garment_meshes, self.garment_vs, self.garment_fs):
                __, frags=self.maskRender(def_garment_mesh)
                # masks=(frags.pix_to_face>=0).float()[...,0]
                # masks = masks>0.
                # masks = masks.detach().cpu().numpy()
                # # debug for segmentt
                # for mask_idx, mask in enumerate(__):
                #     mask = (mask * 255).detach().cpu().numpy().astype(np.uint8)[...,:3]
                #     mask[~masks[0]] = [0,0,0]
                #     cv2.imwrite("./debug/garment/{:04d}.png".format(mask_idx), mask)
                batch_garment_inds,row_garment_inds,col_garment_inds,init_garment_pts,front_garment_face_ids = utils.FindSurfacePs(garment_vs.detach(), garment_fs, frags)
                batch_garment_inds_list.append(batch_garment_inds)
                batch_garment_row_list.append(row_garment_inds)
                batch_garment_col_list.append(col_garment_inds)
                batch_garment_init_pts_list.append(init_garment_pts)
                batch_garment_front_face_list.append(front_garment_face_ids)

        return batch_garment_inds_list,batch_garment_inds_list, batch_garment_row_list, batch_garment_col_list, batch_garment_init_pts_list, batch_garment_front_face_list



    def curve_aware_loss(self, ratio):
        # only for bottom clothes

        ca_loss = 0.
        if self.conf.get_float('pc_weight.curve_aware_weight') ==0.:
            return ca_loss

        if 'upper_bottom' in self.fl_names:
            for fl_idx, name in enumerate(self.fl_names):
                if name == 'upper_bottom':
                    break

            curve_pts = self.inter_free_curve.forward()[fl_idx]
            curve_center = curve_pts.mean(0, keepdim=True)
            curve_mesh_pts = torch.cat([curve_pts, curve_center], dim = 0)
            curve_range  = torch.arange(curve_pts.shape[0])
            center_idx = torch.zeros(curve_pts.shape[0]-1)
            center_idx[...] = curve_pts.shape[0]
            center_idx = center_idx.long()
            faces = torch.cat([curve_range[:-1, None], curve_range[1:, None], center_idx[:,None]], dim = 1)
            faces = torch.cat([faces, torch.tensor([curve_range[-1], curve_range[0], curve_pts.shape[0]])[None,:].long()], dim = 0)
            circle_mesh = trimesh.Trimesh(curve_mesh_pts.detach().cpu(),faces,process =False)
            surface_pts = torch.from_numpy(circle_mesh.sample(50000)).float().cuda()
            mnfld_pred=self.garment_nets[-1](surface_pts,ratio).view(-1)
            circle_sdf_loss=(mnfld_pred+self.sdfShrinkRadius).abs().mean()
            self.info['pc_{}_circle_loss_sdf'.format('upper_bottom')]= circle_sdf_loss.item()
            ca_loss += circle_sdf_loss * (self.conf.get_float('pc_weight.curve_aware_weight') if 'pc_weight' in self.conf else 60.)


        if self.garment_type in CURVE_AWARE:
            target_name = CURVE_AWARE[self.garment_type]
            for fl_idx, name in enumerate(self.fl_names):
                if name == target_name:
                    break

            if self.isfine:
                curve_pts = self.inter_free_curve.forward()[fl_idx]
                curve_center = curve_pts.mean(0, keepdim=True)
                curve_mesh_pts = torch.cat([curve_pts, curve_center], dim = 0)
                curve_range  = torch.arange(curve_pts.shape[0])
                center_idx = torch.zeros(curve_pts.shape[0]-1)
                center_idx[...] = curve_pts.shape[0]
                center_idx = center_idx.long()
                faces = torch.cat([curve_range[:-1, None], curve_range[1:, None], center_idx[:,None]], dim = 1)
                faces = torch.cat([faces, torch.tensor([curve_range[-1], curve_range[0], curve_pts.shape[0]])[None,:].long()], dim = 0)
                circle_mesh = trimesh.Trimesh(curve_mesh_pts.detach().cpu(),faces,process =False)
                surface_pts = torch.from_numpy(circle_mesh.sample(50000)).float().cuda()
                mnfld_pred=self.garment_nets[-1](surface_pts,ratio).view(-1)
                circle_sdf_loss=(mnfld_pred+self.sdfShrinkRadius).abs().mean()
                self.info['pc_{}_circle_loss_sdf'.format(target_name)]= circle_sdf_loss.item()
                ca_loss += circle_sdf_loss * (self.conf.get_float('pc_weight.curve_aware_weight') if 'pc_weight' in self.conf else 60.)

        return ca_loss

    def mask_loss(self, N, H, W, frame_ids, ratio, gt_garment_masks_list, device):
        '''using point clound mask loss to explictly deform garment mesh
        and then compute sdf loss to adjust implict network

        N: batch_size
        H: the heigth of image
        W: the width of image
        frame_ids: cur frame, using to debug
        ratio: position encoding parameters for sdfnetwork, deformernetwork, idr
        gt_garment_masks: ground truth garment mask
        device: cur gpu id

        -------------------------
        Return
            def_garment_meshes:     view-posed mesh
            garment_masks_list:     pred mesh mask
            mgt_garment_masks_list: gt mask
            pc_sdf_loss: sdf loss to adjust implict filed to deformed meshes

        '''
        def __mask_loss(imgs, gtMs):
            N=gtMs.shape[0]
            masks=imgs[...,-1]

            mask_loss=(1.-(masks*gtMs).view(N,-1).sum(1)/(masks+gtMs-masks*gtMs).abs().view(N,-1).sum(1)).mean()

            self.info['pc_loss']['whole_mask_loss']=mask_loss.item()
            loss=mask_loss*(self.conf.get_float('pc_weight.mask_weight') if 'pc_weight.mask_weight' in self.conf else 1.)
            return loss


        def mask_group(whole_garment_vs, whole_garment_masks, frags):
            # zbuf check mask belongs to
            num_verts = whole_garment_vs.shape[0]
            mask_visible = whole_garment_masks[...,0]>0
            zbuf_visible = frags.zbuf[...,0] > 0
            visible = torch.logical_and(zbuf_visible, mask_visible)
            pc_idx = frags.idx
            pc_zbuf = frags.zbuf
            visible_pc_idx = pc_idx[visible]
            near_pc_idx = visible_pc_idx % num_verts
            upper_idx = self.garment_vs[0].shape[0]
            upper_group = near_pc_idx < upper_idx
            bottom_group = torch.logical_and(near_pc_idx > upper_idx, visible_pc_idx>-1)

            # this is computed from alpha compositer
            r = self.pcRender.rasterizer.raster_settings.radius

            # weighted_fs[b,c,i,j] = sum_k cum_alpha_k * features[c,pointsidx[b,k,i,j]]
            # cum_alpha_k = alphas[b,k,i,j] * prod_l=0..k-1 (1 - alphas[b,l,i,j])

            visible_dists2 = frags.dists[visible]
            weights = 1 - visible_dists2 / (r * r)
            upper_weights = (weights* upper_group).sum(-1)
            bottom_weights = (weights * bottom_group).sum(-1)

            garment_masks_group = [torch.zeros_like(whole_garment_masks)[...,0] for i in range(2)]
            garment_masks_group[0][visible] = upper_group_weights
            garment_masks_group[1][visible] = bottom_group_weights
            garment_masks_list = [whole_garment_masks * garment_masks_group[0][...,None],whole_garment_masks * garment_masks_group[1][...,None] ]

            return garment_masks_list

        # TmpVnum
        # the number of vertices after marching cube
        tmp_body_vnum=self.body_vs.shape[0]
        tmp_garment_vnums=[garment_vs.shape[0] for garment_vs in self.garment_vs]
        # d_cond->deform code, rendcond -> color render code
        d_cond_list, poses, trans, rendcond = self.get_grad_parameters(frame_ids, device)
        def_garment_vs = [self.deformer(self.garment_vs[g_i][None,:,:].expand(N,-1,3),[d_cond_list[g_i+1],[poses,trans]],ratio=ratio, offset_type = garment_name)
                for g_i, garment_name in zip(range(self.garment_size), self.garment_names)]
        def_garment_meshes = []
        for tmp_garment_vnum, garment_fs, def_g_vs in zip(tmp_garment_vnums, self.garment_fs, def_garment_vs):
            def_garment_meshes.append(Meshes(verts=[vs.view(tmp_garment_vnum,3) for vs in torch.split(def_g_vs,1)],faces=[garment_fs for _ in range(N)]))

        self.info['pc_loss']={}
        # TODO 8-17
        surface_ps_list = self.find_surface_ps(def_garment_meshes)
        features=[torch.ones(self.body_vs.shape[0],1,device=self.body_vs.device) for _ in range(N)]
        # render body points
        # body_masks,frags=self.pcRender(Pointclouds(points=def_body_meshes.verts_list(),features=features))
        # render garment points
        garment_masks_list = []


        # solve the problem when hand front of bottom garment
        whole_garment_vs = torch.cat(self.garment_vs, dim =0)

        if len(def_garment_meshes) == 1:
            whole_garment_meshes = [torch.cat([garment_vs0], dim =0) for garment_vs0 in def_garment_meshes[0].verts_list()]
        elif len(def_garment_meshes) == 2:
            whole_garment_meshes = [torch.cat([garment_vs0, garment_vs1], dim =0) for garment_vs0, garment_vs1 in zip(def_garment_meshes[0].verts_list(), def_garment_meshes[1].verts_list())]
        else:
            raise NotImplemented('only support less or equal than 2 garment_type')

        whole_garment_features=[torch.ones(whole_garment_vs.shape[0], 1, device=whole_garment_vs.device) for _ in range(N)]
        garment_masks_list,frags=self.pcRender(Pointclouds(points=whole_garment_meshes,features=whole_garment_features), split_size = self.garment_vs[0].shape[0], all_size = whole_garment_vs.shape[0])


        radius=self.pcRender.rasterizer.raster_settings.radius
        radius=int(np.round(radius/2.*float(min(H,W))/1.2))

        # optimizer mask and obtain sdf_loss
        if radius>0:
            # body_masks loss computation
            # gt_body_masks=torch.nn.functional.max_pool2d(gt_body_masks, kernel_size=2*radius+1, stride=1, padding=radius)
            mgt_garment_masks_list = [torch.nn.functional.max_pool2d(gt_garment_masks,kernel_size=2*radius+1,stride=1,padding=radius) for gt_garment_masks in gt_garment_masks_list]
            if len(mgt_garment_masks_list) ==2:
                gt_merge_garment_masks = mgt_garment_masks_list[0] + mgt_garment_masks_list[1]
                gt_merge_garment_masks = (gt_merge_garment_masks>0).float()
            else:
                gt_merge_garment_masks = mgt_garment_masks_list[0]

            # garment_loss = __mask_loss(whole_garment_masks,gt_merge_garment_masks)
            garment_loss = 0.
            for garment_idx, (d_cond, def_garment_mesh, garment_masks, gt_garment_masks, garment_name, garment_vs, garment_fs) in enumerate(zip(d_cond_list[1:],
                    def_garment_meshes, garment_masks_list, mgt_garment_masks_list, self.garment_names, self.garment_vs, self.garment_fs)):
                garment_loss += self.compute_garment_pc_loss(def_garment_mesh,[d_cond, [poses,trans]], garment_masks, gt_garment_masks, ratio, garment_name, garment_vs, garment_fs, garment_idx)
            self.garment_optimizer.zero_grad()
            garment_loss.backward()
            self.garment_optimizer.step()

            # compute sdf loss after explict optimizer
            pc_sdf_loss = 0.

            for garment_idx, (garment_vs, garment_fs, garment_name) in enumerate(zip(self.garment_vs, self.garment_fs, self.garment_names)):
                mnfld_pred=self.garment_nets[garment_idx](garment_vs,ratio).view(-1)
                sdf_loss=(mnfld_pred+self.sdfShrinkRadius).abs().mean()
                self.info['pc_{}_loss_sdf'.format(garment_name)]=sdf_loss.item()
                pc_sdf_loss += sdf_loss*(self.conf.get_float('pc_weight.weight') if 'pc_weight' in self.conf else 60.)

            pc_sdf_loss += self.curve_aware_loss(ratio)



        else:
            raise NotImplemented
            mgtMs=None
            pc_sdf_loss = self.computeTmpPcLoss(defMeshes,[d_cond,[poses,trans]],masks,gtMs,ratio)

        return def_garment_meshes, garment_masks_list, mgt_garment_masks_list, pc_sdf_loss, surface_ps_list

    def sample_train_ray(self, N, sample_pix, gt_garment_masks_list,batch_garment_inds_list, batch_garment_row_list, batch_garment_col_list,
            batch_garment_init_pts_list,frame_ids, device):
        '''sampling pixel ray given camera position

        ---------------------------------------
        Return:
            sample_rays_list:                   rays_list for each garment mesh
            sample_batch_garment_inds_list
            sample_batch_garment_row_list
            sample_batch_garment_col_list
            sample_batch_garment_init_pts_list
            cameras:                            current camera setting

        '''

        sample_pix=self.conf.get_int('sample_pix_num') if 'sample_pix_num' in self.conf else sample_pix


        # for different garment_sdf, we have different energy_functions
        sample_pix = sample_pix // self.garment_size


        sample_batch_garment_inds_list = []
        sample_batch_garment_row_list = []
        sample_batch_garment_col_list = []
        sample_batch_garment_init_pts_list = []

        for gt_garment_masks, batch_inds, row_inds, col_inds, init_pts in zip(gt_garment_masks_list, batch_garment_inds_list, batch_garment_row_list, batch_garment_col_list,
            batch_garment_init_pts_list):
            sel = gt_garment_masks[batch_inds, row_inds, col_inds]>0.

            batch_inds = batch_inds[sel]
            row_inds=row_inds[sel]
            col_inds=col_inds[sel]
            init_pts=init_pts[sel]

            pnum=batch_inds.shape[0]
            # sample_ray
            if pnum>sample_pix*N:
                sel=torch.rand(pnum)<float(sample_pix*N)/float(pnum)

                sel=sel.to(batch_inds.device)
                batch_inds=batch_inds[sel]
                row_inds=row_inds[sel]
                col_inds=col_inds[sel]
                init_pts=init_pts[sel]

            sample_batch_garment_inds_list.append(batch_inds)
            sample_batch_garment_row_list.append(row_inds)
            sample_batch_garment_col_list.append(col_inds)
            sample_batch_garment_init_pts_list.append(init_pts)

        # very important part for SelfRecon
        # rebuild the computation graph from dataset cpu to gpu
        focals,princeple_ps,Rs,Ts,H,W=self.dataset.get_camera_parameters(frame_ids.numel(),device)
        # Rs, Ts is the camera to view matrix
        # the camera to view matrix flip the x and y matrix

        # NOTE that the screen coordinate need flip, y and x, hence the Rs, and Ts gradually is equal to [-1, -1, 1] rotation matrix
        cameras=RectifiedPerspectiveCameras(focals,princeple_ps,Rs,Ts,image_size=[(W, H)]).to(device)
        self.maskRender.rasterizer.cameras=cameras
        if self.pcRender:
            self.pcRender.rasterizer.cameras=cameras

        # sample_rays_list define
        sample_rays_list = []


        for row_inds, col_inds in zip(sample_batch_garment_row_list, sample_batch_garment_col_list):

            rays=cameras.view_rays(torch.cat([col_inds.view(-1,1),row_inds.view(-1,1),torch.ones_like(col_inds.view(-1,1))],dim=-1).float())
            sample_rays_list.append(rays)
        return sample_batch_garment_inds_list, sample_batch_garment_row_list, sample_batch_garment_col_list, sample_batch_garment_init_pts_list, sample_rays_list, cameras

    def opt_garment_surface_ps(self, frame_ids, cameras, ratio, sample_rays_list,sample_batch_garment_init_pts_list, sample_batch_garment_inds_list, device):
        '''Optimal method to find surface pts, in current sdf filed
        as paper said, the constrained sdf and the intesection angle between canonical camera view and surface normal

        -----------------------------------------------
        Return:
            init_garment_pt_list: initalized surface pts
            check_list: setting the threshold to flag valid surface pts
        '''
        d_cond_list, poses, trans, rendcond = self.get_grad_parameters(frame_ids, device)
        # NOTE that from index 1 to end is the garment condition features
        defconds_list=[d_cond_list[1:], [poses,trans]]

        init_garment_ps_list, check_list = utils.OptimizeGarmentSurfacePs(cameras.cam_pos().detach(), [rays.detach() for rays in sample_rays_list],
                sample_batch_garment_init_pts_list, sample_batch_garment_inds_list, self.garment_nets,ratio,
                self.deformer, defconds_list, garment_names = self.garment_names, dthreshold=5.e-5,
                athreshold=self.angThred, w1=3.05, w2=1., times= 20)

        for garment_idx, check in enumerate(check_list):
            self.info['{}_rayInfo'.format(self.garment_names[garment_idx])]=(check.numel(),check.sum().item())




        return init_garment_ps_list, check_list

    def surface_render_loss(self, datas, N, cameras, frame_ids, ratio, check_list, gtCs, init_garment_ps_list, sample_batch_garment_row_list,
            sample_batch_garment_col_list, sample_batch_garment_inds_list, sample_rays_list, device):
        '''using surface rendering method to compute photographics loss
        it contains rgb loss, normal loss, eikonal loss, offset_loss.

        ------------------------
        Return:
            idr rendering loss
        '''
        self.TmpPs=[None for i in range(self.garment_size)]
        self.rays=[None for i in range(self.garment_size)]
        self.batch_inds=[None for i in range(self.garment_size)]
        self.row_inds=[None for i in range(self.garment_size)]
        self.col_inds=[None for i in range(self.garment_size)]
        # decrease sample to save memory
        surface_sample_points  = 4096 // self.garment_size
        total_loss =0.

        d_cond_list, poses, trans, rendcond = self.get_grad_parameters(frame_ids, device)

        for  garment_idx, (init_garment_ps, row_inds, col_inds, batch_inds) in enumerate(zip(init_garment_ps_list,sample_batch_garment_row_list, sample_batch_garment_col_list, sample_batch_garment_inds_list)):
            TmpVs = self.garment_vs[garment_idx]
            TmpVnum = TmpVs.shape[0]
            rays = sample_rays_list[garment_idx]
            garment_name = self.garment_names[garment_idx]
            nonmnfld_pnts=utils.sample_points(torch.cat([init_garment_ps, TmpVs[torch.rand(TmpVnum)<float(surface_sample_points)/float(TmpVnum)].detach()],dim=0), 1.8,0.01)

            nonmnfld_pnts.requires_grad_()
            nonmnfld_pred = self.garment_nets[garment_idx](nonmnfld_pnts,ratio)
            # 9-19 fix the bug when compute eikonal gradient loss
            # nonmnfld_grad= self.sdf.gradient(nonmnfld_pnts,nonmnfld_pred)
            nonmnfld_grad= self.garment_nets[garment_idx].gradient(nonmnfld_pnts,nonmnfld_pred)

            # Eikonal Loss
            grad_loss = ((nonmnfld_grad.norm(2, dim=-1) - 1) ** 2).mean()
            self.info['{}_grad_loss'.format(garment_name)]=grad_loss.item()
            total_loss+=grad_loss*self.conf.get_float('grad_weight')

            d_cond = d_cond_list[garment_idx+1]
            if 'offset_weight' in self.conf and self.conf.get_float('offset_weight')>0.:
                self.deformer.defs[0](nonmnfld_pnts.view(1,-1,3).expand(N,-1,3),d_cond,ratio=ratio)
                def_offloss=self.deformer.defs[0].offset.view(-1,3).norm(p=2,dim=-1).mean()
                self.info['offset_loss']=def_offloss.item()
                total_loss+=def_offloss*self.conf.get_float('offset_weight')
            elif 'offset_weight' in self.conf and self.conf.get_float('offset_weight')==0.:
                with torch.no_grad():
                    self.deformer.defs[0](nonmnfld_pnts.view(1,-1,3).expand(N,-1,3),d_cond,ratio=ratio, offset_type = garment_name)
                    def_offloss=self.deformer.defs[0].offset[garment_name].view(-1,3).norm(p=2,dim=-1).mean()
                    self.info['{}_offset_loss'.format(garment_name)]=def_offloss.item()
            # compute rotation loss
            nonmnfld_pnts=None
            # 0.1
            if 'def_regu' in self.conf and self.conf.get_float('def_regu.weight')>0.:
                if nonmnfld_pnts is None:
                    # initTmps and TmpVs are all in canonical space
                    # Note that only using barycentric pts and TmpVs
                    nonmnfld_pnts=torch.cat([init_garment_ps, TmpVs[torch.rand(TmpVnum)<float(surface_sample_points)/float(TmpVnum)].detach()],dim=0)
                    # without global only surface
                    nonmnfld_pnts=torch.cat([nonmnfld_pnts,utils.sample_points(nonmnfld_pnts,1.8,0.01,0)],dim=0).view(1,-1,3).expand(N,-1,3)
                    nonmnfld_pnts.requires_grad_()

                # this deform only in canonical space
                defVs=self.deformer.defs[0](nonmnfld_pnts,d_cond,ratio=ratio, offset_type = garment_name)
                # core self_reconstrct
                Jacobs=utils.compute_Jacobian(nonmnfld_pnts,defVs,True,True)
                _,s,_=torch.svd(Jacobs.cpu()) #for pytorch, the gpu svd is too slow
                s=torch.log(s.to(device))
                # print(s.norm(dim=1)[0:10])
                # assert(False)
                def_loss=utils.GMRobustError((s*s).sum(1),self.conf.get_float('def_regu.c'),True).mean()
                self.info['def_{}_loss'.format(garment_name)]=def_loss.item()
                total_loss+=def_loss*self.conf.get_float('def_regu.weight')

            # photographics loss
            # rayinfo -> (number_of_ray, valid_ray)
            self.info['{}_color_loss'.format(garment_name)]=-1.0
            check = check_list[garment_idx]
            if self.info['{}_rayInfo'.format(garment_name)][1]>0:

                self.TmpPs[garment_idx] = init_garment_ps[check]
                self.TmpPs[garment_idx].requires_grad=True
                self.rays[garment_idx]=rays[check]
                self.batch_inds[garment_idx]=batch_inds[check]
                self.col_inds[garment_idx]= col_inds[check]
                self.row_inds[garment_idx]= row_inds[check]
                #(tmpVs: xyz + xyz_embeding)
                sdfs=self.garment_nets[garment_idx](self.TmpPs[garment_idx],ratio)

                nx=torch.autograd.grad(sdfs,self.TmpPs[garment_idx],torch.ones_like(sdfs),retain_graph=True,create_graph=True)[0]
                nx=nx/nx.norm(dim=1,keepdim=True)


                # canonical space crays(settting invalid ray equal to zeros, defVs, is the view_pose pt)
                crays, defVs = utils.compute_cardinal_rays(self.deformer,self.TmpPs[garment_idx], self.rays[garment_idx], [d_cond,[poses, trans]],self.batch_inds[garment_idx],ratio,'train', offset_type = garment_name)

                if self.conf.get_float('color_weight')>0.:
                    # render color
                    # inputs ->(render network, template vertices, deform verts, vertice normal in sdf, canonical_rays, sdf_conition, frame condition)
                    # surface rendering
                    colors=utils.compute_netRender_color(self.netRender,self.TmpPs[garment_idx], defVs,nx,crays, self.garment_nets[garment_idx].rendcond,rendcond[self.batch_inds[garment_idx]],ratio)
                # color loss
                # 0.5
                if self.conf.get_float('color_weight')>0.:
                    color_loss=(gtCs[self.batch_inds[garment_idx],self.row_inds[garment_idx],self.col_inds[garment_idx],:]-colors).abs().sum(1)

                    color_loss=scatter(color_loss,self.batch_inds[garment_idx],reduce='mean',dim_size=N).mean()
                    self.info['{}_color_loss'.format(garment_name)]=color_loss.item()
                    total_loss+=self.conf.get_float('color_weight')*color_loss
                if 'normal' in datas and 'normal_weight' in self.conf and self.conf.get_float('normal_weight')>0.:
                    if 'weighted_normal' in self.conf and self.conf.get_bool('weighted_normal'):
                        cnx,_=utils.compute_deformed_normals(self.garment_nets[garment_idx], self.deformer,self.TmpPs[garment_idx],[d_cond, [poses, trans]],self.batch_inds[garment_idx],ratio,'test', offset_type = garment_name)
                        # negative as the ray_view with cnx is oppsite direction
                        weights=torch.clamp((-self.rays[garment_idx]*cnx.detach()).sum(1).detach(),max=1.,min=0.)**2
                    else:
                        weights=torch.ones(nx.shape[0],device=device)
                    gtnormals=datas['normal'].to(device)
                    gtnormals=gtnormals[self.batch_inds[garment_idx],self.row_inds[garment_idx],self.col_inds[garment_idx],:]

                    # flip x, y
                    # why -> [[-1, 0, 0],[0, 1, 0],[0, 0, -1]]
                    # as the camera show, the normal map from pose view is negative x, z, this means it point in the screen.
                    # however the gt normal is point out the screen, hence it needs flip -> x and flip -> z
                    gtnormals=((cameras.R[0]@torch.tensor([[-1.,0.,0.],[0.,1.,0.],[0.,0.,-1.]],device=device))@gtnormals.view(-1,3,1)).view(-1,3)
                    gtnorms=gtnormals.norm(dim=1,keepdim=True)
                    valid_mask=(gtnorms>0.0001)[...,0]

                    gtnormals[valid_mask]=gtnormals[valid_mask]/gtnorms[valid_mask]
                    ds=self.deformer(self.TmpPs[garment_idx],[d_cond, [poses, trans]], self.batch_inds[garment_idx],ratio=ratio, offset_type = garment_name)
                    grad_d_p=utils.compute_Jacobian(self.TmpPs[garment_idx],ds,True,True)

                    gtnormals=(grad_d_p.transpose(-2,-1)@gtnormals.view(-1,3,1)).view(-1,3)
                    normal_loss = (gtnormals - nx).norm(2, dim=1)*weights
                    normal_loss=scatter(normal_loss[valid_mask],self.batch_inds[garment_idx][valid_mask],reduce='mean',dim_size=N).mean()
                    self.info['{}_normal_loss'.format(garment_name)]=normal_loss.item()
                    total_loss+=self.conf.get_float('normal_weight')*normal_loss

        return total_loss

    def dct_poses_loss(self, poses, trans, frame_ids, N):
        '''discreat cosine transfomration loss(DCT)
        NOTE that the change of human pose is slight, so the high frequence of DCT filed is limited to zeros
        hence we define DCT null space to optimize smpl poses and trans parameters

        poses: smpl poses
        trans: smpl translation parameters
        frame_ids: current batch id
        N: batch_size

        ------------------------------
        return:
            DCT poses loss
        '''
        # poses and trans parameters loss
        total_loss = 0.
        if (poses.requires_grad or trans.requires_grad) and self.conf.get_float('dct_weight')>0.:

            klen,Nlen=self.dctnull.shape
            batch_poses,pindices=self.dataset.get_batchframe_data('poses',frame_ids,Nlen)
            batch_trans,tindices=self.dataset.get_batchframe_data('trans',frame_ids,Nlen)
            posedJs=self.deformer.defs[1].posedSkeleton([batch_poses.reshape(N*Nlen,24,3),batch_trans.reshape(N*Nlen,3)])
            # Null spcae is the orthogoal space where, (0-10) lowes frequency, we force the higher frequency to be closed to zero
            # hence dctnull space -> (20,30)  for given 30 frame joints sequence
            dct_loss=self.dctnull[None,:,:].matmul(posedJs.reshape(N,Nlen,72))
            dct_loss=dct_loss.abs().mean()
            total_loss+=dct_loss*self.conf.get_float('dct_weight')
            self.info['dct_loss']=dct_loss.item()

        return total_loss


    def fl_visible_by_sdf_normal(self, d_cond_list, poses, ratio, fl_vs_merge, garment_vs, g_i, garment_name, N):
        '''check whether the feature line points are visible or not using nearest garment_surface normal direction

        d_cond_list: deformer latent code
        poses: transform smpl parameters
        ratio: position encoding latent size
        fl_vs_merge:  merge mesh vertices
        garment_vs:  template garment vertices
        g_i : the number of current garment type
        garment_name: current garment name
        N
        ----------------------------------------
        return:
            view_pose normal from camera-view
        '''

        dist = knn_points(fl_vs_merge[None], garment_vs[None])
        knearest = dist.idx[0,...]
        cano_garment_vs = garment_vs[knearest]
        # compute jacobian matrix
        K_near_num = cano_garment_vs.shape[1]
        cano_garment_vs = cano_garment_vs.view(-1,3)
        def_garment_vs = repeat(cano_garment_vs.view(-1,3),'n c -> b n c', b = N)
        ds=self.deformer(def_garment_vs, [d_cond_list[g_i+1], poses],ratio=ratio, offset_type = garment_name)
        grad_d_p=utils.batch_compute_Jacobian(def_garment_vs, ds, False, False)

        sdfs = self.garment_nets[g_i](cano_garment_vs, ratio)
        # fix the bug when compute eikonal gradient loss
        nx = self.garment_nets[g_i].gradient(cano_garment_vs, sdfs)
        nx = repeat(nx, 'n c -> b n c', b = N)
        grad_d_p_inv,inv_mask=Fast3x3Minv(grad_d_p.view(-1,3,3))
        grad_d_p_inv_trans = grad_d_p_inv.transpose(-2,-1)
        view_pose_nx = (grad_d_p_inv_trans @ nx.reshape(-1,3,1)).view(-1,3)
        view_pose_nx = view_pose_nx / (view_pose_nx.norm(dim = 1, keepdim = True)  + 1e-6)
        view_pose_batch = view_pose_nx.view(N, -1, 3)

        # debug
        # for batch_id, view_pose in enumerate(view_pose_batch):
        #     visible = view_pose[..., -1]>0
        #     invisible = view_pose[..., -1]<0

        #     print(visible.sum(), invisible.sum())

        #     visible_ply = cano_garment_vs[visible]
        #     invisible_ply = cano_garment_vs[invisible]

        #     save_ply('debug_visible/vis_{}'.format(batch_id), visible_ply.detach().cpu())
        #     save_ply('debug_visible/inv_vis_{}'.format(batch_id), invisible_ply.detach().cpu())

        n_inv_mask=~inv_mask

        if n_inv_mask.sum().item()>0:
            nview_pose_nx=torch.zeros_like(view_pose_nx)
            nview_pose_nx[inv_mask]=view_pose_nx[inv_mask]
            view_pose_nx = nview_pose_nx.detach()
        view_pose_nx = view_pose_nx.view(N, -1 , 3)

        return view_pose_nx

    def fl_visible_by_surface_normal(self, d_cond_list, poses, ratio, fl_meshes_dict, fl_names, g_i, garment_name, N):
        '''check whether the feature line points are visible or not using nearest garment_surface normal direction

        d_cond_list: deformer latent code
        poses: transform smpl parameters
        ratio: position encoding latent size
        fl_vs_merge:  merge mesh vertices
        garment_vs:  template garment vertices
        g_i : the number of current garment type
        garment_name: current garment name
        N
        ----------------------------------------
        return:
            view_pose normal from camera-view
        '''

        # compute_meshes normal

        normals_list = []
        verts_list = []
        for fl_name in fl_names:
            fl_mesh = fl_meshes_dict[fl_name]
            center = fl_mesh.verts_packed().mean(0, keepdim =True)
            center_rays = fl_mesh.verts_packed() - center
            verts_normals_packed = fl_mesh.verts_normals_packed()
            angle = F.cosine_similarity(verts_normals_packed, center_rays) < 0
            verts_normals_packed[angle] = - verts_normals_packed[angle]

            verts_list.append(fl_mesh.verts_packed())
            normals_list.append(verts_normals_packed)

        verts = torch.cat(verts_list, dim = 0)
        def_vs = repeat(verts.view(-1,3),'n c -> b n c', b = N)
        ds=self.deformer(def_vs, [d_cond_list[g_i+1], poses], ratio=ratio, offset_type = garment_name)

        grad_d_p=utils.batch_compute_Jacobian(def_vs, ds, False, False)

        nx = torch.cat(normals_list, dim = 0)
        nx = repeat(nx, 'n c -> b n c', b = N)
        grad_d_p_inv,inv_mask = Fast3x3Minv(grad_d_p.view(-1,3,3))
        grad_d_p_inv_trans = grad_d_p_inv.transpose(-2,-1)
        view_pose_nx = (grad_d_p_inv_trans @ nx.reshape(-1,3,1)).view(-1,3)
        view_pose_nx = view_pose_nx / (view_pose_nx.norm(dim = 1, keepdim = True)  + 1e-6)
        view_pose_batch = view_pose_nx.view(N, -1, 3)

        # for batch_id, view_pose in enumerate(view_pose_batch):
        #     visible = view_pose[..., -1]>0
        #     invisible = view_pose[..., -1]<=0
        #     visible_ply = verts[visible]
        #     invisible_ply = verts[invisible]
        #     save_ply('debug_visible/vis_{}.ply'.format(batch_id), visible_ply.detach().cpu())
        #     save_ply('debug_visible/inv_vis_{}.ply'.format(batch_id), invisible_ply.detach().cpu())

        n_inv_mask=~inv_mask
        if n_inv_mask.sum().item()>0:
            nview_pose_nx=torch.zeros_like(view_pose_nx)
            nview_pose_nx[inv_mask]=view_pose_nx[inv_mask]
            view_pose_nx = nview_pose_nx.detach()
        view_pose_nx = view_pose_nx.view(N, -1 , 3)

        return view_pose_nx

    def fl_visible_by_body_zbuff(self, cameras, d_cond_list, poses, ratio,fl_vs_dict, def_fl_vs, cano_smpl_verts_list, fl_names, g_i, garment_name, proj_size, N):
        '''check whether fl can be viewed given camera position
        by
        '''
        def z_buff_check(z_buff, sample_pts):
            '''
            compute the front depth according to z_buff matrix
            '''
            z_buff = rearrange(z_buff, 'b h w c -> b c h w ')
            xy = sample_pts[...,:2]
            uv = xy.unsqueeze(2)  # [B, N, 1, 2]
            samples = torch.nn.functional.grid_sample(z_buff, uv, align_corners=True)  # [B, C, N, 1]

            return samples[:, :, :, 0]  # [B, C, N]

        # check by garment_zbuff

        # [1,1,3]
        cam_pos = self.maskRender.rasterizer.cameras.cam_pos()[None, None].detach().clone()

        def_garment_vs = self.deformer(self.garment_vs[g_i][None,:,:].expand(N,-1,3),[d_cond_list[g_i+1],poses], ratio=ratio, offset_type = garment_name)
        def_garment_meshes = Meshes(verts = [vs.view(-1 ,3) for vs in torch.split(def_garment_vs, 1)], faces=[self.garment_fs[g_i] for _ in range(N)])
        def_fl_vs = torch.cat(def_fl_vs, dim = 1)
        __, garment_frags=self.maskRender(def_garment_meshes)
        garment_zbuf = garment_frags.zbuf
        # zbuf ==-1. is background


        # zbuf
        def_garment_depth = def_garment_vs[..., -1].detach().clone()
        def_garment_zbuff = def_garment_depth - cam_pos[..., -1]

        garment_z_max, __ = def_garment_zbuff[...].max(-1)
        garment_z_max = garment_z_max[:, None, None, None].expand_as(garment_zbuf)
        garment_zbuf[garment_zbuf == -1.] = garment_z_max[garment_zbuf==-1.]

        # x, y, z projection
        screen_pts = cameras.transform_points_screen(def_fl_vs,  proj_size)
        image_width, image_height = proj_size.unbind(1)
        image_width = image_width.view(-1, 1)  # (N, 1)
        image_height = image_height.view(-1, 1)  # (N, 1)
        u = screen_pts[..., 0]
        v = screen_pts[..., 1]
        u  = 2 * u /image_width  - 1
        v  = 2 * v /image_height - 1
        uv = torch.cat([u[..., None],v[..., None]], dim = -1)
        z = def_fl_vs[..., -1]
        garment_surf_buff = z_buff_check(garment_zbuf, uv)[:, 0 , :]
        garment_surf_depth = garment_surf_buff + cam_pos[..., -1]

        garment_check = z - garment_surf_depth

        # check by body zbuff
        # fix the bug the (smpl_verts is not changed by deform filed)
        def_smpl_fl_vs_list = [self.deformer.defs[1](smpl_fl_vert.expand(N,-1,3), poses) for  smpl_fl_vert, fl_name in zip(cano_smpl_verts_list, fl_names)]
        def_smpl_fl_vs = torch.cat(def_smpl_fl_vs_list, dim = 1)

        screen_pts = cameras.transform_points_screen(def_smpl_fl_vs,  proj_size)
        image_width, image_height = proj_size.unbind(1)
        image_width = image_width.view(-1, 1)  # (N, 1)
        image_height = image_height.view(-1, 1)  # (N, 1)
        u = screen_pts[..., 0]
        v = screen_pts[..., 1]
        u  = 2 * u /image_width  - 1
        v  = 2 * v /image_height - 1
        uv = torch.cat([u[..., None],v[..., None]], dim = -1)
        smpl_z = def_smpl_fl_vs[..., -1]





        deform_smpl_verts = self.deformer.defs[1](self.tmpBodyVs.view(1,-1,3).expand(N,-1,3), poses)
        deform_smpl_meshes = Meshes(verts = [deform_body_vs.view(-1,3) for deform_body_vs in torch.split(deform_smpl_verts, 1, dim = 0)], faces = [self.tmpBodyFs for _ in range(N)])
        __, smpl_frags=self.maskRender(deform_smpl_meshes)



        smpl_zbuf = smpl_frags.zbuf
        def_smpl_verts_depth = deform_smpl_verts[..., -1].detach().clone()
        def_smpl_verts_zbuff = def_smpl_verts_depth - cam_pos[..., -1]
        smpl_z_max, __ = def_smpl_verts_zbuff.max(-1)
        smpl_z_max = smpl_z_max[:, None, None, None].expand_as(smpl_zbuf)
        smpl_zbuf[smpl_zbuf == -1.] = smpl_z_max[smpl_zbuf==-1.]
        smpl_surf_buff = z_buff_check(smpl_zbuf, uv)[:, 0 , :]
        smpl_surf_depth = smpl_surf_buff + cam_pos[..., -1]

        smpl_check = smpl_z- smpl_surf_depth



        return torch.cat([garment_check[..., None], smpl_check[..., None]], dim =-1)




    def fl_visible_by_garment_zbuff(self, cameras, d_cond_list, poses, ratio, def_fl_vs, fl_names, g_i, garment_name, proj_size, N):
        '''check whether fl can be viewed given camera position
        by
        '''
        def z_buff_check(z_buff, sample_pts):
            '''
            compute the front depth according to z_buff matrix
            '''


            z_buff = rearrange(z_buff, 'b h w c -> b c h w ')
            xy = sample_pts[...,:2]
            uv = xy.unsqueeze(2)  # [B, N, 1, 2]

            samples = torch.nn.functional.grid_sample(z_buff, uv, align_corners=True)  # [B, C, N, 1]

            return samples[:, :, :, 0]  # [B, C, N]


        def_garment_vs = self.deformer(self.garment_vs[g_i][None,:,:].expand(N,-1,3),[d_cond_list[g_i+1],poses], ratio=ratio, offset_type = garment_name)
        def_garment_meshes = Meshes(verts = [vs.view(-1 ,3) for vs in torch.split(def_garment_vs, 1)], faces=[self.garment_fs[g_i] for _ in range(N)])
        def_fl_vs = torch.cat(def_fl_vs, dim = 1)
        __, frags=self.maskRender(def_garment_meshes)
        zbuf = frags.zbuf
        # zbuf ==-1. is background
        zbuf[zbuf == -1.] = def_garment_vs.max()
        # x, y, z projection
        screen_pts = cameras.transform_points_screen(def_fl_vs,  proj_size)
        image_width, image_height = proj_size.unbind(1)

        image_width = image_width.view(-1, 1)  # (N, 1)
        image_height = image_height.view(-1, 1)  # (N, 1)
        u = screen_pts[..., 0]
        v = screen_pts[..., 1]

        u  = 2 * u /image_width  - 1
        v  = 2 * v /image_height - 1
        uv = torch.cat([u[..., None],v[..., None]], dim = -1)

        z = def_fl_vs[..., -1]
        surf_depth = z_buff_check(zbuf, uv)[:, 0 , :]


        return z - surf_depth


    def deform_feature_line(self,cameras, d_cond_list, fl_masks, poses, ratio, offset_fl_verts_list, gt_fl_pts, proj_img_size, N):
        '''deform feature line, according to lbs

        d_cond_list: it is frame latent code to deform give space 3d points
        fl_masks : ground truth feature line masks
        poses: smpl pose and beta parameters
        ratio: position encoding parameters
        offset_fl_meshes: fl explict mesh
        N: batch_size

        -------------------------------------------
        return:
            def_fl_vs_group: view-pose feature lines
            fl_fs_group: canonical-pose feature lines faces
            fl_vs_group: canonical-pose feature lines verts
            fl_vs_split_group split feature lines to each category
            def_fl_normal_group: view pose feature line normal defined by nearest surface points
        '''
        def merge_fl_verts(fl_vs_dict, fl_names):

            fl_vs_list = [fl_vs_dict[fl_name].expand(-1,3) for fl_name in fl_names]
            fl_vs_merge = torch.cat(fl_vs_list, dim = 0)
            return fl_vs_merge

        gt_fl_pts_list = torch.split(gt_fl_pts, gt_fl_pts.shape[1] // len(self.fl_names), dim=1)

        fl_meshes_dict = {}
        fl_vs_dict = {}
        gt_fl_masks_dict = {}


        def_fl_vs_group = {}
        # canonical_space
        fl_fs_group = {}
        fl_vs_group = {}
        def_fl_normal_group = {}
        fl_vs_split_group = {}

        gt_fl_pts_dict = {}
        gt_fl_pts_group = {}
        gt_fl_masks_group = {}

        gt_fl_masks_list = torch.split(fl_masks, 1, 1)


        for fl_name, offset_fl_verts, gt_fl_pts, gt_fl_masks in zip(self.fl_names, offset_fl_verts_list, gt_fl_pts_list, gt_fl_masks_list):
            fl_meshes_dict[fl_name] = None
            fl_vs_dict[fl_name] = offset_fl_verts
            gt_fl_pts_dict[fl_name] = gt_fl_pts
            gt_fl_masks_dict[fl_name] = gt_fl_masks

        for g_i, (garment_name, garment_vs) in enumerate(zip(self.garment_names, self.garment_vs)):
            fl_names = FL_EXTRACT[garment_name]
            # fl_vs_merge = merge_fl_verts(fl_vs_dict, fl_names)
            # compute curve normal by surface pts
            def_fl_vs = [self.deformer(fl_vs_dict[fl_name].view(-1, 3).expand(N,-1,3), [d_cond_list[g_i+1], poses], ratio = ratio, offset_type = fl_name) for  fl_name in fl_names]
            gt_pts = [gt_fl_pts_dict[fl_name] for fl_name in fl_names]

            # using to smpl zbuff
            cano_smpl_verts_list = self.inter_free_curve.query_canosmpl_verts(fl_names)
            # identity whether the pts is visible
            if self.conf.get_string('fl_visible_method') == 'surface':
                # view_pose_nx = self.fl_visible_by_sdf_normal(d_cond_list, poses, ratio, fl_vs_merge, garment_vs, g_i, garment_name, N)
                # view_pose_nx = self.fl_visible_by_sdf_normal(d_cond_list, poses, ratio, fl_vs_merge, garment_vs, g_i, garment_name, N)
                view_pose_nx = self.fl_visible_by_surface_normal(d_cond_list, poses, ratio, fl_meshes_dict, fl_names, g_i, garment_name, N)

            elif self.conf.get_string('fl_visible_method') == 'zbuff':
                view_pose_nx = self.fl_visible_by_body_zbuff(cameras, d_cond_list, poses, ratio, fl_vs_dict, def_fl_vs, cano_smpl_verts_list, fl_names, g_i, garment_name, proj_img_size, N)
                # view_pose_nx = self.fl_visible_by_garment_zbuff(cameras, d_cond_list, poses, ratio, def_fl_vs, fl_names, g_i, garment_name, proj_img_size, N)
            else:
                raise NotImplemented


            gt_fl_masks_list = [gt_fl_masks_dict[fl_name] for fl_name in fl_names]

            fl_vs = [fl_vs_dict[fl_name].view(-1,3) for fl_name in fl_names]
            fl_vs_split =[vs.shape[1] for vs in def_fl_vs]
            def_fl_normal_group[garment_name] = torch.split(view_pose_nx, fl_vs_split, dim = 1)
            def_fl_vs_group[garment_name] = def_fl_vs
            fl_fs_group[garment_name] = None
            fl_vs_group[garment_name] = fl_vs
            fl_vs_split_group[garment_name] = fl_vs_split
            gt_fl_pts_group[garment_name] = torch.cat(gt_pts, dim = 1)
            gt_fl_masks_group[garment_name] =  torch.cat(gt_fl_masks_list, dim = -1)


        return def_fl_vs_group, fl_fs_group, fl_vs_group, fl_vs_split_group, def_fl_normal_group, gt_fl_pts_group, gt_fl_masks_group

    def compute_fl_proj_loss(self, def_fl_meshes, def_fl_normal_list, defconds, fl_masks, gt_fl_pts, ratio, garment_name, fl_vs_list, fl_fs_list, fl_vs_split, garment_idx, cameras, proj_size, visible_threshold = 0):
        '''
        compute 2d project feature line loss;
        def_fl_meshes: is the feature line sets belong to garment_type
        def_fl_normal: is the feature line normal sets computed from the normal of nearest surface pts belong  to garment_type
        defconds: [cond, smpl_paras]; NOTE cond is the garment condition features
        fl_mask: the visible flag to control pts accumulate to all fl_loss
        gt_fl_pts: ground truth 2d feature line
        fl_vs_list: canonical feature line vertices
        fl_fs_list: canonical feature line faces
        '''
        # def_fl_meshes, [d_cond, [poses,trans]], fl_masks, gt_fl_pts, ratio, garment_name, fl_vs, fl_fs, garment_idx

        # Mask loss
        loss = 0.
        N=fl_masks.shape[0]
        N_fl = len(FL_EXTRACT[garment_name])

        def_fl_verts = torch.cat(def_fl_meshes, dim = 1)
        screen_pts = cameras.transform_points_screen(def_fl_verts,  proj_size)
        # compute cosine similarity
        pixel_pos = screen_pts.detach().view(-1, 3)
        pixel_pos[...,-1] = 1

        view_pixel_rays = -cameras.view_rays(pixel_pos).view(N, -1, 3)
        def_fl_normals = torch.cat(def_fl_normal_list, dim = 1)
        if self.conf.get_string('fl_visible_method') == 'surface':
            # z large 0 is camera rays
            # cosine_sim = torch.einsum('bnc,bnc->bn', view_pixel_rays, def_fl_normals)
            # cosine_visible = cosine_sim > visible_threshold
            cosine_visible = def_fl_normals[...,-1] < 0
        elif self.conf.get_string('fl_visible_method') == 'zbuff':
            z_buff_threshold = torch.zeros_like(def_fl_normals).to(def_fl_normals)
            end_val = np.cumsum(fl_vs_split)
            start_ind = 0
            for end_ind,fl_name in enumerate(FL_EXTRACT[garment_name]):
                z_buff_threshold[..., start_ind: end_val[end_ind],:] = ZBUF_THRESHOLD[fl_name]
                start_ind = end_val[end_ind]

            cosine_visible = def_fl_normals < z_buff_threshold
            garment_visible = cosine_visible[...,0]
            body_visible = cosine_visible[...,1]

            # cosine_visible = torch.logical_and(garment_visible, body_visible)
            cosine_visible = body_visible

        else:
            raise NotImplemented

        gt_fl_pts_list = torch.split(gt_fl_pts, [gt_fl_pts.shape[1] // N_fl for _ in range(N_fl)], dim =1)
        screen_fl_pts_list = list(torch.split(screen_pts, fl_vs_split, dim = 1))
        cosine_visible_list = list(torch.split(cosine_visible, fl_vs_split, dim = 1))
        visible_mask_list = []

        cosine_mask_list = []
        label_mask_list = []

        for fl_idx, (screen_fl_pts, cosine_visible) in enumerate(zip(screen_fl_pts_list, cosine_visible_list)):
            fl_mask = fl_masks[:, None, fl_idx:fl_idx+1].expand_as(screen_fl_pts)
            cosine_visible = cosine_visible[..., None].expand_as(screen_fl_pts)
            visible_mask = torch.logical_and(fl_mask, cosine_visible)
            visible_mask_list.append(visible_mask)
            # using to visualize
            cosine_mask_list.append(cosine_visible)
            label_mask_list.append(fl_mask)



        proj_fl_weights =[self.dataset.fl_weights[fl_name] for fl_name in FL_EXTRACT[garment_name]]

        fl_loss = fl_proj_loss(screen_fl_pts_list, gt_fl_pts_list, visible_mask_list, proj_fl_weights) * (self.conf.get_float('fl_weight.weight') if 'fl_weight.weight' in self.conf else 1.)
        self.info['fl_loss']['{}_project loss'.format(garment_name)] = fl_loss.item()
        loss += fl_loss

        # regularization loss
        reg_loss = self.inter_free_curve.regularization(fl_masks)

        center_offset_loss = reg_loss['center_offset'] * (self.conf.get_float('alpha_weight.center_weight') if 'alpha_weight' in self.conf else 1.)
        diff_a_loss = reg_loss['diff_a_loss'] * (self.conf.get_float('alpha_weight.diff_weight') if 'alpha_weight' in self.conf else 1.)



        loss += center_offset_loss
        loss += diff_a_loss

        self.info['fl_loss']['{}_center loss'.format(garment_name)] = center_offset_loss.item()
        self.info['fl_loss']['{}_diff a loss'.format(garment_name)] = diff_a_loss.item()


        consistent_weight=self.conf.get_float('pc_weight.def_consistent.weight') if 'pc_weight.def_consistent' in self.conf else -1.
        consistent_weight = 0.
        if consistent_weight>0.:
            # defs[1] -> LBSkinner
            # Note that this is consistency the deform template is close to original template
            fl_vs_set = torch.cat(fl_vs_list, dim = 0)
            offset2=(torch.cat(def_fl_meshes, dim = 1) - self.deformer.defs[1](fl_vs_set.view(1,-1,3).expand(N,-1,3),defconds[1]))
            offset2=(offset2*offset2).sum(-1)

            if self.conf.get_float('pc_weight.def_consistent.c')>0.:
                consistent_loss=utils.GMRobustError(offset2,self.conf.get_float('pc_weight.def_consistent.c'),True).mean()
            else:
                consistent_loss=torch.sqrt(offset2).mean()

            self.info['fl_loss']['{}_defconst_loss'.format(garment_name)] = consistent_loss.item()
            loss=loss+consistent_loss*consistent_weight


        return loss, visible_mask_list, label_mask_list



        # # canonical
        # fl_mesh_list = [Meshes(verts=[fl_vs],faces=[fl_fs]) for fl_vs, fl_fs in zip(fl_vs_list, fl_fs_list)]
        # # fl laplacian loss
        # lap_weight = self.conf.get_float('fl_weight.laplacian_weight') if 'fl_weight' in self.conf else -1.

        # if lap_weight>0.:
        #     lap_loss = 0.
        #     for fl_mesh, fl_name in zip(fl_mesh_list, FL_EXTRACT[garment_name]):
        #         fl_lap_loss = lap_weight*mesh_laplacian_smoothing(fl_mesh,method='uniform')
        #         self.info['fl_loss']['{}_{}_laplacian_loss'.format(garment_name, fl_name)]=fl_lap_loss.item()/lap_weight
        #         lap_loss += fl_lap_loss
        #     loss+=lap_loss
        #     self.info['fl_loss']['{}_lap_loss'.format(garment_name)]=lap_loss.item()/lap_weight

        # edge_weight=self.conf.get_float('fl_weight.edge_weight') if 'fl_weight' in self.conf else -1.

        # if edge_weight>0.:
        #     edge_loss = 0.
        #     for fl_mesh, fl_name in zip(fl_mesh_list, FL_EXTRACT[garment_name]):
        #         fl_edge_loss = edge_weight*mesh_edge_loss(fl_mesh)
        #         self.info['fl_loss']['{}_{}_edge_loss'.format(garment_name, fl_name)]=fl_edge_loss.item()/edge_weight
        #         edge_loss += fl_edge_loss
        #     loss=loss+edge_loss
        #     self.info['fl_loss']['{}_edge_loss'.format(garment_name)]=edge_loss.item()/edge_weight

        # norm_weight=self.conf.get_float('pc_weight.norm_weight') if 'pc_weight' in self.conf else -1.
        # if norm_weight>0.:
        #     norm_loss = 0.
        #     for fl_mesh, fl_name in zip(fl_mesh_list, FL_EXTRACT[garment_name]):
        #         fl_norm_loss = norm_weight*mesh_normal_consistency(fl_mesh)
        #         self.info['fl_loss']['{}_{}_norm_loss'.format(garment_name, fl_name)] = fl_norm_loss.item()/norm_weight
        #         norm_loss += fl_norm_loss
        #     loss=loss + norm_loss
        #     self.info['fl_loss']['{}_norm_loss'.format(garment_name)] = norm_loss.item()/edge_weight

        # consistent_weight=self.conf.get_float('pc_weight.def_consistent.weight') if 'pc_weight.def_consistent' in self.conf else -1.
        # if consistent_weight>0.:
        #     # defs[1] -> LBSkinner
        #     # Note that this is consistency the deform template is close to original template

        #     fl_vs_set = torch.cat(fl_vs_list, dim = 0)
        #     offset2=(def_fl_meshes.verts_padded()-self.deformer.defs[1](fl_vs_set.view(1,-1,3).expand(N,-1,3),defconds[1]))
        #     offset2=(offset2*offset2).sum(-1)

        #     if self.conf.get_float('pc_weight.def_consistent.c')>0.:
        #         consistent_loss=utils.GMRobustError(offset2,self.conf.get_float('pc_weight.def_consistent.c'),True).mean()
        #     else:
        #         consistent_loss=torch.sqrt(offset2).mean()

        #     self.info['fl_loss']['{}_defconst_loss'.format(garment_name)] = consistent_loss.item()
        #     loss=loss+consistent_loss*consistent_weight


        # # self.TmpOptimzier is optimal the vertices of template_field
        # return loss, visible_mask_list, label_mask_list


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
        loss = 10. * fl_garment_sdf_loss + 1. * project_loss

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
        # data -> img, mask, normal
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

    def save_debug(self,TmpVs_list,Tmpfs_list, old_fl_deform_vs, defMeshes_list,offset_dict, garment_fl_vs_list, fl_visible_mask_list, gt_fl_visible_mask_list, gt_garment_fl_vs_list, masks_list,gtMs_list,
            mgtMs_list, gtCs,batch_inds_list,row_inds_list,col_inds_list,initTmpPs_list, conds_list, pose_list, rendcond, ratio, N):
        # old_garment_vs, self.garment_vs, def_garment_meshes, self.deformer.defs[0], garment_masks_list, gt_garment_masks_list, mgt_garment_masks_list, gtCs,
        #                 batch_garment_inds_list, batch_garment_row_list, batch_garment_col_list, batch_garment_init_pts_list, d_cond_list, [poses, trans], rendcond, ratio
        '''debug function to help us watch each optimize steps.
        '''

        visualized_imgs_scalar = {}
        if self.root is None:
            return
        # show img
        all_masks = torch.zeros_like(mgtMs_list[0]).to(mgtMs_list[0]).bool()
        for mgtMs in mgtMs_list:
            all_masks |= (mgtMs >0.1)
        for ind, all_mask in enumerate(all_masks):
            all_mask = (all_mask.float()*255.).detach().cpu().numpy().astype(np.uint8)
            cv2.imwrite(osp.join(self.root,'all_{}.png'.format(ind)),all_mask)
            if ind == 0:
                visualized_imgs_scalar['render/masks/all_gt'] = all_mask

        all_masks = torch.zeros_like(masks_list[0]).to(masks_list[0]).bool()
        for pred_mask in masks_list:
            all_masks |= (pred_mask>0.1)
        for ind, all_mask in enumerate(all_masks):
            all_mask = (all_mask.float()*255.).detach().cpu().numpy().astype(np.uint8)
            cv2.imwrite(osp.join(self.root,'pred_{}.png'.format(ind)),all_mask)
            if ind == 0:
                visualized_imgs_scalar['render/masks/all_pred'] = all_mask[...,0]

        garment_idx = 0
        # save def1_fl
        fl_mesh_path = os.path.join(self.root, 'fl_mesh_show')
        garment_mesh_path = os.path.join(self.root, 'garment_mesh_show')
        render_path = os.path.join(self.root, 'render_show')
        os.makedirs(fl_mesh_path, exist_ok = True)
        os.makedirs(garment_mesh_path, exist_ok = True)
        os.makedirs(render_path, exist_ok = True)

        def0_fl_meshes = [verts.squeeze(0) for verts in self.inter_free_curve.inference()]

        # show_fl_curve
        for def0_fl_mesh, fl_name in zip(def0_fl_meshes, self.fl_names):
            offset = offset_dict[fl_name].view(N, -1, 3).detach()
            faces_ids = torch.arange(offset.shape[1])
            faces_ids = torch.cat([faces_ids[:-1, None], faces_ids[1:, None], faces_ids[:-1, None]], dim = -1).to(offset)

            def1_fl_meshes=Meshes(verts=[vs for vs in torch.split(def0_fl_mesh[None,:,:].expand(N,-1,3)+offset,1)], faces=[faces_ids for _ in range(N)],
                    textures=TexturesVertex([torch.ones_like(def0_fl_mesh) for _ in range(N)]))
            for ind,(vs,fs) in enumerate(zip(def1_fl_meshes.verts_list(), def1_fl_meshes.faces_list())):
                mesh = trimesh.Trimesh(vs[0].detach().cpu().numpy(), fs.cpu().numpy())
                mesh.export(osp.join(fl_mesh_path,'def1_{}_{}.ply'.format(fl_name,ind)))


        for TmpVs, Tmpfs, defMeshes, garment_fl_vs, fl_visible_mask, gt_visible_mask, gt_garment_fl_vs, garment_type, masks, gt_masks, mgtms, batch_inds, row_inds, col_inds, initTmpPs, conds in zip(TmpVs_list, Tmpfs_list,
                defMeshes_list, garment_fl_vs_list, fl_visible_mask_list, gt_fl_visible_mask_list, gt_garment_fl_vs_list, self.garment_names, masks_list, gtMs_list, mgtMs_list, batch_inds_list,
                row_inds_list, col_inds_list, initTmpPs_list, conds_list):


            mesh = trimesh.Trimesh(TmpVs.detach().cpu().numpy(), Tmpfs.cpu().numpy())
            mesh.export(osp.join(garment_mesh_path,'tmp_{}.ply').format(garment_type))

            for ind,(vs,fs) in enumerate(zip(defMeshes.verts_list(), defMeshes.faces_list())):
                mesh = trimesh.Trimesh(vs.view(TmpVs.shape[0],3).detach().cpu().numpy(), fs.cpu().numpy())
                mesh.export(osp.join(garment_mesh_path,'def_{}_{}.ply'.format(garment_type, ind)))

            offset = offset_dict[garment_type].view(N,-1,3).detach()
            N=offset.shape[0]
            defMeshes=Meshes(verts=[vs for vs in torch.split(TmpVs[None,:,:].expand(N,-1,3)+offset,1)],faces=[Tmpfs for _ in range(N)],textures=TexturesVertex([torch.ones_like(TmpVs) for _ in range(N)]))

            # draw def1-space garment_mesh
            for ind,(vs,fs) in enumerate(zip(defMeshes.verts_list(),defMeshes.faces_list())):
                mesh = trimesh.Trimesh(vs.view(TmpVs.shape[0],3).detach().cpu().numpy(), fs.cpu().numpy())
                mesh.export(osp.join(garment_mesh_path,'def1_{}_{}.ply'.format(garment_type,ind)))


            images=(masks*255.).detach().cpu().numpy().astype(np.uint8)
            gtMasks=(gt_masks*255.).detach().cpu().numpy().astype(np.uint8)

            # draw fl point, and gt fl points
            draw_board =((gtCs/2.+0.5)*255.).cpu().numpy().astype(np.uint8)[0]
            draw_invisible_board = draw_board.copy()
            draw_gt_board = draw_board.copy()

            draw_label_board = draw_board.copy()
            draw_invisible_label_board = draw_board.copy()
            for fl_name, fl_vs, fl_mask, label_mask in zip(FL_EXTRACT[garment_type], garment_fl_vs, fl_visible_mask, gt_visible_mask):
                fl_color = FL_COLOR[fl_name]
                fl_mask = fl_mask.detach().cpu().numpy()[0, ..., 0].tolist()
                label_mask = label_mask.detach().cpu().numpy()[0, ..., 0].tolist()

                for (fl, visible, label_visible) in zip(fl_vs.detach().cpu().numpy()[0, ...,:2].astype(np.int).tolist(), fl_mask, label_mask):
                    if visible:
                        draw_board = cv2.circle(draw_board, (int(fl[0]), int(fl[1])), 2, fl_color, 1)
                    else:
                        draw_invisible_board = cv2.circle(draw_invisible_board, (int(fl[0]), int(fl[1])), 2, fl_color, 1)

                    if label_visible:
                        draw_label_board = cv2.circle(draw_label_board, (int(fl[0]), int(fl[1])), 2, fl_color, 1)
                    else:
                        draw_invisible_label_board = cv2.circle(draw_invisible_label_board, (int(fl[0]), int(fl[1])), 2, fl_color, 1)

            gt_garment_fl_vs = torch.split(gt_garment_fl_vs, gt_garment_fl_vs.shape[1] // len(FL_EXTRACT[garment_type]), dim = 1)

            for fl_name, fl_vs in zip(FL_EXTRACT[garment_type], gt_garment_fl_vs):
                fl_color = FL_COLOR[fl_name]
                for fl in fl_vs.detach().cpu().numpy()[0, ...,:2].astype(np.int).tolist():
                    draw_gt_board = cv2.circle(draw_gt_board, (int(fl[0]), int(fl[1])), 2, fl_color, 1)

            draw_merge_img = np.concatenate([draw_board, draw_invisible_board, draw_label_board, draw_invisible_label_board, draw_gt_board], axis = 1)

            cv2.imwrite(osp.join(render_path, 'fl_proj_{}.png'.format(garment_type)), draw_merge_img)
            visualized_imgs_scalar['render/fl_proj/{}'.format(garment_type)] = draw_merge_img

            for ind,(img,gtimg) in enumerate(zip(images,gtMasks)):
                cv2.imwrite(osp.join(render_path,'m_{}_{}.png'.format(garment_type, ind)),img)
                if ind ==0:
                    visualized_imgs_scalar['render/masks/{}_pred'.format(garment_type)] = img

                # cv2.imwrite(osp.join(self.root,'gm%d.png'%ind),gtimg)
            if mgtms is not None:
                mgtMasks=(mgtms*255.).detach().cpu().numpy().astype(np.uint8)
                for ind,mgtimg in enumerate(mgtMasks):
                    cv2.imwrite(osp.join(render_path, 'mgm_{}_{}.png'.format(garment_type, ind)),mgtimg)
                    if ind == 0:
                        visualized_imgs_scalar['render/masks/{}_gt'.format(garment_type)] = mgtimg

            #debug draw the images

            if self.draw:
                cameras=self.maskRender.rasterizer.cameras
                with torch.no_grad():
                    rays=cameras.view_rays(torch.cat([col_inds.view(-1,1),row_inds.view(-1,1),torch.ones_like(col_inds.view(-1,1))],dim=-1).float())
                tcolors=[]
                tnormals=[]
                # tlights=[]
                print('draw %d points'%rays.shape[0])
                number=20000
                for ind,(rays_,initTmpPs_,batch_inds_) in enumerate(zip(torch.split(rays,number),torch.split(initTmpPs,number),torch.split(batch_inds,number))):
                    initTmpPs_,check=utils.OptimizeGarmentSurfaceSinlge(cameras.cam_pos().detach(),rays_.detach(),initTmpPs_.clone(),batch_inds_,self.garment_nets[garment_idx],ratio,self.deformer,[conds, pose_list],dthreshold=1.e-4,athreshold=self.angThred,w1=3.05,w2=1.,times=30, offset_type=self.garment_names[garment_idx])
                    # print('%d:(%d,%d)'%(ind,rays_.shape[0],check.sum().item()))
                    initTmpPs_.requires_grad=True
                    sdfs=self.garment_nets[garment_idx](initTmpPs_,ratio)
                    nx=torch.autograd.grad(sdfs,initTmpPs_,torch.ones_like(sdfs),retain_graph=False,create_graph=False)[0]
                    nx=nx/nx.norm(dim=1,keepdim=True)
                    rays_,defVs_=utils.compute_cardinal_rays(self.deformer,initTmpPs_,rays_,[conds, pose_list],batch_inds_,ratio,'test', offset_type = garment_type)

                    with torch.no_grad():
                        tcolors.append(utils.compute_netRender_color(self.netRender,initTmpPs_,defVs_,nx,rays_,self.garment_nets[garment_idx].rendcond,rendcond[batch_inds_],ratio))
                        with torch.enable_grad():
                            nx,_=utils.compute_deformed_normals(self.garment_nets[garment_idx], self.deformer,initTmpPs_,[conds, pose_list] ,batch_inds_,ratio,'test', offset_type = self.garment_names[garment_idx])
                        tnormals.append((torch.tensor([[-1.,0.,0.],[0.,1.,0.],[0.,0.,-1.]],device=nx.device)@cameras.R[0].transpose(0,1)@nx.view(-1,3,1)).view(-1,3))
                tcolors=torch.cat(tcolors,dim=0)


                # print((gtCs[batch_inds,row_inds,col_inds]-tcolors).abs().mean().item())
                tcolors=torch.clamp((tcolors/2.+0.5)*255.,min=0.,max=255.)
                colors=torch.ones_like(gtCs)*255.
                colors[batch_inds,row_inds,col_inds,:]=tcolors
                colors=colors.cpu().numpy().astype(np.uint8)


                tnormals=torch.cat(tnormals,dim=0)
                tnormals=(tnormals*0.5+0.5)*255.
                normals=torch.ones_like(gtCs)*255.
                normals[batch_inds,row_inds,col_inds,:]=tnormals[:,[2,1,0]]
                normals=normals.cpu().numpy().astype(np.uint8)
                gtcolors=((gtCs/2.+0.5)*255.).cpu().numpy().astype(np.uint8)


                for ind,(color,gtcolor, gtmask) in enumerate(zip(colors,gtcolors, gtMasks)):
                    cv2.imwrite(osp.join(render_path, 'rgb_{}_{}.png'.format(garment_type,ind)),color)
                    # cv2.imwrite(osp.join(self.root,'light%d.png'%ind),lights[ind]) if lights is not None else None
                    cv2.imwrite(osp.join(render_path,'gtrgb_{}.png'.format(ind)),gtcolor)
                    cv2.imwrite(osp.join(render_path,'normal_{}_{}.png'.format(garment_type,ind)),normals[ind])
                    if ind == 0:
                        # 9-16 fix the bug for visualized garment error map
                        error_map = np.abs((gtcolor.astype(np.float)/255 -0.5)*2 - (color.astype(np.float)/ 255 - 0.5) * 2).clip(0,1)
                        error_map = (error_map*255).astype(np.uint8)
                        error_jet_map = cv2.applyColorMap(error_map, cv2.COLORMAP_JET)
                        error_jet_map[gtmask == 0] = [255, 255, 255]
                        visualized_imgs_scalar['render/colors/rgb_{}_pred'.format(garment_type)] = color
                        visualized_imgs_scalar['render/colors/rgb_{}_gt'.format(garment_type)] = gtcolor
                        visualized_imgs_scalar['render/colors/normal_{}'.format(garment_type)] = normals[ind]
                        visualized_imgs_scalar['render/colors/error_map_{}'.format(garment_type)] = error_jet_map
            garment_idx+=1
        self.visualizer.add_image(visualized_imgs_scalar, size = 256, rgb = True, step = int(self.opt_times))
        self.draw=False

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
            #for theta
            params=[param for param in self.garment_nets[garment_idx].parameters() if param.requires_grad]
            # grad_l_p @ grad_p_n
            params_grads=torch.autograd.grad(self.garment_nets[garment_idx](p,ratio),params,-rhs_1[:,:,0])
            for param,grad in zip(params,params_grads):
                loss+=(param*grad).sum()
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


    def registration(self, fl_curve, g_vs_list, g_fs_list, ratio,root):
        '''
        registering garment templates to garment_mesh
        '''

        def surface_finder(target_garments, device):
            '''NOTE that sdf can produce wrong surface, we using surface rendering to help us clean noise points
            '''

            print("surface finder start!")

            garment_meshes_masks =[]

            for garment_idx, target_garment in enumerate(target_garments):

                N = 1
                focals,princeple_ps,Rs,Ts,H,W=self.dataset.get_camera_parameters(1, device)
                newTs=self.dataset.trans.mean(0).to(device)[None,:]
                cameras=RectifiedPerspectiveCameras(focals,princeple_ps,torch.tensor([[[-1.,0.,0.],[0.,1.,0.],[0.,0.,-1.]]],device=device).repeat(N,1,1),newTs.repeat(N,1),image_size=[(W, H)]).to(device)
                max_face = target_garment.faces_packed().shape[0]
                raster_settings_silhouette=RasterizationSettings(
                        image_size=(H,W),
                        blur_radius=0.,
                        # blur_radius=np.log(1. / 1e-4 - 1.)*3.e-6,
                        bin_size=(92 if max(H,W)>1024 and max(H,W)<=2048 else None),
                        # NOTE that max_faces_per_bin is very important since rendering noises from pytorch3d
                        max_faces_per_bin= max(10000, max_face),
                        faces_per_pixel=1,
                        perspective_correct=True,
                        clip_barycentric_coords=False,
                        cull_backfaces=self.maskRender.rasterizer.raster_settings.cull_backfaces
                    )
                view_renderer = MeshRendererWithFragments(
                    rasterizer=MeshRasterizer(
                        cameras=cameras,
                        raster_settings=raster_settings_silhouette
                    ),
                        shader=SoftSilhouetteShader()
                )
                view_renderer.to(device)

                color_meshes = Meshes([target_garment[0].verts_packed()], [target_garment.faces_packed()], textures = TexturesVertex([torch.ones_like(target_garment[0].verts_packed())]))
                garment_verts_mask = torch.zeros_like(color_meshes.verts_packed())
                garment_verts = color_meshes.verts_packed()

                with torch.no_grad():
                    for degree in range(0, 360, 30):
                        R_y = Rotate_Y_axis(degree)
                        rotate_color_meshes = pytorch3d_mesh_transformation(color_meshes, R_y)
                        offset = rotate_color_meshes.verts_packed().mean(0, keepdim = True)
                        offset_matrix = torch.eye(4).to(offset)
                        offset_matrix[:3, 3] = -offset[0,...]
                        meshes = pytorch3d_mesh_transformation(rotate_color_meshes, offset_matrix)
                        newcameras=RectifiedPerspectiveCameras(focals,princeple_ps,torch.tensor([[[-1.,0.,0.],[0.,1.,0.],[0.,0.,-1.]]],device=device).repeat(N,1,1),newTs.repeat(N,1),image_size=[(W, H)]).to(device)
                        __, fragment=view_renderer(meshes, cameras=newcameras)
                        view_faces = fragment.pix_to_face[0,...,0]
                        view_faces = view_faces[view_faces !=-1]

                        masks=(fragment.pix_to_face>=0).float()[0,...,0]
                        masks= ((masks>0.).detach().cpu().numpy()*255).astype(np.uint8)

                        view_vertices = meshes.faces_packed()[view_faces]
                        view_vertices = view_vertices.unique()


                        garment_verts_mask[view_vertices] = 1
                save_ply('fuck_{}_{:03d}.ply'.format(garment_idx, degree), color_meshes.verts_packed()[garment_verts_mask[...,0]>0].detach().cpu())

                garment_meshes_masks.append(garment_verts_mask[...,0])
            print("surface finder end!")

            return garment_meshes_masks

        # obtain inital templates
        self.init_template(self.garment_template_path, add_head= False)
        vs=self.tmpBodyVs
        device=vs.device
        garment_templates = self.garment_by_init_smpl()
        # dense_boundary OK
        dense_garment_templates = []
        for garment_idx, garment_template in enumerate(garment_templates):
            # edge dense pc times
            for __ in range(2):
                garment_template = garment_template.dense_boundary()
            dense_garment_templates.append(garment_template)

        curve_list = torch.split(fl_curve, 1, dim =0)
        curve_list = [curve.squeeze(0) for curve in curve_list]

        curve_sets = {}
        for curve, fl_name in zip(curve_list, self.fl_names):
            curve_sets[fl_name] = curve

        target_garments =[Meshes([verts], [faces]) for verts, faces in zip(g_vs_list, g_fs_list)]

        garment_engine = dict(
            fl_init_registry = Laplacian_Optimizer(),
            fl_final_registry = Laplacian_Deform_upper_and_domn_Optimzier(epoch = 1),
            fl_fit_surface_registry = Surface_Intesection(),
            fl_fit_registry = NRICP_Optimizer_AdamW(epoch = 200, dense_pcl=4e4, stiffness_weight = [50, 20, 5, 2, 0.8,0.5,0.35, 0.2, 0.1],
                use_normal = True, inner_iter = 50,mile_stone =[50, 80, 100, 110, 120 , 130, 140, 150],
                # laplacian_weight = [250, 250, 250, 250, 250, 250, 250 , 250, 250]
                laplacian_weight = [250, 250, 250, 250, 250, 250, 250 , 250, 250], threshold = 0.3
                ),

            fl_refine_registry = NRICP_Optimizer_AdamW(epoch = 100, dense_pcl=4e4, stiffness_weight = [2, 0.8,0.5,0.35, 0.2, 0.1],
                use_normal = True, inner_iter = 50,mile_stone =[10, 20 , 30, 40, 80],
                # laplacian_weight = [250, 250, 250, 250, 250, 250, 250 , 250, 250]
                laplacian_weight = [250, 250, 250, 250 , 250, 250], threshold=0.5
                ),
            )

        registry_meshes = []



        for garment_name in self.garment_names:
            if os.path.exists(os.path.join(root,'registry_{}.obj'.format(garment_name))):
                registry_mesh = trimesh.load(os.path.join(root,'registry_{}.obj'.format(garment_name)), process = False)
                # garment_template = garment_template.remesh_garment_mesh(registry_mesh)
                registry_meshes.append(Meshes([torch.from_numpy(registry_mesh.vertices).float()], [torch.from_numpy(registry_mesh.faces).long()] ).to(device))


        if len(registry_meshes) == len(self.garment_names):
            return registry_meshes

        garment_masks = surface_finder(target_garments, device)

        # garment_masks = [None, None]
        for garment_idx, (garment_name, garment_template, target_garment, garment_mask) in enumerate(zip(self.garment_names,dense_garment_templates, target_garments, garment_masks)):

            garment_fl_names = GARMENT_FL_MATCH[garment_name]
            curves = [curve_sets[fl_name] for fl_name in garment_fl_names]
            curve_meshes = [Meshes([curve], [get_curve_faces(curve)]) for curve in curves]

            # curve_mesh = garment_engine['fl_fit_surface_registry'](smpl_slice = curve_meshes, cano_meshes = target_garment, curve_types = )
            # 1.template laplacian deform
            lap_curve_mesh = garment_engine['fl_init_registry'](source_fl_meshes = [garment_template], target_meshes = curve_meshes, source_type = [garment_name], target_fl_type = garment_fl_names, outlayer=True)
            garment_template = lap_curve_mesh['source_fl_meshes'][0]

            # first lap then nricp
            #if garment_idx == 1:
            #    # garment_template = garment_engine['fl_final_registry'](source_fl_meshes = [registry_mesh, garment_template], outlayer = True)
            #    # garment_template = garment_template['source_fl_meshes'][1]
            #    # 2.nricp
            #    # NOTE that due to girl hair influence, we add static points to control nricp
            #    # loss, registry_mesh = garment_engine['fl_fit_registry'](smpl_slice = garment_template, cano_meshes = target_garment, save_path = root, garment_name = garment_name, static_pts_type = ['upper_bottom'], nricp_masks = garment_mask)
            #    # 3.remesh

            #    # registry_mesh = registry_mesh.remesh_garment_mesh(root)
            #else:
            #    # 2.nricp
            #    # NOTE that due to girl hair influence, we add static points to control nricp
            #    # loss, registry_mesh = garment_engine['fl_fit_registry'](smpl_slice = garment_template, cano_meshes = target_garment, save_path = root, garment_name = garment_name, static_pts_type = [], nricp_masks = garment_mask)
            #    # 3.remesh
            #    # registry_mesh = registry_mesh.remesh_garment_mesh(root)


            # nricp
            loss, registry_mesh = garment_engine['fl_fit_registry'](smpl_slice = garment_template, cano_meshes = target_garment, save_path = root, garment_name = garment_name, static_pts_type = [], nricp_masks = garment_mask)
            # remesh
            registry_mesh = registry_mesh.remesh_garment_mesh(root)
            # fine nricp
            loss, registry_mesh = garment_engine['fl_refine_registry'](smpl_slice =registry_mesh, cano_meshes = target_garment, save_path = root, garment_name = garment_name, static_pts_type = [], nricp_masks = garment_mask)
            registry_mesh = registry_mesh.remesh_garment_mesh_no_reduce(root)
            registry_meshes.append(registry_mesh)


        # upper and down merge by laplacian deform

        fine_registry_meshes = []
        # # for garment_idx, (garment_name, garment_template, target_garment, garment_mask) in enumerate(zip(self.garment_names,dense_garment_templates, target_garments, garment_masks)):
        # #     if garment_idx ==1:
        # #         loss, registry_mesh = garment_engine['fl_refine_registry'](smpl_slice =registry_meshes[garment_idx], cano_meshes = target_garment, save_path = root, garment_name = garment_name, static_pts_type = ['upper_bottom'], nricp_masks = garment_mask)
        # #     else:
        # #         loss, registry_mesh = garment_engine['fl_refine_registry'](smpl_slice =registry_meshes[garment_idx], cano_meshes = target_garment, save_path = root, garment_name = garment_name, static_pts_type = [], nricp_masks = garment_mask)

        # #     fine_registry_meshes.append(Meshes([registry_mesh.vertices], [registry_mesh.faces] ).to(device))



        # for garment_idx, (garment_name, garment_template, target_garment, garment_mask) in enumerate(zip(self.garment_names,dense_garment_templates, target_garments, garment_masks)):
        #     if garment_idx ==0:
        #         loss, registry_mesh = garment_engine['fl_refine_registry'](smpl_slice =registry_meshes[garment_idx], cano_meshes = target_garment, save_path = root, garment_name = garment_name, static_pts_type = [], nricp_masks = garment_mask)
        #     else:
        #         registry_mesh = registry_meshes[garment_idx]
        #     fine_registry_meshes.append(Meshes([registry_mesh.vertices], [registry_mesh.faces] ).to(device))


        # final merge
        # garment_template = garment_engine['fl_final_registry'](source_fl_meshes = registry_meshes, outlayer = True)
        # registry_meshes = [registry_mesh for registry_mesh in  garment_template['source_fl_meshes']]





        fine_registry_meshes= [Meshes([registry_mesh.vertices], [registry_mesh.faces] ).to(device) for registry_mesh in registry_meshes]
        return fine_registry_meshes




    def offset_filter(self, registry_meshes, ratio, device):

        N = 1
        offset_filter = defaultdict(list)
        with torch.no_grad():
            for frame_id in range(len(self.dataset)):

                d_cond_list, poses, trans, rendcond = self.get_grad_parameters(frame_id, device)
                d_cond_list = d_cond_list[1:]

                for garment_idx, (registry_mesh, garment_name) in enumerate(zip(registry_meshes, self.garment_names)):
                    TmpVs = registry_mesh.verts_packed()
                    Tmpfs = registry_mesh.faces_packed()
                    TmpVnum=TmpVs.shape[0]

                    d_cond = d_cond_list[garment_idx]

                    self.deformer.defs[0](TmpVs[None,:,:].expand(N,-1,3),d_cond[None], ratio=ratio, offset_type = garment_name)

                    offset_filter[garment_name].append(self.deformer.defs[0].offset[garment_name].detach().cpu())

        self.filter_list = {}
        for garment_name in self.garment_names:
            offset_datas = torch.cat(offset_filter[garment_name], dim = 0)
            mean, var = offset_datas.mean(0), offset_datas.var(0, unbiased = True)
            offset_filter[garment_name] = [mean, var]

            query_list = [0]
            pre_statis_idx = 0
            for i in range(1, offset_datas.shape[0]):
                var_matrix = torch.sqrt(((offset_datas[i] - mean)**2 / var.mean(dim = 0, keepdim = True)))
                var_max = var_matrix.max()
                outlier = (var_matrix > 3).sum() / 3

                if outlier > 500:
                    query_list.append(pre_statis_idx)
                else:
                    pre_statis_idx = i
                    query_list.append(i)

                print("frame:{:04d} 3sigma:{:.4f} var:{:.4f}".format(i, outlier, var_max))
            self.filter_list[garment_name] = query_list






    def smooth_trans(self, H, W, device):
        '''
        smooth predict smpl poses avoid jitter
        '''
        d_cond_list, poses, trans_hat, __ = self.get_grad_parameters(torch.arange(len(self.dataset)), device)
        tcmr_predict_smooth = torch.load('./a_pose_female_process/anran/anran_tcmr_smpl/latest.pth')
        tcmr_poses  = tcmr_predict_smooth['poses']
        # trans_hat = tcmr_predict_smooth['trans'].cuda()

        # poses_hat[...,18:,:] =0
        # poses_hat = smooth_poses(poses_hat)

        smooth_range =  SMOOTH_TRANS[self.garment_type]
        d_cond_hat_list = [d_cond.detach().clone() for d_cond in d_cond_list]
        for smooth_list in smooth_range:
            start = smooth_list[0]
            end = smooth_list[-1]
            dist = (trans_hat[end] - trans_hat[start]) /(end-start)
            bin_size = dist[None]*(torch.arange(1,end-start)[:,None].cuda())
            update_trans = trans_hat[start][None] + bin_size
            trans_hat[start+1:end] = update_trans
            # d_cond_list_smooth
            for d_idx in range(len(d_cond_list)):
                fl_dist =  (d_cond_list[d_idx][end] - d_cond_list[d_idx][start])/(end-start)
                fl_bin_size = fl_dist[None]*(torch.arange(1,end-start)[:,None].cuda())
                update_fl = d_cond_list[d_idx][start][None] + fl_bin_size
                # update_fl = d_cond_list[d_idx][start]
                d_cond_hat_list[d_idx][start+1:end] = update_fl

        poses_hat = tcmr_poses.cuda()
        poses_hat = smooth_poses(poses_hat)

        # poses_hat  = torch.from_numpy(self.dataset.tcmr_poses.reshape(-1,24,3)).cuda()
        # for lingteng dance
        # poses_hat[:poses.shape[0], 16:18, :] = poses[0,16:18,:]


        return poses_hat, trans_hat, d_cond_hat_list

        # debug see smooth
        N = 1
        for frame_id in range(len(self.dataset)):
            __, outs = self.dataset[frame_id]

            try:
                imgs=outs['img']
                imgs  = (imgs+1) /2. * 255
            except:
                imgs = torch.ones((H,W,3)).float()
                imgs  = (imgs+1) /2. * 255
            trans = trans_hat[frame_id]
            poses  = poses_hat[frame_id].unsqueeze(0)

            focals,princeple_ps,Rs,Ts,H,W=self.dataset.get_camera_parameters(1, device)

            cameras= RectifiedPerspectiveCameras(focals,princeple_ps,Rs,Ts,image_size=[(W, H)]).to(device)

            def_body_vs=self.deformer.defs[1](self.tmpBodyVs[None,:,:].expand(N,-1,3),[poses,trans])

            proj_img_size = repeat(torch.Tensor([W,H]).float(), 'c -> b c', b= N).to(device)

            proj_pts = cameras.transform_points_screen(def_body_vs, proj_img_size)[0]


            imgs = imgs.detach().cpu().numpy().astype(np.uint8)
            for proj_pt in proj_pts.detach().cpu().numpy().astype(np.int):
                # draw_board = cv2.circle(draw_board, (int(fl[0]), int(fl[1])), 2, fl_color, 1)
                imgs = cv2.circle(imgs, (proj_pt[0], proj_pt[1]), 2, (0,0,255), 1)


            print('./debug/smooth/{:04d}.png'.format(frame_id))
            cv2.imwrite('./debug/smooth/{:04d}.png'.format(frame_id),imgs)
        xxx

        return poses_hat, trans_hat, d_cond_hat_list


    def not_smooth_trans(self, H, W, device):
        d_cond_hat_list, poses_hat, trans_hat, __ = self.get_grad_parameters(torch.arange(self.dataset.all_size()), device)
        # poses_hat  = torch.from_numpy(self.dataset.tcmr_poses.reshape(-1,24,3)).cuda()

        return poses_hat, trans_hat, d_cond_hat_list

        # debug to show
        N = 1
        for frame_id in range(len(self.dataset)):
            data_id, outs = self.dataset[frame_id]
            try:
                imgs=outs['img']
                imgs  = (imgs+1) /2. * 255
            except:
                imgs = torch.ones((H,W,3)).float()
                imgs  = (imgs+1) /2. * 255


            print(data_id)
            trans = trans_hat[data_id]
            poses  = poses_hat[data_id].unsqueeze(0)

            focals,princeple_ps,Rs,Ts,H,W=self.dataset.get_camera_parameters(1, device)

            cameras= RectifiedPerspectiveCameras(focals,princeple_ps,Rs,Ts,image_size=[(W, H)]).to(device)

            def_body_vs=self.deformer.defs[1](self.tmpBodyVs[None,:,:].expand(N,-1,3),[poses,trans])

            proj_img_size = repeat(torch.Tensor([W,H]).float(), 'c -> b c', b= N).to(device)

            proj_pts = cameras.transform_points_screen(def_body_vs, proj_img_size)[0]


            imgs = imgs.detach().cpu().numpy().astype(np.uint8)
            for proj_pt in proj_pts.detach().cpu().numpy().astype(np.int):
                # draw_board = cv2.circle(draw_board, (int(fl[0]), int(fl[1])), 2, fl_color, 1)
                imgs = cv2.circle(imgs, (proj_pt[0], proj_pt[1]), 2, (0,0,255), 1)


            print('./debug/smooth/{:04d}.png'.format(frame_id))
            cv2.imwrite('./debug/smooth/{:04d}.png'.format(frame_id),imgs)
        xxxx

        return poses_hat, trans_hat, d_cond_hat_list

    def smooth_poses_debug(self,poses_y, H, W, device):
        '''
        smooth predict smpl poses avoid jitter
        '''

        d_cond_list, _, trans, rendcond = self.get_grad_parameters(torch.arange(self.dataset.origin_size()), device)
        trans = trans.mean(0, keepdim = True)

        # debug see smooth
        N = 1
        for frame_id in range(len(self.dataset)):
            __, outs = self.dataset[frame_id]

            try:
                imgs=outs['img']
                imgs  = (imgs+1) /2. * 255
            except:
                imgs = torch.ones((H,W,3)).float()
                imgs  = (imgs+1) /2. * 255

            poses = torch.from_numpy(self.dataset.anim_poses[frame_id])[None].float().to(device)
            focals,princeple_ps,Rs,Ts,H,W=self.dataset.get_camera_parameters(1, device)

            cameras= RectifiedPerspectiveCameras(focals,princeple_ps,Rs,Ts,image_size=[(W, H)]).to(device)
            def_body_vs=self.deformer.defs[1](self.tmpBodyVs[None,:,:].expand(N,-1,3),[poses,trans])

            proj_img_size = repeat(torch.Tensor([W,H]).float(), 'c -> b c', b= N).to(device)

            proj_pts = cameras.transform_points_screen(def_body_vs, proj_img_size)[0]


            imgs = imgs.detach().cpu().numpy().astype(np.uint8)
            for proj_pt in proj_pts.detach().cpu().numpy().astype(np.int):
                # draw_board = cv2.circle(draw_board, (int(fl[0]), int(fl[1])), 2, fl_color, 1)
                imgs = cv2.circle(imgs, (proj_pt[0], proj_pt[1]), 2, (0,0,255), 1)


            print('./debug/smooth/{:04d}.png'.format(frame_id))
            cv2.imwrite('./debug/smooth/{:04d}.png'.format(frame_id),imgs)

    def infer_garment_animation(self,TmpVs_list,Tmpfs_list, poses_y, H,W,ratio,frame_ids, root = None):

        device=TmpVs_list[0].device

        d_cond_list, _, trans, rendcond = self.get_grad_parameters(torch.arange(self.dataset.origin_size()), device)
        if self.registry_meshes == None:
            fl_curve = self.inter_free_curve.inference()


            self.registry_meshes = self.registration(fl_curve, TmpVs_list, Tmpfs_list, ratio, root)
            max_face = max([mesh.faces_packed().shape[0] for mesh in self.registry_meshes])

            raster_settings_silhouette=RasterizationSettings(
                    image_size=(H,W),
                    blur_radius=0.,
                    # blur_radius=np.log(1. / 1e-4 - 1.)*3.e-6,
                    bin_size=(92 if max(H,W)>1024 and max(H,W)<=2048 else None),
                    # NOTE that max_faces_per_bin is very important since rendering noises from pytorch3d
                    max_faces_per_bin= max(10000, max_face ),
                    faces_per_pixel=1,
                    perspective_correct=True,
                    clip_barycentric_coords=False,
                    cull_backfaces=self.maskRender.rasterizer.raster_settings.cull_backfaces
                )

            self.maskRender.rasterizer.raster_settings=raster_settings_silhouette
            # smooth filter
            # self.smooth_poses_debug(poses_y, H, W, device)

        registry_meshes = self.registry_meshes
        focals,princeple_ps,Rs,Ts,H,W=self.dataset.get_camera_parameters(frame_ids.numel(),device)

        cameras= RectifiedPerspectiveCameras(focals,princeple_ps,Rs,Ts,image_size=[(W, H)]).to(device)
        self.maskRender.rasterizer.cameras=cameras

        if self.pcRender:
            self.pcRender.rasterizer.cameras=cameras

        d_cond_list, _, trans, rendcond = self.get_grad_parameters(torch.arange(self.dataset.origin_size()), device)
        d_cond_list = [d_cond.mean(0, keepdim = True) for d_cond in d_cond_list]
        trans = trans.mean(0, keepdim = True)
        rendcond = rendcond.mean(0, keepdim = True)
        d_cond_list = d_cond_list[1:]
        poses = poses_y.to(device)


        # d_cond_list_merge = []
        # for garment_idx, garment_name in enumerate(self.garment_names):
        #     d_cond_list, __, __, rendcond= self.get_grad_parameters([self.filter_list[garment_name][frame_ids[0]]], device)
        #     d_cond_list = d_cond_list[1:]
        #     d_cond_list_merge.append(d_cond_list[garment_idx])

        # d_cond_list = d_cond_list_merge

        colors_list = []
        imgs_list = []
        defMeshVs_list = []

        N=frame_ids.numel()
        def_smpl_body_vs =  self.deformer.defs[1](self.tmpBodyVs[None,:,:].expand(N,-1,3),[poses,trans])
        smpl_meshes = Meshes(verts = [def_smpl_body_vs[0]], faces = [self.tmpBodyFs], textures = TexturesVertex([torch.ones_like(self.tmpBodyVs)]))
        smpl_imgs, fragment = self.maskRender(smpl_meshes)

        smpl_imgs=smpl_imgs[...,:3]
        # if 'image' in gts:
        #     imgs[~masks]=gts['image'][~masks][:,[2,1,0]]
        smpl_imgs=torch.clamp(smpl_imgs*255.,min=0.,max=255.).detach().cpu().numpy().astype(np.uint8)

        for garment_idx, (registry_mesh, garment_name) in enumerate(zip(registry_meshes, self.garment_names)):
            with torch.no_grad():
                TmpVs = registry_mesh.verts_packed()
                Tmpfs = registry_mesh.faces_packed()
                TmpVnum=TmpVs.shape[0]

                N=frame_ids.numel()
                d_cond = d_cond_list[garment_idx]
                defTmpVs=self.deformer(TmpVs[None,:,:].expand(N,-1,3),[d_cond,[poses,trans]], ratio=ratio, offset_type = garment_name)
                defMeshes=Meshes(verts=[vs.view(TmpVnum,3) for vs in torch.split(defTmpVs,1)],faces=[Tmpfs for _ in range(N)],textures=TexturesVertex([torch.ones_like(TmpVs) for _ in range(N)]))
                defMeshVs = defTmpVs.detach().cpu().numpy()

                # tmp = trimesh.load('./train_mesh.obj')
                # defMeshes = Meshes([torch.from_numpy(tmp.vertices).float()],[torch.from_numpy(tmp.faces).long()]).cuda()
                imgs,frags=self.maskRender(defMeshes)

                masks=(frags.pix_to_face>=0).float()[...,0]
                masks=masks>0.
                imgs=imgs[...,:3]
                # if 'image' in gts:
                #     imgs[~masks]=gts['image'][~masks][:,[2,1,0]]
                imgs=torch.clamp(imgs*255.,min=0.,max=255.).cpu().numpy().astype(np.uint8)

                imgs = np.concatenate([smpl_imgs,imgs], axis = 2)

                # color render
                defTmpVs=self.deformer.defs[0](TmpVs[None,:,:].expand(N,-1,3),d_cond,ratio=ratio, offset_type = garment_name)
                defMeshes=Meshes(verts=[vs.view(TmpVnum,3) for vs in torch.split(defTmpVs,1)],faces=[Tmpfs for _ in range(N)],textures=TexturesVertex([torch.ones_like(TmpVs) for _ in range(N)]))

                newTs=self.dataset.trans.mean(0).to(device)[None,:]
                newcameras=RectifiedPerspectiveCameras(focals,princeple_ps,torch.tensor([[[-1.,0.,0.],[0.,1.,0.],[0.,0.,-1.]]],device=device).repeat(N,1,1),newTs.repeat(N,1),image_size=[(W, H)]).to(device)
                batch_inds,row_inds,col_inds,initTmpPs,_=utils.FindSurfacePs(TmpVs.detach(),Tmpfs,frags)

                cameras=self.maskRender.rasterizer.cameras
                rays=cameras.view_rays(torch.cat([col_inds.view(-1,1),row_inds.view(-1,1),torch.ones_like(col_inds.view(-1,1))],dim=-1).float())

                defconds=[d_cond.detach(),[poses.detach(),trans.detach()]]
            tcolors=[]
            print('draw %d points'%rays.shape[0])
            for ind,(rays_,initTmpPs_,batch_inds_) in enumerate(zip(torch.split(rays,10000),torch.split(initTmpPs,10000),torch.split(batch_inds,10000))):
                initTmpPs_,check=utils.OptimizeGarmentSurfaceSinlge(cameras.cam_pos().detach(),rays_.detach(),initTmpPs_.clone(),batch_inds_,self.garment_nets[garment_idx],ratio,self.deformer, defconds, dthreshold=1.e-4,athreshold=self.angThred,w1=3.05,w2=1.,times=30, offset_type=self.garment_names[garment_idx])
                # print('%d:(%d,%d)'%(ind,rays_.shape[0],check.sum().item()))
                initTmpPs_.requires_grad=True

                sdfs=self.garment_nets[garment_idx](initTmpPs_,ratio)
                nx=torch.autograd.grad(sdfs,initTmpPs_,torch.ones_like(sdfs),retain_graph=False,create_graph=False)[0]
                nx=nx/nx.norm(dim=1,keepdim=True)
                rays_,defVs_=utils.compute_cardinal_rays(self.deformer,initTmpPs_,rays_,defconds,batch_inds_,ratio,'test', offset_type= garment_name)
                with torch.no_grad():
                    tcolors.append(utils.compute_netRender_color(self.netRender,initTmpPs_,defVs_,nx,rays_,self.garment_nets[garment_idx].rendcond,rendcond[batch_inds_],ratio))
            tcolors=torch.cat(tcolors,dim=0)
            # print((gtCs[batch_inds,row_inds,col_inds]-tcolors).abs().mean().item())
            tcolors=torch.clamp((tcolors/2.+0.5)*255.,min=0.,max=255.)
            colors=torch.ones(N,H,W,3,device=device)*255.
            colors[batch_inds,row_inds,col_inds,:]=tcolors
            # if gts and 'image' in gts:
            #     colors[~masks]=gts['image'][~masks][:,:3]*255.

            colors=colors.cpu().numpy().astype(np.uint8)
            colors_list.append(colors)
            imgs_list.append(imgs)
            defMeshVs_list.append(defMeshVs)
        return colors_list, imgs_list, defMeshVs_list

    def infer_garment_fl(self,TmpVs_list,Tmpfs_list,H,W,ratio,frame_ids,notcolor=False,gts=None, root = None):
        '''
        this function is generate alpha_curve clinder given ordered points
        '''
        try:
            self.fl_curve_meshes
        except:
            #  upper_obj_path = os.path.join(root,'registry_short_sleeve_upper.obj')
            #  upper_obj = trimesh.load(upper_obj_path)
            #  verts = upper_obj.vertices
            #  boundaries = upper_obj.outline()
            #  b_pts_list = [entity.points for entity in boundaries.entities]
            #  curve_verts = [verts[b_pts] for b_pts in b_pts_list]
            #  curve_verts = [uniformsample3d(curve_vert, 2000) for curve_vert in curve_verts]
            #  self.fl_curve_meshes = self.inter_free_curve.curve_to_mesh(curve_verts = curve_verts, curve_idx = [0,1,2,3],target_idx = [3,2,0,1])

            self.fl_curve_meshes = self.inter_free_curve.curve_to_mesh()

        fl_curve_meshes = [mesh.clone() for mesh in self.fl_curve_meshes]

        fl_map = {}
        for fl_idx, fl_name in enumerate(self.fl_names):

            fl_map[fl_name] = fl_idx


        device=TmpVs_list[0].device
        d_cond_list, poses, trans, rendcond = self.get_grad_parameters(frame_ids, device)
        N = frame_ids.numel()
        # avaerage offset ot show only pose
        # d_cond_list, _, __, ___ = self.get_grad_parameters(torch.arange(len(self.dataset)), device)
        # d_cond_list = [d_cond.mean(0, keepdim = True) for d_cond in d_cond_list]
        d_cond_list = d_cond_list[1:]


        def_fl_verts_list = []
        def_fl_faces_list = []

        for garment_idx, garment_name in enumerate(self.garment_names):
            with torch.no_grad():
                fl_target = FL_EXTRACT[garment_name]
                curr_fl_mesh_list =[]
                for fl in fl_target:
                    curr_fl_mesh_list.append(fl_curve_meshes[fl_map[fl]])


                curr_fl_mesh_verts = [curr_fl_mesh.verts_packed() for curr_fl_mesh in curr_fl_mesh_list]
                curr_fl_mesh_faces = [curr_fl_mesh.faces_packed() for curr_fl_mesh in curr_fl_mesh_list]
                curr_fl_mesh_size = [curr_fl_mesh.verts_packed().shape[0] for curr_fl_mesh in curr_fl_mesh_list]
                TmpVs = torch.cat(curr_fl_mesh_verts, dim=0).cuda()



                d_cond = d_cond_list[garment_idx]
                defTmpVs=self.deformer(TmpVs[None,:,:].expand(N,-1,3),[d_cond,[poses,trans]], ratio=ratio, offset_type = garment_name)[0]
                def_fl_verts = torch.split(defTmpVs, curr_fl_mesh_size)


                def_fl_verts_list.extend(def_fl_verts)
                def_fl_faces_list.extend(curr_fl_mesh_faces)
        pre_face  = 0

        for f_idx in range(len(def_fl_faces_list)):
            def_fl_faces_list[f_idx] +=pre_face
            pre_face += def_fl_verts_list[f_idx].shape[0]

        def_verts = torch.cat(def_fl_verts_list)
        faces = torch.cat(def_fl_faces_list)

        fl_mesh = trimesh.Trimesh(def_verts.detach().cpu(), faces.detach().cpu(), process = False)




        return fl_mesh














    def infer_garment(self,TmpVs_list,Tmpfs_list,H,W,ratio,frame_ids,notcolor=False,gts=None, root = None, smooth =True):
        device=TmpVs_list[0].device

        smpl_vs, smpl_fs = self.tmpBodyVs.cuda(), self.tmpBodyFs.cuda()


        if self.registry_meshes == None:
            fl_curve = self.inter_free_curve.inference()

            for fl_idx, fl in enumerate(fl_curve):
                save_ply(os.path.join('fl_debug', '{:05}.ply'.format(fl_idx)), fl.view(-1,3).detach().cpu())




            self.registry_meshes = self.registration(fl_curve, TmpVs_list, Tmpfs_list, ratio, root)
            max_face = max([mesh.faces_packed().shape[0] for mesh in self.registry_meshes])

            raster_settings_silhouette=RasterizationSettings(
                    image_size=(H,W),
                    blur_radius=0.,
                    # blur_radius=np.log(1. / 1e-4 - 1.)*3.e-6,
                    bin_size=(92 if max(H,W)>1024 and max(H,W)<=2048 else None),
                    # NOTE that max_faces_per_bin is very important since rendering noises from pytorch3d
                    max_faces_per_bin= max(10000, max_face ),
                    faces_per_pixel=1,
                    perspective_correct=True,
                    clip_barycentric_coords=False,
                    cull_backfaces=self.maskRender.rasterizer.raster_settings.cull_backfaces
                )

            self.maskRender.rasterizer.raster_settings=raster_settings_silhouette

            # smooth filter
            smooth  = False
            if smooth:
                poses_hat, trans_hat, d_cond_hat_list = self.smooth_trans(H, W, device)
                # self.offset_filter(self.registry_meshes, ratio, device)
            else:
                poses_hat, trans_hat, d_cond_hat_list = self.not_smooth_trans(H, W, device)


            self.poses_hat, self.trans_hat, self.d_cond_hat_list = poses_hat, trans_hat, d_cond_hat_list


            d_cond_list, _, __, ___ = self.get_grad_parameters(torch.arange(len(self.dataset)), device)
            d_cond_list = [d_cond.mean(0, keepdim = True) for d_cond in d_cond_list]
            d_cond_list = d_cond_list[1:]
            for garment_idx, (registry_mesh, garment_name) in enumerate(zip(self.registry_meshes, self.garment_names)):

                if os.path.exists(os.path.join(root,'registry_{}.obj'.format(garment_name))):
                    continue
                TmpVs = registry_mesh.verts_packed()
                Tmpfs = registry_mesh.faces_packed()

                tri_meshes = trimesh.Trimesh(TmpVs.detach().cpu(), Tmpfs.detach().cpu(), process = False)
                tri_meshes.export(os.path.join(root,'registry_{}.obj'.format(garment_name)))

                d_cond = d_cond_list[garment_idx]
                N=frame_ids.numel()
                defTmpVs=self.deformer.defs[0](TmpVs[None,:,:].expand(N,-1,3),d_cond,ratio=ratio, offset_type = garment_name)
                verts_colors = self.deformer.defs[1].query_skinning_weights_colors(defTmpVs)

                defMeshes=Meshes(verts=[vs.view(-1,3) for vs in torch.split(defTmpVs,1)],faces=[Tmpfs for _ in range(N)])

                tri_meshes = trimesh.Trimesh(defMeshes.verts_packed().detach().cpu(), defMeshes.faces_packed().detach().cpu(), process = False)
                tri_meshes.visual.vertex_colors = verts_colors
                tri_meshes.export(os.path.join(root,'registry_{}_color.obj'.format(garment_name)))


                # save_obj(os.path.join(root,'registry_{}.obj'.format(garment_name)), defMeshes.verts_packed(), defMeshes.faces_packed())

        registry_meshes = self.registry_meshes




        focals,princeple_ps,Rs,Ts,H,W=self.dataset.get_camera_parameters(frame_ids.numel(),device)
        cameras= RectifiedPerspectiveCameras(focals,princeple_ps,Rs,Ts,image_size=[(W, H)]).to(device)
        self.maskRender.rasterizer.cameras=cameras


        if self.pcRender:
            self.pcRender.rasterizer.cameras=cameras

        d_cond_list, poses, trans, rendcond = self.get_grad_parameters(frame_ids, device)

        # avaerage offset ot show only pose
        # d_cond_list, _, __, ___ = self.get_grad_parameters(torch.arange(len(self.dataset)), device)
        # d_cond_list = [d_cond.mean(0, keepdim = True) for d_cond in d_cond_list]


        poses = self.poses_hat[frame_ids[0]].unsqueeze(0)
        trans = self.trans_hat[frame_ids[0]].unsqueeze(0)

        d_cond_hat_list = [d_cond_hat[frame_ids[0]][None] for d_cond_hat in self.d_cond_hat_list]


        print(frame_ids)

        d_cond_list = d_cond_hat_list[1:]



        # d_cond_list_merge = []
        # for garment_idx, garment_name in enumerate(self.garment_names):
        #     d_cond_list, __, __, rendcond= self.get_grad_parameters([self.filter_list[garment_name][frame_ids[0]]], device)
        #     d_cond_list = d_cond_list[1:]
        #     d_cond_list_merge.append(d_cond_list[garment_idx])

        # d_cond_list = d_cond_list_merge

        colors_list = []
        imgs_list = []
        def1imgs_list = []
        defMeshVs_list = []

        def_smpl_list = []

        N=frame_ids.numel()
        def_smpl_vs = self.deformer.defs[1](smpl_vs[None,:,:].expand(N,-1,3),[poses,trans])
        def_smpl_list= [Meshes([def_smpl_vs[0]], [smpl_fs])]

        color_map = RENDER_COLORS[self.garment_type]



        # merge render

        vs_list = []
        fs_list = []
        color_list = []

        add_faces= 0
        with torch.no_grad():
            for garment_idx, (registry_mesh, garment_name) in enumerate(zip(registry_meshes, self.garment_names)):
                    TmpVs = registry_mesh.verts_packed()
                    Tmpfs = registry_mesh.faces_packed()+add_faces
                    TmpVnum=TmpVs.shape[0]

                    d_cond = d_cond_list[garment_idx]
                    defTmpVs=self.deformer(TmpVs[None,:,:].expand(N,-1,3),[d_cond,[poses,trans]], ratio=ratio, offset_type = garment_name)

                    color = torch.ones_like(TmpVs)
                    color[...,:] = torch.tensor(color_map[garment_idx]).float() /255


                    vs_list.append(defTmpVs.view(-1,3))
                    fs_list.append(Tmpfs.view(-1,3))
                    color_list.append(color.view(-1,3))
                    add_faces += TmpVs.shape[0]


            defMeshes=Meshes(verts=[torch.cat(vs_list, dim = 0)], faces  = [torch.cat(fs_list, dim = 0)], textures=TexturesVertex([torch.cat(color_list, dim =0)]))



            imgs,frags=self.maskRender(defMeshes)

            masks=(frags.pix_to_face>=0).float()[...,0]
            gtMs=gts['mask']
            gts['maskE']=(1.-(masks*gtMs).view(N,-1).sum(1)/(masks+gtMs-masks*gtMs).abs().view(N,-1).sum(1)).cpu().numpy()
            masks=masks>0.
            imgs=imgs[...,:3]
            # if 'image' in gts:
            #     imgs[~masks]=gts['image'][~masks][:,[2,1,0]]

            merge_imgs=torch.clamp(imgs*255.,min=0.,max=255.).cpu().numpy().astype(np.uint8)[0]










        for garment_idx, (registry_mesh, garment_name) in enumerate(zip(registry_meshes, self.garment_names)):
            with torch.no_grad():
                TmpVs = registry_mesh.verts_packed()
                Tmpfs = registry_mesh.faces_packed()
                TmpVnum=TmpVs.shape[0]

                d_cond = d_cond_list[garment_idx]
                defTmpVs=self.deformer(TmpVs[None,:,:].expand(N,-1,3),[d_cond,[poses,trans]], ratio=ratio, offset_type = garment_name)


                color = torch.ones_like(TmpVs)
                color[...,:] = torch.tensor(color_map[garment_idx]).float() /255

                defMeshes=Meshes(verts=[vs.view(TmpVnum,3) for vs in torch.split(defTmpVs,1)],faces=[Tmpfs for _ in range(N)],textures=TexturesVertex([color for _ in range(N)]))
                defMeshVs = defTmpVs.detach().cpu().numpy()

                # verts_colors = self.deformer.defs[1].query_skinning_weights_colors(registry_mesh.verts_packed())
                # tri_meshes = trimesh.Trimesh(defMeshes.verts_packed().detach().cpu(), defMeshes.faces_packed().detach().cpu(), process = False)
                # tri_meshes.visual.vertex_colors = verts_colors
                # tri_meshes.export(os.path.join(root,'debug_{}_color.obj'.format(garment_name)))



                # tmp = trimesh.load('./train_mesh.obj')
                # defMeshes = Meshes([torch.from_numpy(tmp.vertices).float()],[torch.from_numpy(tmp.faces).long()]).cuda()
                imgs,frags=self.maskRender(defMeshes)

                masks=(frags.pix_to_face>=0).float()[...,0]
                gtMs=gts['mask']
                gts['maskE']=(1.-(masks*gtMs).view(N,-1).sum(1)/(masks+gtMs-masks*gtMs).abs().view(N,-1).sum(1)).cpu().numpy()
                masks=masks>0.
                imgs=imgs[...,:3]
                # if 'image' in gts:
                #     imgs[~masks]=gts['image'][~masks][:,[2,1,0]]


                imgs=torch.clamp(imgs*255.,min=0.,max=255.).cpu().numpy().astype(np.uint8)

                # color render
                def1TmpVs=self.deformer.defs[0](TmpVs[None,:,:].expand(N,-1,3),d_cond,ratio=ratio, offset_type = garment_name)
                def1Meshes=Meshes(verts=[vs.view(TmpVnum,3) for vs in torch.split(def1TmpVs,1)],faces=[Tmpfs for _ in range(N)],textures=TexturesVertex([torch.ones_like(TmpVs) for _ in range(N)]))


                newTs=self.dataset.trans.mean(0).to(device)[None,:]
                newcameras=RectifiedPerspectiveCameras(focals,princeple_ps,torch.tensor([[[-1.,0.,0.],[0.,1.,0.],[0.,0.,-1.]]],device=device).repeat(N,1,1),newTs.repeat(N,1),image_size=[(W, H)]).to(device)
                def1imgs,_=self.maskRender(def1Meshes,cameras=newcameras,lights=PointLights(device=device,location=((0, 1, newTs[0,2].item()),)))
                def1imgs=torch.clamp(def1imgs*255.,min=0.,max=255.).cpu().numpy().astype(np.uint8)

                batch_inds,row_inds,col_inds,initTmpPs,_=utils.FindSurfacePs(TmpVs.detach(),Tmpfs,frags)

                cameras=self.maskRender.rasterizer.cameras
                rays=cameras.view_rays(torch.cat([col_inds.view(-1,1),row_inds.view(-1,1),torch.ones_like(col_inds.view(-1,1))],dim=-1).float())

                defconds=[d_cond.detach(),[poses.detach(),trans.detach()]]
            if notcolor:
                return None,imgs,def1imgs,defMeshVs, def_smpl_list
            tcolors=[]
            print('draw %d points'%rays.shape[0])
            for ind,(rays_,initTmpPs_,batch_inds_) in enumerate(zip(torch.split(rays,10000),torch.split(initTmpPs,10000),torch.split(batch_inds,10000))):
                initTmpPs_,check=utils.OptimizeGarmentSurfaceSinlge(cameras.cam_pos().detach(),rays_.detach(),initTmpPs_.clone(),batch_inds_,
                        self.garment_nets[garment_idx],ratio,self.deformer, defconds,
                        dthreshold=1.e-4,athreshold=self.angThred,w1=3.05,w2=1.,times=30, offset_type=self.garment_names[garment_idx])
                # print('%d:(%d,%d)'%(ind,rays_.shape[0],check.sum().item()))
                initTmpPs_.requires_grad=True

                sdfs=self.garment_nets[garment_idx](initTmpPs_,ratio)
                nx=torch.autograd.grad(sdfs,initTmpPs_,torch.ones_like(sdfs),retain_graph=False,create_graph=False)[0]
                nx=nx/nx.norm(dim=1,keepdim=True)
                rays_,defVs_=utils.compute_cardinal_rays(self.deformer,initTmpPs_,rays_,defconds,batch_inds_,ratio,'test', offset_type= garment_name)
                with torch.no_grad():
                    tcolors.append(utils.compute_netRender_color(self.netRender,initTmpPs_,defVs_,nx,rays_,self.garment_nets[garment_idx].rendcond,
                        rendcond[batch_inds_],ratio))
            tcolors=torch.cat(tcolors,dim=0)
            # print((gtCs[batch_inds,row_inds,col_inds]-tcolors).abs().mean().item())
            tcolors=torch.clamp((tcolors/2.+0.5)*255.,min=0.,max=255.)
            colors=torch.ones(N,H,W,3,device=device)*255.
            colors[batch_inds,row_inds,col_inds,:]=tcolors

            # if gts and 'image' in gts:
            #     colors[~masks]=gts['image'][~masks][:,:3]*255.

            colors=colors.cpu().numpy().astype(np.uint8)
            colors_list.append(colors)
            imgs_list.append(imgs)
            def1imgs_list.append(def1imgs)
            defMeshVs_list.append(defMeshes)
        return colors_list, imgs_list, def1imgs_list, defMeshVs_list, def_smpl_list, merge_imgs

    def infer(self,TmpVs_list,Tmpfs_list,H,W,ratio,frame_ids,notcolor=False,gts=None):

        device=TmpVs_list[0].device
        focals,princeple_ps,Rs,Ts,H,W=self.dataset.get_camera_parameters(frame_ids.numel(),device)
        cameras=RectifiedPerspectiveCameras(focals,princeple_ps,Rs,Ts,image_size=[(W, H)]).to(device)
        self.maskRender.rasterizer.cameras=cameras

        if self.pcRender:
            self.pcRender.rasterizer.cameras=cameras

        d_cond_list, poses, trans, rendcond = self.get_grad_parameters(frame_ids, device)
        # idx 0 for body lant code
        d_cond_list = d_cond_list[1:]


        colors_list = []
        imgs_list = []
        def1imgs_list = []
        defMeshVs_list = []

        # inference garment_mesh
        for garment_idx, (TmpVs, Tmpfs, garment_name) in enumerate(zip(TmpVs_list, Tmpfs_list, self.garment_names)):
            with torch.no_grad():

                TmpVnum=TmpVs.shape[0]
                N=frame_ids.numel()
                d_cond = d_cond_list[garment_idx]
                defTmpVs=self.deformer(TmpVs[None,:,:].expand(N,-1,3),[d_cond,[poses,trans]], ratio=ratio, offset_type = garment_name)
                defMeshes=Meshes(verts=[vs.view(TmpVnum,3) for vs in torch.split(defTmpVs,1)],faces=[Tmpfs for _ in range(N)],textures=TexturesVertex([torch.ones_like(TmpVs) for _ in range(N)]))
                defMeshVs = defTmpVs.detach().cpu().numpy()
                # tmp = trimesh.load('./train_mesh.obj')
                # defMeshes = Meshes([torch.from_numpy(tmp.vertices).float()],[torch.from_numpy(tmp.faces).long()]).cuda()
                imgs,frags=self.maskRender(defMeshes)

                if gts:
                    masks=(frags.pix_to_face>=0).float()[...,0]
                    gtMs=gts['mask']
                    gts['maskE']=(1.-(masks*gtMs).view(N,-1).sum(1)/(masks+gtMs-masks*gtMs).abs().view(N,-1).sum(1)).cpu().numpy()
                    masks=masks>0.
                    imgs=imgs[...,:3]
                    if 'image' in gts:
                        imgs[~masks]=gts['image'][~masks][:,[2,1,0]]

                imgs=torch.clamp(imgs*255.,min=0.,max=255.).cpu().numpy().astype(np.uint8)

                # color render
                defTmpVs=self.deformer.defs[0](TmpVs[None,:,:].expand(N,-1,3),d_cond,ratio=ratio, offset_type = garment_name)
                defMeshes=Meshes(verts=[vs.view(TmpVnum,3) for vs in torch.split(defTmpVs,1)],faces=[Tmpfs for _ in range(N)],textures=TexturesVertex([torch.ones_like(TmpVs) for _ in range(N)]))


                newTs=self.dataset.trans.mean(0).to(device)[None,:]
                newcameras=RectifiedPerspectiveCameras(focals,princeple_ps,torch.tensor([[[-1.,0.,0.],[0.,1.,0.],[0.,0.,-1.]]],device=device).repeat(N,1,1),newTs.repeat(N,1),image_size=[(W, H)]).to(device)
                def1imgs,_=self.maskRender(defMeshes,cameras=newcameras,lights=PointLights(device=device,location=((0, 1, newTs[0,2].item()),)))
                def1imgs=torch.clamp(def1imgs*255.,min=0.,max=255.).cpu().numpy().astype(np.uint8)

                batch_inds,row_inds,col_inds,initTmpPs,_=utils.FindSurfacePs(TmpVs.detach(),Tmpfs,frags)

                cameras=self.maskRender.rasterizer.cameras
                rays=cameras.view_rays(torch.cat([col_inds.view(-1,1),row_inds.view(-1,1),torch.ones_like(col_inds.view(-1,1))],dim=-1).float())

                defconds=[d_cond.detach(),[poses.detach(),trans.detach()]]
            if notcolor:
                return None,imgs,def1imgs,defMeshVs
            tcolors=[]
            print('draw %d points'%rays.shape[0])
            for ind,(rays_,initTmpPs_,batch_inds_) in enumerate(zip(torch.split(rays,10000),torch.split(initTmpPs,10000),torch.split(batch_inds,10000))):
                initTmpPs_,check=utils.OptimizeGarmentSurfaceSinlge(cameras.cam_pos().detach(),rays_.detach(),initTmpPs_.clone(),batch_inds_,self.garment_nets[garment_idx],ratio,self.deformer, defconds, dthreshold=1.e-4,athreshold=self.angThred,w1=3.05,w2=1.,times=30, offset_type=self.garment_names[garment_idx])
                # print('%d:(%d,%d)'%(ind,rays_.shape[0],check.sum().item()))
                initTmpPs_.requires_grad=True

                sdfs=self.garment_nets[garment_idx](initTmpPs_,ratio)
                nx=torch.autograd.grad(sdfs,initTmpPs_,torch.ones_like(sdfs),retain_graph=False,create_graph=False)[0]
                nx=nx/nx.norm(dim=1,keepdim=True)
                rays_,defVs_=utils.compute_cardinal_rays(self.deformer,initTmpPs_,rays_,defconds,batch_inds_,ratio,'test', offset_type= garment_name)
                with torch.no_grad():
                    tcolors.append(utils.compute_netRender_color(self.netRender,initTmpPs_,defVs_,nx,rays_,self.garment_nets[garment_idx].rendcond,rendcond[batch_inds_],ratio))
            tcolors=torch.cat(tcolors,dim=0)
            # print((gtCs[batch_inds,row_inds,col_inds]-tcolors).abs().mean().item())
            tcolors=torch.clamp((tcolors/2.+0.5)*255.,min=0.,max=255.)
            colors=torch.ones(N,H,W,3,device=device)*255.
            colors[batch_inds,row_inds,col_inds,:]=tcolors

            if gts and 'image' in gts:
                colors[~masks]=gts['image'][~masks][:,:3]*255.

            colors=colors.cpu().numpy().astype(np.uint8)
            colors_list.append(colors)
            imgs_list.append(imgs)
            def1imgs_list.append(def1imgs)
            defMeshVs_list.append(defMeshVs)
        return colors_list, imgs_list, def1imgs_list, defMeshVs_list


    def draw_loss(self, steps,  **kwarges):
        if self.visualizer == None:
            return
        else:
            info = self.info
            info.update(kwarges)
            scalar_infos = make_recursive_meta_func(info)
            self.visualizer.add_scalar(scalar_infos, int(steps))


    @make_nograd_func
    def visualize_curve_mesh(self,save_fl_mesh_path, epoch, device):
        '''
        visualize_mesh and feature curve results
        '''
        def render_canonical_mesh(meshes, offset):
            N = len(meshes)
            newTs=self.dataset.trans.mean(0).to(device)[None,:]
            offset_matrix = torch.eye(4).to(offset)
            offset_matrix[:3, 3] = -offset[0,...]
            meshes = pytorch3d_mesh_transformation(meshes, offset_matrix)
            newTs[...,:-1] =0
            newTs[...,-1]+=1

            newcameras=RectifiedPerspectiveCameras(focals,princeple_ps,torch.tensor([[[-1.,0.,0.],[0.,1.,0.],[0.,0.,-1.]]],device=device).repeat(N,1,1),newTs.repeat(N,1),image_size=[(W, H)]).to(device)
            def1imgs, fragment=self.maskRender(meshes, cameras=newcameras,lights=PointLights(device=device,location=((0, 1, newTs[0,2].item()),)))
            def1imgs=def1imgs[...,:3]

            return def1imgs, fragment

        def show_garment_with_feature_line(cur_imgs, cur_masks, garment_imgs, garment_masks, degree):
            featureline_batch = {}
            garment_batch = []
            for img, mask, name in zip(cur_imgs, cur_masks, FL_INFOS[self.garment_type]):
                featureline_batch[name]= [img, mask]

            for garment_img, garment_name in zip(garment_imgs, self.garment_names):
                for match_fl in GARMENT_FL_MATCH[garment_name]:
                    fl_img, fl_mask = featureline_batch[match_fl]
                    fl_mask = fl_mask[..., None]
                    garment_img  = garment_img * (1-fl_mask) + fl_img * fl_mask

                garment_img = torch.clamp(garment_img * 255.,min=0.,max=255.).cpu().numpy().astype(np.uint8)
                garment_batch.append(garment_img)

            return garment_batch

        def show_smpl_with_feature_line(cur_imgs, cur_masks, smpl_imgs, smpl_masks, degree):
            smpl_img = smpl_imgs[0]
            for cur_mask, cur_img  in zip(cur_masks, cur_imgs):
                cur_mask = cur_mask[..., None]
                smpl_img  = smpl_img * (1-cur_mask) + cur_img * cur_mask

            smpl_img = torch.clamp(smpl_img * 255.,min=0.,max=255.).cpu().numpy().astype(np.uint8)

            return smpl_img

        def build_curve_mesh(curves):
            curve_verts_list = []
            curve_faces_list = []
            for curve in curves:
                curve_center = curve.mean(dim=0, keepdim = True)
                nump = torch.arange(curve.shape[0])
                center_id = nump[-1]+1
                face_list = []

                for p_id in range(curve.shape[0]-1):
                    face_list.append([p_id, p_id+1, center_id])

                faces = torch.tensor(face_list).long().to(curve)
                verts = torch.cat([curve, curve_center], dim =0).float()

                curve_verts_list.append(verts)
                curve_faces_list.append(faces)




            color_curve_meshes = Meshes(verts = curve_verts_list,
            faces = curve_faces_list,
            textures=TexturesVertex([red_like(curve_verts) for curve_verts in curve_verts_list]))

            return color_curve_meshes




        # choose hardphong shader
        self.maskRender.shader=HardPhongShader(device,self.maskRender.rasterizer.cameras)
        focals,princeple_ps,Rs,Ts,H,W=self.dataset.get_camera_parameters(1, device)

        cano_curve_verts = self.inter_free_curve.inference()
        cano_color_curve_meshes = build_curve_mesh(cano_curve_verts)

        smpl_meshes = Meshes(verts = [self.tmpBodyVs], faces = [self.tmpBodyFs], textures = TexturesVertex([torch.ones_like(self.tmpBodyVs)]))


        rigid_cano_fl_verts = torch.split(cano_curve_verts,1, dim =0)
        rigid_cano_fl_verts = [rigid_cano_fl_vert.squeeze(0) for rigid_cano_fl_vert in rigid_cano_fl_verts]
        cano_smpl_fl_verts = self.cano_fl_to_body_trans(rigid_cano_fl_verts, self.fl_names)
        cano_color_smpl_curve_meshes = build_curve_mesh(cano_smpl_fl_verts)

        # save canonical-view fl
        save_ply(os.path.join(save_fl_mesh_path, 'canonical_fl.ply'), cano_curve_verts.view(-1,3).detach().cpu())
        save_ply(os.path.join(save_fl_mesh_path, 'cano_smpl_fl.ply'), torch.cat(cano_smpl_fl_verts, dim = 0).view(-1,3).detach().cpu())


        skinning_textures  = [self.deformer.defs[1].query_skinning_weights_colors(garment_vs[None]).to(garment_vs) for garment_vs in self.garment_vs]
        skinning_textures  = [skinning_texture.float() for skinning_texture in skinning_textures]


        color_meshes =Meshes(verts = [vs for vs in self.garment_vs],
                faces = [fs for fs in self.garment_fs],
                textures=TexturesVertex([skinning_texture for skinning_texture in skinning_textures]))

        title_list = self.garment_names[:]
        title_list.append('body')
        show_board = {}

        # for degree in [0, 120, 240]:
        for degree in [0, 120,240]:
            degree_board = []
            R_y = Rotate_Y_axis(degree)

            rotate_color_meshes = pytorch3d_mesh_transformation(color_meshes, R_y)
            rotate_color_meshes_mean = rotate_color_meshes.verts_packed().mean(0, keepdim = True)



            rotate_cano_color_curve_meshes = pytorch3d_mesh_transformation(cano_color_curve_meshes, R_y)
            cur_imgs, cur_fragment = render_canonical_mesh(rotate_cano_color_curve_meshes, rotate_color_meshes_mean)
            cur_masks=(cur_fragment.pix_to_face>=0).float()[...,0]

            garment_imgs, garment_fragment = render_canonical_mesh(rotate_color_meshes, rotate_color_meshes_mean)
            garment_masks = (garment_fragment.pix_to_face>=0).float()[...,0]
            img_batch = show_garment_with_feature_line(cur_imgs, cur_masks, garment_imgs, garment_masks, degree)
            degree_board.extend(img_batch)

            rotate_color_smpl_meshes = pytorch3d_mesh_transformation(smpl_meshes, R_y)
            rotate_color_smpl_mean = rotate_color_smpl_meshes.verts_packed().mean(0, keepdim = True)
            smpl_imgs, smpl_fragment = render_canonical_mesh(rotate_color_smpl_meshes, rotate_color_smpl_mean)
            smpl_masks = (smpl_fragment.pix_to_face>=0).float()[...,0]

            rotate_cano_color_smpl_curve_meshes = pytorch3d_mesh_transformation(cano_color_smpl_curve_meshes, R_y)
            cur_imgs, cur_fragment = render_canonical_mesh(rotate_cano_color_smpl_curve_meshes, rotate_color_smpl_mean)
            cur_masks=(cur_fragment.pix_to_face>=0).float()[...,0]
            smpl_batch = show_smpl_with_feature_line(cur_imgs, cur_masks, smpl_imgs, smpl_masks, degree)
            degree_board.append(smpl_batch)
            show_board['{:03d}'.format(degree)] = degree_board

        columns = ['degree']
        columns.extend(title_list)

        image_table_row = []
        for key in show_board.keys():
            image_table_col = []
            image_table_col.append(key)
            image_table_col.extend(show_board[key])
            image_table_row.append(image_table_col)

        title = 'render canonical mesh'
        self.visualizer.log_images_to_wandb(image_table_row, columns, step= epoch, size=512, title = title)

        render_path = os.path.join(self.root, 'render_show')
        os.makedirs(render_path, exist_ok= True)
        for key in show_board.keys():
            board = np.concatenate(show_board[key], axis = 1)
            cv2.imwrite(os.path.join(render_path, 'render_cur_{}.png'.format(key)), board)

        # NOTE that the following code is very important
        # return to mask render
        self.maskRender.shader=SoftSilhouetteShader()




    def align_fl(self, fl_align_path, epoch = 0):
        def merge_fl_meshes(fl_mesh_list):
            merge_fl = fl_mesh_list[0]
            for merge_fl_idx in range(1, len(fl_mesh_list)):
                merge_fl.update(fl_mesh_list[merge_fl_idx])
            return merge_fl

        vs=self.tmpBodyVs
        device=vs.device
        garment_templates = self.garment_by_init_smpl()
        # dense_boundary OK
        dense_garment_templates = []
        for garment_idx, garment_template in enumerate(garment_templates):
            # edge dense pc times
            for __ in range(2):
                garment_template = garment_template.dense_boundary()
            dense_garment_templates.append(garment_template)
        # extract_garment_template_mesh

        garment_fl_templates = [dense_garment_template.extract_featurelines() for dense_garment_template in dense_garment_templates]

        fl_meshes = merge_fl_meshes(garment_fl_templates)
        fl_meshes = tocuda([fl_meshes[fl_name] for fl_name in FL_INFOS[self.garment_type]])



        trans_matrix = torch.load(fl_align_path)
        rigid_R = trans_matrix['rigid_R'].to(device)
        rigid_T = trans_matrix['rigid_T'].to(device)
        rigid_scale = trans_matrix['rigid_scale'].to(device)
        self.cano_fl_to_body_trans = Inverse_Fl_Body(fl_meshes, self.fl_names, rigid_T, rigid_scale)

        if 'rigid_scale' in trans_matrix.keys():
            rigid_scale = trans_matrix['rigid_scale'].to(device)
            meshes_vertices = scale_icp_rotate_center_transform(fl_meshes, rigid_R, rigid_T, rigid_scale)
        else:
            raise NotImplemented
            meshes_vertices = icp_rotate_center_transform(fl_meshes, rigid_R, rigid_T)

        rigid_center_list = [ meshes_vertice.mean(0, keepdim = True) for meshes_vertice in meshes_vertices]
        self.cano_fl_to_body_trans.set_rigid_center(rigid_center_list, self.fl_names)

        meshes_faces = [fl_mesh.faces_packed() for fl_mesh in fl_meshes]

        fl_meshes = [Meshes([verts], [faces]) for verts, faces in zip(meshes_vertices, meshes_faces)]
        # feature line curve explict meshes
        self.fl_meshes = fl_meshes

        fl_verts = [fl_mesh.verts_packed() for fl_mesh in self.fl_meshes]
        cano_smpl_fl_verts = self.cano_fl_to_body_trans(fl_verts, self.fl_names)

        fl_meshes = Meshes([verts for verts in meshes_vertices], [faces for faces in meshes_faces])


        save_obj('./debug/rigid_fl.obj', fl_meshes.verts_packed(), fl_meshes.faces_packed())
        fl_meshes = Meshes([verts for verts in cano_smpl_fl_verts], [faces for faces in meshes_faces])
        save_obj('./debug/cano_fl.obj', fl_meshes.verts_packed(), fl_meshes.faces_packed())

        self.inter_free_curve = Intersect_Free_Curve(fl_verts, cano_smpl_fl_verts, meshes_faces, self.fl_names, self.cano_fl_to_body_trans, sample_num=200)


        cano_verts = self.inter_free_curve()


