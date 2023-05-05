'''
copy from self-recon
'''
import numpy as np
import torch.nn as nn
import torch
from torch.nn.parameter import Parameter
import torch.autograd.functional as F
from FastMinv import Fast3x3Minv
from .Embedder import get_embedder
import utils
import os
import cv2
from pytorch3d.io import save_obj, save_ply
from pytorch3d.renderer.points.rasterizer import PointsRasterizationSettings
import pdb
from pytorch3d.transforms.transform3d import Transform3d
from engineer.networks.OptimNetwork import OptimNetwork
from engineer.networks.OptimGarmentNetwork import OptimGarmentNetwork
from engineer.networks.OptimGarmentNetwork_Large_Pose import OptimGarmentNetwork_LargePose
import trimesh
from utils.constant import TEMPLATE_GARMENT
from engineer.optimizer.lap_deform_optimizer import Laplacian_Optimizer
from engineer.core.beta_optimizer import smpl_beta_optimizer
from engineer.utils.skinning_weights import weights2colors

class ImplicitNetwork(nn.Module):
    def __init__(
            self,
            feature_vector_size,
            d_in,
            d_out,
            dims,
            geometric_init=True,
            bias=1.0,
            skip_in=(),
            weight_norm=True,
            multires=0
    ):
        super().__init__()



        dims = [d_in] + dims + [d_out + feature_vector_size]

        self.d_out=d_out
        self.embed_fn = None
        self.multires=multires
        if multires > 0:
            # position embedding
            embed_fn, input_ch = get_embedder(multires)

            self.embed_fn = embed_fn
            dims[0] = input_ch

        self.num_layers = len(dims)
        self.skip_in = skip_in

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                    torch.nn.init.constant_(lin.bias, -bias)
                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.softplus = nn.Softplus(beta=100)

    def forward(self, input, ratio):
        ratio=ratio if type(ratio)==float or type(ratio)==int else ratio['sdfRatio']


        if self.embed_fn is not None:
            if ratio is None:    #one weight, all equal one
                input = self.embed_fn(input)
            elif ratio<=0: #zero weight
                input = self.embed_fn(input, [0. for _ in range(self.multires*2)])
            else:
                input = self.embed_fn(input, utils.annealing_weights(self.multires,ratio))
        x = input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, input], 1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.softplus(x)

        if x.shape[-1]>self.d_out:
            self.rendcond=x[:,self.d_out:]
            x=x[:,0:self.d_out]
        else:
            self.rendcond=None

        return x

    def gradient(self, x,y=None):
        x.requires_grad_(True)
        if y is None:
            y = self.forward(x)
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return gradients.view(-1,3)

def getTmpSdf(device,multires,bias=0.6,feature_vector_size=256):
    # get tmp sdf based on implicit network
    # bias is the radius of sphere given sdf network
    net= ImplicitNetwork(feature_vector_size=feature_vector_size,d_in=3,d_out=1,dims = [ 512, 512, 512, 512, 512, 512, 512, 512 ],geometric_init= True,bias = bias,skip_in = [4],weight_norm=True,multires=multires)


    return net.to(device)

import MCGpu
from pytorch3d.structures import Meshes,Pointclouds,join_meshes_as_batch
from pytorch3d.loss import (
    chamfer_distance,
    mesh_edge_loss,
    mesh_laplacian_smoothing,
    mesh_normal_consistency,
)
from .CameraMine import PointsRendererWithFrags
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
    AlphaCompositor
)
#debug
import cv2
import trimesh
import openmesh as om
import os.path as osp
from MCAcc import Seg3dLossless
from .Deformer import initialLBSkinner,getTranslatorNet,CompositeDeformer,LBSkinner
from .CameraMine import RectifiedPerspectiveCameras
# from pytorch3d.renderer import (
#     RasterizationSettings,
#     MeshRasterizer,
#     SoftSilhouetteShader,
#     BlendParams
# )
from pytorch3d.renderer.mesh.renderer import MeshRendererWithFragments
def getOptNet(dataset, save_folder, N,bmins,bmaxs,resolutions,device,conf,use_initial_sdf=True,use_initial_skinner=True, visualizer = None, opt_large = False):


    # 6 256
    sdf_multires=conf.get_int('sdf_net.multires')
    garment_type =conf.get_string('train.garment_type')
    garment_net = []
    for template_name in TEMPLATE_GARMENT[garment_type]:
        sdf_multires=conf.get_int('garment_sdf_net.multires')
        condlen=conf.get_int('render_net.condlen')
        garment_net.append(getTmpSdf(device,sdf_multires,0.6,condlen))


    # garment_sdf_initialized()
    sdf_multires=conf.get_int('sdf_net.multires')
    condlen=conf.get_int('render_net.condlen')
    # output 257 sdf-value and embeding
    tmpSdf=getTmpSdf(device,sdf_multires,0.6,condlen)

    # -1200, 1, initial_sdf_idr_6_1
    sdf_initialized=conf.get_int('train.initial_iters')
    init_pose_type=conf.get_int('train.skinner_pose_type') if 'train.skinner_pose_type' in conf else 0
    initial_sdf_file = osp.join(dataset.root, save_folder, 'initial_sdf_idr' +'_%d_%d.pth'%(sdf_multires,init_pose_type))
    initial_sdf_path = osp.join(dataset.root, save_folder, 'initial_sdf_idr' +'_%d_%d.ply'%(sdf_multires,init_pose_type))


    if osp.isfile(initial_sdf_file) and use_initial_sdf:
        tmpSdf.load_state_dict(torch.load(initial_sdf_file,map_location='cpu'))
        tmp_sdf_mesh = trimesh.load(initial_sdf_path)

        if (len(garment_net)):
            for garment_idx, garment_name in enumerate(TEMPLATE_GARMENT[garment_type]):
                init_garment_file = initial_sdf_file.replace('sdf', 'sdf_{}'.format(garment_name))

                assert osp.isfile(init_garment_file)
                garment_net[garment_idx].load_state_dict(torch.load(init_garment_file,map_location='cpu'))
        sdf_initialized=-1
    elif sdf_initialized<=0:
        sdf_initialized=1200
        tmp_sdf_mesh = None

    skinner_pth_name=osp.join(dataset.root, save_folder, 'initial_skinner_%d.pth'%init_pose_type)

    if osp.isfile(skinner_pth_name) and use_initial_skinner:
        data=torch.load(skinner_pth_name,map_location='cpu')
        dataset.shape = data['betas']
        fite_skinning_path = os.path.join(dataset.root, 'diffused_skinning_weights.npy')
        fite_skinning = torch.from_numpy(np.load(fite_skinning_path)).float()
        # initPose=torch.from_numpy(utils.smpl_tmp_cPose()).view(1,24,3)
        skinner=LBSkinner(fite_skinning[None],data['bmins'],data['bmaxs'],data['Js'],data['parents'],
                init_pose = data['init_pose'],align_corners=False, extra_trans = data['extra_trans'],
                bbox_center= data['bbox_center'],bbox_extend=data['bbox_extend'])
        # skinner=LBSkinner(data['ws'],data['bmins'],data['bmaxs'],data['Js'],data['parents'], init_pose=data['init_pose'],align_corners=False, extra_trans = data['extra_trans'])
        tmpBodyVs=data['tmpBodyVs']
        tmpBodyFs=data['tmpBodyFs']

        # smpl_body_path = osp.join(dataset.root, save_folder, 'A-smpl_body.obj' )
        # trim_smpl_body = trimesh.Trimesh(tmpBodyVs.detach().cpu(), tmpBodyFs.detach().cpu(), process = False)
        # trim_smpl_body.export(smpl_body_path)


        # xxx
        # debug for skinning_weights
        # skinner.cuda()
        # skinning_colors = skinner.query_skinning_weights_colors(tmpBodyVs.cuda())
        # tmp_body = trimesh.Trimesh(tmpBodyVs, tmpBodyFs, process =False)
        # tmp_body.visual.vertex_colors = skinning_colors
        # tmp_body.export('fuck.obj')
        # pdb.set_trace()

    else:
        # initilize initPose to A pose to save volume space
        initPose=torch.from_numpy(utils.smpl_tmp_Apose(init_pose_type)).view(1,24,3).to(device)
        # optimize beta, extra trans residual by human pose

        if dataset.gt_joints2d is not None:
            betas, extra_trans = smpl_beta_optimizer(dataset.gender, initPose, dataset, device)
            extra_trans = extra_trans.detach().cpu()
        else:
            betas = dataset.shape.detach().clone()
            extra_trans = None

        betas.requires_grad = False
        dataset.shape = betas.detach().cpu()

        skinner,tmpBodyVs,tmpBodyFs=initialLBSkinner(dataset.gender,dataset.shape.to(device),initPose,(128+1, 224+1, 64+1),bmins,bmaxs, extra_trans)
        torch.save({'ws':skinner.ws,'bmins':skinner.b_min,'bmaxs':skinner.b_max,'Js':skinner.Js,
                    'parents':skinner.parents,'init_pose':skinner.init_pose,
                    'tmpBodyVs':tmpBodyVs,'tmpBodyFs':tmpBodyFs, 'betas':betas, 'extra_trans': extra_trans ,'bbox_center': skinner.bbox_center,
                    'bbox_extend':skinner.bbox_extend},
                    skinner_pth_name)

        smpl_body_path = osp.join(dataset.root, save_folder, 'A-smpl_body.obj' )
        trim_smpl_body = trimesh.Trimesh(tmpBodyVs.detach().cpu(), tmpBodyFs.detach().cpu(), process = False)
        trim_smpl_body.export(smpl_body_path)
        ### NOTE that need to compute diffused skinning weights

    #use False: weight norm can influence weight initialization, can not produce small weights as initialization
    # deformer=MLPTranslator(dataset.conds[dataset.cond_ns.index('deformer')].shape[-1],conf.get_int('mlp_deformer.multires'),False)
    # 128 + 39
    deformer=getTranslatorNet(device,conf.get_config('mlp_deformer'))
    deformer=CompositeDeformer([deformer,skinner]).to(device)
    cam_data=dataset.camera_params
    cameras=RectifiedPerspectiveCameras(cam_data['focal_length'].view(1,2).expand(N,2),cam_data['princeple_points'].view(1,2).expand(N,2),
                utils.quat2mat(cam_data['cam2world_coord_quat'].view(1,4)).expand(N,3,3),cam_data['world2cam_coord_trans'].view(1,3).expand(N,3),
                image_size=[(dataset.W, dataset.H)]).to(device)

    b_min, b_max = skinner.bbox_size()



    engine = Seg3dLossless(
            query_func=None,
            b_min = b_min.tolist(),
            b_max = b_max.tolist(),
            resolutions=resolutions,
            align_corners=False,
            balance_value=0.0, # be careful
            device=device,
            visualize=False,
            debug=False,
            use_cuda_impl=False,
            faster=False
        )

    raster_settings_silhouette = RasterizationSettings(
    image_size=(dataset.H,dataset.W),
    blur_radius=0.,
    bin_size=int(2 ** max(np.ceil(np.log2(max(dataset.H,dataset.W))) - 4, 4)),
    faces_per_pixel=1,
    perspective_correct=True,
    clip_barycentric_coords=False,
    cull_backfaces=False
    )
    renderer = MeshRendererWithFragments(
    rasterizer=MeshRasterizer(
        cameras=cameras,
        raster_settings=raster_settings_silhouette
    ),
    shader=SoftSilhouetteShader()
    )
    rendnet=RenderNet.getRenderNet(device,conf.get_config('render_net'))

    # fl init registry
    # TODO garment registry setting
    garment_engine = dict(fl_init_registry = Laplacian_Optimizer())


    # rendnet=getattr(RenderNet,conf.get_string('render_net.type'))(conf.get_int('render_net.condlen'),d_in=9,d_out=3,dims = [ 512, 512, 512, 512 ],mode='idr',weight_norm=True,multires=conf.get_int('render_net.multires')).to(device)
    # optNet=OptimNetwork(tmpSdf,deformer,engine,renderer,rendnet,conf=conf.get_config('loss_coarse'))
    # optNet.register_buffer('tmpBodyVs',tmpBodyVs)
    # optNet.register_buffer('tmpBodyFs',tmpBodyFs)
    if not opt_large:
        optNet=OptimGarmentNetwork(tmpSdf,deformer,engine,renderer,rendnet,conf=conf, garment_net = garment_net, garment_type = garment_type, tmpBodyFs= tmpBodyFs, tmpBodyVs = tmpBodyVs)
    else:
        # training large pose network
        optNet=OptimGarmentNetwork_LargePose(tmpSdf,deformer,engine,renderer,rendnet,conf=conf, garment_net = garment_net, garment_type = garment_type, tmpBodyFs= tmpBodyFs, tmpBodyVs = tmpBodyVs)


    optNet.visualizer = visualizer
    if optNet.garment_engine == None:
        optNet.garment_engine = garment_engine


    if tmp_sdf_mesh is not None:
        optNet.load_init_sdf_vertices(tmp_sdf_mesh)
    optNet.remesh_intersect=conf.get_int('train.coarse.point_render.remesh_intersect')

    tmp=om.TriMesh(points=tmpBodyVs.cpu().numpy(),face_vertex_indices=tmpBodyFs.cpu().numpy())
    tmp.request_face_normals()
    tmp.request_vertex_normals()
    tmp.update_normals()
    optNet.register_buffer('tmpBodyNs',torch.from_numpy(tmp.vertex_normals()))
    optNet=optNet.to(device)
    optNet.dataset=dataset
    if dataset.poses.requires_grad or dataset.trans.requires_grad:
        # range from 10-30 discreate cosine tranformation
        optNet.dctnull=utils.DCTNullSpace(10,30).to(device)

    return optNet, sdf_initialized
