"""
@File: CameraMine.py
@Author: Lingteng Qiu
@Email: qiulingteng@link.cuhk.edu.cn
@Date: 2022-10-25
@Desc: Deformer codes modified from SelfRecon
"""
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from smpl_pytorch.util import batch_rodrigues
from smpl_pytorch.SMPL import SMPL,getSMPL
from .Embedder import get_embedder
from MCAcc import GridSamplerMine3dFunction
import utils
from pytorch3d.io import save_obj, save_ply
import pdb
from  pytorch3d.ops.knn import knn_points
from engineer.utils.skinning_weights import weights2colors

class CompositeDeformer(nn.Module):
    def __init__(self,deformers):
        super().__init__()
        self.N=len(deformers)
        self.defs=nn.ModuleList(deformers)
    def forward(self,ps,conds,batch_inds=None,**kwargs):
        assert(self.N==len(conds))
        out=ps
        for cond,deformer in zip(conds,self.defs):
            out=deformer(out,cond,batch_inds,**kwargs)


        return out

class Inverse_Fl_Body(nn.Module):
    def __init__(self, cano_fl_meshes, fl_names, rigid_t_list, rigid_scale_list):
        '''
        canonical feature line in body-space
        '''
        super().__init__()

        self.fl_names = fl_names
        self.initialize_canonical_coordinate(cano_fl_meshes, rigid_t_list, rigid_scale_list)
    def set_rigid_center(self, rigid_center_list, fl_names):
        self.rigid_center_dict = {}
        assert len(rigid_center_list) == len(fl_names)
        for rigid_center, fl_name in zip(rigid_center_list, fl_names):
            self.rigid_center_dict[fl_name] = rigid_center


    def initialize_canonical_coordinate(self, cano_fl_meshes, rigid_t_list, rigid_scale_list):
        '''compute canonical scale_init, scale parameters
        '''

        def get_center_scale_init(v):
            center = v.mean(0, keepdim = True)
            v_dirs = (v-center) / ((v- center).norm(dim=1, keepdim = True) + 1e-6)
            init_scale = ((v - center) * v_dirs).sum(dim = -1, keepdim = True)

            return center, v_dirs, init_scale

        self.v_dirs_dict = {}
        self.init_scale_dict = {}
        self.verts_dict = {}
        self.center_dict = {}
        self.rigid_t_dict = {}
        self.rigid_scale_dict = {}


        for fl_name, cano_fl_mesh, rigid_t, rigid_s in zip(self.fl_names, cano_fl_meshes, rigid_t_list, rigid_scale_list):
            center, v_dirs, init_scale = get_center_scale_init(cano_fl_mesh.verts_packed())
            self.v_dirs_dict[fl_name] = v_dirs
            self.init_scale_dict[fl_name] = init_scale
            self.verts_dict[fl_name] = cano_fl_mesh.verts_packed()
            self.center_dict[fl_name] = center
            self.rigid_t_dict[fl_name] =  rigid_t
            self.rigid_scale_dict[fl_name] = rigid_s


    def query_scale_init(self, rigid_v_dirs, v_dirs, init_scale):

        pdb.set_trace()
        dist = knn_points(rigid_v_dirs[None], v_dirs[None], K = 5)
        dists = dist.dists[0]
        idx = dist.idx[0]
        std = 2. * 0.1 ** 2
        xyz_neighbs_weight = torch.exp(-dists / std)
        xyz_neighbs_weight = xyz_neighbs_weight / xyz_neighbs_weight.sum(-1, keepdim=True)

        nearest_v_dirs = v_dirs[idx]
        rigid_v_dirs = rigid_v_dirs[:, None, :].expand_as(nearest_v_dirs)

        nearest_init_scale = init_scale[idx][..., 0]

        frac_cosine_value = 1./ (rigid_v_dirs * nearest_v_dirs).sum(-1)
        nearest_init_scale *= frac_cosine_value

        query_init_scale = torch.einsum('nc,nc->n', xyz_neighbs_weight, nearest_init_scale)


    def forward(self, rigid_cano_fl_verts, fl_names):
        '''
        rigid_cano_fl: rigid transformation would change the position in canonical body space
        forward function inverse rigid_cano_fl to canonical space fl
        fl_names: the name of feature line
        '''
        cano_fl_verts_list = []
        for rigid_cano_fl_verts, fl_name in zip(rigid_cano_fl_verts, fl_names):
            # parameters in canonical body space
            center = self.center_dict[fl_name]
            v_dirs = self.v_dirs_dict[fl_name]
            verts = self.verts_dict[fl_name]
            init_scale = self.init_scale_dict[fl_name]
            rigid_t = self.rigid_t_dict[fl_name]
            rigid_scale = self.rigid_scale_dict[fl_name]

            rigid_center = self.rigid_center_dict[fl_name]
            rigid_v_dirs = (rigid_cano_fl_verts - rigid_center) / ((rigid_cano_fl_verts- rigid_center).norm(dim=1, keepdim = True) + 1e-6)
            cano_fl_verts = ((rigid_cano_fl_verts - rigid_t) - center) /rigid_scale + center

            cano_fl_verts_list.append(cano_fl_verts)
            # query scale init value of rigid feature line
            # self.query_scale_init(rigid_v_dirs, v_dirs, init_scale)

        return cano_fl_verts_list














class MLPTranslator(nn.Module):
    def __init__(self,feature_vector_size,multires,weight_norm=False):
        super().__init__()

        dims = [3+feature_vector_size,512,512,512,512,3]
        self.feature_vector_size=feature_vector_size
        self.embed_fn = None
        self.multires=multires
        if multires > 0:
            # 3 * 6* 2 +3 = 39
            embed_fn, input_ch = get_embedder(multires)
            self.embed_fn = embed_fn
            dims[0] = input_ch + feature_vector_size

        self.num_layers = len(dims)

        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)
            if weight_norm:
            #     lin = nn.utils.weight_norm(lin)
                print('MLPTranslator:weight norm can influence weight initialization, can not produce small weights as initialization. Now do not use weight_norm')
            # initialize with zeros translation
            if l==self.num_layers-2:
                torch.nn.init.normal_(lin.weight, mean=0., std=0.001)
                torch.nn.init.constant_(lin.bias, 0.)
            setattr(self, "lin" + str(l), lin)

        self.relu = nn.ReLU()
        self.offset = {}
    def forward(self,ps,conds,batch_inds=None,**kwargs):
        '''
        Ps: input implicit verts: [Batch, V, 3]
        conds
        '''
        ratio=kwargs['ratio']['deformerRatio']
        offset_type = kwargs['offset_type']

        if self.embed_fn is not None:
            if ratio is None:    #one weight, all equal one
                ps = self.embed_fn(ps)
            elif ratio<=0: #zero weight
                ps = self.embed_fn(ps, [0. for _ in range(self.multires*2)])
            else:
                # NOTE that the annealing weights strategy, will control how many feature embeding used
                # for example if if ratio ==1, means all feature embeeding used , while  ratio =0.5, only half feature embedding employed
                ps = self.embed_fn(ps, utils.annealing_weights(self.multires,ratio))

        if batch_inds is not None:
            x=torch.cat([ps,conds[batch_inds]],dim=1)
        else:
            x=torch.cat([ps,conds.view(-1,1,self.feature_vector_size).expand(-1,ps.shape[1],self.feature_vector_size)],dim=-1).view(-1,ps.shape[-1]+self.feature_vector_size)

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))
            x = lin(x)

            if l < self.num_layers - 2:
                x = self.relu(x)

        if batch_inds is not None:
            self.offset[offset_type] = x
            return ps[...,:3]+x
        else:
            self.offset[offset_type]=x.view(ps.shape[0],ps.shape[1],3)
            return ps[...,:3]+x.view(ps.shape[0],ps.shape[1],3)


def getTranslatorNet(device,conf):
    if 'type' in conf:
        return globals()[conf.get_string('type')](conf.get_int('condlen'),multires=conf.get_int('multires')).to(device)
    else:
        return MLPTranslator(conf.get_int('condlen'),multires=conf.get_int('multires')).to(device)

#based on smpl skeleton lbs grid deformer
class LBSkinner(nn.Module):
    def __init__(self,ws,bmins,bmaxs,Js,parents,init_pose=None,align_corners=False, extra_trans = None, bbox_extend = None, bbox_center = None):
        '''
        b_min,
        b_max,
        ws,
        Js
        init_pose

        '''
        super().__init__()
        if type(bmins) is list:
            self.register_buffer('b_min',torch.tensor(bmins,dtype=torch.float).view(1,3))
        elif type(bmins) is np.ndarray:
            self.register_buffer('b_min',torch.from_numpy(bmins.astype(np.float32)).view(1,3))
        else:
            self.register_buffer('b_min',bmins.view(1,3))
        if type(bmaxs) is list:
            self.register_buffer('b_max',torch.tensor(bmaxs,dtype=torch.float).view(1,3))
        elif type(bmaxs) is np.ndarray:
            self.register_buffer('b_max',torch.from_numpy(bmaxs.astype(np.float32)).view(1,3))
        else:
            self.register_buffer('b_max',bmaxs.view(1,3))

        if type(ws) is np.ndarray:
            self.register_buffer('ws',torch.from_numpy(ws.astype(np.float32)))
        else:
            self.register_buffer('ws',ws.to(torch.float))


        if extra_trans == None:
            extra_trans = torch.full([1,3], 0.).float()
        self.register_buffer('extra_trans', extra_trans.to(torch.float))

        self.register_buffer('bbox_extend', bbox_extend.to(torch.float))
        self.register_buffer('bbox_center', bbox_center.to(torch.float))

        self.align_corners=align_corners; assert(align_corners==False)
        self.register_buffer('Js',Js.view(24,3))
        self.parents=parents



        if init_pose is None:
            # self.init_pose=None
            self.register_buffer('init_pose',None)
        else:
            if type(init_pose)==np.ndarray:
                init_pose=torch.from_numpy(init_pose.astype(np.float32))
            if init_pose.numel()==24*3:
                init_pose=batch_rodrigues(init_pose.view(-1,3)).view(24,3,3)
                # init_pose.shape
                # [B, 24, 3,3]
                self.init_pose_inverse(init_pose,self.Js)
            # read computed init_pose
            else:
                self.register_buffer('init_pose',init_pose.view(24,4,4))

    def bbox_size(self):
        '''obtain bbox_size, as we scale them

        '''
        margin=torch.tensor([0.15,0.15,0.20]).to(self.b_min)

        return self.b_min - margin, self.b_max + margin

    def init_pose_inverse(self,init_pose,Js):
        resultsR = [init_pose[0]]
        resultsT = [Js[0]]

        for i in range(1, self.parents.shape[0]):
            j_here = Js[i] - Js[self.parents[i]]
            R_here = init_pose[i]
            resultsR.append(resultsR[self.parents[i]].matmul(R_here))
            resultsT.append(resultsR[self.parents[i]].matmul(j_here.view(-1,1)).view(-1)+resultsT[self.parents[i]])

        # NOTE THAT origianl R, T matrix is map the original points to T, which is center by joints
        invs=[]
        for R,T in zip(resultsR,resultsT):
            inv=torch.zeros(4,4)
            inv[3,3]=1.
            inv[:3,:3]=R.transpose(0,1)
            inv[:3,3]=(-T.view(1,-1).matmul(R)).view(-1)
            invs.append(inv)
            # Note that the invs coordinate -R.T @ k

        # invs matrix is center by world coordinate
        # hence the invs[:3,3] is the joints coordinate in world-space
        self.register_buffer('init_pose',torch.stack(invs,dim=0))




    def posedSkeleton(self,conds):
        poses,trans=conds
        batch_size=poses.shape[0]
        assert(batch_size==trans.shape[0])
        poses=batch_rodrigues(poses.view(-1,3)).view(batch_size,24,3,3)
        root_rotation = poses[:, 0, :, :]
        Js = self.Js.view(1,24,3,1).expand(batch_size,24,3,1)
        def make_A(R, t):
            R_homo = F.pad(R, [0, 0, 0, 1, 0, 0])
            t_homo = torch.cat([t, torch.ones(R.shape[0], 1, 1).to(R.device)], dim = 1)
            return torch.cat([R_homo, t_homo], 2)
        A0 = make_A(root_rotation, Js[:, 0])
        results = [A0]
        parent=self.parents
        for i in range(1, parent.shape[0]):
            j_here = Js[:, i] - Js[:, parent[i]]
            A_here = make_A(poses[:, i], j_here)
            res_here = torch.matmul(results[parent[i]], A_here)
            results.append(res_here)
        results = torch.stack(results, dim = 1) #batch_size,24,4,4
        new_J = results[:, :, :3, 3]
        return new_J
    def query_skinning_weights_colors(self, tps):
        nps = self.inv_transform_v(tps, self.bbox_extend, self.bbox_center).view(-1,3)
        ps_ws=GridSamplerMine3dFunction.apply(self.ws, nps.reshape(1,1,1,-1,3)).view(-1,nps.shape[0]).transpose(0,1)
        ps_colors = weights2colors(ps_ws.detach().cpu())





        return ps_colors

    def inv_transform_v(self, v, scale_grid, transl):
        """
        v: [b, n, 3]
        """


        v = v - transl[None, None]
        v = v / scale_grid
        v = v * 2




        return v



    def forward(self,ps,conds,batch_inds=None,**kwargs):
        if type(ps)==list:
            tps,ps=ps
        else:
            tps=ps

        poses,trans=conds
        trans = trans + self.extra_trans


        batch_size=poses.shape[0]
        assert(batch_size==trans.shape[0])

        poses=batch_rodrigues(poses.view(-1,3)).view(batch_size,24,3,3)

        root_rotation = poses[:, 0, :, :]
        Js = self.Js.view(1,24,3,1).expand(batch_size,24,3,1)



        def make_A(R, t):
            R_homo = F.pad(R, [0, 0, 0, 1, 0, 0])
            t_homo = torch.cat([t, torch.ones(R.shape[0], 1, 1).to(R.device)], dim = 1)
            return torch.cat([R_homo, t_homo], 2)

        A0 = make_A(root_rotation, Js[:, 0])
        results = [A0]
        parent=self.parents

        for i in range(1, parent.shape[0]):
            j_here = Js[:, i] - Js[:, parent[i]]
            A_here = make_A(poses[:, i], j_here)
            res_here = torch.matmul(results[parent[i]], A_here)
            results.append(res_here)

        results = torch.stack(results, dim = 1) #batch_size,24,4,4

        new_J = results[:, :, :3, 3]
        # hence our implicit from A-pose ,hence init pose is matrix make A-pose -> canonical pose
        if self.init_pose is None:
            Js_w0 = torch.cat([Js, torch.zeros(batch_size, 24, 1, 1).to(poses.device)], dim = 2)
            init_bone = torch.matmul(results, Js_w0)
            init_bone = F.pad(init_bone, [3, 0, 0, 0, 0, 0, 0, 0])
            A = results - init_bone
        else:
            # A-pose -> canonical -> view-pose
            A = torch.matmul(results,self.init_pose.view(1,24,4,4).expand(batch_size,24,4,4))

        # normalized tps -> [-1, 1]
        # mod
        # nps=2.*(tps.reshape(-1,3)-self.b_min)/(self.b_max-self.b_min)-1.

        nps= self.inv_transform_v(tps, self.bbox_extend, self.bbox_center).view(-1,3)





        # use mine grid_sample to support double backward
        # print(self.ws.shape)

        # blender weight
        ps_ws=GridSamplerMine3dFunction.apply(self.ws, nps.reshape(1,1,1,-1,3)).view(-1,nps.shape[0]).transpose(0,1)


        if batch_inds is None:
            batch_size2,pnum,_=ps.shape
            assert(batch_size==batch_size2)
            ps_ws=ps_ws.view(batch_size,pnum,24)
            T = torch.matmul(ps_ws, A.view(batch_size, 24, 16)).view(batch_size, pnum, 4, 4)
            v_posed_homo = torch.cat([ps, torch.ones(batch_size, pnum, 1, device = ps.device)], dim = 2)
            v_homo = torch.matmul(T, v_posed_homo.unsqueeze(-1))
            v_homo=v_homo[:,:,:3,0]+trans.view(-1,1,3)

            return v_homo
        else:
            ps=ps.reshape(-1,3)
            v_homo=torch.zeros_like(ps)
            assert(batch_inds.numel()==ps.shape[0])
            for bid in range(batch_size):
                check=batch_inds==bid
                if check.any().item():
                    T=torch.matmul(ps_ws[check],A[bid].view(24,16)).view(-1,4,4)
                    tmp=torch.matmul(T,F.pad(ps[check],(0,1),mode='constant',value=1).unsqueeze(-1))[:,:3,0]
                    v_homo[check]=tmp
            v_homo=v_homo+trans[batch_inds]
            return v_homo
    def repose(self,ps,conds,batch_inds=None,**kwargs):
        if type(ps)==list:
            tps,ps=ps
        else:
            tps=ps

        poses,trans=conds


        batch_size=poses.shape[0]
        assert(batch_size==trans.shape[0])

        poses=batch_rodrigues(poses.view(-1,3)).view(batch_size,24,3,3)

        root_rotation = poses[:, 0, :, :]
        Js = self.Js.view(1,24,3,1).expand(batch_size,24,3,1)



        def make_A(R, t):
            R_homo = F.pad(R, [0, 0, 0, 1, 0, 0])
            t_homo = torch.cat([t, torch.ones(R.shape[0], 1, 1).to(R.device)], dim = 1)
            return torch.cat([R_homo, t_homo], 2)

        A0 = make_A(root_rotation, Js[:, 0])
        results = [A0]
        parent=self.parents

        for i in range(1, parent.shape[0]):
            j_here = Js[:, i] - Js[:, parent[i]]
            A_here = make_A(poses[:, i], j_here)
            res_here = torch.matmul(results[parent[i]], A_here)
            results.append(res_here)

        results = torch.stack(results, dim = 1) #batch_size,24,4,4

        new_J = results[:, :, :3, 3]
        # hence our implicit from A-pose ,hence init pose is matrix make A-pose -> canonical pose
        if self.init_pose is None:
            Js_w0 = torch.cat([Js, torch.zeros(batch_size, 24, 1, 1).to(poses.device)], dim = 2)
            init_bone = torch.matmul(results, Js_w0)
            init_bone = F.pad(init_bone, [3, 0, 0, 0, 0, 0, 0, 0])
            A = results - init_bone
        else:
            # A-pose -> canonical -> view-pose
            A = torch.matmul(results,self.init_pose.view(1,24,4,4).expand(batch_size,24,4,4))

        # normalized tps -> [-1, 1]
        # mod
        # nps=2.*(tps.reshape(-1,3)-self.b_min)/(self.b_max-self.b_min)-1.

        nps= self.inv_transform_v(tps, self.bbox_extend, self.bbox_center).view(-1,3)





        # use mine grid_sample to support double backward
        # print(self.ws.shape)

        # blender weight
        ps_ws=GridSamplerMine3dFunction.apply(self.ws, nps.reshape(1,1,1,-1,3)).view(-1,nps.shape[0]).transpose(0,1)


        if batch_inds is None:
            batch_size2,pnum,_=ps.shape
            assert(batch_size==batch_size2)
            ps_ws=ps_ws.view(batch_size,pnum,24)
            T = torch.matmul(ps_ws, A.view(batch_size, 24, 16)).view(batch_size, pnum, 4, 4)
            v_posed_homo = torch.cat([ps, torch.ones(batch_size, pnum, 1, device = ps.device)], dim = 2)
            v_homo = torch.matmul(T, v_posed_homo.unsqueeze(-1))
            v_homo=v_homo[:,:,:3,0]+trans.view(-1,1,3)

            return v_homo
        else:
            ps=ps.reshape(-1,3)
            v_homo=torch.zeros_like(ps)
            assert(batch_inds.numel()==ps.shape[0])
            for bid in range(batch_size):
                check=batch_inds==bid
                if check.any().item():
                    T=torch.matmul(ps_ws[check],A[bid].view(24,16)).view(-1,4,4)
                    tmp=torch.matmul(T,F.pad(ps[check],(0,1),mode='constant',value=1).unsqueeze(-1))[:,:3,0]
                    v_homo[check]=tmp
            v_homo=v_homo+trans[batch_inds]
            return v_homo

def smooth_weights(weights,times=3):

    for _ in range(times):
        mean=(weights[:,:,2:,1:-1,1:-1]+weights[:,:,:-2,1:-1,1:-1]+\
            weights[:,:,1:-1,2:,1:-1]+weights[:,:,1:-1,:-2,1:-1]+\
            weights[:,:,1:-1,1:-1,2:]+weights[:,:,1:-1,1:-1,:-2])/6.0

        weights[:,:,1:-1,1:-1,1:-1]=(weights[:,:,1:-1,1:-1,1:-1]-mean)*0.7+mean
        sums=weights.sum(1,keepdim=True)
        weights=weights/sums
    # weights[weights<5.e-3]=0.0
    return weights

def compute_lbswField(bmins,bmaxs,resolutions,smpl_verts,smpl_ws,align_corners=False,mean_neighbor=5,smooth_times=30):
    #
    # ws=compute_lbswField(bmins,bmaxs,resolution,verts.view(6890,3),smpl.weight.view(6890,24),align_corners=False,mean_neighbor=30,smooth_times=30)
    device=smpl_verts.device
    bmins=torch.tensor(bmins).float().to(device).view(1,-1)
    bmaxs=torch.tensor(bmaxs).float().to(device).view(1,-1)
    W,H,D=resolutions

    resolutions = torch.tensor(resolutions).float().to(device).view(-1)
    arrangeX = torch.linspace(0, W-1, W).long().to(device)
    arrangeY = torch.linspace(0, H-1, H).long().to(device)
    arrangeZ = torch.linspace(0, D-1, D).long().to(device)

    gridD, gridH, gridW = torch.meshgrid([arrangeZ, arrangeY, arrangeX])


    coords = torch.stack([gridW, gridH, gridD]) # [3, steps[0], steps[1], steps[2]]
    coords = coords.view(3, -1).t() # [N, 3]


    if align_corners:
        coords2D = coords.float() / (resolutions[None,:] - 1)
    else:
        step = 1.0 / resolutions[None,:].float()
        coords2D = coords.float() / resolutions[None,:] + step / 2
    coords2D = coords2D * (bmaxs - bmins) + bmins
    fws=[]

    for ind,tmp in enumerate(torch.split(coords2D,50000)):
        # if ind/10==0:
        #     print(ind)
        # (50000, 30)
        dists,indices=(tmp[:,None,:]-smpl_verts[None,:,:]).norm(dim=-1).topk(mean_neighbor,dim=-1,largest=False)
        dists=torch.clamp(dists,0.0001,1.)
        ws=1./dists
        ws=ws/ws.sum(-1,keepdim=True)
        ws=(smpl_ws[indices.view(-1)]*ws.view(-1,1)).reshape(ws.shape[0],mean_neighbor,-1).sum(1)
        fws.append(ws)

    fws=torch.cat(fws,dim=0)
    #[1, blend_weight_joints, k]
    fws=fws.transpose(0,1).reshape(1,-1,D,H,W)
    fws=smooth_weights(fws,smooth_times)

    # [1, 24, D, H, W]
    return fws


def initialLBSkinner(gender,shape,pose,resolution,bmins=None,bmaxs=None, extra_trans = None):
    smpl=getSMPL(gender).to(shape.device)
    Js,_=smpl.skeleton(shape.view(1,-1),True)
    verts,_,_=smpl(shape.view(1,-1),pose.view(1,24,3),True)
    # using for debug
    # faces = torch.tensor(smpl.faces).long()
    # save_obj('a_pose.obj', verts[0],faces)

    # build the data
    if bmins is None or bmaxs is None: #adaptive generate box
        bbox_data_min = verts[0].min(0).values
        bbox_data_max = verts[0].max(0).values
        bbox_data_extend = (bbox_data_max - bbox_data_min).max()
        bbox_grid_extend = bbox_data_extend * 1.1
        grid_center = (bbox_data_min + bbox_data_max) / 2


    # Resolution defined before training (129 225 65)
    ws=compute_lbswField(bbox_data_min.tolist(),bbox_data_max,resolution,verts.view(6890,3),smpl.weight.view(6890,24),align_corners=False,mean_neighbor=30,smooth_times=30)

    # LBSkinner
    # it represents
    #    b_min,
    #    b_max,
    #    ws,
    #    Js
    #    init_pose

    return LBSkinner(ws,bbox_data_min,bbox_data_max, Js, smpl.parents, init_pose=pose,align_corners=False, extra_trans  = extra_trans,
            bbox_extend = bbox_grid_extend, bbox_center = grid_center),verts.view(6890,3),torch.tensor(smpl.faces,dtype=torch.long,device=verts.device)


