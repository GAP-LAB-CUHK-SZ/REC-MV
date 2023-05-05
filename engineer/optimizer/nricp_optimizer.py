"""
@File: nricp_optimizer.
@Author: Lingteng Qiu
@Email: qiulingteng@link.cuhk.edu.cn
@Date: 2022-10-13
@Desc: NRICP optimizer with normal check, modified from https://github.com/wuhaozhe/pytorch-nicp
"""
import torch.nn as nn
import torch
from .base_optimzier import _Base_Optimizer
from utils.common_utils import make_nograd_func
import time
from pytorch3d.structures import Meshes
from pytorch3d.io import save_obj
from engineer.utils.mesh_utils import mesh_boundary
from pytorch3d.ops import (
    corresponding_points_alignment,
    knn_points,
    knn_gather
)
import torch.nn.functional as F
from FastMinv import Fast3x3Minv
from pytorch3d.loss import mesh_laplacian_smoothing
from pytorch3d.io import save_obj
import numpy as np
from einops import rearrange
from scipy import sparse
import numpy as np
from einops import repeat
import pdb
from pytorch3d.loss import chamfer_distance
import os
# from sksparse.cholmod import cholesky
# from sksparse.cholmod import cholesky_AAt
class Local_Affine(nn.Module):
    def __init__(self, num_points, batch_size = 1, edges = None, gamma= 1):
        '''
            specify the number of points, the number of points should be constant across the batch
            and the edges torch.Longtensor() with shape N * 2
            the local affine operator supports batch operation
            batch size must be constant
            add additional pooling on top of w matrix
        '''
        super(Local_Affine, self).__init__()
        #[b, n ,3, 3]
        self.A = nn.Parameter(torch.eye(3).unsqueeze(0).unsqueeze(0).repeat(batch_size, num_points, 1, 1))
        #[b, n ,3, 1]
        self.b = nn.Parameter(torch.zeros(3).unsqueeze(0).unsqueeze(0).unsqueeze(3).repeat(batch_size, num_points, 1, 1))

        G = torch.eye(4).unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1, 1)
        G[...,3,3]= gamma
        self.register_buffer('G', G)


        self.edges = edges
        self.num_points = num_points

    def stiffness(self):
        '''
            calculate the stiffness of local affine transformation
            f norm get infinity gradient when w is zero matrix,
        '''
        if self.edges is None:
            raise Exception("edges cannot be none when calculate stiff")
        idx1 = self.edges[:, 0]
        idx2 = self.edges[:, 1]


        affine_weight = torch.cat((self.A, self.b), dim = 3)

        w1 = torch.index_select(affine_weight, dim = 1, index = idx1)
        w2 = torch.index_select(affine_weight, dim = 1, index = idx2)

        w_diff = (w1 - w2)
        w_diff = torch.einsum('bnhw,bnwj->bnhj',w_diff, self.G.repeat(1, w_diff.shape[1], 1, 1))


        return w_diff **2

    def forward(self, x, pool_num = 0, return_stiff = False):
        '''
            x should have shape of B * N * 3
        '''
        x = x.unsqueeze(3)
        out_x = torch.matmul(self.A, x)
        out_x = out_x + self.b


        out_x.squeeze_(3)
        if return_stiff:
            stiffness = self.stiffness()
            return out_x, stiffness
        else:
            return out_x

    def forward_normal(self, x):
        '''
            x should have shape of B * N * 3
        '''
        x = x.unsqueeze(3)

        b, n,_,__ = self.A.shape
        A = self.A.view(-1,3,3)

        A_inv,A_inv_mask = Fast3x3Minv(A)
        A_inv_transpose = A_inv.transpose(-1,-2)

        normal = A_inv_transpose.view(b,n,3,3) @ x

        return normal.squeeze(-1), A_inv_mask[None]



class Local_Affine_Matrix(nn.Module):
    def __init__(self, source_v, num_points, batch_size = 1, edges = None, gamma= 1):
        '''
            specify the number of points, the number of points should be constant across the batch
            and the edges torch.Longtensor() with shape N * 2
            the local affine operator supports batch operation
            batch size must be constant
            add additional pooling on top of w matrix
        '''
        super(Local_Affine_Matrix, self).__init__()

        # build D matrix
        v = F.pad(source_v, (0, 1, 0, 0),mode='constant', value =1 )
        idx = torch.arange(num_points)[..., None]
        jdx = idx*4
        jdx = torch.cat([jdx, jdx+1, jdx+2, jdx+3], dim = 1)
        idx = repeat(idx,'n 1 -> n 4').contiguous()

        D = sparse.csr_matrix((v.view(-1).contiguous().numpy(), (idx.view(-1).contiguous().numpy(), jdx.view(-1).contiguous().numpy())), shape=(num_points, num_points*4))


        A = torch.eye(3).unsqueeze(0).unsqueeze(0).repeat(batch_size, num_points, 1, 1)
        #[b, n ,3, 1]
        b = torch.zeros(3).unsqueeze(0).unsqueeze(0).unsqueeze(3).repeat(batch_size, num_points, 1, 1)
        G = torch.eye(4)
        G[...,3,3]= gamma

        self.register_buffer('A', A)
        self.register_buffer('b', b)


        self.register_buffer('G', G)
        self.edges = edges
        self.num_points = num_points

        # build edge_verts matrix
        edge_idx = torch.arange(self.edges.shape[0])[..., None]
        inde = -torch.ones((edge_idx.shape[0],1))
        outde = torch.ones((edge_idx.shape[0],1))


        edge_idx = repeat(edge_idx, 'n 1 -> n 2').contiguous()
        value = torch.cat([inde,outde], dim =-1).contiguous()
        M = sparse.csr_matrix((value.view(-1).contiguous().numpy(), (edge_idx.view(-1).contiguous().numpy(), self.edges.view(-1).contiguous().numpy())), shape=(self.edges.shape[0], num_points))
        kron_M = sparse.kron(M,self.G)

        self.D = D
        self.kron_M = kron_M



    def affine_matrix(self):
        return torch.cat([self.A,self.b], dim=-1)


    def stiffness(self):
        '''
            calculate the stiffness of local affine transformation
            f norm get infinity gradient when w is zero matrix,
        '''
        if self.edges is None:
            raise Exception("edges cannot be none when calculate stiff")
        idx1 = self.edges[:, 0]
        idx2 = self.edges[:, 1]


        affine_weight = torch.cat((self.A, self.b), dim = 3)

        w1 = torch.index_select(affine_weight, dim = 1, index = idx1)
        w2 = torch.index_select(affine_weight, dim = 1, index = idx2)

        w_diff = (w1 - w2)
        w_diff = torch.einsum('bnhw,bnwj->bnhj',w_diff, self.G.repeat(1, w_diff.shape[1], 1, 1))

        return w_diff **2

    def forward(self, x):
        '''
            x should have shape of B * N * 3
        '''
        x = x.unsqueeze(3)
        out_x = torch.matmul(self.A, x)
        out_x = out_x + self.b


        out_x.squeeze_(3)

        return out_x

    def forward_normal(self, x):
        '''
            x should have shape of B * N * 3
        '''
        x = x.unsqueeze(3)

        b, n,_,__ = self.A.shape
        A = self.A.view(-1,3,3)

        A_inv,A_inv_mask = Fast3x3Minv(A)
        A_inv_transpose = A_inv.transpose(-1,-2)

        normal = A_inv_transpose.view(b,n,3,3) @ x

        return normal.squeeze(-1), A_inv_mask[None]

    def normal_solve(self, A, b):

        s = time.time()
        factor = cholesky_AAt(A.T)
        x = factor(A.T * b)

        # about 10s solve this normal equaitons
        print('solver Time: {:.4f}s'.format(time.time()-s))

        x = torch.from_numpy(x).float()
        batch = x.shape[0] //4

        x = rearrange(x, '(n w) h -> n w h', n=batch)
        x = x.transpose(2,1)


        R  = x[...,:,:3]
        t  = x[...,:,3:]

        self.register_buffer('A', R[None])
        self.register_buffer('b', t[None])


class NRICP_Optimizer_AdamW(_Base_Optimizer):
    '''
    This is NRICP Optimizer using AdamW optimizer to registry templte mesh to target mesh
    more details see:
    Optimal Step Nonrigid ICP Algorithms for Surface Registration, cvpr2007
    '''
    def __init__(self, epoch, dense_pcl, use_normal, stiffness_weight = [], mile_stone = [],inner_iter = 10, laplacian_weight = 0., gamma = 1, threshold = 0.5,optimizer_setting = None, device = 'cuda:0'):
        '''
        epoch: hook training epoch
        optimizer_setting: is optimizer setting, NRICP u could use adam, adamw, ....
        '''
        super(NRICP_Optimizer_AdamW, self).__init__(optimizer_setting)

        self.name = 'NRICP_Optimizer_GPU'
        self.dense_pcl = int(dense_pcl)
        self.use_normal = use_normal
        self.mile_times = 0
        self.mile_idx = 0
        self.stiffness_weight = stiffness_weight
        self.laplacian_weight = laplacian_weight
        self.inner_iter = inner_iter
        self.mile_stone = mile_stone
        # hook stop time
        self.epoch = epoch
        self.device = device
        self.gamma = gamma

        # learnable parameters
        self.local_affine = None
        self.threshold = threshold

        assert len(self.mile_stone) == len(self.stiffness_weight) - 1

    def __collect_data(self, inputs):
        '''the data fomat we only process to Garemnet Mesh and Garment Polygons class

        return
            source_idx: vertices need to deform to target position
            cano_meshes: canonical target meshes
        '''
        smpl_slice = inputs['smpl_slice']
        cano_meshes = inputs['cano_meshes']
        save_path = inputs['save_path']
        garment_name = inputs['garment_name']
        static_pts_type = inputs['static_pts_type']
        nricp_masks = inputs['nricp_masks']


        while smpl_slice.vertices.squeeze(0).shape[0] < self.dense_pcl:
            smpl_slice = smpl_slice.dense_pcl(type='edge')
        inputs['smpl_slice'] = smpl_slice

        static_pts_ids = smpl_slice.get_outlayer_idx(*static_pts_type)


        if len(static_pts_ids) == 0:
            static_pts_ids = None
        else:
            static_pts_ids = torch.cat(static_pts_ids)
        # smpl_slice.save_obj('./dense.obj')
        # save_obj('./cano_mesh.obj', cano_meshes.verts_packed(), cano_meshes.faces_packed())
        # for filed in smpl_slice.get_fields():
        #     if filed == 'back_ground':
        #         continue
        #     smpl_slice.save_boudnary('./{}.ply'.format(filed),filed)

        return smpl_slice, cano_meshes, static_pts_ids, save_path, garment_name, nricp_masks

    def __encode_results(self, inputs, new_verts):
        '''the data return depend on your format
        '''
        smpl_slice = inputs['smpl_slice']
        smpl_slice.update_padded(new_verts.to(smpl_slice.vertices))



        return smpl_slice


    def fitting(self, inputs):
        ''' NRICP fitting functions
        Given source point clounds and target point clounds
        this function will lead to optimizer local affine matrix to obtain ICP deform's results
        NOTE that we modified NRICP by adding normal check and nricp masks to clean noise points
        '''
        self.mile_times = 0
        self.mile_idx = 0

        smpl_slice, cano_meshes, static_pts_ids, save_path, garment_name, nricp_masks = self.__collect_data(inputs)
        if save_path is not None:
            save_path = os.path.join(save_path,'nricp_deform',str(garment_name))
            os.makedirs(save_path, exist_ok= True)

        source_v = smpl_slice.vertices[None]
        target_v = cano_meshes.verts_packed()[None]
        source_f = smpl_slice.faces[None]

        if self.use_normal:
            source_normals = smpl_slice.verts_normals_packed().to(self.device)[None]
            target_normals = cano_meshes.verts_normals_packed().to(self.device)[None]
        # ignore boudnary pts
        # NOTE that the boundary exists template mesh. Also, we align featureline before nricp, so we ignore the template boundary


        boundary_mask = mesh_boundary(smpl_slice.faces, smpl_slice.vertices.shape[0])
        inner_mask = torch.logical_not(boundary_mask).to(self.device)[None]

        source_edges = smpl_slice.edges_packed().to(self.device)
        self.local_affine = Local_Affine(source_v.shape[1], 1, source_edges, gamma= self.gamma).to(self.device)

        source_v = source_v.to(self.device)
        target_v = target_v.to(self.device)
        source_f = source_f.to(self.device)


        target_v = target_v[nricp_masks[None]>0].unsqueeze(0)
        target_normals = target_normals[nricp_masks[None]>0].unsqueeze(0)

        if static_pts_ids is not None:
            static_joints = source_v[0, static_pts_ids].detach()[None]


        for i in range(self.epoch):
            # forward affine_network
            new_source_v,stiffness = self.local_affine(source_v, pool_num=0, return_stiff = True)
            old_source_v = new_source_v.detach().clone()
            # warp normal map
            with torch.no_grad():
                new_source_normals, inv_mask = self.local_affine.forward_normal(source_normals)
                inner_inv_mask = torch.logical_and(inv_mask, inner_mask)

            inner_optimizer = torch.optim.AdamW([{'params': self.local_affine.parameters()}], lr=1e-4, amsgrad=True)
            knn =knn_points(new_source_v, target_v)
            close_points = knn_gather(target_v, knn.idx)[:,:,0]
            close_normals = knn_gather(target_normals, knn.idx)[:,:,0]

            stiffness_weight = self.stiffness_weight[self.mile_idx]
            laplacian_weight = self.laplacian_weight[self.mile_idx]

            iterations = range(100) if i == 0 else range(self.inner_iter)
            for inner_i in iterations:
                inner_optimizer.zero_grad()
                with torch.no_grad():
                    normal_cos_sim = F.cosine_similarity(close_normals, new_source_normals, dim = 2)
                    # threshold -> pi/3

                    weight_mask = torch.logical_and(inner_inv_mask, normal_cos_sim > self.threshold)
                #     xxx
                vert_distance = (new_source_v - close_points) ** 2

                bsize = vert_distance.shape[0]
                if static_pts_ids is not None:
                    new_boundary_pts = new_source_v[0, static_pts_ids][None]
                    static_loss = (new_boundary_pts - static_joints) ** 2
                    static_sum = torch.sum(static_loss)/ bsize
                else:
                    static_sum = torch.tensor(0.)






                # vert_distance_mask = torch.sum(vert_distance, dim =2) < 0.04 **2
                # weight_mask = torch.logical_and(inner_mask, vert_distance_mask)
                vert_distance = weight_mask[...,None] * vert_distance
                vert_distance = vert_distance.view(bsize, -1)
                vert_sum = torch.sum(vert_distance) / bsize


                stiffness = stiffness.view(bsize, -1)
                stiffness_sum = torch.sum(stiffness) * stiffness_weight / bsize
                laplacian_loss = mesh_laplacian_smoothing(Meshes(new_source_v, source_f)) * laplacian_weight

                loss = torch.sqrt(vert_sum + stiffness_sum + static_sum) + laplacian_loss
                loss.backward()
                inner_optimizer.step()
                new_source_v,stiffness = self.local_affine(source_v, pool_num=0, return_stiff = True)
                # warp normal map
                with torch.no_grad():
                    new_source_normals, inv_mask = self.local_affine.forward_normal(source_normals)
                    inner_inv_mask = torch.logical_and(inv_mask, inner_mask)
            distance = torch.mean(torch.sqrt(torch.sum((old_source_v - new_source_v) ** 2, dim = 2)))
            print("current {:03d} NRICP avg_update:{:.5f} valid{:d}/{:d}: dis:{:.4f}, stiffness:{:.4f}, laplacian:{:.4f}, total:{:.4f}"
                    " laplacian_weight:{:.4f}, stiffness_weight:{:.4f}, static_sum:{:.4f}".format(self.mile_times,distance.item(), weight_mask.sum(),
                        weight_mask.numel(), vert_sum.item(), stiffness_sum.item(),
                        laplacian_loss.item(), loss.item(), laplacian_weight,
                        stiffness_weight, static_sum.item()))
            self.mile_times+=1

            if self.mile_times in self.mile_stone:
                self.mile_idx+=1
                with torch.no_grad():
                    new_source_v = self.local_affine(source_v, pool_num=0, return_stiff = False)
                    # if save_path is not None:
                    #     save_obj(os.path.join(save_path, '{}.obj'.format(self.mile_times)), new_source_v.detach().cpu()[0], smpl_slice.faces.detach().cpu(),)

        with torch.no_grad():
            new_source_v = self.local_affine(source_v, pool_num=0, return_stiff = False)
            if save_path is not None:
                save_obj(os.path.join(save_path, '{}.obj'.format(self.mile_times)), new_source_v.detach().cpu()[0], smpl_slice.faces.detach().cpu(),)


        smpl_slice =  self.__encode_results(inputs, new_source_v)


        return loss, smpl_slice





class NRICP_Optimizer_Matrix(_Base_Optimizer):
    '''
    This is NRICP Optimizer using pesudo inverse solver to registry templte mesh to target mesh
    more details see:
    Optimal Step Nonrigid ICP Algorithms for Surface Registration, cvpr2007
    '''
    def __init__(self, epoch, dense_pcl, use_normal, stiffness_weight = [], mile_stone = [],  gamma = 1, optimizer_setting = None):
        '''
        epoch: hook training epoch
        optimizer_setting: is optimizer setting, NRICP u could use adam, adamw, ....
        '''
        super(NRICP_Optimizer_Matrix, self).__init__(optimizer_setting)

        self.name = 'NRICP_Optimizer_GPU'
        self.dense_pcl = int(dense_pcl)
        self.use_normal = use_normal
        self.mile_times = 0
        self.mile_idx = 0
        self.stiffness_weight = stiffness_weight
        self.mile_stone = mile_stone
        # hook stop time
        self.epoch = epoch
        self.gamma = gamma



        # learnable parameters
        self.local_affine = None

        assert len(self.mile_stone) == len(self.stiffness_weight) - 1

    def __collect_data(self, inputs):
        '''the data fomat we only process to Garemnet Mesh and Garment Polygons class

        return
            source_idx: vertices need to deform to target position
            cano_meshes: canonical target meshes
        '''
        smpl_slice = inputs['smpl_slice']
        cano_meshes = inputs['cano_meshes']
        while smpl_slice.vertices.squeeze(0).shape[0] < self.dense_pcl:
            smpl_slice = smpl_slice.dense_pcl(type='edge')
        inputs['smpl_slice'] = smpl_slice

        # smpl_slice.save_obj('./dense.obj')
        # save_obj('./cano_mesh.obj', cano_meshes.verts_packed(), cano_meshes.faces_packed())
        # for filed in smpl_slice.get_fields():
        #     if filed == 'back_ground':
        #         continue
        #     smpl_slice.save_boudnary('./{}.ply'.format(filed),filed)

        return smpl_slice, cano_meshes

    def __encode_results(self, inputs, new_verts):
        '''the data return depend on your format
        '''
        smpl_slice = inputs['smpl_slice']
        smpl_slice.update_padded(new_verts.to(smpl_slice.vertices))

    def registry_local_affine_matrix(self, source_v, num_points, batch_size = 1, edges = None, gamma= 1):
        return Local_Affine_Matrix(source_v, num_points,batch_size, edges, gamma)

    def find_neighbour(self,source_v, target_v, target_normals):

        knn =knn_points(source_v.cuda(), target_v.cuda())

        close_points = knn_gather(target_v, knn.idx)[:,:,0]
        close_normals = knn_gather(target_normals, knn.idx)[:,:,0]

        return close_points, close_normals

    def solver(self, source_v, target_v, source_normals, target_normals ,mask):
        old_x = 10*self.local_affine.affine_matrix()
        max_cnt = 20
        cnt = 0


        while(torch.norm(old_x - self.local_affine.affine_matrix()))>1e-2:

            self.local_affine.cuda()
            source_v = source_v.cuda()
            target_v = target_v.cuda()
            mask = mask.cuda()
            source_normals = source_normals.cuda()
            target_normals = target_normals.cuda()

            new_source_v = self.local_affine(source_v)
            # NOTE affine normals using the inverse transpose of normals
            new_source_normals, inv_mask = self.local_affine.forward_normal(source_normals)
            inv_W = torch.logical_and(inv_mask, mask)
            # close_points, close_normals = self.find_neighbour(new_source_v, target_v, target_normals)
            close_points, close_normals = self.find_neighbour(new_source_v, target_v, target_normals)

            normal_cos_sim = F.cosine_similarity(close_normals, new_source_normals, dim = 2)
            # threshold -> pi/3
            W = torch.logical_and(inv_W, normal_cos_sim > 0.5)
            weight_mask = W.clone().cpu()

            # solve normal equations
            W = W.cpu().numpy().astype(np.float32)[0]
            w_i = np.arange(W.shape[0])
            w_j = np.arange(W.shape[0])
            sparse_W = sparse.csr_matrix((W, (w_i, w_j)), shape=(W.shape[0],W.shape[0]))
            WD = sparse_W @ self.local_affine.D
            stiffness_weight = self.stiffness_weight[self.mile_idx]
            kron_M = stiffness_weight * self.local_affine.kron_M
            kron_b = np.zeros((kron_M.shape[0],3))

            U = close_points.cpu().numpy()[0]
            U = sparse_W @ U

            b = np.concatenate([kron_b, U],axis= 0)
            A = sparse.vstack([kron_M,WD])

            # solve Ax = b
            # x = A'A \ A' b
            self.local_affine.cpu()
            old_x = self.local_affine.affine_matrix().clone()

            self.local_affine.normal_solve(A,b)

            source_v = source_v.cpu()
            close_points = close_points.cpu()

            new_source_v = self.local_affine(source_v)
            stiffness = self.local_affine.stiffness()

            vert_distance = (new_source_v - close_points) ** 2
            vert_distance = weight_mask[...,None] * vert_distance

            bsize = vert_distance.shape[0]
            vert_distance = vert_distance.view(bsize, -1)
            vert_sum = torch.sum(vert_distance) / bsize

            stiffness = stiffness.view(bsize, -1)
            stiffness_sum = torch.sum(stiffness) * stiffness_weight / bsize

            cnt+=1
            if cnt>max_cnt:
                break
            print("energy function: valid:{}/{}\tdistance:{:04f}\tstiffness:{:04f}\taffine_w:{:04f}\tstiffness_weight:{:.4f}".format(weight_mask.sum(),weight_mask.numel(), vert_sum.item(), stiffness_sum.item(),torch.norm(old_x - self.local_affine.affine_matrix()),  stiffness_weight))

        loss = torch.sqrt(vert_sum + stiffness_sum)
        return loss




    @make_nograd_func
    def fitting(self, inputs):
        ''' NRICP fitting functions
        Given source point clounds and target point clounds
        this function will lead to optimizer local affine matrix to obtain local-ICP deform's results
        '''

        begin = False
        smpl_slice, cano_meshes = self.__collect_data(inputs)
        source_v = smpl_slice.vertices[None]
        target_v = cano_meshes.verts_packed()[None]
        source_f = smpl_slice.faces[None]

        if self.use_normal:
            source_normals = smpl_slice.verts_normals_packed()
            target_normals = cano_meshes.verts_normals_packed()

        # ignore boudnary pts
        # NOTE that the boundary exists template mesh. Also, we align featureline before nricp, so we ignore the template boundary
        boundary_mask = mesh_boundary(smpl_slice.faces, smpl_slice.vertices.shape[0])
        mask = torch.logical_not(boundary_mask)[None]

        if self.local_affine == None:
            source_edges = smpl_slice.edges_packed()
            self.local_affine = self.registry_local_affine_matrix(smpl_slice.vertices,source_v.shape[1], 1, source_edges, gamma= self.gamma)

        loss = self.solver(source_v, target_v, source_normals[None], target_normals[None], mask)

        self.mile_times+=1
        if self.mile_times in self.mile_stone:
            self.mile_idx+=1
            new_source_v = self.local_affine(source_v)
            save_obj('./results/smpl_clothes/fitting_scans/nricp_optimizer/{}.obj'.format(self.mile_times), new_source_v.detach().cpu()[0], smpl_slice.faces.detach().cpu(),)
        if self.mile_times == self.epoch:
            # with torch.no_grad():
            #     new_source_v = self.local_affine(source_v, pool_num=0, return_stiff = False)
            #     save_obj('./results/smpl_clothes/fitting_scans/nricp_optimizer/final.obj'.format(self.mile_times), new_source_v.detach().cpu()[0], smpl_slice.faces.detach().cpu(),)
            # TODO update final mesh
            raise NotImplemented
        return loss

