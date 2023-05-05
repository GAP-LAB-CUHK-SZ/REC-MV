
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
import trimesh
from trimesh.ray import ray_triangle
from trimesh.ray import ray_pyembree
import time
from collections import defaultdict

class Surface_Intesection(_Base_Optimizer):
    '''
    '''
    def __init__(self, ray_dirs = [0.,0.,-1], optimizer_setting = None, use_normal = True):
        '''
        using surface intesection pts to deform template
        '''
        super(Surface_Intesection, self).__init__(optimizer_setting)

        self.name = 'Surface_Intesection'
        self.use_normal = use_normal


        self.ray_dirs = ray_dirs

    def __collect_data(self, inputs):
        '''the data fomat we only process to Garemnet Mesh and Garment Polygons class

        return
            source_idx: vertices need to deform to target position
            cano_meshes: canonical target meshes
        '''
        smpl_slice = inputs['smpl_slice']
        cano_meshes = inputs['cano_meshes']

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




    def fitting(self, inputs):
        ''' NRICP fitting functions
        Given source point clounds and target point clounds
        this function will lead to optimizer local affine matrix to obtain ICP deform's results
        '''

        smpl_slice, cano_meshes = self.__collect_data(inputs)
        source_v = smpl_slice.vertices[None]
        target_v = cano_meshes.verts_packed()[None]
        source_f = smpl_slice.faces[None]


        target_mesh = trimesh.Trimesh(cano_meshes.verts_packed().detach().cpu().numpy(), cano_meshes.faces_packed().detach().cpu().numpy())
        ray_tracing = ray_pyembree.RayMeshIntersector(target_mesh)


        if self.use_normal:
            source_normals = smpl_slice.verts_normals_packed()
            target_normals = cano_meshes.verts_normals_packed()


        source_ori = source_v.detach().cpu().numpy()[0]
        source_ray_dirs = source_normals.detach().cpu().numpy()


        ray_index_sets = defaultdict(list)

        start = time.time()
        locations_a, index_ray_a, __ = ray_tracing.intersects_location(source_ori, source_ray_dirs)
        locations_b, index_ray_b, __ = ray_tracing.intersects_location(source_ori, -source_ray_dirs)


        locations = np.concatenate([locations_a, locations_b], axis = 0)
        index_ray = np.concatenate([index_ray_a, index_ray_b], axis = 0)

        print(time.time()-start)




        pdb.set_trace()
        # ignore boudnary pts
        # NOTE that the boundary exists template mesh. Also, we align featureline before nricp, so we ignore the template boundary
        boundary_mask = mesh_boundary(smpl_slice.faces, smpl_slice.vertices.shape[0])
        inner_mask = torch.logical_not(boundary_mask).to(self.device)[None]

        if self.local_affine == None:
            source_edges = smpl_slice.edges_packed().to(self.device)
            self.local_affine = Local_Affine(source_v.shape[1], 1, source_edges, gamma= self.gamma).to(self.device)
            begin = True
        source_v = source_v.to(self.device)
        target_v = target_v.to(self.device)
        source_f = source_f.to(self.device)



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
            laplacian_weight = self.laplacian_weight

            iterations = range(100) if begin else range(self.inner_iter)
            for inner_i in iterations:
                inner_optimizer.zero_grad()
                with torch.no_grad():
                    normal_cos_sim = torch.abs(F.cosine_similarity(close_normals, new_source_normals, dim = 2))
                    # threshold -> pi/3
                    weight_mask = torch.logical_and(inner_inv_mask, normal_cos_sim > 0.5)
                #     xxx
                vert_distance = (new_source_v - close_points) ** 2
                # vert_distance_mask = torch.sum(vert_distance, dim =2) < 0.04 **2
                # weight_mask = torch.logical_and(inner_mask, vert_distance_mask)
                vert_distance = weight_mask[...,None] * vert_distance
                bsize = vert_distance.shape[0]
                vert_distance = vert_distance.view(bsize, -1)
                vert_sum = torch.sum(vert_distance) / bsize
                stiffness = stiffness.view(bsize, -1)
                stiffness_sum = torch.sum(stiffness) * stiffness_weight / bsize
                laplacian_loss = mesh_laplacian_smoothing(Meshes(new_source_v, source_f)) * laplacian_weight

                loss = torch.sqrt(vert_sum + stiffness_sum) + laplacian_loss
                loss.backward()
                inner_optimizer.step()
                new_source_v,stiffness = self.local_affine(source_v, pool_num=0, return_stiff = True)
                # warp normal map
                with torch.no_grad():
                    new_source_normals, inv_mask = self.local_affine.forward_normal(source_normals)
                    inner_inv_mask = torch.logical_and(inv_mask, inner_mask)
            distance = torch.mean(torch.sqrt(torch.sum((old_source_v - new_source_v) ** 2, dim = 2)))
            print("current {:03d} NRICP avg_update:{:.5f} valid{:d}/{:d}: dis:{:.4f}, stiffness:{:.4f}, laplacian:{:.4f}, total:{:.4f}"
                    " laplacian_weight:{:.4f}, stiffness_weight:{:.4f}".format(self.mile_times,distance.item(), weight_mask.sum(),
                        weight_mask.numel(), vert_sum.item(), stiffness_sum.item(),
                        laplacian_loss.item(), loss.item(), laplacian_weight,
                        stiffness_weight))
            self.mile_times+=1

            if self.mile_times in self.mile_stone:
                self.mile_idx+=1
                with torch.no_grad():
                    new_source_v = self.local_affine(source_v, pool_num=0, return_stiff = False)
                    save_obj('./debug/nricp_optimizer/{}.obj'.format(self.mile_times), new_source_v.detach().cpu()[0], smpl_slice.faces.detach().cpu(),)
        return loss



