"""
@File: lap_deform_optimizer.
@Author: Lingteng Qiu
@Email: qiulingteng@link.cuhk.edu.cn
@Date: 2023-01-02
@Desc: laplacian deformation to align template surface and featurelines
"""
import torch.nn as nn
import torch
from .base_optimzier import _Base_Optimizer
from utils.common_utils import make_nograd_func
from pytorch3d.ops.knn import knn_gather, knn_points
from pytorch3d.structures import Meshes
from utils.common_utils import tensor2numpy
from torch import sparse_coo_tensor
import pdb
from pytorch3d.io import save_ply
import os
from sksparse.cholmod import cholesky_AAt
import scipy.sparse as sp




class Laplacian_Optimizer(_Base_Optimizer):
    '''
    This is Laplacian deform Optimizer using to fit two different cloud setting
    '''
    def __init__(self, epoch=3, constrain_weight = 1., optimizer_setting = None, smooth = True):
        '''
        epoch: hook training epoch
        optimizer_setting: is optimizer setting, in the icp registry this node is set to 0
        smooth: each step using laplacian smooth
        '''
        super(Laplacian_Optimizer, self).__init__(optimizer_setting)
        self.name = "Laplacian_Deform_Optimzier"
        # hook stop time
        self.epoch = epoch
        # energy function to compute
        # define by laplacian editting
        self.constrain_weights = constrain_weight
        self.smooth = smooth


    def energy_func(self,L, u, t, W):
        '''Energy function to check optimal results
        '''

        return torch.trace((L@u- t).T @ W  @(L@ u- t))

    def __encode_results(self, inputs, new_s, idx):
        '''the data return depend on your format
        '''

        inputs['source_fl_meshes'][idx].update_vertices(new_s)


    def __collect_data(self, inputs):
        '''the data fomat we only process to Garemnet Mesh and Garment Polygons class

        return
            source_idx: vertices need to deform to target position
            target: target vertices
            laplacian matrix: D^-1 (A- D) defined by pytorch3d
            vertices: source mesh' vertices
        '''

        def merge_source_target_map(s_t_list, static_pts_list):

            new_source_target_map =[]
            for source_target_map, static_pts in zip(s_t_list, static_pts_list):

                if static_pts == None:
                    new_source_target_map.append(source_target_map)
                else:
                    new_source_target_map.append([torch.cat([source_target_map[0], static_pts[0]], dim = 0),
                            torch.cat([source_target_map[1], static_pts[1]], dim = 0)])


            return new_source_target_map


        source_garment_templates = inputs['source_fl_meshes']
        target_meshes = inputs['target_meshes']
        target_fl_names = inputs['target_fl_type']
        best_match_type = inputs['outlayer']


        target_packed = dict()

        for target_mesh, target_fl_name in zip(target_meshes, target_fl_names):
            target_packed[target_fl_name] = target_mesh.verts_packed()

        target_meshes_verts = [target_mesh.verts_packed() for target_mesh in target_meshes]
        source_garment_laplacian_matrix = [source_garment_template.laplacian_packed() for source_garment_template in source_garment_templates]


        source_target_map_list = [source_garment_template.best_match(target_packed, source_garment_template.device, best_match_type) for source_garment_template in source_garment_templates]
        static_pts_list =[source_garment_template.static_pts_match() for source_garment_template in source_garment_templates]

        source_target_map_list = merge_source_target_map(source_target_map_list, static_pts_list)

        template_vertices = [source_garment_template.vertices for source_garment_template in source_garment_templates]


        return source_target_map_list, source_garment_laplacian_matrix, template_vertices


    def solver(self, L, t, W):
        # (A^t w A)^-1 A^t w
        pseudo_inverse = torch.linalg.inv(L.T @ W @ L) @ L.T@ W
        return pseudo_inverse @ t


    def __call__(self, **inputs):
        return self.fitting(inputs)



    @make_nograd_func
    def fitting(self, inputs):
        '''Laplacian fitting functions
        Given source point clounds and target point clounds
        this funciton find global R and t to make the distance(e.g. Euclidean distance) between source points and target points match
        '''

        print('Laplacian deformation it would be spent 1~3 minutes ')

        for epoch in range(self.epoch):
            s_t_map_list, laplacian_matrix_list, vertices_list = self.__collect_data(inputs)

            for s_id, s_t_map in enumerate(s_t_map_list):
                source_idx, target = s_t_map
                vertices = vertices_list[s_id]
                source_pts = vertices[source_idx]
                os.makedirs('./debug/register/', exist_ok = True)

                save_ply('./debug/register/source_{}.ply'.format(s_id), source_pts)
                save_ply('./debug/register/target_{}.ply'.format(s_id), target)




            for template_idx, (s_t_map, laplacian_matrix, vertices) in enumerate(zip(s_t_map_list, laplacian_matrix_list, vertices_list)):
                source_idx, target = s_t_map


                source = vertices[source_idx]
                device = source.device
                laplacian_verts = laplacian_matrix @ vertices




                u = target
                n,_ = laplacian_matrix.shape
                m,_ = source.shape


                indices= torch.cat([torch.arange(source.shape[0])[None], source_idx[None]], dim =0)
                values = torch.ones_like(source_idx).float()
                constrain_matrix = sparse_coo_tensor(indices= indices, values=values,size=(m,n),device=device)


                # W for weighted least-square
                W = torch.ones((laplacian_matrix.shape[0]), device = device)
                constrain_W = torch.ones((constrain_matrix.shape[0]), device = device)
                constrain_W[:] = self.constrain_weights
                W = torch.cat([W, constrain_W])
                W = torch.diag(W)

                L = torch.cat([laplacian_matrix, constrain_matrix]).to_dense()
                t = torch.cat([laplacian_verts, u], dim = 0)
                # argmin u' |Lu' - t| --> u' = L
                new_s = self.solver(L,t,W)



                if self.smooth:
                    laplacian_matrix = laplacian_matrix.to_dense()
                    laplacian_matrix[torch.arange(laplacian_matrix.shape[0]),torch.arange(laplacian_matrix.shape[0])] = 0
                    new_s = laplacian_matrix @ new_s
                self.__encode_results(inputs,new_s, template_idx)


        return inputs





class Laplacian_Deform_upper_and_domn_Optimzier(Laplacian_Optimizer):
    '''
    This is Laplacian deform Optimizer using to fit two different cloud setting
    '''
    def __init__(self, epoch=3, constrain_weight = 1., optimizer_setting = None, smooth = True):
        '''
        epoch: hook training epoch
        optimizer_setting: is optimizer setting, in the icp registry this node is set to 0
        smooth: each step using laplacian smooth
        '''
        super(Laplacian_Optimizer, self).__init__(optimizer_setting)
        self.name = "Laplacian_Deform_Optimzier"
        # hook stop time
        self.epoch = epoch
        # energy function to compute
        # define by laplacian editting
        self.constrain_weights = constrain_weight
        self.smooth = smooth


    def energy_func(self,L, u, t, W):
        '''Energy function to check optimal results
        '''

        return torch.trace((L@u- t).T @ W  @(L@ u- t))

    def __encode_results(self, inputs, new_s, idx):
        '''the data return depend on your format
        '''

        inputs['source_fl_meshes'][idx].update_vertices(new_s)


    def __collect_data(self, inputs):
        '''the data fomat we only process to Garemnet Mesh and Garment Polygons class

        return
            source_idx: vertices need to deform to target position
            target: target vertices
            laplacian matrix: D^-1 (A- D) defined by pytorch3d
            vertices: source mesh' vertices
        '''



        source_meshes = inputs['source_fl_meshes']
        target_mesh, source_mesh = source_meshes[0], source_meshes[1]
        best_match_type = inputs['outlayer']
        target_packed = dict()


        for field in target_mesh.get_fields():
            target_packed[field] = torch.cat(target_mesh.get_boundary(field))


        source_target_map = source_mesh.single_best_match(target_packed,source_mesh.device, 'upper_bottom',best_match_type)
        source_garment_laplacian_matrix = source_mesh.laplacian_packed()

        def merge_source_target_map(s_t_list, static_pts_list):

            new_source_target_map =[]
            for source_target_map, static_pts in zip(s_t_list, static_pts_list):

                if static_pts == None:
                    new_source_target_map.append(source_target_map)
                else:

                    new_source_target_map.append([torch.cat([source_target_map[0], static_pts[0]], dim = 0),
                            torch.cat([source_target_map[1], static_pts[1]], dim = 0)])

            return new_source_target_map


        static_fields = []
        for field in source_mesh.get_fields():

            if field != 'upper_bottom':
                static_fields.append(field)

        static_idx = torch.cat(source_mesh.get_outlayer_idx(*static_fields))
        static_pts = torch.cat(source_mesh.get_outlayer_boundary(*static_fields))
        static_pts_list = [static_idx, static_pts]
        source_target_map = merge_source_target_map([source_target_map],[static_pts_list])[0]
        template_vertices = source_mesh.vertices


        return [source_target_map], [source_garment_laplacian_matrix], [template_vertices]
    def solver(self, L, t, W):


        sparse_l = sp.csr_matrix(L)


        factor = cholesky_AAt(sparse_l.T)
        solve = factor(sparse_l.T @ t)


        return torch.from_numpy(solve).float()


    def __call__(self, **inputs):
        return self.fitting(inputs)



    @make_nograd_func
    def fitting(self, inputs):
        '''Laplacian fitting functions
        Given source point clounds and target point clounds
        this funciton find global R and t to make the distance(e.g. Euclidean distance) between source points and target points match
        '''


        if len(inputs['source_fl_meshes'])  == 1:
            return inputs



        for epoch in range(self.epoch):
            s_t_map_list, laplacian_matrix_list, vertices_list = self.__collect_data(inputs)

            for s_id, s_t_map in enumerate(s_t_map_list):
                source_idx, target = s_t_map
                vertices = vertices_list[s_id]
                source_pts = vertices[source_idx]
                save_ply('./debug/register/source_{}.ply'.format(s_id), source_pts)
                save_ply('./debug/register/target_{}.ply'.format(s_id), target)




            for template_idx, (s_t_map, laplacian_matrix, vertices) in enumerate(zip(s_t_map_list, laplacian_matrix_list, vertices_list)):
                source_idx, target = s_t_map
                source = vertices[source_idx]
                device = source.device
                laplacian_verts = laplacian_matrix @ vertices




                u = target
                n,_ = laplacian_matrix.shape
                m,_ = source.shape


                indices= torch.cat([torch.arange(source.shape[0])[None], source_idx[None]], dim =0)
                values = torch.ones_like(source_idx).float()
                constrain_matrix = sparse_coo_tensor(indices= indices, values=values,size=(m,n),device=device)


                # W for weighted least-square
                W = torch.ones((laplacian_matrix.shape[0]), device = device)
                constrain_W = torch.ones((constrain_matrix.shape[0]), device = device)
                constrain_W[:] = self.constrain_weights
                W = torch.cat([W, constrain_W])
                W = torch.diag(W)

                L = torch.cat([laplacian_matrix, constrain_matrix]).to_dense()
                t = torch.cat([laplacian_verts, u], dim = 0)
                # argmin u' |Lu' - t| --> u' = L
                new_s = self.solver(L,t,W)

                if self.smooth:
                    laplacian_matrix = laplacian_matrix.to_dense()
                    laplacian_matrix[torch.arange(laplacian_matrix.shape[0]),torch.arange(laplacian_matrix.shape[0])] = 0
                    new_s = laplacian_matrix @ new_s
                self.__encode_results(inputs,new_s, 1)


        return inputs


