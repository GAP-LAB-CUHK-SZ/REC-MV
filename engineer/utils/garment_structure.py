"""
@File: garment_structure.py
@Author: Lingteng Qiu
@Email: qiulingteng@link.cuhk.edu.cn
@Date: 2022-07-12
@Desc: some data structures are used to process REC-MV data
1. Intesection-free Curves
2. Garment Mesh Datas
3. Garment polygon Datas
"""

import torch
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.io import save_obj, save_ply
from engineer.utils.transformation import pytorch3d_mesh_transformation
from scipy.optimize import linear_sum_assignment as linear_assignment
import ot
import numpy as np
import trimesh
import pymeshlab
import openmesh as om
from engineer.utils.matrix_utils import faster_matrix_power
from utils.constant import GARMENT_FL_MATCH, FL_EXTRACT
from collections import defaultdict
import pdb
import torch.nn as nn
from engineer.utils.polygons import uniformsample3d
import torch.nn.functional as F
import torch
from pytorch3d.ops.knn import knn_gather, knn_points
import os.path as osp
from einops import rearrange
from pytorch3d.loss import chamfer_distance


class Intersect_Free_Curve(nn.Module):
    def __init__(self, verts_list, cano_smpl_verts_list, faces_list, fl_names, cano2canosmpl, sample_num = 200):
        '''Alpha Curve define

        verts_list: verts in canonical space after fl_align
        cano_smpl_verts_list: verts in canonical smpl before fl align
        '''
        super(Intersect_Free_Curve, self).__init__()

        self.cano2canosmpl = cano2canosmpl
        self.fl_names = fl_names
        self.sample_num = sample_num
        device = verts_list[0].device

        uni_curve_verts_list = self.extract_edge(verts_list, faces_list, cano_smpl_verts_list, sample_num)

        uni_curve_verts_list = [torch.from_numpy(uni_curve_verts).to(device) for uni_curve_verts in uni_curve_verts_list]
        self.initialize_parameters(uni_curve_verts_list)



    def query_canosmpl_verts(self, query_names):

        query_list = []
        cano_smpl_verts_list = torch.split(self.cano_smpl_verts, 1, dim=0)
        curve_dict = {}
        for fl_name, cano_smpl_verts in zip(self.fl_names, cano_smpl_verts_list):
            curve_dict[fl_name] = cano_smpl_verts.squeeze(0)

        for query_name in query_names:
            query_pts = curve_dict[query_name]
            query_list.append(query_pts)

        return query_list


    def initialize_parameters(self, uni_curve_verts_list):
        '''
        registry the parameters of Alpha curve to train
        '''

        cano_smpl_verts_list = self.cano2canosmpl(uni_curve_verts_list, self.fl_names)
        cano_smpl_verts_list = [cano_smpl_verts[None] for cano_smpl_verts in cano_smpl_verts_list]


        cano_smpl_verts_tensor = torch.cat(cano_smpl_verts_list, dim = 0)
        self.register_buffer('cano_smpl_verts', cano_smpl_verts_tensor)
        uni_curve_verts_list = [uni_curve_verts[None] for uni_curve_verts in uni_curve_verts_list]
        cano_verts_tensor = torch.cat(uni_curve_verts_list, dim = 0)
        cano_verts_center = cano_verts_tensor.mean(1, keepdim = True)
        self.register_buffer('cano_verts_center', cano_verts_center)

        cano_v_dirs = (cano_verts_tensor - cano_verts_center) / ((cano_verts_tensor- cano_verts_center).norm(dim=-1, keepdim = True) + 1e-6)
        nx = torch.cross(cano_v_dirs[:,:-1,:], cano_v_dirs[:,1:,:])
        nx = nx / nx.norm(dim = -1, keepdim = True)
        nx = nx.mean(dim=1, keepdim = True)

        self.register_buffer('cano_nx', nx)
        self.register_buffer('cano_v_dirs', cano_v_dirs)


        init_scale = ((cano_verts_tensor - cano_verts_center) * cano_v_dirs).sum(dim = -1, keepdim = True)
        init_scale = torch.clamp_min(init_scale,0.)
        self.register_buffer('init_scale', init_scale)
        device = init_scale.device
        # learnable parameters
        fl_scale = nn.Parameter(torch.full(init_scale.shape, 1.0, device = device, requires_grad = True))
        nx_scale = nn.Parameter(torch.full(init_scale.shape, 0., device = device, requires_grad = True))

        self.register_parameter('scale', fl_scale)
        self.register_parameter('nx_scale', nx_scale)

    def forward(self):

        v_dir_offset = self.cano_v_dirs * self.init_scale * F.relu(self.scale)
        nx_offset = self.nx_scale* self.cano_nx

        return self.cano_verts_center + v_dir_offset + nx_offset


    def inference(self):
        with torch.no_grad():
            v_dir_offset = self.cano_v_dirs * self.init_scale * F.relu(self.scale)
            nx_offset = self.nx_scale* self.cano_nx

            return self.cano_verts_center + v_dir_offset + nx_offset


    def regularization(self, fl_masks):
        '''
        regularization energy function
        '''
        # [start -> end ->start] is a loop
        cano_verts = self.forward()
        # center should not change too much
        used_flag = fl_masks.sum()
        used_flag = (used_flag >0).float()

        center_loss = used_flag * abs(cano_verts.mean(1, keepdim = True) - self.cano_verts_center).sum()

        # B, N - 1, 3
        diff_a = cano_verts[:, :-1, :] - cano_verts[:, 1:, :]
        diff_b = cano_verts[:,-1:,:] - cano_verts[:,0:1, :]
        diff_c = cano_verts[:,0:1,:] - cano_verts[:,1:2, :]
        diff_a =  torch.cat([diff_a, diff_b, diff_c], dim = 1)
        diff_a = diff_a / (diff_a.norm(dim = -1, keepdim = True) + 1e-6)

        diff_a_loss = (1 - F.cosine_similarity(diff_a[:,:-1, :], diff_a[:,1:,:], dim=-1))

        return {'center_offset': 0 * center_loss, 'diff_a_loss': diff_a_loss.sum()}



    def extract_edge(self, verts_list, faces_list, cano_smpl_verts_list, sample_num = 200):
        # extract curve_edge



        verts_list = [verts.detach().cpu().numpy() for verts in verts_list]
        faces_list = [faces.detach().cpu().numpy() for faces in faces_list]
        curve_list = [trimesh.Trimesh(vertices = verts , faces = faces, process =False) for verts, faces in zip(verts_list, faces_list)]

        uni_curve_verts_list = []

        for curve, verts in zip(curve_list, verts_list):
            boundaries = curve.outline()
            b_pts_list = [entity.points for entity in boundaries.entities]
            b_face_ids_list = [entity.nodes for entity in boundaries.entities]
            assert len(b_pts_list) == 2
            curve_edge_idx = b_pts_list[0] if len(b_pts_list[0]) > len(b_pts_list[1]) else b_pts_list[1]
            curve_verts = verts[curve_edge_idx]
            uni_curve_verts = uniformsample3d(curve_verts, sample_num)

            uni_curve_verts_list.append(uni_curve_verts)



        return uni_curve_verts_list


    def curve_to_mesh(self, curve_radius =0.002, num_joints = 6, curve_verts = None, curve_idx = None, target_idx = None):


        if not curve_verts == None:
            fl_scale = self.scale.detach().clone()
            fl_scale[target_idx] = fl_scale[target_idx].mean()
            fl_scale = nn.Parameter(fl_scale, requires_grad=True)
            self.register_parameter('scale', fl_scale)

            fl_optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4)
            self.init_scale[target_idx,...] *=  2

            for i in range(20000):
                fl_optimizer.zero_grad()
                curves = self.forward()

                loss = 0.
                cham_loss = 0.
                diff_a_loss = 0.
                for c_i, t_i in zip(curve_idx, target_idx):

                    gt = torch.from_numpy(curve_verts[c_i]).float().cuda()
                    cham_loss += 1000*chamfer_distance(curves[t_i][None], gt[None])[0]


                    # regularization loss
                    diff_a = curves[t_i, :-1, :] - curves[t_i, 1:, :]
                    diff_b = curves[t_i,-1:,:] - curves[t_i,0:1, :]
                    diff_a =  torch.cat([diff_a, diff_b], dim = 0)
                    diff_a = diff_a / (diff_a.norm(dim = -1, keepdim = True) + 1e-6)
                    diff_a_loss += (1 - F.cosine_similarity(diff_a[:-1, :], diff_a[1:,:], dim=-1)).sum()

                loss = 0.1* diff_a_loss + cham_loss
                print('{:06d}/{:06d}: {:.4f}\t{:.4f}'.format(i,20000, diff_a_loss, cham_loss))

                loss.backward()
                fl_optimizer.step()

        curves = self.inference().detach().cpu()
        cano_nx = self.cano_nx.detach().cpu()[:,0,:]
        N = curves.shape[0]

        diff_a = curves[:, :-1, :] - curves[:, 1:, :]
        diff_b = curves[:,-1:,:] - curves[:,0:1, :]
        diff_a =  torch.cat([diff_a, diff_b], dim = 1)
        diff_a = diff_a / (diff_a.norm(dim = -1, keepdim = True) + 1e-6)
        cano_nx = cano_nx[:, None, :].expand_as(diff_a)


        # rotate
        cross_n = torch.cross(diff_a, cano_nx)
        dot_n = diff_a * (diff_a * cano_nx)

        rotate_nx_list = []
        for i in range(0, 360, 360 // num_joints):
            radius = torch.tensor(np.radians(i)).float()
            rotate_nx = cano_nx * torch.cos(radius)  + cross_n * torch.sin(radius) + dot_n*(1-torch.cos(radius))
            rotate_nx_list.append(rotate_nx[..., None, :])

        rotate_nx = torch.cat(rotate_nx_list, dim=-2)
        curve_mesh_verts = curves[..., None,:] + curve_radius * rotate_nx
        face_idx = torch.arange(curve_mesh_verts.view(-1,3).shape[0]).view(*curve_mesh_verts.shape[:-1])

        # adj face
        batch_faces = []
        for f_i in range(curve_mesh_verts.shape[1] ):
            start_face = f_i % curve_mesh_verts.shape[1]
            end_face = (f_i+1) % curve_mesh_verts.shape[1]


            start_face = face_idx[:,start_face]
            end_face = face_idx[:,end_face]

            face_list = []
            for v_i in range(start_face.shape[1]):
                start_v_i  = (v_i) % start_face.shape[1]
                end_v_i = (v_i +1) % start_face.shape[1]


                face_a = torch.cat([start_face[:,start_v_i:start_v_i+1], end_face[:, start_v_i:start_v_i+1], end_face[:, end_v_i:end_v_i+1]], dim = -1)
                face_b = torch.cat([start_face[:,start_v_i:start_v_i+1], end_face[:, end_v_i:end_v_i+1],start_face[:, end_v_i:end_v_i+1]],  dim = -1)
                face_list.append(face_a)
                face_list.append(face_b)

            faces = torch.cat(face_list, dim =-1).view(N, num_joints *2,3)
            batch_faces.append(faces)



        curve_faces = torch.cat(batch_faces, dim =1)


        for curve_idx, curve_face in enumerate(curve_faces):
            curve_faces[curve_idx] -= curve_face.min()

        curve_verts = curve_mesh_verts.view(N,-1,3)
        curve_meshes = [Meshes([curve_vert],[curve_face]) for curve_vert, curve_face in zip(curve_verts, curve_faces)]

        return curve_meshes



def close_hole(mesh):
    '''naive close hole,
    only connect curve center pts

    mesh: Trimesh structure
    ----------
    return
        close hole trimesh structure
    '''

    boundaries = mesh.outline()

    b_pts_list = [entity.points for entity in boundaries.entities]
    b_face_ids_list = [entity.nodes for entity in boundaries.entities]
    vertices = mesh.vertices
    faces = mesh.faces
    origin_faces_size = faces.shape[0]
    verts_id = vertices.shape[0]

    new_vertices_list = []
    new_faces_list = []
    faces_list = faces.reshape(-1).tolist()

    for b_pts, b_face_ids in zip(b_pts_list, b_face_ids_list):

        bool_faces_list = [face_id in b_pts for face_id in faces_list]
        bool_faces = np.asarray(bool_faces_list).reshape(-1, 3)
        boundary_f_idx = np.where(bool_faces.sum(-1) == 2)[0]

        boundary_bool_faces = bool_faces[boundary_f_idx]
        boundary_faces = faces[boundary_f_idx]
        boundary_line = boundary_faces[boundary_bool_faces].reshape(-1, 2)
        center = vertices[b_pts].mean(0, keepdims= True)
        f_size = b_face_ids.shape[0]
        add_f_idx = np.asarray([verts_id])[None].repeat(f_size, axis=0)
        add_faces = np.concatenate([boundary_line, add_f_idx], axis= -1)
        add_faces = add_faces[...,[0,2,1]]
        new_vertices_list.append(center)
        new_faces_list.append(add_faces)
        verts_id+=1

    add_new_vertices = np.concatenate(new_vertices_list, axis= 0)
    add_new_faces = np.concatenate(new_faces_list, axis= 0)
    new_vertices = np.concatenate([vertices, add_new_vertices], axis= 0)
    new_faces = np.concatenate([faces, add_new_faces], axis= 0)

    new_faces_id = list(range(origin_faces_size, new_faces.shape[0]))
    for i in range(2):
        new_vertices, new_faces = trimesh.remesh.subdivide(new_vertices, new_faces, new_faces_id)
        new_faces_id = list(range(origin_faces_size, new_faces.shape[0]))


    return  trimesh.Trimesh(new_vertices, new_faces, process = False)


def reorder_face(vertices_id, faces_id):
    new_faces_id = np.zeros_like(faces_id)
    vert_id ={}
    for new_id, ver_id in enumerate(vertices_id):
        vert_id[ver_id] = new_id
    for i in range(faces_id.shape[0]):
        for j in range(faces_id.shape[1]):
            new_faces_id[i][j] = vert_id[faces_id[i][j]]

    return new_faces_id



class _Base_Garment(object):
    def __init__(self):
        super(_Base_Garment, self).__init__()
    @property
    def device(self):
        return self.vertices.device





class Garment_Mesh(_Base_Garment):
    """Docstring for Garment Mesh """
    def __init__(self, vertices, faces, colors, color_decode_fun, boundary_color_map, garment_type, static_pts_idx  = None):
        """TODO: to be defined. """

        super(Garment_Mesh, self).__init__()
        self.colors = colors
        self.garment_type= garment_type
        self.garment = Meshes(vertices, faces)
        self.color_decode_fun = color_decode_fun
        self.boundary_color_map = boundary_color_map
        self.__init_boundary(boundary_color_map)

        self.outlayer()

        self.static_pts_idx =  static_pts_idx
        self.registry_staitc_pts(static_pts_idx)



    def registry_staitc_pts(self, static_pts_idx):
        if static_pts_idx == None:
            return
        self.static_pts_map = np.zeros(self.garment.verts_packed().shape[0], dtype = np.int)
        self.static_pts_map[static_pts_idx] = 1


    def static_pts_match(self):
        if self.static_pts_idx == None:
            return None

        boundary_idx = self.get_boundary_idx(*self.get_fields())

        boundary_set = set(torch.cat(boundary_idx, dim = 0).detach().cpu().tolist())
        static_pts_set = set(np.where(self.static_pts_map>0)[0].tolist())

        iou = static_pts_set & boundary_set
        # if exist iou with boundary, we set False static match
        if len(iou)>0:
            return None
        else:

            return [torch.from_numpy(np.asarray(self.static_pts_idx)).long(), self.vertices[self.static_pts_idx]]


    def remesh_garment_mesh_no_reduce(self, tmp_path):
        '''remesh garment mesh using Meshlab algorithm
        '''

        remesh_mesh = trimesh.Trimesh(self.vertices.detach().cpu(), self.faces.detach().cpu(), process = False)
        # mesh.export('./tmp_obj.obj')
        # ms = pymeshlab.MeshSet()
        # ms.load_new_mesh('./tmp_obj.obj')
        # ms.meshing_close_holes(maxholesize=300000)
        # ms.meshing_isotropic_explicit_remeshing()
        # ms.save_current_mesh('./tmp_obj.obj')
        # mesh = trimesh.load('./tmp_obj.obj', process = False)

        boundaries = remesh_mesh.outline()
        b_pts_list = [entity.points for entity in boundaries.entities]
        outlayer_set = sorted(set(np.concatenate(b_pts_list,axis=0).tolist()))

        new_verts = torch.from_numpy(remesh_mesh.vertices).float()
        new_faces = torch.from_numpy(remesh_mesh.faces).long()

        my_outlayer_set = sorted(self.outlayer_set)
        my_vs = self.vertices[my_outlayer_set]
        remesh_vs = new_verts[outlayer_set]
        knn_results = knn_points(remesh_vs[None], my_vs[None])
        nearest_idx = knn_results.idx[0, ...,0]


        nearest_idx = np.asarray(my_outlayer_set)[nearest_idx]

        colors_copy = torch.zeros_like(new_verts, dtype = torch.int32)[None , :, 0:1]
        colors_copy[...]= self.boundary_color_map['back_ground']
        colors_copy[:, outlayer_set] = self.colors[:, nearest_idx, :]


        return Garment_Mesh([new_verts], [new_faces], colors_copy, self.color_decode_fun, self.boundary_color_map, self.garment_type, static_pts_idx  = None)



    def remesh_garment_mesh(self, tmp_path):
        '''remesh garment mesh using Meshlab algorithm
        '''

        trm = trimesh.Trimesh(self.vertices.detach().cpu(), self.faces.detach().cpu(), process = False)
        trm.export(osp.join(tmp_path,'nricp_coarse.obj'))

        # mesh.export('./tmp_obj.obj')
        # ms = pymeshlab.MeshSet()
        # ms.load_new_mesh('./tmp_obj.obj')
        # ms.meshing_close_holes(maxholesize=300000)
        # ms.meshing_isotropic_explicit_remeshing()
        # ms.save_current_mesh('./tmp_obj.obj')
        # mesh = trimesh.load('./tmp_obj.obj', process = False)
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(osp.join(tmp_path, 'nricp_coarse.obj'))
        ms.meshing_isotropic_explicit_remeshing()
        ms.meshing_surface_subdivision_loop()
        ms.save_current_mesh(osp.join(tmp_path,'./remesh.obj'))
        remesh_mesh = trimesh.load(osp.join(tmp_path, 'remesh.obj'))

        boundaries = remesh_mesh.outline()
        b_pts_list = [entity.points for entity in boundaries.entities]
        outlayer_set = sorted(set(np.concatenate(b_pts_list,axis=0).tolist()))

        new_verts = torch.from_numpy(remesh_mesh.vertices).float()
        new_faces = torch.from_numpy(remesh_mesh.faces).long()

        my_outlayer_set = sorted(self.outlayer_set)
        my_vs = self.vertices[my_outlayer_set]
        remesh_vs = new_verts[outlayer_set]
        knn_results = knn_points(remesh_vs[None], my_vs[None])
        nearest_idx = knn_results.idx[0, ...,0]


        nearest_idx = np.asarray(my_outlayer_set)[nearest_idx]

        colors_copy = torch.zeros_like(new_verts, dtype = torch.int32)[None , :, 0:1]
        colors_copy[...]= self.boundary_color_map['back_ground']
        colors_copy[:, outlayer_set] = self.colors[:, nearest_idx, :]


        return Garment_Mesh([new_verts], [new_faces], colors_copy, self.color_decode_fun, self.boundary_color_map, self.garment_type, static_pts_idx  = None)





    def outlayer(self):
        '''
        the outlayer edge pts
        '''

        meshes = trimesh.Trimesh(self.vertices.detach().cpu().numpy(), self.faces.detach().cpu().numpy(), process = False)

        boundaries = meshes.outline()
        b_pts_list = [entity.points for entity in boundaries.entities]
        self.outlayer_set = set(np.concatenate(b_pts_list,axis=0).tolist())


    def update_vertices(self, new_vertices):

        offset = new_vertices - self.vertices
        self.garment = self.garment.offset_verts(offset)



    def __init_boundary(self, color_map):
        '''help us to find current garment_mesh's feature vertice_id
        '''
        self.boundary_fields = dict()
        for key in color_map.keys():
            _, indx, _ = torch.where(self.colors == color_map[key])
            self.boundary_fields[key] = indx

    def get_fields(self, background = False):
        if background:
            return sorted(list(self.boundary_fields.keys()))
        else:
            keys = sorted(list(self.boundary_fields.keys()))
            return list(filter(lambda x: not x =='back_ground', keys))

    def obtain_fl_mesh(self, field):

        boundary_idx = self.boundary_fields[field]
        boundary_verts = self.vertices[ boundary_idx]
        device = boundary_verts.device

        face_edges = self.faces.detach().cpu().numpy().reshape(-1)
        boundary_edges = np.asarray([edge in boundary_idx for edge in face_edges]).reshape(-1, 3)
        boundary_conds = boundary_edges.sum(-1)
        valid_face_id = np.where(boundary_conds ==3)[0]
        boundary_faces = self.faces[valid_face_id]



        ordered_faces_id = torch.from_numpy(reorder_face(boundary_idx.numpy(), boundary_faces.numpy())).long().to(device)
        fl_mesh = Meshes([boundary_verts], [ordered_faces_id]).to(device)


        return fl_mesh



    def extract_featurelines(self):

        fl_fields = FL_EXTRACT[self.garment_type]

        ret_dict = {}
        for fl in fl_fields:
            ret_dict[fl] = self.obtain_fl_mesh(fl)


        return ret_dict


    def get_outlayer_boundary(self, *args):
        ret = []

        for key in args:
            boundary_idx = set(self.boundary_fields[key].detach().cpu().numpy().tolist())
            boundary_idx = sorted(list(self.outlayer_set & boundary_idx))
            boundary = self.vertices[ boundary_idx]


            ret.append(boundary)
        return ret


    def single_best_match(self,target, device, field,is_outlayer = False):

        match_target = [ ]
        source_idx = []
        for field in [field]:
            if False == is_outlayer :
                source_bo = self.get_boundary(field)
            else:
                source_bo = self.get_outlayer_boundary(field)
            target_bo = target[field]


            source_bo = torch.cat(source_bo)



            if target_bo.shape[0]>  source_bo.shape[0]:
                idx = torch.arange(0, target_bo.shape[0], (target_bo.shape[0]-1) /source_bo.shape[0]).long()
                target_bo = target_bo[idx]
            else:
                idx = torch.arange(0, target_bo.shape[0], (target_bo.shape[0]-1) /source_bo.shape[0]).long()
                target_bo = target_bo[idx]


            source_bo = source_bo.detach().cpu().numpy()
            target_bo = target_bo.detach().cpu().numpy()



            distance = ot.dist(source_bo, target_bo )
            source_match_idx, target_match_idx = linear_assignment(distance)


            source_c= source_bo.mean(axis = 0)
            target_c= target_bo.mean(axis = 0)
            source_n = source_bo - source_c
            target_n = (target_bo - target_c)[target_match_idx]

            source_n_norm = np.linalg.norm(source_n, axis=1, keepdims=True)
            target_n_norm = np.linalg.norm(target_n, axis=1, keepdims=True)

            similiarity = ((source_n * target_n)/(source_n_norm * target_n_norm)).sum(axis=-1)
            norm_mask = (similiarity >0.5)
            source_match_idx = source_match_idx[norm_mask]
            target_match_idx= target_match_idx[norm_mask]





            target_v = target_bo[target_match_idx]

            if False == is_outlayer:
                source_id = self.get_boundary_idx(field)
            else:
                source_id = self.get_outlayer_idx(field)


            source_idx.append(source_id[0][source_match_idx])
            match_target.append(target_v)

        match_target = np.concatenate(match_target, axis=0)

        match_target = torch.from_numpy(match_target).float().to(device)


        return torch.cat(source_idx,dim=0), match_target



    def bottom_color_set(self):
        trm = trimesh.Trimesh(self.vertices.detach().cpu(), self.faces.detach().cpu(), process = False)
        boundaries = trm.outline()
        b_pts_list = [entity.points for entity in boundaries.entities]




    def best_match(self, target, device, is_outlayer = False):
        '''
        find the best match method make the source pts match target pts
        '''

        garment_type = self.garment_type
        fl_fields = GARMENT_FL_MATCH[garment_type]

        match_target = [ ]
        source_idx = []


        for field in fl_fields:

            if False == is_outlayer :
                source_bo = self.get_boundary(field)
            else:


                source_bo = self.get_outlayer_boundary(field)
            target_bo = target[field]


            source_bo = torch.cat(source_bo)



            if target_bo.shape[0]>  source_bo.shape[0]:
                idx = torch.arange(0, target_bo.shape[0], (target_bo.shape[0]-1) /source_bo.shape[0]).long()
                target_bo = target_bo[idx]
            else:
                idx = torch.arange(0, target_bo.shape[0], (target_bo.shape[0]-1) /source_bo.shape[0]).long()
                target_bo = target_bo[idx]


            source_bo = source_bo.detach().cpu().numpy()
            target_bo = target_bo.detach().cpu().numpy()



            distance = ot.dist(source_bo, target_bo )
            source_match_idx, target_match_idx = linear_assignment(distance)


            source_c= source_bo.mean(axis = 0)
            target_c= target_bo.mean(axis = 0)
            source_n = source_bo - source_c
            target_n = (target_bo - target_c)[target_match_idx]

            source_n_norm = np.linalg.norm(source_n, axis=1, keepdims=True)
            target_n_norm = np.linalg.norm(target_n, axis=1, keepdims=True)

            similiarity = ((source_n * target_n)/(source_n_norm * target_n_norm)).sum(axis=-1)
            norm_mask = (similiarity >0.5)
            source_match_idx = source_match_idx[norm_mask]
            target_match_idx= target_match_idx[norm_mask]





            target_v = target_bo[target_match_idx]

            if False == is_outlayer:
                source_id = self.get_boundary_idx(field)
            else:
                source_id = self.get_outlayer_idx(field)


            source_idx.append(source_id[0][source_match_idx])
            match_target.append(target_v)

        match_target = np.concatenate(match_target, axis=0)

        match_target = torch.from_numpy(match_target).float().to(device)


        return torch.cat(source_idx,dim=0), match_target


    def get_outlayer_idx(self, *args):
        ret = []


        for key in args:
            if key not in self.get_fields():
                continue
            boundary_idx = set(self.boundary_fields[key].detach().cpu().numpy().tolist())
            boundary_idx = torch.tensor(sorted(list(self.outlayer_set & boundary_idx))).long()
            ret.append(boundary_idx)

        return ret

    def get_boundary_idx(self, *args):
        ret = []

        for key in args:
            boundary_idx = self.boundary_fields[key]
            ret.append(boundary_idx)

        return ret

    def get_boundary(self, *args):
        ''' obtain boudnary_mesh.
        generally speaking, we do not considering the faces, as the bounary is polygons
        '''
        ret = []

        for key in args:
            boundary_idx = self.boundary_fields[key]
            boundary = self.vertices[ boundary_idx]



            ret.append(boundary)
        return ret

    def save_obj(self, save_path):
        save_obj(save_path, self.vertices, self.faces)

    def save_boudnary(self, save_path, *fileds):

        boudary_pts = self.get_boundary(*fileds)

        boudary_pts = torch.cat(boudary_pts, dim =0)
        save_ply(save_path, boudary_pts)


    def close_hole(self, save_name):
        '''
        Using meshlab meshing_close_holes to fill the hold of udf garment
        Return Mesh, close the hole of slice mesh(Trimesh)
        '''


        vertices = self.vertices.detach().cpu().numpy()
        faces = self.faces.detach().cpu().numpy()
        mesh = trimesh.Trimesh(vertices,faces, process= False)

        # NOTE thta pymeshlab sometimes connot close big hole,

        # mesh.export('./tmp_obj.obj')
        # ms = pymeshlab.MeshSet()
        # ms.load_new_mesh('./tmp_obj.obj')
        # ms.meshing_close_holes(maxholesize=300000)
        # ms.meshing_isotropic_explicit_remeshing()
        # ms.save_current_mesh('./tmp_obj.obj')
        # mesh = trimesh.load('./tmp_obj.obj', process = False)
        # load the normal map

        mesh = close_hole(mesh)
        mesh.export('./{}.obj'.format(save_name))

        vert_normal = mesh.vertex_normals[...] / (np.sqrt(np.sum(mesh.vertex_normals **2, axis=1, keepdims = True)))

        tmp=om.TriMesh(points=mesh.vertices,face_vertex_indices=mesh.faces)
        tmp.request_face_normals()
        tmp.request_vertex_normals()
        tmp.update_normals()
        mesh_normal = torch.from_numpy(tmp.vertex_normals()).float()
        mesh_verts = torch.from_numpy(tmp.points()).float()

        return [mesh_verts, mesh_normal]

    def get_adj_matrix(self, verts, edges):
        """
        Computes the adj matrix.
        The definition of the laplacian is
        L[i, j] = 1 / deg(i)  , if (i, j) is an edge
        L[i, j] =    0        , otherwise
        where deg(i) is the degree of the i-th vertex in the graph.

        Args:
            verts: tensor of shape (V, 3) containing the vertices of the graph
            edges: tensor of shape (E, 2) containing the vertex indices of each edge
        Returns:
            L: Sparse FloatTensor of shape (V, V)
        """
        V = verts.shape[0]
        e0, e1 = edges.unbind(1)
        e_self = torch.arange(V, device=verts.device)


        idx01 = torch.stack([e0, e1], dim=1)  # (E, 2)
        idx10 = torch.stack([e1, e0], dim=1)  # (E, 2)
        idx00 = torch.stack([torch.arange(V, device=verts.device), torch.arange(V, device=verts.device)], dim=1)
        idx = torch.cat([idx01, idx10, idx00], dim=0).t()  # (2, 2*E)

        # First, we construct the adjacency matrix,
        # i.e. A[i, j] = 1 if (i,j) is an edge, or
        # A[e0, e1] = 1 &  A[e1, e0] = 1
        ones = torch.ones(idx.shape[1], dtype=torch.float32, device=verts.device)
        A = torch.sparse.FloatTensor(idx, ones, (V, V))
        deg = torch.sparse.sum(A, dim=1).to_dense()


        # We construct the Laplacian matrix by adding the non diagonal values
        # i.e. L[i, j] = 1 ./ deg(i) if (i, j) is an edge
        deg0 = deg[e0]
        deg0 = torch.where(deg0 > 0.0, 1.0 / deg0, deg0)
        deg1 = deg[e1]
        deg1 = torch.where(deg1 > 0.0, 1.0 / deg1, deg1)

        deg_self = deg[e_self]
        deg_self = torch.where(deg_self > 0.0, 1.0 / deg_self, deg_self)
        val = torch.cat([deg0, deg1, deg_self])
        adj_matrix  = torch.sparse.FloatTensor(idx, val, (V, V))

        return adj_matrix

    def dense_boundary(self, type='edge', k = 3):

        '''
        add points at boundary near by K neighbour using edge_method
        '''

        with torch.no_grad():
            verts_packed = self.garment.verts_packed()  # (sum(V_n), 3)
            edges_packed = self.garment.edges_packed()  # (sum(E_n), 3)
            adj_matrix = self.get_adj_matrix(verts_packed, edges_packed)
            statistics_matrix = faster_matrix_power(adj_matrix,3).to_dense()

        boundary_idx = self.get_boundary_idx(*self.get_fields(background = False))
        boundary_idx = torch.cat(boundary_idx, dim=-1)

        boundary_transfer = statistics_matrix[boundary_idx, ...]
        _, edge_idx = torch.where(boundary_transfer>0.)
        knear_pts = torch.unique(edge_idx).detach().cpu().numpy()

        faces_packed = self.garment.faces_packed()
        vertices_packed = self.garment.verts_packed()
        edge_packed = self.garment.edges_packed()

        edge_nodes = edge_packed.view(-1).detach().cpu().numpy()


        # compute valid edge
        edge_idx_nodes = np.asarray([edge_node in knear_pts for edge_node in edge_nodes]).reshape(-1,2)
        edge_idx = edge_idx_nodes.sum(-1)
        valid_edge_idx = np.where(edge_idx==2)[0]

        # compute_valid face
        face_edges = self.garment.faces_packed_to_edges_packed().detach().cpu().numpy().reshape(-1)
        face_edges = np.asarray([edge in valid_edge_idx  for edge in face_edges]).reshape(-1, 3)
        face_edges = face_edges.sum(-1)
        valid_face_id = np.where(face_edges == 3)[0]
        single_face_id = np.where(face_edges == 1)[0]
        invalid_face_id = np.where(face_edges != 3)[0]

        # map old face_edges  to new face_edges
        valid_edge_idx_reorder = {}
        for new_id, old_id in enumerate(valid_edge_idx):
            valid_edge_idx_reorder[old_id] = new_id

        valid_old_faces_edges = self.garment.faces_packed_to_edges_packed()[valid_face_id].view(-1).detach().numpy().tolist()

        # compute boundary_edges and add boundary edge face

        valid_single_faces_nodes = self.garment.faces_packed_to_edges_packed()[single_face_id].view(-1).detach().numpy().tolist()
        valid_single_faces_edges= torch.from_numpy(np.asarray([node in valid_old_faces_edges for node in valid_single_faces_nodes])).view(-1, 3)
        batch_id, verts_id = torch.where(valid_single_faces_edges ==True)


        # NOTE face_packed 1, 2,3  faces2edge:[3-2/2-3, 3-1/1-3, 1-2/2-1]
        # guarantee face clock-order
        # face_dir_edge1 = torch.cat([faces_packed[...,1:2], faces_packed[...,2:3]], dim=-1)
        # face_dir_edge2 = torch.cat([faces_packed[...,2:3], faces_packed[...,0:1]], dim=-1)
        # face_dir_edge3 = torch.cat([faces_packed[...,0:1], faces_packed[...,1:2]], dim=-1)
        face_dir_edge1 = torch.cat([faces_packed[...,2:3], faces_packed[...,1:2]] , dim=-1)
        face_dir_edge2 = torch.cat([faces_packed[...,0:1], faces_packed[...,2:3]] , dim=-1)
        face_dir_edge3 = torch.cat([faces_packed[...,1:2], faces_packed[...,0:1]] , dim=-1)
        face_dir_edge = torch.cat([face_dir_edge1[:, None, :], face_dir_edge2[:, None, :], face_dir_edge3[:, None, :]], dim = 1)
        single_face_dir_edge = face_dir_edge[single_face_id]
        boundary_line_verts_id = single_face_dir_edge[batch_id, verts_id]
        boundary_edge = self.garment.faces_packed_to_edges_packed()[single_face_id][batch_id, verts_id]

        # using to compute the boundary_edge center
        boundary_line_center_valid_idx = torch.Tensor([valid_edge_idx_reorder[old_idx] for old_idx in boundary_edge.detach().cpu().numpy().tolist()]).long()

        valid_faces_edges = torch.from_numpy(np.asarray(list(map(lambda x:valid_edge_idx_reorder[x], valid_old_faces_edges))).reshape(-1, 3))
        valid_edge_packed = edge_packed[valid_edge_idx]
        e_v = (vertices_packed[valid_edge_packed]).mean(1)
        valid_faces_edges = valid_faces_edges + vertices_packed.shape[0]
        valid_line_center_verts_id = (boundary_line_center_valid_idx  + vertices_packed.shape[0])[..., None]
        boundary_faces = torch.cat([boundary_line_verts_id,valid_line_center_verts_id], dim = -1)


        colors_packed = self.colors.view(-1,1)



        # compute new mesh faces
        invalid_faces_packed = faces_packed[invalid_face_id]
        valid_faces_packed = faces_packed[valid_face_id]

        valid_small_face0 = torch.cat([valid_faces_edges[...,0:1], valid_faces_packed[...,2:3], valid_faces_edges[...,1:2]],dim=-1)
        valid_small_face1 = torch.cat([valid_faces_edges[...,1:2], valid_faces_packed[...,0:1], valid_faces_edges[...,2:3]],dim=-1)
        valid_small_face2 = torch.cat([valid_faces_edges[...,2:3], valid_faces_packed[...,1:2], valid_faces_edges[...,0:1]],dim=-1)
        new_small_faces = torch.cat([valid_small_face0,valid_small_face1,valid_small_face2], dim =0)
        new_faces_packed = torch.cat([new_small_faces, valid_faces_edges, invalid_faces_packed, boundary_faces], dim = 0)
        # new_faces_packed = torch.cat([boundary_faces], dim = 0)

        # compute new garment colors
        node1_color = colors_packed[valid_edge_packed[..., 0]]
        node2_color = colors_packed[valid_edge_packed[..., 1]]

        node1 = node1_color == self.boundary_color_map['back_ground']
        node2 = node2_color == self.boundary_color_map['back_ground']
        node_colors = torch.cat([node1_color,node2_color], dim =1)
        # NOTE there not exists the two different boundary edge
        boundary_weight = torch.cat([torch.logical_not(node1),torch.logical_not(node2)], dim=1).float()
        boundary_weight = boundary_weight / boundary_weight.sum(1, keepdim = True)
        node_colors = (node_colors * (boundary_weight)).sum(1, keepdim = True).to(colors_packed)
        back_indx = torch.logical_and(node1,node2)
        new_colors = torch.zeros((e_v.shape[0],1), dtype=colors_packed.dtype)
        new_colors[back_indx] =self.boundary_color_map['back_ground']


        new_colors[torch.logical_not(back_indx)] = node_colors[torch.logical_not(back_indx)]
        assert not torch.isnan(new_colors.sum())
        new_colors_packed = torch.cat([colors_packed, new_colors], dim = 0)
        new_vertices_packed = torch.cat([vertices_packed, e_v], dim = 0)


        # origin edge


        new_garment_mesh = Garment_Mesh(new_vertices_packed[None], new_faces_packed[None], new_colors_packed[None], self.color_decode_fun, self.boundary_color_map, garment_type = self.garment_type, static_pts_idx= self.static_pts_idx)


        return new_garment_mesh



    def dense_pcl(self,type='edge'):
        '''
        dense mesh by edge method
        '''
        if type =='edge':
            return self.__edge_dense_pcl()
        else:
            raise NotImplemented
    def __edge_dense_pcl(self):
        '''
        clound point dense method
        edge dense_method
        '''
        faces_packed = self.garment.faces_packed()
        vertices_packed = self.garment.verts_packed()
        edge_packed = self.garment.edges_packed()
        colors_packed = self.colors.view(-1,1)

        e_v = (vertices_packed[edge_packed]).mean(1)
        face_edges = self.garment.faces_packed_to_edges_packed() + vertices_packed.shape[0]

        node1_color = colors_packed[edge_packed[..., 0]]
        node2_color = colors_packed[edge_packed[..., 1]]
        new_vertices_packed = torch.cat([vertices_packed, e_v], dim = 0)

        # need to guarantee clock order
        # NOTE that pytorch3d edge is order by r-b-l


        small_face0 = torch.cat([face_edges[...,0:1], faces_packed[...,2:3], face_edges[...,1:2]],dim=-1)
        small_face1 = torch.cat([face_edges[...,1:2], faces_packed[...,0:1], face_edges[...,2:3]],dim=-1)
        small_face2 = torch.cat([face_edges[...,2:3], faces_packed[...,1:2], face_edges[...,0:1]],dim=-1)
        new_small_faces = torch.cat([small_face0,small_face1,small_face2], dim =0)
        new_faces_packed = torch.cat([new_small_faces, face_edges], dim = 0)

        # Color remesh
        node1 = node1_color == self.boundary_color_map['back_ground']
        node2 = node2_color == self.boundary_color_map['back_ground']
        node_colors = torch.cat([node1_color,node2_color], dim =1)
        # NOTE there not exists the two different boundary edge
        boundary_weight = torch.cat([torch.logical_not(node1),torch.logical_not(node2)], dim=1).float()
        boundary_weight = boundary_weight / boundary_weight.sum(1, keepdim = True)
        node_colors = (node_colors * (boundary_weight)).sum(1, keepdim = True).to(colors_packed)
        back_indx = torch.logical_and(node1,node2)
        new_colors = torch.zeros((e_v.shape[0],1), dtype=colors_packed.dtype)
        new_colors[back_indx] =self.boundary_color_map['back_ground']


        new_colors[torch.logical_not(back_indx)] = node_colors[torch.logical_not(back_indx)]
        assert not torch.isnan(new_colors.sum())
        new_colors_packed = torch.cat([colors_packed, new_colors], dim = 0)

        new_garment_mesh = Garment_Mesh(new_vertices_packed[None], new_faces_packed[None], new_colors_packed[None], self.color_decode_fun, self.boundary_color_map, garment_type = self.garment_type,
                static_pts_idx = self.static_pts_idx)
        return new_garment_mesh


    @property
    def vertices(self):
        return self.garment.verts_packed()

    @property
    def faces(self):
        return self.garment.faces_packed()

    def update_padded(self, new_verts):
        self.garment =  self.garment.update_padded(new_verts)

    def __len__(self):
        return len(self.garment)

    def verts_normals_packed(self):
        return self.garment.verts_normals_packed()

    def edges_packed(self):
        return self.garment.edges_packed()


    def laplacian_packed(self):
        """Docstring for Laplacian_Loss.

        Returns: TODO

        """
        return self.garment.laplacian_packed()
    def transform_R_t(self, R,t):
        ''''transform function:
        Parameters
            R:[3,3],
            t:[1,3]
        '''

        device = R.device

        transform_matrix = torch.eye(4).to(device)
        transform_matrix[:3,:3] = R
        transform_matrix[:3,3:] = t.T
        self.garment = pytorch3d_mesh_transformation(self.garment,transform_matrix)



class Garment_Polygons(_Base_Garment):
    """Docstring for Garment Polygn using to represent featureline"""
    def __init__(self, **kwargs):
        """TODO: to be defined. """
        super(Garment_Polygons, self).__init__()
        self.boundary_fields = dict()

        for key,val in kwargs.items():
            self.boundary_fields[key] = Pointclouds(val[None])


    def get_fields(self):
        return self.boundary_fields.keys()

    def get_boundary(self, *args):
        ''' obtain boudnary_mesh.
        generally speaking, we do not considering the faces, as the bounary is polygons
        '''
        ret = []
        for key in args:
            boundary = (self.boundary_fields[key]).points_packed()
            ret.append(boundary)
        return ret

    def save_boundary_ply(self, save_path, seq_id):
        save_path.mkdir(parents =True, exist_ok = True)
        for key in self.get_fields():
            boundary = self.boundary_fields[key].points_packed()
            save_name = (str(save_path)+'/{}_{}.ply'.format(key, seq_id))
            save_ply(save_name, boundary)


    @property
    def vertices(self):
        ret = []
        for key in self.get_fileds():
            ret = self.boundary_fields[key].points_packed()

        return torch.cat(ret, dim = 0)




    def __len__(self):
        return len(self.garment)

