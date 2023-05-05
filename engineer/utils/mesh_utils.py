import numpy as np
from utils.common_utils import numpy2tensor
from engineer.utils.garment_structure import Garment_Mesh
import trimesh
from pytorch3d.structures import Meshes
from einops import repeat
import torch
import pdb
def slice_garment_mesh(vertices_id, faces_id, smpl_verts, vertices_colors, decode_color, boundary_color_map, garment_type, static_pts_idx = None) :
    """the funciton to arrange the vertices id -> start from 0

    @vertices_id: TODO
    @faces_id: TODO

    Returns: TODO

    """

    slice_smpl_faces = reorder_face(vertices_id, faces_id).astype(np.int32)
    slice_smpl_faces = numpy2tensor(slice_smpl_faces).unsqueeze(0)


    if  static_pts_idx is not None:
        static_set = set(static_pts_idx.tolist())
        vertices_set = set(vertices_id)
        # need static belongs to meshes
        if not len(static_set & vertices_set) == len(static_set):
            static_pts_idx = None
        else:
            static_pts_idx = static_pts_idx.tolist()
            static_pts_idx = reorder_verts(vertices_id, static_pts_idx)




    return Garment_Mesh(smpl_verts, slice_smpl_faces, vertices_colors, decode_color, boundary_color_map = boundary_color_map, garment_type= garment_type, static_pts_idx = static_pts_idx )

def reorder_face(vertices_id, faces_id):
    new_faces_id = np.zeros_like(faces_id)
    vert_id ={}
    for new_id, ver_id in enumerate(vertices_id):
        vert_id[ver_id] = new_id
    for i in range(faces_id.shape[0]):
        for j in range(faces_id.shape[1]):
            new_faces_id[i][j] = vert_id[faces_id[i][j]]

    return new_faces_id


def reorder_verts(vertices_id, vid_list):
    vert_id = {}
    for new_id, ver_id in enumerate(vertices_id):
        vert_id[ver_id] = new_id
    ret = []
    for vid in  vid_list:
        ret.append(vert_id[vid])

    return ret


def merge_meshes(verts, faces):

    if faces == None:
        return verts

    verts_container = []
    faces_container = []

    vert_cnt = 0
    for vert, face in zip(verts, faces):
        face = face.clone()
        face+=vert_cnt


        if len(face.shape) != len(vert.shape):
            batch_size = vert.shape[0]
            face = repeat(face, 'n c -> b n c', b = batch_size)

        verts_container.append(vert)
        faces_container.append(face)
        vert_cnt += vert.shape[1]

    verts = torch.cat(verts_container, dim = 1)
    faces = torch.cat(faces_container, dim = 1)

    return Meshes(verts, faces)

def mesh_boundary(in_faces: torch.LongTensor, num_verts: int):
    '''
    input:
        in edges: N * 3, is the vertex index of each face, where N is number of faces
        num_verts: the number of vertexs mesh
    return:
        boundary_mask: bool tensor of num_verts, if true, point is on the boundary, else not
    '''
    in_x = in_faces[:, 0]
    in_y = in_faces[:, 1]
    in_z = in_faces[:, 2]
    in_xy = in_x * (num_verts) + in_y
    in_yx = in_y * (num_verts) + in_x
    in_xz = in_x * (num_verts) + in_z
    in_zx = in_z * (num_verts) + in_x
    in_yz = in_y * (num_verts) + in_z
    in_zy = in_z * (num_verts) + in_y
    in_xy_hash = torch.minimum(in_xy, in_yx)
    in_xz_hash = torch.minimum(in_xz, in_zx)
    in_yz_hash = torch.minimum(in_yz, in_zy)
    in_hash = torch.cat((in_xy_hash, in_xz_hash, in_yz_hash), dim = 0)
    output, count = torch.unique(in_hash, return_counts = True, dim = 0)
    boundary_edge = output[count == 1]
    boundary_vert1 = boundary_edge // num_verts
    boundary_vert2 = boundary_edge % num_verts
    boundary_mask = torch.zeros(num_verts).bool().to(in_faces.device)
    boundary_mask[boundary_vert1] = True
    boundary_mask[boundary_vert2] = True
    return boundary_mask
