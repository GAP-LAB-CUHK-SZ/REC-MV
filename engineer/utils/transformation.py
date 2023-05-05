import torch
import trimesh
import math
import pytorch3d

import torch.nn.functional as F

def Rotate_Y_axis(degree):
    return torch.from_numpy(trimesh.transformations.rotation_matrix(math.radians(degree), [0, 1, 0])).float()

def pytorch3d_mesh_transformation(mesh, matrix):
    new_mesh = mesh.clone()
    device = mesh.device
    matrix = matrix.to(device)
    return  pytorch3d_mesh_transformation_(new_mesh, matrix)

def pytorch3d_mesh_transformation_(mesh, matrix):
    """
    Add an offset to the vertices of this Meshes. In place operation.
    If normals are present they may be recalculated.

    Args:
        vert_offsets_packed: A Tensor of shape (3,) or the same shape as
                            self.verts_packed, giving offsets to be added
                            to all vertices.
    Returns:
        self.
    """
    verts_packed = mesh.verts_packed()
    update_normals = True

    homo_verts_packed = transform_vertices(matrix[None], verts_packed[None])
    verts_packed = homo_to_3d(homo_verts_packed)[0]


    mesh._verts_packed = verts_packed
    new_verts_list = list(
        mesh._verts_packed.split(mesh.num_verts_per_mesh().tolist(), 0)
    )
    # update verts list
    # Note that since _compute_packed() has been executed, verts_list
    # cannot be None even if not provided during construction.
    mesh._verts_list = new_verts_list

    # update verts padded
    if mesh._verts_padded is not None:
        for i, verts in enumerate(new_verts_list):
            if len(verts) > 0:
                mesh._verts_padded[i, : verts.shape[0], :] = verts

    # update face areas and normals and vertex normals
    # only if the original attributes are present
    if update_normals and any(
        v is not None
        for v in [mesh._faces_areas_packed, mesh._faces_normals_packed]
    ):
        mesh._compute_face_areas_normals(refresh=True)
    if update_normals and mesh._verts_normals_packed is not None:
        mesh._compute_vertex_normals(refresh=True)

    return mesh




def offset_verts(self, vert_offsets_packed):
    """
    Out of place offset_verts.

    Args:
        vert_offsets_packed: A Tensor of the same shape as self.verts_packed
            giving offsets to be added to all vertices.
    Returns:
        new Meshes object.
    """
    new_mesh = self.clone()
    return new_mesh.offset_verts_(vert_offsets_packed)



def transform_vertices(camera, pts):
    '''
    Input:
        camera: batch * 4*  4
        pts: batch* n *  3
    '''
    homo_pts = F.pad(pts,(0,1,0,0,0,0),'constant', 1.)

    return transform_homo_vertices(camera, homo_pts)

def transform_matrices_vertices(pts, *matrices):
    '''this function to compute a series of matrices transformation,

    '''
    homo_pts = F.pad(pts,(0,1,0,0,0,0),'constant', 1.)
    for matrix in matrices[::-1]:
        homo_pts = transform_homo_vertices(matrix, homo_pts)
    return homo_pts

def transform_homo_vertices(camera, homo_pts):

    return (camera @ homo_pts.transpose(2,1)).transpose(2,1)

def batch_transform_vertices(camera, pts):
    homo_pts = F.pad(pts,(0,1,0,0,0,0),'constant', 1.)


    return batch_transform_homo_vertices(camera, homo_pts)

def batch_transform_matrices_vertices(pts, *matrices):
    '''this function to compute a series of matrices transformation,

    '''
    if pts.shape[-1] == 3:
        homo_pts = F.pad(pts,(0,1,0,0,0,0),'constant', 1.)
    else:
        homo_pts =pts
    assert homo_pts.shape[-1] == 4, 'the input dim of coordinate need 3 or 4 dimention'
    for matrix in matrices[::-1]:
        homo_pts = batch_transform_homo_vertices(matrix, homo_pts)
    return homo_pts

def batch_transform_homo_vertices(camera, homo_pts):
    return torch.einsum('bhw,bnw->bnh', camera, homo_pts)

def batch_rotate_normal(camera, normals):
    R = camera[..., :3, :3]
    return torch.einsum('bhw,bnw->bnh', R, normals)

def homo_to_3d(vertices):
    vertices[...,:3] /= (vertices[...,3:].clone())
    return vertices[...,:3]





def batch_rodrigues(axisang):
    # This function is borrowed from https://github.com/MandyMo/pytorch_HMR/blob/master/src/util.py#L37
    # axisang N x 3
    axisang_norm = torch.norm(axisang + 1e-8, p=2, dim=1)
    angle = torch.unsqueeze(axisang_norm, -1)
    axisang_normalized = torch.div(axisang, angle)
    angle = angle * 0.5
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    quat = torch.cat([v_cos, v_sin * axisang_normalized], dim=1)
    rot_mat = quat2mat(quat)
    rot_mat = rot_mat.view(rot_mat.shape[0], 9)


    return rot_mat

def batch_axisang2quat(axisang):
    # This function is borrowed from https://github.com/MandyMo/pytorch_HMR/blob/master/src/util.py#L37
    # axisang N x 3
    axisang_norm = torch.norm(axisang + 1e-8, p=2, dim=1)
    angle = torch.unsqueeze(axisang_norm, -1)
    axisang_normalized = torch.div(axisang, angle)
    angle = angle * 0.5
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    quat = torch.cat([v_cos, v_sin * axisang_normalized], dim=1)
    return  quat

def quat2mat(quat):
    """
    This function is borrowed from https://github.com/MandyMo/pytorch_HMR/blob/master/src/util.py#L50

    Convert quaternion coefficients to rotation matrix.
    Args:
        quat: size = [batch_size, 4] 4 <===>(w, x, y, z)
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [batch_size, 3, 3]
    """
    norm_quat = quat
    norm_quat = norm_quat / norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:, 0], norm_quat[:, 1], norm_quat[:,
                                                             2], norm_quat[:,
                                                                           3]

    batch_size = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    rotMat = torch.stack([
        w2 + x2 - y2 - z2, 2 * xy - 2 * wz, 2 * wy + 2 * xz, 2 * wz + 2 * xy,
        w2 - x2 + y2 - z2, 2 * yz - 2 * wx, 2 * xz - 2 * wy, 2 * wx + 2 * yz,
        w2 - x2 - y2 + z2
    ],
                         dim=1).view(batch_size, 3, 3)
    return rotMat

def get_uv_camera(w, h):
    uv_camera = torch.eye(4).float()
    uv_camera[0, 0] = w //2
    uv_camera[1, 1] = h  //2
    uv_camera[0, 3] = w //2
    uv_camera[1, 3] = h //2

    return uv_camera


def Rotate_X_axis(degree):

    return torch.from_numpy(trimesh.transformations.rotation_matrix(math.radians(degree), [1, 0, 0])).float()


def flip_Y_axis():

    y_flip = torch.eye(4).float()
    y_flip[1,1] = -1

    return y_flip



def rotate_normal(camera,normals):
    R = camera[:3,:3]
    return (R @ normals.T).T