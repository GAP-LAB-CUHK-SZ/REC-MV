import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from engineer.utils.mesh_loss import Project_Loss, Batch_Project_Loss, Laplacian_Loss, Edge_Loss, batch_EMD_loss
from einops import repeat
import pytorch3d
from pytorch_rendering.camera import set_orthogonal_camera, get_project_matrix, get_ndc_to_screen_transform
from pytorch3d.structures import join_meshes_as_batch
import os
from engineer.utils.visualization import (
    visualized_project_verts,
    save_mesh,
    save_ply,
    save_mesh_transform,
    save_mesh_face_transform,
    visualized_RT_img,
    visualized_center_RT_img,
    visualized_deforms,
    visualized_canonical_feature_lines,
)
from collections import defaultdict
from torch.autograd import Variable
import logging
from engineer.utils.matrix_transform import compute_rotation_matrix_from_ortho6d, compute_rotation_matrix_from_quaternion
from pytorch3d.structures import Meshes
logger = logging.getLogger('logger.trainer')
from engineer.utils.transformation import batch_transform_vertices, batch_rotate_normal,batch_transform_matrices_vertices, homo_to_3d
import cv2
from utils.constant import RAY_DIRS, FL_IDX
from utils.constant import Z_RAY
import utils.optimizer as optim
from pytorch3d.structures import Pointclouds, Meshes, join_meshes_as_batch
from engineer.utils.smpl_utils import visible_z_buff
import torch
from utils.logger import get_meta_parameters
from engineer.utils.data_util import collect_meta_data
torch.autograd.set_detect_anomaly(True)


def update_mesh(meshes, deforms):

    new_meshes = [mesh.offset_verts(deform_verts) for mesh, deform_verts in zip(meshes,deforms)]
    return join_meshes_as_batch(new_meshes)

def icp_rotate_transfrom(meshes, R_pack, T_Pack):
    def transform(v, R, T):
        return (R @ v.T).T + T

    if isinstance(meshes[0], pytorch3d.structures.Meshes):
        vertices = [mesh.verts_packed() for mesh in meshes]
    else:
        vertices = meshes

    transformed_vertices = [transform(vert, R, T) for vert, R, T in zip(vertices, R_pack, T_Pack)]

    return transformed_vertices


def project_points(pts,proj_matrix):
    homo = torch.ones((pts.shape[0],1)).float().cuda()
    homo_pts = torch.cat([pts,homo], dim = 1)
    proj_pts = (proj_matrix @ homo_pts.T).T


    return proj_pts

def visible_or_not(gt_feature_lines, visible_threshold = 0.2):

    visibles = []

    for gt_feature_line in gt_feature_lines:
        if gt_feature_line.sum()/ gt_feature_line.shape[0] < visible_threshold:
            visibles.append([False, 0.])
        else:
            visibles.append([True, torch.clamp(gt_feature_line.sum()/ gt_feature_line.shape[0] / 0.5, max = 1.)])

    return visibles


def rotate_normal(camera,normals):
    R = camera[:3,:3]
    return (R @ normals.T).T


def split_rotate_from_global_R(rigid_R, feature_line_normal, pts_indx):


    new_feature_line_normals = []
    feature_lines = torch.split(feature_line_normal, pts_indx)
    global_R = rigid_R.detach().clone().cpu()
    for R, feature_line in zip(global_R, feature_lines):
        new_feature_line_normals.append((R @ feature_line.T).T)
    return torch.cat(new_feature_line_normals, dim=0)



def rigid_optimizer(smpl_model, train_data_dataloder, test_dataloader, template_catogories, save_path ,cfg):
    '''
    optimize global R,T given feature line with project heatmap
    '''
    # begin at rigid optimizer
    logger.info('begining at rigid optimizer ')
    if cfg.rigid_R == 'o6':
        initial_pose = torch.full((len(template_catogories),6 ), 0.0, device='cuda:0', requires_grad= False)
        initial_pose[:,0]  = 1.
        initial_pose[:,4] = 1.
    else:
        raise NotImplementedError


    meta_name = get_meta_parameters(cfg.meta_hyper)
    #update_save_path by meta_name
    save_path = os.path.join(save_path, meta_name)
    os.makedirs(save_path, exist_ok= True)


    rigid_pose =  Variable(initial_pose, requires_grad = False)
    rigid_T = torch.full((len(template_catogories),1 ,3), 0.0, device='cuda:0', requires_grad=True)

    # optimizier setting
    rigid_T_optimizer = cfg.optim_para['rigid_T_optimizer']


    if 'SGD' == rigid_T_optimizer['type']:
        rigid_T_optimizer = torch.optim.SGD([rigid_T], lr= rigid_T_optimizer['lr'], momentum= rigid_T_optimizer['momentum'])
    elif 'Adam' == rigid_T_optimizer['type']:
        rigid_T_optimizer = torch.optim.Adam([rigid_T], lr= rigid_T_optimizer['lr'])

    rigid_template_catogories = ['rigid_' + temp for temp in template_catogories]

    all_save_mesh_path =  os.path.join(save_path, 'mesh_exp')
    save_mesh_path = os.path.join(save_path, 'rigid_mesh_exp')
    save_img_path = os.path.join(save_path, 'proj_img')

    os.makedirs(save_mesh_path, exist_ok= True)
    os.makedirs(save_img_path, exist_ok= True)

    meta_loss = defaultdict(list)
    loss_weight = cfg.loss_weight

    logger.info("training global rigid_T!")
    T_epoch = 0
    # optimizer_t
    for T_epoch in range(cfg.rigid_optimizer_epoch):
        # visualizaion

        # visualized_RT_img(smpl_model, test_dataloader, rigid_T, rigid_pose, save_img_path, epoch = T_epoch, all_epoch = False)
        # NOTE that the following is meta parameters,
        # meta = ['name','polygons','image','world2ndc','ndc2screen','smpl_pose', 'smpl_beta', 'smpl_trans', 'polygons_mask', 'z_buff', 'valid_polygons']
        for batch_id, batch in enumerate(train_data_dataloder):

            rigid_T_optimizer.zero_grad()

            image_names, image_batch, world2ndc, ndc2screen, trans_scale_matrix, smpl_pose, smpl_trans, gt_polygons, valid_gt_polygons, gt_polygons_mask, z_buff = collect_meta_data(batch)
            rigid_R = compute_rotation_matrix_from_ortho6d(rigid_pose)

            # TODO Z-buff infos
            #smpl_model.featureline_z_buff(smpl_pose, z_buff, world_to_ndc)

            meshes, meshes_vertices, meshes_normals = smpl_model.smpl_feature_line_rigid_transform(rigid_R, rigid_T)
            pose_xyz, pose_normals = smpl_model.forward_skinning(meshes_vertices, meshes_normals, smpl_pose, trans = smpl_trans)
            pose_indx = [xyz.shape[1] for xyz in pose_xyz]
            pose_xyz = torch.cat(pose_xyz, dim = 1)
            pose_normals = torch.cat(pose_normals, dim = 1)
            uv_xyz = batch_transform_matrices_vertices(pose_xyz, world2ndc, trans_scale_matrix)
            uv_xyz = homo_to_3d(uv_xyz)
            front_xyz = visible_z_buff(z_buff, uv_xyz)

            uv_xyz = torch.split(uv_xyz, pose_indx, dim=1)
            front_xyz = torch.split(front_xyz, pose_indx, dim=1)
            loss = 0.
            loss_dict = {}

            if cfg.meta_hyper.EMD:
                loss_dict.update(batch_EMD_loss(uv_xyz, gt_polygons, FL_IDX, gt_polygons_mask, valid_gt_polygons))
            else:
                loss_dict.update(Batch_Project_Loss(uv_xyz, gt_polygons, front_xyz, FL_IDX, gt_polygons_mask, valid_gt_polygons))
            for key in loss_dict.keys():
                loss += loss_weight[key.split('_')[0]] * loss_dict[key]
                meta_loss[key].append(loss_weight[key.split('_')[0]] * loss_dict[key].item())


            loss.backward()
            rigid_T_optimizer.step()
            logger.info("rigid_T Loss:{:.4f}".format(loss.item()))

    logger.info("T is already !")
    T_epoch+=1
    visualized_RT_img(smpl_model, test_dataloader, rigid_T, rigid_pose, save_img_path, epoch = T_epoch, all_epoch= False)

    # begin to optimize global rigid_R
    logger.info("training global rigid_R!")
    rigid_T.requires_grad = False
    rigid_R = compute_rotation_matrix_from_ortho6d(rigid_pose)
    # meshes_faces = [mesh.faces_packed() for mesh in meshes]
    # static_poses = icp_rotate_transfrom(meshes, rigid_R, rigid_T)

    # update featureline to postion after rigid_T
    # NOTE the initial rotation matrix is identity
    smpl_model.update_feature_line_by_RT(rigid_R, rigid_T)
    # compute each feature_line center, prepare to rigid_R fitting

    rigid_pose =  Variable(initial_pose.clone(), requires_grad = True)

    # zero transformation
    rigid_R_T = torch.full((len(template_catogories),1 ,3), 0.0, device='cuda:0', requires_grad = False)

    # optimizier setting
    rigid_R_optimizer = cfg.optim_para['rigid_R_optimizer']

    if 'SGD' == rigid_R_optimizer['type']:
        rigid_R_optimizer = torch.optim.SGD([rigid_pose], lr= rigid_R_optimizer['lr'], momentum= rigid_R_optimizer['momentum'])
    elif 'Adam' == rigid_R_optimizer['type']:
        rigid_R_optimizer = torch.optim.Adam([rigid_pose], lr= rigid_R_optimizer['lr'])


    for R_epoch in range(cfg.rigid_optimizer_epoch):
        for batch_id, batch in enumerate(train_data_dataloder):
            loss = 0.
            rigid_R_optimizer.zero_grad()

            image_names, image_batch, world2ndc, ndc2screen, trans_scale_matrix, smpl_pose, smpl_trans, gt_polygons, valid_gt_polygons, gt_polygons_mask, z_buff = collect_meta_data(batch)
            rigid_R = compute_rotation_matrix_from_ortho6d(rigid_pose)
            # center rotated
            meshes, meshes_vertices, meshes_normals = smpl_model.smpl_feature_line_center_transform(rigid_R, rigid_R_T)
            pose_xyz, pose_normals = smpl_model.forward_skinning(meshes_vertices, meshes_normals, smpl_pose, trans = smpl_trans)


            pose_indx = [xyz.shape[1] for xyz in pose_xyz]
            pose_xyz = torch.cat(pose_xyz, dim = 1)
            pose_normals = torch.cat(pose_normals, dim = 1)
            uv_xyz = batch_transform_matrices_vertices(pose_xyz, world2ndc, trans_scale_matrix)
            uv_xyz = homo_to_3d(uv_xyz)
            front_xyz = visible_z_buff(z_buff, uv_xyz)

            uv_xyz = torch.split(uv_xyz, pose_indx, dim=1)
            front_xyz = torch.split(front_xyz, pose_indx, dim=1)

            loss = 0.
            loss_dict = {}

            if cfg.meta_hyper.EMD:
                loss_dict.update(batch_EMD_loss(uv_xyz, gt_polygons, FL_IDX, gt_polygons_mask, valid_gt_polygons))
            else:
                loss_dict.update(Batch_Project_Loss(uv_xyz, gt_polygons, front_xyz, FL_IDX, gt_polygons_mask, valid_gt_polygons))

            for key in loss_dict.keys():
                loss += loss_weight[key.split('_')[0]] * loss_dict[key]
                meta_loss[key].append(loss_dict[key].item())

            loss.backward()
            rigid_R_optimizer.step()
            logger.info("rigid_R Loss:{:.4f}".format(loss.item()))

        # NOTE that visualize the rigid pose after training as the initial pose that is the end of T trans
        visualized_center_RT_img(smpl_model, test_dataloader, rigid_R_T, rigid_pose, save_img_path, epoch = R_epoch+T_epoch+1, all_epoch = False)

    logger.info("R is already !")
    smpl_model.update_feature_line_by_center_RT(rigid_R, rigid_R_T)

    #save meta_loss
    torch.save(meta_loss, os.path.join(save_path,'meta_RT.pth'))

    return R_epoch + T_epoch + 1



def vertices_optimizer(smpl_model, train_data_dataloder, test_dataloader, template_catogories, device, global_epoch, save_path ,cfg):
    '''this code use to leran vertice deformation by the loss
    '''

    print(global_epoch)

    meta_name = get_meta_parameters(cfg.meta_hyper)
    #update_save_path by meta_name
    save_path = os.path.join(save_path, meta_name)

    # update this files save the fitting results to vertices optimizer
    save_path = os.path.join(save_path, 'vertice_deform')
    os.makedirs(save_path, exist_ok= True)

    loss_weight = cfg.loss_weight
    meshes, meshes_normals = smpl_model.feature_line_to_meshes(device)
    # define learnable parameters
    deform_verts = [torch.full(mesh.verts_packed().shape, 0.0, device = device, requires_grad = True) for mesh in meshes]
    optimizer = torch.optim.Adam(deform_verts, lr= cfg.LR)

    meta_loss = defaultdict(list)


    save_img_path = os.path.join(save_path, 'deform_verts')
    save_last_epoch_path = os.path.join(save_path,'final_deform_verts')
    os.makedirs(save_img_path, exist_ok = True)
    os.makedirs(save_last_epoch_path, exist_ok = True)



    for epoch in range(cfg.num_epoch):
        # leraning rate update
        optim.adjust_learning_rate(optimizer, epoch, cfg)
        visualized_canonical_feature_lines(smpl_model, deform_verts, save_path, epoch)

        if epoch % 5 ==0:
            visualized_deforms(smpl_model, test_dataloader, deform_verts , save_img_path, epoch = global_epoch+epoch+1, device = device, all_epoch =False)


        for batch_id, batch in enumerate(train_data_dataloder):
            loss = 0.
            optimizer.zero_grad()

            image_names, image_batch, world2ndc, ndc2screen, trans_scale_matrix, smpl_pose, smpl_trans, gt_polygons, valid_gt_polygons, gt_polygons_mask, z_buff = collect_meta_data(batch)

            # move verts
            feature_line_meshes = update_mesh(meshes,deform_verts)
            meshes_vertices = [mesh.verts_packed() for mesh in feature_line_meshes]

            pose_xyz, pose_normals = smpl_model.forward_skinning(meshes_vertices, meshes_normals, smpl_pose, trans = smpl_trans)
            pose_indx = [xyz.shape[1] for xyz in pose_xyz]
            pose_xyz = torch.cat(pose_xyz, dim = 1)
            pose_normals = torch.cat(pose_normals, dim = 1)

            uv_xyz = batch_transform_matrices_vertices(pose_xyz, world2ndc, trans_scale_matrix)
            uv_xyz = homo_to_3d(uv_xyz)
            front_xyz = visible_z_buff(z_buff, uv_xyz)

            uv_xyz = torch.split(uv_xyz, pose_indx, dim=1)
            front_xyz = torch.split(front_xyz, pose_indx, dim=1)

            loss = 0.
            loss_dict = {}


            if cfg.meta_hyper.EMD:
                loss_dict.update(batch_EMD_loss(uv_xyz, gt_polygons, FL_IDX, gt_polygons_mask, valid_gt_polygons))
            else:
                loss_dict.update(Batch_Project_Loss(uv_xyz, gt_polygons, front_xyz, FL_IDX, gt_polygons_mask, valid_gt_polygons))



            mesh_loss = Laplacian_Loss(feature_line_meshes)
            # edge loss
            edge_loss = Edge_Loss(feature_line_meshes)
            loss_dict.update(mesh_loss)
            loss_dict.update(edge_loss)






            for key in loss_dict.keys():
                loss += loss_weight[key.split('_')[0]] * loss_dict[key]
                meta_loss[key].append(loss_weight[key.split('_')[0]] * loss_dict[key].item())



            loss.backward()

            optimizer.step()
            logger.info("epoch{:03d},vertice Loss:{:.4f}".format(epoch, loss.item()))

    visualized_deforms(smpl_model, test_dataloader, deform_verts , save_last_epoch_path, epoch = global_epoch+epoch+1, device = device)


    torch.save(meta_loss, os.path.join(save_path,'meta_vertices.pth'))
