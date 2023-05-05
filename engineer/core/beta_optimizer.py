"""
@File: beta_optimizer.
@Author: Lingteng Qiu
@Email: qiulingteng@link.cuhk.edu.cn
@Date: 2023-02-01
@Desc: optimizing beta parameters since the results estimated by the video-avater are not very correct
"""
# THE ORDER of COCOPLUS
#  0: RAnkle
#  1: RKnee
#  2: RHip
#  3: LHip
#  4: LKnee
#  5: LAnkle
#  6: RWrist
#  7: RElbow
#  8: RShoulder
#  9: LShoulder
# 10: LElbow
# 11: LWrist
# 12: Neck
# 13: Head (top of)
# 14: Nose
# 15: LEye
# 16: REye
# 17: LEar
# 18: REar


# THE ORDER of COCO
# 0: “nose”,
# 1: “left_eye”,
# 2: “right_eye”,
# 3: “left_ear”,
# 4: “right_ear”,
# 5: “left_shoulder”,
# 6: “right_shoulder”,
# 7: “left_elbow”,
# 8: “right_elbow”,
# 9: “left_wrist”,
# 10: “right_wrist”,
# 11: “left_hip”,
# 12: “right_hip”,
# 13: “left_knee”,
# 14: “right_knee”,
# 15: “left_ankle”,
# 16: “right_ankle”
import torch
from torch.autograd import Variable
import os
import glob
from model.CameraMine import PointsRendererWithFrags,RectifiedPerspectiveCameras
from pytorch3d.structures import Meshes
from engineer.utils.matrix_transform import compute_rotation_matrix_from_ortho6d, icp_rotate_transfrom, center_transform, icp_rotate_center_transform, scale_icp_rotate_transfrom, scale_icp_rotate_center_transform
from utils.common_utils import tocuda
from einops import repeat
import cv2
import numpy as np
from pytorch3d.loss import chamfer_distance
import pdb
from smpl_pytorch.SMPL import SMPL,getSMPL
from pytorch3d.io import save_obj
from einops import repeat

COCOPLUS2COCO = [
    14, 15 ,16 , 17, 18 , 9, 8, 10, 7, 11, 6, 3, 2, 4, 1, 5, 0
        ]

def batch_kp_2d_l1_loss(real_2d_kp, predict_2d_kp):
    kp_gt = real_2d_kp.view(-1, 3)
    kp_pred = predict_2d_kp.contiguous().view(-1, 2)


    vis = kp_gt[:, 2]
    k = torch.sum(vis) * 2.0 + 1e-8
    dif_abs = torch.abs(kp_gt[:, :2] - kp_pred).sum(1)


    return torch.matmul(dif_abs, vis) * 1.0 / k

class RandomSampler(torch.utils.data.Sampler):
    def __init__(self,data_source,intersect,shuffle):
        self.length=len(data_source)
        self.intersect=intersect

        self.shuffle=shuffle
        self.n=(self.length-1)//self.intersect+1
        self.start=self.length-self.intersect*(self.n-1)

    def __iter__(self):
        # return iter([0,55,85,595,380]*48)
        # return iter([55]*240)
        if self.shuffle:
            start=random.sample(list(range(0,self.start)),1)[0]
            index=torch.arange(start,self.length,self.intersect)
            index=index[torch.randperm(self.n)]
        else:
            index=torch.arange(0,self.length,self.intersect)
        assert(index.numel()==self.n)
        return iter(index.view(-1).tolist())
    def __len__(self):
        return self.n



def update_feature_line_mesh(fl_meshes, rigid_R, rigid_T):
    meshes_vertices = icp_rotate_transfrom(fl_meshes, rigid_R, rigid_T)

    return [mesh.update_padded(mesh_vertices[None]) for mesh, mesh_vertices in zip(fl_meshes, meshes_vertices)]

def update_scale_feature_line_mesh(fl_meshes, rigid_R, rigid_T, rigid_scale):
    meshes_vertices = scale_icp_rotate_transfrom(fl_meshes, rigid_R, rigid_T, rigid_scale)

    return [mesh.update_padded(mesh_vertices[None]) for mesh, mesh_vertices in zip(fl_meshes, meshes_vertices)]


def fl_proj_loss(fl_pts_list,gt_fl_pts_list, fl_masks):

    loss = 0.
    for fl_pts, gt_fl_pts, fl_mask in zip(fl_pts_list, gt_fl_pts_list, fl_masks):
        screen_fl_pts = fl_pts[...,:2]
        screen_fl_mask = fl_mask[...,:2]
        screen_fl_pts = screen_fl_pts * screen_fl_mask
        cham_loss, __ = chamfer_distance(screen_fl_pts, gt_fl_pts, point_reduction = 'sum')

        if screen_fl_mask.sum() !=0.:
            loss+=cham_loss / (screen_fl_mask.sum() //2)
        else:
            loss +=0.
    return loss / len(fl_pts_list)

def smpl_beta_optimizer(gender, initPose, dataset, device = 'cuda:0'):
    '''
    initialzie beta parameters due to videoavatar is not align with size
    '''
    # begin at rigid optimizer

    smpl=getSMPL(gender).to(device)
    betas = dataset.shape.to(device).clone()
    betas.requires_grad = True
    extra_trans = torch.full((1 ,3), 0., device='cuda:0', requires_grad=True)


    beta_optimizer = torch.optim.Adam([betas, extra_trans], lr= 0.005, weight_decay= 0.)


    sampler = RandomSampler(dataset,1,True)
    dataloader=torch.utils.data.DataLoader(dataset, 8 ,sampler,num_workers=8)

    iter_cnt = 0



    beta_epoch = 150 // len(dataloader)


    for epoch in range(beta_epoch):
        for frame_ids, batch in dataloader:

            imgs = batch['img']
            batch_gt_joints2d = batch['gt_joints2d'].to(device)
            poses,trans,d_cond,rendcond= dataset.get_grad_parameters(frame_ids,device)

            trans+=extra_trans

            batch_size = poses.shape[0]
            focals,princeple_ps,Rs,Ts,H,W= dataset.get_camera_parameters(frame_ids.numel(),device)
            proj_img_size = repeat(torch.Tensor([W,H]).float(), 'c -> b c', b= batch_size).to(device)
            cameras=RectifiedPerspectiveCameras(focals,princeple_ps,Rs,Ts,image_size=[(W, H)]).to(device)
            batch_betas = betas[None].expand([batch_size, -1])

            verts, _, __ = smpl(batch_betas, poses, True)
            verts = verts + trans.view(-1,1,3)

            # smpl2 cocoplus
            joint_x = torch.matmul(verts[:, :, 0], smpl.joint_regressor)
            joint_y = torch.matmul(verts[:, :, 1], smpl.joint_regressor)
            joint_z = torch.matmul(verts[:, :, 2], smpl.joint_regressor)
            joints = torch.stack([joint_x, joint_y, joint_z], dim = 2)

            batch_joints_pts = cameras.transform_points_screen(joints, proj_img_size)
            batch_joints_pts = batch_joints_pts[:, COCOPLUS2COCO, :]

            beta_optimizer.zero_grad()
            loss = batch_kp_2d_l1_loss(batch_gt_joints2d, batch_joints_pts[..., :2])
            print("iteration step {:04d}: {:.4f}".format(iter_cnt, loss.item()))
            loss.backward()
            beta_optimizer.step()
            iter_cnt+=1

            # for ind, (img, screen_pts) in enumerate(zip(imgs, batch_verts_pts)):
            #     for joint in screen_pts.detach().cpu().numpy().astype(np.int):
            #         img = cv2.circle(img, (int(joint[0]), int(joint[1])), 2, (0, 255, 0), 1)

            # for ind, (img, joints_pts, gt_joints2d) in enumerate(zip(imgs, batch_joints_pts, batch_gt_joints2d)):
            #     cnt = 0
            #     for joint, gt_joint in zip(joints_pts.detach().cpu().numpy().astype(np.int), gt_joints2d.detach().cpu().numpy().astype(np.int)):


            #         img = cv2.circle(img, (int(joint[0]), int(joint[1])), 2, (0, 0, 255), 1)
            #         img = cv2.circle(img, (int(gt_joint[0]), int(gt_joint[1])), 2, (255, 0, 0), 1)
            #         cv2.imwrite('./debug/smpl_beta/{:04d}.png'.format(cnt), img)
            #         cnt+=1

    dataloader=torch.utils.data.DataLoader(dataset, 1 ,sampler,num_workers=8)
    cnt=0
    os.makedirs('./debug/smpl_beta/imgs/', exist_ok = True)
    for frame_ids, batch in dataloader:
        imgs = batch['img']
        batch_gt_joints2d = batch['gt_joints2d'].to(device)

        poses,trans,d_cond,rendcond= dataset.get_grad_parameters(frame_ids,device)
        trans+=extra_trans

        batch_size = poses.shape[0]
        focals,princeple_ps,Rs,Ts,H,W= dataset.get_camera_parameters(frame_ids.numel(),device)
        proj_img_size = repeat(torch.Tensor([W,H]).float(), 'c -> b c', b= batch_size).to(device)
        cameras=RectifiedPerspectiveCameras(focals,princeple_ps,Rs,Ts,image_size=[(W, H)]).to(device)
        batch_betas = betas[None].expand([batch_size, -1])

        verts, _, __ = smpl(batch_betas, poses, True)
        verts = verts + trans.view(-1,1,3)

        # smpl2 cocoplus
        joint_x = torch.matmul(verts[:, :, 0], smpl.joint_regressor)
        joint_y = torch.matmul(verts[:, :, 1], smpl.joint_regressor)
        joint_z = torch.matmul(verts[:, :, 2], smpl.joint_regressor)
        joints = torch.stack([joint_x, joint_y, joint_z], dim = 2)

        batch_joints_pts = cameras.transform_points_screen(joints, proj_img_size)
        batch_joints_pts = batch_joints_pts[:, COCOPLUS2COCO, :]
        batch_verts_pts = cameras.transform_points_screen(verts, proj_img_size)
        imgs =((imgs / 2. + 0.5) *255.).cpu().numpy().astype(np.uint8)

        for ind, (img, screen_pts) in enumerate(zip(imgs, batch_verts_pts)):
            for joint in screen_pts.detach().cpu().numpy().astype(np.int):
                img = cv2.circle(img, (int(joint[0]), int(joint[1])), 2, (0, 255, 0), 1)

        cv2.imwrite('./debug/smpl_beta/imgs/{:04d}.png'.format(frame_ids[0]), img)
        print('visualized dataset {:04d}!'.format(frame_ids[0]))
        cnt+=1

    print('optimizing transl is :')
    print(extra_trans)
    return betas.detach().clone(), extra_trans.detach().clone()
