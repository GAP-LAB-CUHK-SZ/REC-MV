"""
@File: fl_optimizer.
@Author: Lingteng Qiu
@Email: qiulingteng@link.cuhk.edu.cn
@Date: 2023-05-01
@Desc: featureline optimization: such as scale, translation
initialize fl parts
"""
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
from pytorch3d.io import save_obj, save_ply
from einops import rearrange
from utils.constant import INI_FL_SCALE





def check_zbuf_body(smpl_mesh, N, deform_lbs, poses, trans, cameras, mask_render, img_size, screen_pts):
    # zbuf visible check
    smpl_verts = smpl_mesh.verts_packed()[None].expand(N, -1 ,3)
    deform_smpl_vertices = deform_lbs(smpl_verts, [poses,trans])
    deform_smpl_mesh = Meshes(verts = [verts.squeeze(0) for verts in torch.split(deform_smpl_vertices, 1, dim = 0)], faces = [smpl_mesh.faces_packed() for _ in range(N)])
    mask_render.rasterizer.cameras=cameras

    __, body_frag = mask_render(deform_smpl_mesh)
    body_zbuf = body_frag.zbuf
    body_z_max, __ = deform_smpl_vertices[..., -1].max(-1)
    body_z_max = body_z_max[:, None, None, None].expand_as(body_zbuf)
    body_zbuf[body_zbuf == -1.] = body_z_max[body_zbuf==-1.]

    image_width, image_height = img_size.unbind(1)
    image_width = image_width.view(-1, 1)  # (N, 1)
    image_height = image_height.view(-1, 1)  # (N, 1)
    u = screen_pts[..., 0]
    v = screen_pts[..., 1]
    u  = 2 * u /image_width  - 1
    v  = 2 * v /image_height - 1
    uv = torch.cat([u[..., None],v[..., None]], dim = -1)



    z_buff = rearrange(body_zbuf, 'b h w c -> b c h w ')
    uv = uv[...,:2]
    uv = uv.unsqueeze(2)  # [B, N, 1, 2]
    samples = torch.nn.functional.grid_sample(z_buff, uv, align_corners=True)  # [B, C, N, 1]

    return samples[:, 0 , :, 0]

def update_feature_line_mesh(fl_meshes, rigid_R, rigid_T):
    meshes_vertices = icp_rotate_transfrom(fl_meshes, rigid_R, rigid_T)

    return [mesh.update_padded(mesh_vertices[None]) for mesh, mesh_vertices in zip(fl_meshes, meshes_vertices)]

def update_scale_feature_line_mesh(fl_meshes, rigid_R, rigid_T, rigid_scale):
    meshes_vertices = scale_icp_rotate_transfrom(fl_meshes, rigid_R, rigid_T, rigid_scale)

    return [mesh.update_padded(mesh_vertices[None]) for mesh, mesh_vertices in zip(fl_meshes, meshes_vertices)]


def fl_proj_loss(fl_pts_list,gt_fl_pts_list, fl_masks, proj_fl_weights = None):

    loss = torch.tensor(0.).to(fl_pts_list[0])


    if proj_fl_weights == None:
        proj_fl_weights = [1. for _ in range(len(fl_pts_list))]


    for fl_pts, gt_fl_pts, fl_mask, proj_fl_weight in zip(fl_pts_list, gt_fl_pts_list, fl_masks, proj_fl_weights):
        screen_fl_pts = fl_pts[...,:2]
        screen_fl_masks = fl_mask[...,:2]


        batch_check = fl_mask[...,0].sum(dim=-1)
        valid_batch = (batch_check>0).float().sum()




        batch_loss = 0.

        batch_size = screen_fl_masks.shape[0]
        for screen_fl_pt, screen_fl_mask, gt_fl_pt in zip(screen_fl_pts, screen_fl_masks, gt_fl_pts):
            screen_fl_pt = screen_fl_pt[screen_fl_mask == 1].view(1,-1,2)
            gt_fl_pt = gt_fl_pt.view(1, -1 ,2)

            cham_loss, __ =  chamfer_distance(screen_fl_pt, gt_fl_pt, point_reduction = 'sum')
            batch_loss+= proj_fl_weight * cham_loss

        if valid_batch != 0:
            batch_loss /= valid_batch

        if screen_fl_masks.sum() !=0.:
            loss += batch_loss  / (screen_fl_masks.sum() //2)
        else:
            loss += batch_loss

    return loss / len(fl_pts_list)
def scale_rigid_optimizer(deform_lbs, fl_meshes, smpl_mesh, mask_render, dataset, data_dataloader, save_path, fl_infos, rigid_R_type = 'o6', device = 'cuda:0'):
    '''
    rigid-transform fitting for fl_meshes, to align initial mesh
    using CD loss, add scale parameters
    '''

    # begin at rigid optimizer
    init_trans_matrix_path = os.path.join(save_path, 'init_trans_matrix.pth')
    template_categories = fl_meshes.keys()

    train_data_dataloader = dataset.get_init_fl_datasets(data_dataloader.batch_size, data_dataloader.sampler, data_dataloader.num_workers)

    T_epoch = max(150 // len(train_data_dataloader),2)
    S_epoch = min(max(150 // len(train_data_dataloader),2),10)

    # T_epoch = 0
    # S_epoch = 0

    print('begining at rigid optimizer ')
    if rigid_R_type == 'o6':
        initial_pose = torch.full((len(template_categories),6 ), 0.0, device='cuda:0', requires_grad= False)
        initial_pose[:,0]  = 1.
        initial_pose[:,4] = 1.
    else:
        raise NotImplementedError

    rigid_pose =  Variable(initial_pose, requires_grad = False)
    rigid_T = torch.full((len(template_categories),1 ,3), 0.0, device='cuda:0', requires_grad=True)


    rigid_scale = [INI_FL_SCALE[category] for category in template_categories ]
    rigid_scale = torch.tensor(rigid_scale, device= 'cuda:0', requires_grad = True).float()


    # rigid_scale = torch.full((len(template_categories),1 ,1), 1., device='cuda:0', requires_grad=True)


    # optimizier setting
    rigid_scale_T_optimizer = torch.optim.Adam([rigid_T, rigid_scale], lr= 0.005, weight_decay= 0.)
    rigid_template_catogories = ['rigid_' + temp for temp in template_categories]

    all_save_mesh_path =  os.path.join(save_path, 'mesh_exp')
    save_mesh_path = os.path.join(save_path, 'rigid_mesh_exp')
    save_img_path = os.path.join(save_path, 'proj_img')

    os.makedirs(save_mesh_path, exist_ok= True)
    os.makedirs(save_img_path, exist_ok= True)
    fl_meshes = tocuda([fl_meshes[fl_name] for fl_name in fl_infos])

    rigid_R = compute_rotation_matrix_from_ortho6d(rigid_pose)
    init_meshes_vertices = scale_icp_rotate_transfrom(fl_meshes, rigid_R, rigid_T, rigid_scale)

    if os.path.exists(init_trans_matrix_path):
        trans_matrix = torch.load(init_trans_matrix_path)
        rigid_R = trans_matrix['rigid_R'].to(device)
        rigid_T = trans_matrix['rigid_T'].to(device)
        rigid_scale = trans_matrix['rigid_scale'].to(device)
        meshes_vertices = scale_icp_rotate_center_transform(fl_meshes, rigid_R, rigid_T, rigid_scale)

        # for batch_id, (frame_ids, batch) in enumerate(data_dataloader):
        #     imgs = batch['img']
        #     imgs = ((imgs/2)+0.5)*255
        #     imgs = imgs.detach().cpu().numpy().astype(np.uint8)
        #     frame_ids= frame_ids.long().to(device)
        #     N = frame_ids.numel()

        #     focals,princeple_ps,Rs,Ts,H,W=dataset.get_camera_parameters(frame_ids.numel(),device)
        #     img_size = repeat(torch.Tensor([W,H]).float(), 'c -> b c', b= N).to(device)
        #     cameras=RectifiedPerspectiveCameras(focals,princeple_ps,Rs,Ts,image_size=[(W, H)]).to(device)
        #     batch_fl_vertices = [repeat(meshes_vertice,'n c -> b n c', b = N) for meshes_vertice in meshes_vertices]

        #     fl_split_size =[batch_fl_vertice.shape[1] for batch_fl_vertice in batch_fl_vertices]

        #     batch_vertices = torch.cat(batch_fl_vertices, dim = 1)
        #     poses, trans, _, __ = dataset.get_grad_parameters(frame_ids,device)
        #     deform_batch_vertices = deform_lbs(batch_vertices, [poses,trans])
        #     screen_pts = cameras.transform_points_screen(deform_batch_vertices,  img_size)

        #     batch_init_fl_vertices = [repeat(init_meshes_vertice,'n c -> b n c', b = N) for init_meshes_vertice in init_meshes_vertices]
        #     batch_init_vertices = torch.cat(batch_init_fl_vertices, dim = 1)
        #     deform_batch_init_vertices = deform_lbs(batch_init_vertices, [poses,trans])
        #     init_screen_pts = cameras.transform_points_screen(deform_batch_init_vertices,  img_size)
        #     zbuf_body = check_zbuf_body(smpl_mesh, N, deform_lbs, poses, trans, cameras, mask_render, img_size, init_screen_pts)

        #     z = deform_batch_init_vertices[..., -1]
        #     ref_v = z - zbuf_body
        #     visible = ref_v <= 0.01

        #     for img_id, (frame_id, screen_pt, img, vis_mask) in enumerate(zip(frame_ids, screen_pts, imgs, visible)):
        #         for pt, vis in zip(screen_pt.detach().cpu().numpy().astype(np.int), vis_mask.detach().cpu().numpy().astype(np.bool)):
        #             if not vis:
        #                 continue
        #             img = cv2.circle(img, (pt[0], pt[1]),2, (0,0,255), 2)
        #         cv2.imwrite('./debug/debug_proj/debug_fl_proj_{:04d}.png'.format(frame_id), img)

        #         print('save: {:04d}.png'.format(frame_id))
        # xxxx
        return [mesh.update_padded(mesh_vertices[None]) for mesh, mesh_vertices in zip(fl_meshes, meshes_vertices)]

    print("training global rigid_T!")
    for epoch in range(T_epoch):
        # visualizaion
        # visualized_RT_img(smpl_model, test_dataloader, rigid_T, rigid_pose, save_img_path, epoch = T_epoch, all_epoch = False)
        # NOTE that the following is meta parameters,
        # meta = ['name','polygons','image','world2ndc','ndc2screen','smpl_pose', 'smpl_beta', 'smpl_trans', 'polygons_mask', 'z_buff', 'valid_polygons']
        for batch_id, (frame_ids, batch) in enumerate(train_data_dataloader):


            rigid_scale_T_optimizer.zero_grad()

            frame_ids= frame_ids.long().to(device)
            N = frame_ids.numel()
            gt_fl_pts = batch['fl_pts'].to(device)
            fl_masks = batch['fl_masks'].to(device)
            N_fl = fl_masks.shape[-1]

            fl_masks = torch.split(fl_masks, [1 for _ in range(fl_masks.shape[-1])], dim=1)
            focals,princeple_ps,Rs,Ts,H,W=dataset.get_camera_parameters(frame_ids.numel(),device)

            img_size = repeat(torch.Tensor([W,H]).float(), 'c -> b c', b= N).to(device)
            cameras = RectifiedPerspectiveCameras(focals,princeple_ps,Rs,Ts,image_size=[(W, H)]).to(device)
            rigid_R = compute_rotation_matrix_from_ortho6d(rigid_pose)
            meshes_vertices = scale_icp_rotate_transfrom(fl_meshes, rigid_R, rigid_T, rigid_scale)
            batch_fl_vertices = [repeat(meshes_vertice,'n c -> b n c', b = N) for meshes_vertice in meshes_vertices]
            fl_split_size =[batch_fl_vertice.shape[1] for batch_fl_vertice in batch_fl_vertices]
            batch_vertices = torch.cat(batch_fl_vertices, dim = 1)
            poses, trans, _, __ = dataset.get_grad_parameters(frame_ids,device)
            deform_batch_vertices = deform_lbs(batch_vertices, [poses,trans])
            screen_pts = cameras.transform_points_screen(deform_batch_vertices,  img_size)

            # zbuf check
            batch_init_fl_vertices = [repeat(init_meshes_vertice,'n c -> b n c', b = N) for init_meshes_vertice in init_meshes_vertices]
            batch_init_vertices = torch.cat(batch_init_fl_vertices, dim = 1)
            deform_batch_init_vertices = deform_lbs(batch_init_vertices, [poses.detach(),trans.detach()])
            init_screen_pts = cameras.transform_points_screen(deform_batch_init_vertices,  img_size)
            zbuf_body = check_zbuf_body(smpl_mesh, N, deform_lbs, poses.detach(), trans.detach(), cameras, mask_render, img_size, init_screen_pts)

            z = deform_batch_init_vertices[..., -1]
            ref_v = z - zbuf_body
            visible = ref_v < 0.01
            visible_list = list(torch.split(visible, fl_split_size, dim = 1))
            visible_list = [visible_mask.float() for visible_mask in visible_list]


            gt_fl_pts_list = torch.split(gt_fl_pts, [gt_fl_pts.shape[1] // N_fl for _ in range(N_fl)], dim =1)
            screen_fl_pts_list = list(torch.split(screen_pts, fl_split_size, dim = 1))
            fl_masks = [fl_mask[:,None,:].expand_as(screen_fl_pts).float() for (fl_mask, screen_fl_pts) in zip(fl_masks, screen_fl_pts_list)]

            visible_fl_masks =[fl_mask * visible_mask[...,None]  for fl_mask, visible_mask in zip(fl_masks, visible_list)]

            fl_loss = fl_proj_loss(screen_fl_pts_list, gt_fl_pts_list, visible_fl_masks)

            fl_loss.backward()
            rigid_scale_T_optimizer.step()


            print(rigid_scale)


            print("{}/{}/{}: fl_loss: {}".format(epoch, batch_id, len(train_data_dataloader), fl_loss.item()))



    rigid_T.requires_grad = False
    rigid_scale_T_optimizer = torch.optim.Adam([rigid_scale], lr= 0.005, weight_decay= 0.)

    print("training global rigid_scale!")
    for epoch in range(S_epoch):
        # visualizaion
        # visualized_RT_img(smpl_model, test_dataloader, rigid_T, rigid_pose, save_img_path, epoch = T_epoch, all_epoch = False)
        # NOTE that the following is meta parameters,
        # meta = ['name','polygons','image','world2ndc','ndc2screen','smpl_pose', 'smpl_beta', 'smpl_trans', 'polygons_mask', 'z_buff', 'valid_polygons']
        for batch_id, (frame_ids, batch) in enumerate(train_data_dataloader):
            rigid_scale_T_optimizer.zero_grad()

            frame_ids= frame_ids.long().to(device)
            N = frame_ids.numel()
            gt_fl_pts = batch['fl_pts'].to(device)
            fl_masks = batch['fl_masks'].to(device)
            N_fl = fl_masks.shape[-1]




            fl_masks = torch.split(fl_masks, [1 for _ in range(fl_masks.shape[-1])], dim=1)
            focals,princeple_ps,Rs,Ts,H,W=dataset.get_camera_parameters(frame_ids.numel(),device)

            img_size = repeat(torch.Tensor([W,H]).float(), 'c -> b c', b= N).to(device)
            cameras = RectifiedPerspectiveCameras(focals,princeple_ps,Rs,Ts,image_size=[(W, H)]).to(device)
            rigid_R = compute_rotation_matrix_from_ortho6d(rigid_pose)
            meshes_vertices = scale_icp_rotate_transfrom(fl_meshes, rigid_R, rigid_T, rigid_scale)
            batch_fl_vertices = [repeat(meshes_vertice,'n c -> b n c', b = N) for meshes_vertice in meshes_vertices]

            fl_split_size =[batch_fl_vertice.shape[1] for batch_fl_vertice in batch_fl_vertices]

            batch_vertices = torch.cat(batch_fl_vertices, dim = 1)
            poses, trans, _, __ = dataset.get_grad_parameters(frame_ids,device)
            deform_batch_vertices = deform_lbs(batch_vertices, [poses,trans])
            screen_pts = cameras.transform_points_screen(deform_batch_vertices,  img_size)


            # zbuf check
            batch_init_fl_vertices = [repeat(init_meshes_vertice,'n c -> b n c', b = N) for init_meshes_vertice in init_meshes_vertices]
            batch_init_vertices = torch.cat(batch_init_fl_vertices, dim = 1)
            poses, trans, _, __ = dataset.get_grad_parameters(frame_ids,device)
            deform_batch_init_vertices = deform_lbs(batch_init_vertices, [poses,trans])
            init_screen_pts = cameras.transform_points_screen(deform_batch_init_vertices,  img_size)
            zbuf_body = check_zbuf_body(smpl_mesh, N, deform_lbs, poses.detach(), trans.detach(), cameras, mask_render, img_size, init_screen_pts)

            z = deform_batch_init_vertices[..., -1]
            ref_v = z - zbuf_body
            visible = ref_v < 0.01

            visible_list = list(torch.split(visible, fl_split_size, dim = 1))
            visible_list = [visible_mask.float() for visible_mask in visible_list]

            gt_fl_pts_list = torch.split(gt_fl_pts, [gt_fl_pts.shape[1] // N_fl for _ in range(N_fl)], dim =1)

            screen_fl_pts_list = list(torch.split(screen_pts, fl_split_size, dim = 1))
            fl_masks = [fl_mask[:,None,:].expand_as(screen_fl_pts).float() for (fl_mask, screen_fl_pts) in zip(fl_masks, screen_fl_pts_list)]

            visible_fl_masks =[fl_mask * visible_mask[...,None]  for fl_mask, visible_mask in zip(fl_masks, visible_list)]
            fl_loss = fl_proj_loss(screen_fl_pts_list, gt_fl_pts_list, visible_fl_masks)
            fl_loss.backward()
            rigid_scale_T_optimizer.step()

            print(rigid_scale)

            print("{}/{}/{}: fl_loss: {}".format(epoch, batch_id, len(train_data_dataloader), fl_loss.item()))


    # rigid_R = compute_rotation_matrix_from_ortho6d(rigid_pose)
    # meshes_vertices = scale_icp_rotate_transfrom(fl_meshes, rigid_R, rigid_T, rigid_scale)
    # for batch_id, (frame_ids, batch) in enumerate(train_data_dataloader):

    #     imgs = batch['img']
    #     imgs = ((imgs/2)+0.5)*255
    #     imgs = imgs.detach().cpu().numpy().astype(np.uint8)
    #     frame_ids= frame_ids.long().to(device)
    #     N = frame_ids.numel()

    #     focals,princeple_ps,Rs,Ts,H,W=dataset.get_camera_parameters(frame_ids.numel(),device)
    #     img_size = repeat(torch.Tensor([W,H]).float(), 'c -> b c', b= N).to(device)
    #     cameras=RectifiedPerspectiveCameras(focals,princeple_ps,Rs,Ts,image_size=[(W, H)]).to(device)
    #     batch_fl_vertices = [repeat(meshes_vertice,'n c -> b n c', b = N) for meshes_vertice in meshes_vertices]

    #     fl_split_size =[batch_fl_vertice.shape[1] for batch_fl_vertice in batch_fl_vertices]

    #     batch_vertices = torch.cat(batch_fl_vertices, dim = 1)
    #     poses, trans, _, __ = dataset.get_grad_parameters(frame_ids,device)
    #     deform_batch_vertices = deform_lbs(batch_vertices, [poses,trans])
    #     screen_pts = cameras.transform_points_screen(deform_batch_vertices,  img_size)
    #     for img_id, (screen_pt, img) in enumerate(zip(screen_pts, imgs)):
    #         for pt in screen_pt.detach().cpu().numpy().astype(np.int):
    #             img = cv2.circle(img, (pt[0], pt[1]),2, (0,0,255), 2)
    #         cv2.imwrite('./debug/debug_fl_proj_{:04d}.png'.format(img_id), img)
    #     xxxx
    #     xxxxxx
    # xxxx
    print("training global rigid_R!")

    rigid_T.requires_grad = False
    rigid_scale.requires_grad = False

    rigid_scale  = rigid_scale.detach().cpu()
    # left and right need the same as
    map_fl = {}
    for fl_idx , fl_info in enumerate(fl_infos):
        map_fl[fl_info] = fl_idx



    if 'left_cuff' in fl_infos:
        max_cuff = max(rigid_scale[map_fl['left_cuff']], rigid_scale[map_fl['right_cuff']])
        rigid_scale[map_fl['left_cuff']] = max_cuff
        rigid_scale[map_fl['right_cuff']] = max_cuff


    if 'left_pant' in fl_infos:
        max_pant = max(rigid_scale[map_fl['left_pant']], rigid_scale[map_fl['right_pant']])
        rigid_scale[map_fl['left_pant']] = max_pant
        rigid_scale[map_fl['right_pant']] = max_pant


    rigid_scale = rigid_scale.cuda()







    rigid_R = compute_rotation_matrix_from_ortho6d(rigid_pose)
    fl_meshes =update_scale_feature_line_mesh(fl_meshes, rigid_R, rigid_T, rigid_scale)



    rigid_pose =  Variable(initial_pose.clone(), requires_grad = True)
    rigid_R_T = torch.full((len(template_categories),1 ,3), 0.0, device='cuda:0', requires_grad = False)
    rigid_R_optimizer = torch.optim.Adam([rigid_pose], lr= 0.001, weight_decay= 0.)

    for r_epoch in range(T_epoch):
        for batch_id, (frame_ids, batch) in enumerate(train_data_dataloader):
            loss = 0.
            rigid_R_optimizer.zero_grad()

            frame_ids= frame_ids.long().to(device)
            N = frame_ids.numel()

            gt_fl_pts = batch['fl_pts'].to(device)
            fl_masks = batch['fl_masks'].to(device)
            N_fl = fl_masks.shape[-1]

            fl_masks = torch.split(fl_masks, [1 for _ in range(fl_masks.shape[-1])], dim=1)
            focals,princeple_ps,Rs,Ts,H,W=dataset.get_camera_parameters(frame_ids.numel(),device)

            img_size = repeat(torch.Tensor([W,H]).float(), 'c -> b c', b= N).to(device)
            cameras=RectifiedPerspectiveCameras(focals,princeple_ps,Rs,Ts,image_size=[(W, H)]).to(device)
            rigid_R = compute_rotation_matrix_from_ortho6d(rigid_pose)
            # center rotated
            meshes_vertices = center_transform(fl_meshes, rigid_R, rigid_R_T)
            batch_fl_vertices = [repeat(meshes_vertice,'n c -> b n c', b = N) for meshes_vertice in meshes_vertices]
            fl_split_size =[batch_fl_vertice.shape[1] for batch_fl_vertice in batch_fl_vertices]

            batch_vertices = torch.cat(batch_fl_vertices, dim = 1)
            poses, trans, _, __ = dataset.get_grad_parameters(frame_ids,device)
            deform_batch_vertices = deform_lbs(batch_vertices, [poses,trans])
            screen_pts = cameras.transform_points_screen(deform_batch_vertices,  img_size)

            batch_init_fl_vertices = [repeat(init_meshes_vertice,'n c -> b n c', b = N) for init_meshes_vertice in init_meshes_vertices]
            batch_init_vertices = torch.cat(batch_init_fl_vertices, dim = 1)
            poses, trans, _, __ = dataset.get_grad_parameters(frame_ids,device)
            deform_batch_init_vertices = deform_lbs(batch_init_vertices, [poses,trans])
            init_screen_pts = cameras.transform_points_screen(deform_batch_init_vertices,  img_size)
            zbuf_body = check_zbuf_body(smpl_mesh, N, deform_lbs, poses.detach(), trans.detach(), cameras, mask_render, img_size, init_screen_pts)

            z = deform_batch_init_vertices[..., -1]
            ref_v = z - zbuf_body
            visible = ref_v < 0.01

            visible_list = list(torch.split(visible, fl_split_size, dim = 1))
            visible_list = [visible_mask.float() for visible_mask in visible_list]

            gt_fl_pts_list = torch.split(gt_fl_pts, [gt_fl_pts.shape[1] // N_fl for _ in range(N_fl)], dim =1)

            screen_fl_pts_list = list(torch.split(screen_pts, fl_split_size, dim = 1))
            fl_masks = [fl_mask[:,None,:].expand_as(screen_fl_pts).float() for (fl_mask, screen_fl_pts) in zip(fl_masks, screen_fl_pts_list)]

            visible_fl_masks =[fl_mask * visible_mask[...,None]  for fl_mask, visible_mask in zip(fl_masks, visible_list)]
            fl_loss = fl_proj_loss(screen_fl_pts_list, gt_fl_pts_list, visible_fl_masks)
            fl_loss.backward()

            rigid_R_optimizer.step()

            print("{}/{}/{} rigid_R Loss:{:.4f}".format(r_epoch, batch_id,len(train_data_dataloader), fl_loss.item()))

    # debug for projection
    rigid_R = compute_rotation_matrix_from_ortho6d(rigid_pose)
    meshes_vertices =  center_transform(fl_meshes, rigid_R, rigid_R_T)
    for batch_id, (frame_ids, batch) in enumerate(data_dataloader):

        imgs = batch['img']
        imgs = ((imgs/2)+0.5)*255
        imgs = imgs.detach().cpu().numpy().astype(np.uint8)
        frame_ids= frame_ids.long().to(device)
        N = frame_ids.numel()

        focals,princeple_ps,Rs,Ts,H,W=dataset.get_camera_parameters(frame_ids.numel(),device)
        img_size = repeat(torch.Tensor([W,H]).float(), 'c -> b c', b= N).to(device)
        cameras=RectifiedPerspectiveCameras(focals,princeple_ps,Rs,Ts,image_size=[(W, H)]).to(device)
        batch_fl_vertices = [repeat(meshes_vertice,'n c -> b n c', b = N) for meshes_vertice in meshes_vertices]

        fl_split_size =[batch_fl_vertice.shape[1] for batch_fl_vertice in batch_fl_vertices]

        batch_vertices = torch.cat(batch_fl_vertices, dim = 1)
        poses, trans, _, __ = dataset.get_grad_parameters(frame_ids,device)
        deform_batch_vertices = deform_lbs(batch_vertices, [poses,trans])
        screen_pts = cameras.transform_points_screen(deform_batch_vertices,  img_size)

        batch_init_fl_vertices = [repeat(init_meshes_vertice,'n c -> b n c', b = N) for init_meshes_vertice in init_meshes_vertices]
        batch_init_vertices = torch.cat(batch_init_fl_vertices, dim = 1)
        deform_batch_init_vertices = deform_lbs(batch_init_vertices, [poses,trans])
        init_screen_pts = cameras.transform_points_screen(deform_batch_init_vertices,  img_size)
        zbuf_body = check_zbuf_body(smpl_mesh, N, deform_lbs, poses, trans, cameras, mask_render, img_size, init_screen_pts)

        z = deform_batch_init_vertices[..., -1]
        ref_v = z - zbuf_body
        visible = ref_v <= 0.01

        for img_id, (frame_id, screen_pt, img, vis_mask) in enumerate(zip(frame_ids, screen_pts, imgs, visible)):
            for pt, vis in zip(screen_pt.detach().cpu().numpy().astype(np.int), vis_mask.detach().cpu().numpy().astype(np.bool)):
                if not vis:
                    continue
                img = cv2.circle(img, (pt[0], pt[1]),2, (0,0,255), 2)
            cv2.imwrite('./debug/debug_proj/debug_fl_proj_{:04d}.png'.format(frame_id), img)

            print('save: {:04d}.png'.format(frame_id))


    # left right need to the same



    torch.save({'rigid_R' : rigid_R.detach().cpu(), 'rigid_T': rigid_T.detach().cpu(), 'rigid_scale': rigid_scale.detach().cpu() }, init_trans_matrix_path)
    meshes_vertices =  center_transform(fl_meshes, rigid_R, rigid_R_T)


    return [mesh.update_padded(mesh_vertices[None]) for mesh, mesh_vertices in zip(fl_meshes, meshes_vertices)]

def rigid_optimizer(deform_lbs, fl_meshes, dataset, train_data_dataloader, save_path, fl_infos, rigid_R_type = 'o6', device = 'cuda:0'):
    '''
    rigid-transform fitting for fl_meshes, to align initial mesh
    using CD loss
    '''
    # begin at rigid optimizer
    init_trans_matrix_path = os.path.join(save_path, 'init_trans_matrix.pth')

    template_categories = fl_meshes.keys()
    print('begining at rigid optimizer ')
    if rigid_R_type == 'o6':
        initial_pose = torch.full((len(template_categories),6 ), 0.0, device='cuda:0', requires_grad= False)
        initial_pose[:,0]  = 1.
        initial_pose[:,4] = 1.
    else:
        raise NotImplementedError

    rigid_pose =  Variable(initial_pose, requires_grad = False)
    rigid_T = torch.full((len(template_categories),1 ,3), 0.0, device='cuda:0', requires_grad=True)
    # optimizier setting
    rigid_T_optimizer = torch.optim.Adam([rigid_T], lr= 0.005, weight_decay= 0.)
    rigid_template_catogories = ['rigid_' + temp for temp in template_categories]

    all_save_mesh_path =  os.path.join(save_path, 'mesh_exp')
    save_mesh_path = os.path.join(save_path, 'rigid_mesh_exp')
    save_img_path = os.path.join(save_path, 'proj_img')

    os.makedirs(save_mesh_path, exist_ok= True)
    os.makedirs(save_img_path, exist_ok= True)

    fl_meshes = tocuda([fl_meshes[fl_name] for fl_name in fl_infos])
    if os.path.exists(init_trans_matrix_path):
        trans_matrix = torch.load(init_trans_matrix_path)
        rigid_R = trans_matrix['rigid_R'].to(device)
        rigid_T = trans_matrix['rigid_T'].to(device)
        meshes_vertices = icp_rotate_center_transform(fl_meshes, rigid_R, rigid_T)
        return [mesh.update_padded(mesh_vertices[None]) for mesh, mesh_vertices in zip(fl_meshes, meshes_vertices)]
        # debug to show imgs
        # for batch_id, (frame_ids, batch) in enumerate(train_data_dataloader):

        #     imgs = batch['img']
        #     imgs = ((imgs/2)+0.5)*255
        #     imgs = imgs.detach().cpu().numpy().astype(np.uint8)
        #     frame_ids= frame_ids.long().to(device)
        #     N = frame_ids.numel()

        #     focals,princeple_ps,Rs,Ts,H,W=dataset.get_camera_parameters(frame_ids.numel(),device)
        #     img_size = repeat(torch.Tensor([W,H]).float(), 'c -> b c', b= N).to(device)
        #     cameras=RectifiedPerspectiveCameras(focals,princeple_ps,Rs,Ts,image_size=[(W, H)]).to(device)
        #     batch_fl_vertices = [repeat(meshes_vertice,'n c -> b n c', b = N) for meshes_vertice in meshes_vertices]

        #     fl_split_size =[batch_fl_vertice.shape[1] for batch_fl_vertice in batch_fl_vertices]

        #     batch_vertices = torch.cat(batch_fl_vertices, dim = 1)
        #     poses, trans, _, __ = dataset.get_grad_parameters(frame_ids,device)
        #     deform_batch_vertices = deform_lbs(batch_vertices, [poses,trans])
        #     screen_pts = cameras.transform_points_screen(deform_batch_vertices,  img_size)
        #     for img_id, (screen_pt, img) in enumerate(zip(screen_pts, imgs)):
        #         for pt in screen_pt.detach().cpu().numpy().astype(np.int):
        #             img = cv2.circle(img, (pt[0], pt[1]),2, (0,0,255), 2)
        #         cv2.imwrite('./debug_fl_proj_{:04d}.png'.format(img_id), img)
        #     xxxx

    print("training global rigid_T!")
    T_epoch = 0
    # optimizer_t
    for T_epoch in range(2):
        # visualizaion
        # visualized_RT_img(smpl_model, test_dataloader, rigid_T, rigid_pose, save_img_path, epoch = T_epoch, all_epoch = False)
        # NOTE that the following is meta parameters,
        # meta = ['name','polygons','image','world2ndc','ndc2screen','smpl_pose', 'smpl_beta', 'smpl_trans', 'polygons_mask', 'z_buff', 'valid_polygons']
        for batch_id, (frame_ids, batch) in enumerate(train_data_dataloader):
            rigid_T_optimizer.zero_grad()

            frame_ids= frame_ids.long().to(device)
            N = frame_ids.numel()
            gt_fl_pts = batch['fl_pts'].to(device)
            fl_masks = batch['fl_masks'].to(device)
            N_fl = fl_masks.shape[-1]

            fl_masks = torch.split(fl_masks, [1 for _ in range(fl_masks.shape[-1])], dim=1)
            focals,princeple_ps,Rs,Ts,H,W=dataset.get_camera_parameters(frame_ids.numel(),device)

            img_size = repeat(torch.Tensor([W,H]).float(), 'c -> b c', b= N).to(device)
            cameras=RectifiedPerspectiveCameras(focals,princeple_ps,Rs,Ts,image_size=[(W, H)]).to(device)
            rigid_R = compute_rotation_matrix_from_ortho6d(rigid_pose)
            meshes_vertices = icp_rotate_transfrom(fl_meshes, rigid_R, rigid_T)
            batch_fl_vertices = [repeat(meshes_vertice,'n c -> b n c', b = N) for meshes_vertice in meshes_vertices]

            fl_split_size =[batch_fl_vertice.shape[1] for batch_fl_vertice in batch_fl_vertices]

            batch_vertices = torch.cat(batch_fl_vertices, dim = 1)
            poses, trans, _, __ = dataset.get_grad_parameters(frame_ids,device)
            deform_batch_vertices = deform_lbs(batch_vertices, [poses,trans])
            screen_pts = cameras.transform_points_screen(deform_batch_vertices,  img_size)

            gt_fl_pts_list = torch.split(gt_fl_pts, [gt_fl_pts.shape[1] // N_fl for _ in range(N_fl)], dim =1)



            screen_fl_pts_list = list(torch.split(screen_pts, fl_split_size, dim = 1))
            fl_masks = [fl_mask[:,None,:].expand_as(screen_fl_pts).float() for (fl_mask, screen_fl_pts) in zip(fl_masks, screen_fl_pts_list)]

            fl_loss = fl_proj_loss(screen_fl_pts_list, gt_fl_pts_list, fl_masks)
            fl_loss.backward()
            rigid_T_optimizer.step()

            print("{}/{}: fl_loss: {}".format(batch_id, len(train_data_dataloader), fl_loss.item()))

    print("training global rigid_R!")
    rigid_T.requires_grad = False
    rigid_R = compute_rotation_matrix_from_ortho6d(rigid_pose)
    fl_meshes = update_feature_line_mesh(fl_meshes, rigid_R, rigid_T)

    rigid_pose =  Variable(initial_pose.clone(), requires_grad = True)
    rigid_R_T = torch.full((len(template_categories),1 ,3), 0.0, device='cuda:0', requires_grad = False)
    rigid_R_optimizer = torch.optim.Adam([rigid_pose], lr= 0.001, weight_decay= 0.)

    for R_epoch in range(2):
        for batch_id, (frame_ids, batch) in enumerate(train_data_dataloader):
            loss = 0.
            rigid_R_optimizer.zero_grad()

            frame_ids= frame_ids.long().to(device)
            N = frame_ids.numel()

            gt_fl_pts = batch['fl_pts'].to(device)
            fl_masks = batch['fl_masks'].to(device)
            N_fl = fl_masks.shape[-1]

            fl_masks = torch.split(fl_masks, [1 for _ in range(fl_masks.shape[-1])], dim=1)
            focals,princeple_ps,Rs,Ts,H,W=dataset.get_camera_parameters(frame_ids.numel(),device)

            img_size = repeat(torch.Tensor([W,H]).float(), 'c -> b c', b= N).to(device)
            cameras=RectifiedPerspectiveCameras(focals,princeple_ps,Rs,Ts,image_size=[(W, H)]).to(device)
            rigid_R = compute_rotation_matrix_from_ortho6d(rigid_pose)
            # center rotated
            meshes_vertices = center_transform(fl_meshes, rigid_R, rigid_R_T)
            batch_fl_vertices = [repeat(meshes_vertice,'n c -> b n c', b = N) for meshes_vertice in meshes_vertices]
            fl_split_size =[batch_fl_vertice.shape[1] for batch_fl_vertice in batch_fl_vertices]

            batch_vertices = torch.cat(batch_fl_vertices, dim = 1)
            poses, trans, _, __ = dataset.get_grad_parameters(frame_ids,device)
            deform_batch_vertices = deform_lbs(batch_vertices, [poses,trans])
            screen_pts = cameras.transform_points_screen(deform_batch_vertices,  img_size)
            gt_fl_pts_list = torch.split(gt_fl_pts, [gt_fl_pts.shape[1] // N_fl for _ in range(N_fl)], dim =1)

            screen_fl_pts_list = list(torch.split(screen_pts, fl_split_size, dim = 1))
            fl_masks = [fl_mask[:,None,:].expand_as(screen_fl_pts).float() for (fl_mask, screen_fl_pts) in zip(fl_masks, screen_fl_pts_list)]

            fl_loss = fl_proj_loss(screen_fl_pts_list, gt_fl_pts_list, fl_masks)
            fl_loss.backward()
            rigid_R_optimizer.step()

            print("{}/{} rigid_R Loss:{:.4f}".format(batch_id,len(train_data_dataloader), fl_loss.item()))
    torch.save({'rigid_R' : rigid_R.detach().cpu(), 'rigid_T': rigid_T.detach().cpu()}, init_trans_matrix_path)
    meshes_vertices =  center_transform(fl_meshes, rigid_R, rigid_R_T)
    return [mesh.update_padded(mesh_vertices[None]) for mesh, mesh_vertices in zip(fl_meshes, meshes_vertices)]
