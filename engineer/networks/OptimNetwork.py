import numpy as np
import torch.nn as nn
import torch
from torch.nn.parameter import Parameter
import torch.autograd.functional as F
from FastMinv import Fast3x3Minv
from model.Embedder import get_embedder
import utils
import os
import cv2
from pytorch3d.io import save_obj, save_ply
from pytorch3d.renderer.points.rasterizer import PointsRasterizationSettings
import pdb
from pytorch3d.transforms.transform3d import Transform3d

from utils.constant import TEMPLATE_GARMENT
import MCGpu
from pytorch3d.structures import Meshes,Pointclouds,join_meshes_as_batch
from pytorch3d.loss import (
    chamfer_distance,
    mesh_edge_loss,
    mesh_laplacian_smoothing,
    mesh_normal_consistency,
)
from model.CameraMine import PointsRendererWithFrags,RectifiedPerspectiveCameras, PointsRendererWithFrags_Split
from pytorch3d.io import load_objs_as_meshes
from torch_scatter import scatter
import model.RenderNet as RenderNet
from pytorch3d.renderer import (
    RasterizationSettings,
    MeshRasterizer,
    SoftSilhouetteShader,
    TexturesVertex,
    BlendParams,
    PointsRasterizationSettings,
    # PointsRenderer,
    PointsRasterizer,
    PointLights,
    AlphaCompositor
)
#debug
import cv2
import trimesh
import openmesh as om
import os.path as osp
from MCAcc import Seg3dLossless
from model.Deformer import initialLBSkinner,getTranslatorNet,CompositeDeformer,LBSkinner
# from pytorch3d.renderer import (
#     RasterizationSettings,
#     MeshRasterizer,
#     SoftSilhouetteShader,
#     BlendParams
# )
from pytorch3d.renderer.mesh.renderer import MeshRendererWithFragments
class OptimNetwork(nn.Module):
    def __init__(self,TmpSdf,Deformer,accEngine,maskRender,netRender,conf=None):
        super().__init__()
        self.conf=conf
        # self.camera=Cam
        self.sdf=TmpSdf
        self.deformer=Deformer
        self.maskRender=maskRender
        self.netRender=netRender
        self.engine=accEngine
        self.angThred=self.maskRender.rasterizer.cameras.angThreshold(0.5)
        print('camera ang threshold is %f'%self.angThred)
        self.TmpVs=None
        self.Tmpfs=None
        self.forward_time=0
        self.remesh_intersect=30
        self.remesh_time=0.
        self.next_conf=None
        self.next_train_conf=None
        self.pcRender=None
        self.draw=False
        self.enable_mesh_color=True
        self.sdfShrinkRadius=0.0

    def update_hierarchical_config(self,device, max_verts = 10000, max_face = 10000):

        if self.next_conf is not None:
            self.conf=self.next_conf
            self.forward_time=0
            rasterizer=self.maskRender.rasterizer
            H,W=(rasterizer.raster_settings.image_size[0],rasterizer.raster_settings.image_size[1])

            raster_settings_silhouette = PointsRasterizationSettings(
                image_size=(H,W),
                radius=self.next_train_conf.get_float('point_render.radius'),
                bin_size=(92 if max(H,W)>=1024 and max(H,W)<=2048 else None),
                # max_points_per_bin = max(max_verts, 10000),
                points_per_pixel=50,
                )
            self.pcRender=PointsRendererWithFrags_Split(
                rasterizer=PointsRasterizer(
                    cameras=rasterizer.cameras,
                    raster_settings=raster_settings_silhouette
                ),
                    compositor=AlphaCompositor(background_color=None)
                ).to(device)
            self.remesh_intersect=self.next_train_conf.get_int('point_render.remesh_intersect')
            self.sdfShrinkRadius=0.0
            raster_settings_silhouette=RasterizationSettings(
                    image_size=(H,W),
                    blur_radius=0.,
                    # blur_radius=np.log(1. / 1e-4 - 1.)*3.e-6,
                    bin_size=(92 if max(H,W)>=1024 and max(H,W)<=2048 else None),
                    # max_faces_per_bin= max(10000, max_face),
                    faces_per_pixel=1,
                    perspective_correct=True,
                    clip_barycentric_coords=False,
                    cull_backfaces=self.maskRender.rasterizer.raster_settings.cull_backfaces
                )

            self.maskRender.rasterizer.raster_settings=raster_settings_silhouette
            self.next_conf=None
            self.next_train_conf=None

    def initializeTmpSDF(self,nepochs,save_name,with_normals=False):
        network=self.sdf
        # change back to train mode
        network.train()
        optimizer = torch.optim.Adam(
            [
                {
                    "params": network.parameters(),
                    "lr": 0.005,
                    "weight_decay": 0
                }
            ])
        sche=torch.optim.lr_scheduler.StepLR(optimizer,500,0.5)
        vs=self.tmpBodyVs
        device=vs.device


        if with_normals and self.tmpBodyNs is None:
            with_normals=False
        if with_normals:
            ns=self.tmpBodyNs
        else:
            ns=torch.ones_like(vs)/np.sqrt(3)

        batch_size=5000


        for epoch in range(1, nepochs + 1):
            permute=torch.randperm(vs.shape[0])
            evs=vs[permute]
            ens=ns[permute]

            evs=torch.split(evs,batch_size)
            ens=torch.split(ens,batch_size)




            for data_index,(mnfld_pnts, normals) in enumerate(zip(evs,ens)):
                mnfld_pnts = mnfld_pnts.to(device)

                if with_normals:
                    normals = normals.to(device)
                # sample pts, containing global sampling and local smapling with 0.01 sigma
                nonmnfld_pnts = utils.sample_points(mnfld_pnts, 1.8,0.01)

                # forward pass

                mnfld_pnts.requires_grad_()
                nonmnfld_pnts.requires_grad_()

                mnfld_pred = network(mnfld_pnts,-1)
                nonmnfld_pred = network(nonmnfld_pnts,-1)


                mnfld_grad = network.gradient(mnfld_pnts, mnfld_pred)
                nonmnfld_grad = network.gradient(nonmnfld_pnts, nonmnfld_pred)

                # manifold loss
                mnfld_loss = (mnfld_pred.abs()).mean()
                # eikonal loss
                grad_loss = ((nonmnfld_grad.norm(2, dim=-1) - 1) ** 2).mean()

                loss = mnfld_loss + 0.1 * grad_loss
                # loss=0
                # normals loss
                if with_normals:
                    normals = normals.view(-1, 3)
                    # print(mnfld_grad[:10,:])
                    # print(mnfld_grad.norm(2, dim=1).mean())
                    normals_loss = ((mnfld_grad - normals).abs()).norm(2, dim=1).mean()
                    loss = loss + 1.0 * normals_loss
                else:
                    normals_loss = torch.zeros(1)

                # back propagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # print status
                if data_index == len(evs)-1:
                    print('Train Epoch: {}\tTrain Loss: {:.6f}\tManifold loss: {:.6f}'
                          '\tGrad loss: {:.6f}\tNormals Loss: {:.6f}'.format(
                        epoch, loss.item(), mnfld_loss.item(), grad_loss.item(), normals_loss.item()))
            sche.step()
        torch.save(network.state_dict(),save_name)

    def discretizeSDF(self,ratio,engine=None,balance_value=0.):
        def query_func(points):
            with torch.no_grad():
                return self.sdf.forward(points.reshape(-1,3),ratio).reshape(1,1,-1)
        if engine is None:
            engine=self.engine

        engine.balance_value=balance_value
        engine.query_func=query_func
        sdfs=engine.forward()

        # each bins size, in our space
        verts, faces=MCGpu.mc_gpu(sdfs[0,0].permute(2,1,0).contiguous(),engine.spacing_x,engine.spacing_y,engine.spacing_z,engine.bx,engine.by,engine.bz,balance_value)
        return verts,faces

    def infer(self,TmpVs,Tmpfs,H,W,ratio,frame_ids,notcolor=False,gts=None):
        device=TmpVs.device
        with torch.no_grad():
            focals,princeple_ps,Rs,Ts,H,W=self.dataset.get_camera_parameters(frame_ids.numel(),device)
            cameras=RectifiedPerspectiveCameras(focals,princeple_ps,Rs,Ts,image_size=[(W, H)]).to(device)
            self.maskRender.rasterizer.cameras=cameras
            self.pcRender.rasterizer.cameras=cameras

            TmpVnum=TmpVs.shape[0]
            N=frame_ids.numel()
            poses,trans,d_cond,rendcond=self.dataset.get_grad_parameters(frame_ids,device)
            defTmpVs=self.deformer(TmpVs[None,:,:].expand(N,-1,3),[d_cond,[poses,trans]],ratio=ratio)
            defMeshes=Meshes(verts=[vs.view(TmpVnum,3) for vs in torch.split(defTmpVs,1)],faces=[Tmpfs for _ in range(N)],textures=TexturesVertex([torch.ones_like(TmpVs) for _ in range(N)]))
            defMeshVs = defTmpVs.detach().cpu().numpy()
            imgs,frags=self.maskRender(defMeshes)
            if gts:
                masks=(frags.pix_to_face>=0).float()[...,0]
                gtMs=gts['mask']
                gts['maskE']=(1.-(masks*gtMs).view(N,-1).sum(1)/(masks+gtMs-masks*gtMs).abs().view(N,-1).sum(1)).cpu().numpy()
                masks=masks>0.
                imgs=imgs[...,:3]
                if 'image' in gts:
                    imgs[~masks]=gts['image'][~masks][:,[2,1,0]]


            imgs=torch.clamp(imgs*255.,min=0.,max=255.).cpu().numpy().astype(np.uint8)
            defTmpVs=self.deformer.defs[0](TmpVs[None,:,:].expand(N,-1,3),d_cond,ratio=ratio)
            defMeshes=Meshes(verts=[vs.view(TmpVnum,3) for vs in torch.split(defTmpVs,1)],faces=[Tmpfs for _ in range(N)],textures=TexturesVertex([torch.ones_like(TmpVs) for _ in range(N)]))


            newTs=self.dataset.trans.mean(0).to(device)[None,:]
            newcameras=RectifiedPerspectiveCameras(focals,princeple_ps,torch.tensor([[[-1.,0.,0.],[0.,1.,0.],[0.,0.,-1.]]],device=device).repeat(N,1,1),newTs.repeat(N,1),image_size=[(W, H)]).to(device)
            def1imgs,_=self.maskRender(defMeshes,cameras=newcameras,lights=PointLights(device=device,location=((0, 1, newTs[0,2].item()),)))
            def1imgs=torch.clamp(def1imgs*255.,min=0.,max=255.).cpu().numpy().astype(np.uint8)


            batch_inds,row_inds,col_inds,initTmpPs,_=utils.FindSurfacePs(TmpVs.detach(),Tmpfs,frags)

            cameras=self.maskRender.rasterizer.cameras
            rays=cameras.view_rays(torch.cat([col_inds.view(-1,1),row_inds.view(-1,1),torch.ones_like(col_inds.view(-1,1))],dim=-1).float())
            defconds=[d_cond.detach(),[poses.detach(),trans.detach()]]
        if notcolor:
            return None,imgs,def1imgs,defMeshVs
        tcolors=[]
        print('draw %d points'%rays.shape[0])
        for ind,(rays_,initTmpPs_,batch_inds_) in enumerate(zip(torch.split(rays,10000),torch.split(initTmpPs,10000),torch.split(batch_inds,10000))):
            initTmpPs_,check=utils.OptimizeSurfacePs(cameras.cam_pos().detach(),rays_.detach(),initTmpPs_.clone(),batch_inds_,self.sdf,ratio,self.deformer,defconds,dthreshold=1.e-4,athreshold=self.angThred,w1=3.05,w2=1.,times=30)
            # print('%d:(%d,%d)'%(ind,rays_.shape[0],check.sum().item()))
            initTmpPs_.requires_grad=True

            sdfs=self.sdf(initTmpPs_,ratio)
            nx=torch.autograd.grad(sdfs,initTmpPs_,torch.ones_like(sdfs),retain_graph=False,create_graph=False)[0]
            nx=nx/nx.norm(dim=1,keepdim=True)
            rays_,defVs_=utils.compute_cardinal_rays(self.deformer,initTmpPs_,rays_,defconds,batch_inds_,ratio,'test')
            with torch.no_grad():
                tcolors.append(utils.compute_netRender_color(self.netRender,initTmpPs_,defVs_,nx,rays_,self.sdf.rendcond,rendcond[batch_inds_],ratio))

        tcolors=torch.cat(tcolors,dim=0)
        # print((gtCs[batch_inds,row_inds,col_inds]-tcolors).abs().mean().item())
        tcolors=torch.clamp((tcolors/2.+0.5)*255.,min=0.,max=255.)
        colors=torch.ones(N,H,W,3,device=device)*255.
        colors[batch_inds,row_inds,col_inds,:]=tcolors
        if gts and 'image' in gts:
            colors[~masks]=gts['image'][~masks][:,:3]*255.

        colors=colors.cpu().numpy().astype(np.uint8)
        return colors,imgs,def1imgs,defMeshVs

    def save_debug(self,TmpVs,Tmpfs,defMeshes,offset,masks,gtMs,mgtMs,gtCs,batch_inds,row_inds,col_inds,initTmpPs,defconds,rendcond,ratio):
        if self.root is None:
            return

        mesh = trimesh.Trimesh(TmpVs.detach().cpu().numpy(), Tmpfs.cpu().numpy())
        mesh.export(osp.join(self.root,'tmp.ply'))

        for ind,(vs,fs) in enumerate(zip(defMeshes.verts_list(),defMeshes.faces_list())):
            mesh = trimesh.Trimesh(vs.view(TmpVs.shape[0],3).detach().cpu().numpy(), fs.cpu().numpy())
            mesh.export(osp.join(self.root,'def_%d.ply'%ind))
        N=offset.shape[0]
        defMeshes=Meshes(verts=[vs for vs in torch.split(TmpVs[None,:,:].expand(N,-1,3)+offset,1)],faces=[Tmpfs for _ in range(N)],textures=TexturesVertex([torch.ones_like(TmpVs) for _ in range(N)]))
        for ind,(vs,fs) in enumerate(zip(defMeshes.verts_list(),defMeshes.faces_list())):
            mesh = trimesh.Trimesh(vs.view(TmpVs.shape[0],3).detach().cpu().numpy(), fs.cpu().numpy())
            mesh.export(osp.join(self.root,'def1_%d.ply'%ind))

        images=(masks*255.).detach().cpu().numpy().astype(np.uint8)
        gtMasks=(gtMs*255.).detach().cpu().numpy().astype(np.uint8)
        for ind,(img,gtimg) in enumerate(zip(images,gtMasks)):
            cv2.imwrite(osp.join(self.root,'m%d.png'%ind),img)
            # cv2.imwrite(osp.join(self.root,'gm%d.png'%ind),gtimg)
        if mgtMs is not None:
            mgtMasks=(mgtMs*255.).detach().cpu().numpy().astype(np.uint8)
            for ind,mgtimg in enumerate(mgtMasks):
                cv2.imwrite(osp.join(self.root,'mgm%d.png'%ind),mgtimg)

        #debug draw the images
        if self.draw:
            cameras=self.maskRender.rasterizer.cameras
            with torch.no_grad():
                rays=cameras.view_rays(torch.cat([col_inds.view(-1,1),row_inds.view(-1,1),torch.ones_like(col_inds.view(-1,1))],dim=-1).float())

            tcolors=[]
            tnormals=[]
            # tlights=[]
            print('draw %d points'%rays.shape[0])
            number=20000
            for ind,(rays_,initTmpPs_,batch_inds_) in enumerate(zip(torch.split(rays,number),torch.split(initTmpPs,number),torch.split(batch_inds,number))):
                initTmpPs_,check=utils.OptimizeSurfacePs(cameras.cam_pos().detach(),rays_.detach(),initTmpPs_.clone(),batch_inds_,self.sdf,ratio,self.deformer,defconds,dthreshold=1.e-4,athreshold=self.angThred,w1=3.05,w2=1.,times=30)
                # print('%d:(%d,%d)'%(ind,rays_.shape[0],check.sum().item()))
                initTmpPs_.requires_grad=True
                sdfs=self.sdf(initTmpPs_,ratio)
                nx=torch.autograd.grad(sdfs,initTmpPs_,torch.ones_like(sdfs),retain_graph=False,create_graph=False)[0]
                nx=nx/nx.norm(dim=1,keepdim=True)
                rays_,defVs_=utils.compute_cardinal_rays(self.deformer,initTmpPs_,rays_,defconds,batch_inds_,ratio,'test')

                with torch.no_grad():
                    tcolors.append(utils.compute_netRender_color(self.netRender,initTmpPs_,defVs_,nx,rays_,self.sdf.rendcond,rendcond[batch_inds_],ratio))
                    with torch.enable_grad():
                        nx,_=utils.compute_deformed_normals(self.sdf,self.deformer,initTmpPs_,defconds,batch_inds_,ratio,'test')
                    tnormals.append((torch.tensor([[-1.,0.,0.],[0.,1.,0.],[0.,0.,-1.]],device=nx.device)@cameras.R[0].transpose(0,1)@nx.view(-1,3,1)).view(-1,3))

            tcolors=torch.cat(tcolors,dim=0)
            # print((gtCs[batch_inds,row_inds,col_inds]-tcolors).abs().mean().item())
            tcolors=torch.clamp((tcolors/2.+0.5)*255.,min=0.,max=255.)
            colors=torch.ones_like(gtCs)*255.
            colors[batch_inds,row_inds,col_inds,:]=tcolors
            colors=colors.cpu().numpy().astype(np.uint8)

            tnormals=torch.cat(tnormals,dim=0)
            tnormals=(tnormals*0.5+0.5)*255.
            normals=torch.ones_like(gtCs)*255.
            normals[batch_inds,row_inds,col_inds,:]=tnormals[:,[2,1,0]]
            normals=normals.cpu().numpy().astype(np.uint8)


            gtcolors=((gtCs/2.+0.5)*255.).cpu().numpy().astype(np.uint8)
            for ind,(color,gtcolor) in enumerate(zip(colors,gtcolors)):
                cv2.imwrite(osp.join(self.root,'rgb%d.png'%ind),color)
                # cv2.imwrite(osp.join(self.root,'light%d.png'%ind),lights[ind]) if lights is not None else None
                cv2.imwrite(osp.join(self.root,'gtrgb%d.png'%ind),gtcolor)
                cv2.imwrite(osp.join(self.root,'normal%d.png'%ind),normals[ind])

            self.draw=False


    # def forward(self,gtCs,gtMs,gtAs,sample_pix,ratio,frame_ids,root=None,**kwargs):
    def forward(self,datas,sample_pix,ratio,frame_ids,root=None,**kwargs):

        # data -> img, mask, normal
        device=frame_ids.device
        gtCs=datas['img'].to(device)
        gtMs=datas['mask'].to(device)

        # rebuild the computation graph from dataset cpu to gpu
        focals,princeple_ps,Rs,Ts,H,W=self.dataset.get_camera_parameters(frame_ids.numel(),device)

        cameras=RectifiedPerspectiveCameras(focals,princeple_ps,Rs,Ts,image_size=[(W, H)]).to(device)


        self.maskRender.rasterizer.cameras=cameras

        if self.pcRender:
            self.pcRender.rasterizer.cameras=cameras
        self.info={}
        self.root=None

        # every remesh_intersect time it will update marching cube to rebuild mesh
        if self.TmpVs is None or self.Tmpfs is None or self.forward_time%self.remesh_intersect==0:
            self.update_hierarchical_config(device)
            self.TmpVs,self.Tmpfs=self.discretizeSDF(ratio,None,-self.sdfShrinkRadius)
            if self.TmpVs.shape[0]==0:
                print('tmp sdf vanished...')
                assert(False)
            self.remesh_time=1.+np.floor(self.remesh_time)
            self.TmpVs.requires_grad=True
            # optimizer the sdf_vertices
            self.TmpOptimizer=torch.optim.SGD([self.TmpVs], lr=0.05, momentum=0.9)
            tm=om.TriMesh(self.TmpVs.detach().cpu().numpy(),self.Tmpfs.detach().cpu().numpy())
            # this vertices belongs to which faces
            TmpFid=torch.from_numpy(tm.vertex_face_indices().astype(np.int64)).to(device)
            TmpVid=torch.arange(0,TmpFid.shape[0]).view(-1,1).repeat(1,TmpFid.shape[1]).to(device)
            sel=TmpFid>=0


            # this is indices -> Fid, belongs to which face, Vid belongs to which vertices_id
            self.TmpFid=TmpFid[sel].view(-1)
            self.TmpVid=TmpVid[sel].view(-1)
            self.root=root


        oldTmpVs=self.TmpVs.detach().clone() if self.root else None
        # TmpVnum
        # the number of vertices after marching cube
        TmpVnum=self.TmpVs.shape[0]
        #N=3
        N=gtCs.shape[0]
        # d_cond->deform code, rendcond -> color render code
        poses,trans,d_cond,rendcond=self.dataset.get_grad_parameters(frame_ids,device)

        # poses (N, 24, 3)
        # trans (3 , 3)

        # NOTE d_cond is the deformer net by deform network
        # [pose trans] is

        # the pose fitted would flip x-axis , y -axis
        defTmpVs=self.deformer(self.TmpVs[None,:,:].expand(N,-1,3),[d_cond,[poses,trans]],ratio=ratio)
        # for def_idx, defTmpv in enumerate(defTmpVs):
        #     save_obj('./debug/{:04d}.obj'.format(def_idx), defTmpv, self.Tmpfs)

        defMeshes=Meshes(verts=[vs.view(TmpVnum,3) for vs in torch.split(defTmpVs,1)],faces=[self.Tmpfs for _ in range(N)])

        batch_inds=None
        self.info['pc_loss']={}
        # which
        with torch.no_grad():
            _,frags=self.maskRender(defMeshes)
            batch_inds,row_inds,col_inds,initTmpPs,front_face_ids=utils.FindSurfacePs(self.TmpVs.detach(),self.Tmpfs,frags)



        features=[torch.ones(self.TmpVs.shape[0],1,device=self.TmpVs.device) for _ in range(N)]
        # render points
        masks,frags=self.pcRender(Pointclouds(points=defMeshes.verts_list(),features=features))

        # for mask_idx, mask in enumerate(masks):
        #     mask = (mask * 255).detach().cpu().numpy().astype(np.uint8)[...,0]
        #     cv2.imwrite("./debug/{:04d}.png".format(mask_idx), mask)


        radius=self.pcRender.rasterizer.raster_settings.radius
        radius=int(np.round(radius/2.*float(min(H,W))/1.2))
        if radius>0:
            mgtMs=torch.nn.functional.max_pool2d(gtMs,kernel_size=2*radius+1,stride=1,padding=radius)
            # for mask_idx, mask in enumerate(mgtMs):
            #     mask = (mask * 255).detach().cpu().numpy().astype(np.uint8)
            #     cv2.imwrite("./debug/gt_{:04d}.png".format(mask_idx), mask)
            # optimal marching cube -> and then compute loss with sdf
            total_loss=self.computeTmpPcLoss(defMeshes,[d_cond,[poses,trans]],masks,mgtMs,ratio)
        else:
            mgtMs=None
            total_loss=self.computeTmpPcLoss(defMeshes,[d_cond,[poses,trans]],masks,gtMs,ratio)
        # in the deformer network also record offset lerarning from oldTmpVs
        self.save_debug(oldTmpVs,self.Tmpfs,defMeshes,self.deformer.defs[0].offset.view(N,-1,3).detach(),masks,gtMs,mgtMs,gtCs,batch_inds,row_inds,col_inds,initTmpPs,[d_cond.detach(),[poses.detach(),trans.detach()]],rendcond,ratio)


        # batch_inds, row_inds, col_inds -> is the rendered pixel has mesh using implict sdf mesh
        sel=gtMs[batch_inds,row_inds,col_inds]>0. #color loss only compute render mask and gt mask intersected part
        batch_inds=batch_inds[sel]
        row_inds=row_inds[sel]
        col_inds=col_inds[sel]
        initTmpPs=initTmpPs[sel]
        # pnum the number of pixel the implict filed overlap between gt mesh
        pnum=batch_inds.shape[0]
        #2000
        sample_pix=self.conf.get_int('sample_pix_num') if 'sample_pix_num' in self.conf else sample_pix




        # sample_ray
        if pnum>sample_pix*N:
            sel=torch.rand(pnum)<float(sample_pix*N)/float(pnum)
            sel=sel.to(batch_inds.device)
            batch_inds=batch_inds[sel]
            row_inds=row_inds[sel]
            col_inds=col_inds[sel]
            initTmpPs=initTmpPs[sel]
            pnum=batch_inds.shape[0]



        # very important part for SelfRecon
        # rebuild the computation graph from dataset cpu to gpu
        focals,princeple_ps,Rs,Ts,H,W=self.dataset.get_camera_parameters(frame_ids.numel(),device)
        # Rs, Ts is the camera to view matrix
        # the camera to view matrix flip the x and y matrix

        # NOTE that the screen coordinate need flip, y and x, hence the Rs, and Ts gradually is equal to [-1, -1, 1] rotation matrix
        cameras=RectifiedPerspectiveCameras(focals,princeple_ps,Rs,Ts,image_size=[(W, H)]).to(device)



        self.maskRender.rasterizer.cameras=cameras
        if self.pcRender:
            self.pcRender.rasterizer.cameras=cameras
        # cameras=self.maskRender.rasterizer.cameras


        rays=cameras.view_rays(torch.cat([col_inds.view(-1,1),row_inds.view(-1,1),torch.ones_like(col_inds.view(-1,1))],dim=-1).float())
        # TODO debug rays

        poses,trans,d_cond,rendcond=self.dataset.get_grad_parameters(frame_ids,device)

        defconds=[d_cond,[poses,trans]]

        # initTmpPs is the centbary pts ray intersect
        # athreshold forces the deformer network is 1 pixels
        # This function optimal surface face barycentric pts, satisfied close to surface and also arsin deform and original direction is limited to 1 pixel
        # replace the sphere intecsection by optimizeSurfacePs


        initTmpPs,check=utils.OptimizeSurfacePs(cameras.cam_pos().detach(),rays.detach(),initTmpPs,batch_inds,self.sdf,ratio,self.deformer,defconds,dthreshold=5.e-5,athreshold=self.angThred,w1=3.05,w2=1.,times=10)




        self.info['rayInfo']=(check.numel(),check.sum().item())
        self.TmpPs=None
        # decrease sample to save memory
        nonmnfld_pnts=utils.sample_points(torch.cat([initTmpPs,self.TmpVs[torch.rand(TmpVnum)<4096./float(TmpVnum)].detach()],dim=0), 1.8,0.01)

        nonmnfld_pnts.requires_grad_()
        nonmnfld_pred = self.sdf(nonmnfld_pnts,ratio)
        nonmnfld_grad=self.sdf.gradient(nonmnfld_pnts,nonmnfld_pred)
        # Eikonal Loss
        grad_loss = ((nonmnfld_grad.norm(2, dim=-1) - 1) ** 2).mean()
        self.info['grad_loss']=grad_loss.item()
        total_loss+=grad_loss*self.conf.get_float('grad_weight')



        if 'offset_weight' in self.conf and self.conf.get_float('offset_weight')>0.:
            self.deformer.defs[0](nonmnfld_pnts.view(1,-1,3).expand(N,-1,3),d_cond,ratio=ratio)
            def_offloss=self.deformer.defs[0].offset.view(-1,3).norm(p=2,dim=-1).mean()
            self.info['offset_loss']=def_offloss.item()
            total_loss+=def_offloss*self.conf.get_float('offset_weight')
        elif 'offset_weight' in self.conf and self.conf.get_float('offset_weight')==0.:
            with torch.no_grad():
                self.deformer.defs[0](nonmnfld_pnts.view(1,-1,3).expand(N,-1,3),d_cond,ratio=ratio)
                def_offloss=self.deformer.defs[0].offset.view(-1,3).norm(p=2,dim=-1).mean()
                self.info['offset_loss']=def_offloss.item()

        nonmnfld_pnts=None
        # 0.1
        if 'def_regu' in self.conf and self.conf.get_float('def_regu.weight')>0.:
            if nonmnfld_pnts is None:
                # initTmps and TmpVs are all in canonical space
                # Note that only using barycentric pts and TmpVs
                nonmnfld_pnts=torch.cat([initTmpPs,self.TmpVs[torch.rand(TmpVnum)<4096./float(TmpVnum)].detach()],dim=0)
                # without global only surface
                nonmnfld_pnts=torch.cat([nonmnfld_pnts,utils.sample_points(nonmnfld_pnts,1.8,0.01,0)],dim=0).view(1,-1,3).expand(N,-1,3)
                nonmnfld_pnts.requires_grad_()



            # this deform only in canonical space
            defVs=self.deformer.defs[0](nonmnfld_pnts,d_cond,ratio=ratio)
            # core self_reconstrct
            Jacobs=utils.compute_Jacobian(nonmnfld_pnts,defVs,True,True)

            _,s,_=torch.svd(Jacobs.cpu()) #for pytorch, the gpu svd is too slow

            s=torch.log(s.to(device))

            # print(s.norm(dim=1)[0:10])
            # assert(False)


            def_loss=utils.GMRobustError((s*s).sum(1),self.conf.get_float('def_regu.c'),True).mean()
            self.info['def_loss']=def_loss.item()
            total_loss+=def_loss*self.conf.get_float('def_regu.weight')


        # True, True
        if (poses.requires_grad or trans.requires_grad) and self.conf.get_float('dct_weight')>0.:

            klen,Nlen=self.dctnull.shape
            batch_poses,pindices=self.dataset.get_batchframe_data('poses',frame_ids,Nlen)
            batch_trans,tindices=self.dataset.get_batchframe_data('trans',frame_ids,Nlen)
            posedJs=self.deformer.defs[1].posedSkeleton([batch_poses.reshape(N*Nlen,24,3),batch_trans.reshape(N*Nlen,3)])
            # Null spcae is the orthogoal space where, (0-10) lowes frequency, we force the higher frequency to be closed to zero
            # hence dctnull space -> (20,30)  for given 30 frame joints sequence
            dct_loss=self.dctnull[None,:,:].matmul(posedJs.reshape(N,Nlen,72))
            dct_loss=dct_loss.abs().mean()
            total_loss+=dct_loss*self.conf.get_float('dct_weight')
            self.info['dct_loss']=dct_loss.item()

        self.info['color_loss']=-1.0


        # rayinfo -> (number_of_ray, valid_ray)
        if self.info['rayInfo'][1]>0:
            self.TmpPs=initTmpPs[check]
            self.TmpPs.requires_grad=True
            self.rays=rays[check]
            self.batch_inds=batch_inds[check]
            self.col_inds=col_inds[check]
            self.row_inds=row_inds[check]
            #(tmpVs: xyz + xyz_embeding)
            sdfs=self.sdf(self.TmpPs,ratio)

            nx=torch.autograd.grad(sdfs,self.TmpPs,torch.ones_like(sdfs),retain_graph=True,create_graph=True)[0]
            nx=nx/nx.norm(dim=1,keepdim=True)

            # canonical space crays(settting invalid ray equal to zeros, defVs, is the view_pose pt)
            crays,defVs=utils.compute_cardinal_rays(self.deformer,self.TmpPs,self.rays,defconds,self.batch_inds,ratio,'train')
            if self.conf.get_float('color_weight')>0.:
                # render color
                # inputs ->(render network, template vertices, deform verts, vertice normal in sdf, canonical_rays, sdf_conition, frame condition)
                # surface rendering
                colors=utils.compute_netRender_color(self.netRender,self.TmpPs,defVs,nx,crays,self.sdf.rendcond,rendcond[self.batch_inds],ratio)

            # color loss
            # 0.5
            if self.conf.get_float('color_weight')>0.:
                color_loss=(gtCs[self.batch_inds,self.row_inds,self.col_inds,:]-colors).abs().sum(1)

                color_loss=scatter(color_loss,self.batch_inds,reduce='mean',dim_size=N).mean()
                self.info['color_loss']=color_loss.item()
                total_loss+=self.conf.get_float('color_weight')*color_loss

            if 'normal' in datas and 'normal_weight' in self.conf and self.conf.get_float('normal_weight')>0.:
                if 'weighted_normal' in self.conf and self.conf.get_bool('weighted_normal'):
                    cnx,_=utils.compute_deformed_normals(self.sdf,self.deformer,self.TmpPs,defconds,self.batch_inds,ratio,'test')
                    # negative as the ray_view with cnx is oppsite direction
                    weights=torch.clamp((-self.rays*cnx.detach()).sum(1).detach(),max=1.,min=0.)**2
                else:
                    weights=torch.ones(nx.shape[0],device=device)
                gtnormals=datas['normal'].to(device)
                gtnormals=gtnormals[self.batch_inds,self.row_inds,self.col_inds,:]
                # flip x, y
                # why -> [[-1, 0, 0],[0, 1, 0],[0, 0, -1]]
                # as the camera show, the normal map from pose view is negative x, z, this means it point in the screen.
                # however the gt normal is point out the screen, hence it needs flip -> x and flip -> z
                gtnormals=((cameras.R[0]@torch.tensor([[-1.,0.,0.],[0.,1.,0.],[0.,0.,-1.]],device=device))@gtnormals.view(-1,3,1)).view(-1,3)
                gtnorms=gtnormals.norm(dim=1,keepdim=True)
                valid_mask=(gtnorms>0.0001)[...,0]
                gtnormals[valid_mask]=gtnormals[valid_mask]/gtnorms[valid_mask]
                ds=self.deformer(self.TmpPs,defconds,self.batch_inds,ratio=ratio)
                grad_d_p=utils.compute_Jacobian(self.TmpPs,ds,True,True)
                gtnormals=(grad_d_p.transpose(-2,-1)@gtnormals.view(-1,3,1)).view(-1,3)
                normal_loss = (gtnormals - nx).norm(2, dim=1)*weights
                normal_loss=scatter(normal_loss[valid_mask],self.batch_inds[valid_mask],reduce='mean',dim_size=N).mean()
                self.info['normal_loss']=normal_loss.item()
                total_loss+=self.conf.get_float('normal_weight')*normal_loss


        self.remesh_time=np.floor(self.remesh_time)+float(self.forward_time%self.remesh_intersect)/float(self.remesh_intersect)
        self.info['remesh']=self.remesh_time
        self.forward_time+=1
        return total_loss


    def computeTmpPcLoss(self,defMeshes,defconds,imgs,gtMs,ratio):
        # Mask loss
        N=gtMs.shape[0]
        masks=imgs[...,-1]
        mask_loss=(1.-(masks*gtMs).view(N,-1).sum(1)/(masks+gtMs-masks*gtMs).abs().view(N,-1).sum(1)).mean()



        self.info['pc_loss']['mask_loss']=mask_loss.item()
        loss=mask_loss*(self.conf.get_float('pc_weight.mask_weight') if 'pc_weight.mask_weight' in self.conf else 1.)

        tmpMesh=Meshes(verts=[self.TmpVs],faces=[self.Tmpfs])
        lap_weight=self.conf.get_float('pc_weight.laplacian_weight') if 'pc_weight' in self.conf else -1.

        # template loss : laplacian_loss, edge_loss, norm loss
        if lap_weight>0.:
            lap_loss=lap_weight*mesh_laplacian_smoothing(tmpMesh,method='uniform')
            loss=loss+lap_loss
            self.info['pc_loss']['lap_loss']=lap_loss.item()/lap_weight
        edge_weight=self.conf.get_float('pc_weight.edge_weight') if 'pc_weight' in self.conf else -1.
        if edge_weight>0.:
            edge_loss=edge_weight*mesh_edge_loss(tmpMesh,target_length=0.)
            loss=loss+edge_loss
            self.info['pc_loss']['edge_loss']=edge_loss.item()/edge_weight
        norm_weight=self.conf.get_float('pc_weight.norm_weight') if 'pc_weight' in self.conf else -1.
        if norm_weight>0.:
            norm_loss=norm_weight*mesh_normal_consistency(tmpMesh)
            loss=loss+norm_loss
            self.info['pc_loss']['norm_loss']=norm_loss.item()/norm_weight


        consistent_weight=self.conf.get_float('pc_weight.def_consistent.weight') if 'pc_weight.def_consistent' in self.conf else -1.
        if consistent_weight>0.:

            # defs[1] -> LBSkinner
            # Note that this is consistency the deform template is close to original template
            offset2=(defMeshes.verts_padded()-self.deformer.defs[1](self.TmpVs.view(1,-1,3).expand(N,-1,3),defconds[1]))
            offset2=(offset2*offset2).sum(-1)
            if self.conf.get_float('pc_weight.def_consistent.c')>0.:
                consistent_loss=utils.GMRobustError(offset2,self.conf.get_float('pc_weight.def_consistent.c'),True).mean()
            else:
                consistent_loss=torch.sqrt(offset2).mean()
            self.info['pc_loss']['defconst_loss']=consistent_loss.item()
            loss=loss+consistent_loss*consistent_weight


        # self.TmpOptimzier is optimal the vertices of template_field
        self.TmpOptimizer.zero_grad()
        loss.backward()
        self.TmpOptimizer.step()

        mnfld_pred=self.sdf(self.TmpVs,ratio).view(-1)
        sdf_loss=(mnfld_pred+self.sdfShrinkRadius).abs().mean()
        self.info['pc_loss_sdf']=sdf_loss.item()
        loss=sdf_loss*(self.conf.get_float('pc_weight.weight') if 'pc_weight' in self.conf else 60.)

        return loss

    # this func must be invoked after forward() getting loss and has run loss.backward()
    # this func propagates grad(TmpPs) to sdf, deformer and camera parameters and conds grad(if requires_grad)
    # to do: check whether torch Function can handle this problem, that will simplify the use
    def propagateTmpPsGrad(self,frame_ids,ratio):
        if self.TmpPs is None or self.TmpPs.grad is None:
            self.info['invInfo']=(-1,-1)
            return

        device=self.TmpPs.device
        poses,trans,d_cond,_=self.dataset.get_grad_parameters(frame_ids,device)

        defconds=[d_cond,[poses,trans]]
        focals,princeple_ps,Rs,Ts,H,W=self.dataset.get_camera_parameters(frame_ids.numel(),device)
        self.maskRender.rasterizer.cameras=RectifiedPerspectiveCameras(focals,princeple_ps,Rs,Ts,image_size=[(W, H)]).to(device)
        # pl_pp, loss partial L partial p
        grad_l_p=self.TmpPs.grad


        # to make v can backward to camera parameters
        # after loss.backward(), the computation graph of self.rays to camera params has been destroyed, need to be recomputed
        if self.rays.requires_grad:
            v=self.maskRender.rasterizer.cameras.view_rays(torch.cat([self.col_inds.view(-1,1),self.row_inds.view(-1,1),torch.ones_like(self.col_inds.view(-1,1))],dim=-1).float())
        else:
            v=self.rays.detach()

        c=self.maskRender.rasterizer.cameras.cam_pos()
        p=self.TmpPs #[N,3]
        f=self.sdf(p,ratio)

        grad_f_p=torch.autograd.grad(f,p,torch.ones_like(f),retain_graph=False)[0] #[N,3]

        if hasattr(self.deformer.defs[0],'enableSdfcond'):
            self.sdf(p,ratio)
            d=self.deformer(p,[[defconds[0],self.sdf.rendcond],defconds[1]],self.batch_inds,ratio=ratio)
        else:
            d=self.deformer(p,defconds,self.batch_inds,ratio=ratio) #[N,3]


        inputs=[p]
        grad_d_p=[]
        opt_defconds=[]


        # deform embedding, trans, and pose condition
        for defcond in defconds:
            if type(defcond) is list:
                for defc in defcond:
                    if defc.requires_grad:
                        opt_defconds.append(defc)
            else:
                if defcond.requires_grad:
                    opt_defconds.append(defcond)
        grad_outputs=torch.ones_like(d[...,0])

        outx=torch.autograd.grad(d[...,0],inputs,grad_outputs,retain_graph=True)
        grad_d_p.append(outx[0].view(-1,1,3))
        # if len(opt_defconds):
        #     grad_d_z.append([grad.view(-1,1,opt_defcond.shape[-1]) for grad,opt_defcond in zip(outx[1:],opt_defconds)])
        outy=torch.autograd.grad(d[...,1],inputs,grad_outputs,retain_graph=True)
        grad_d_p.append(outy[0].view(-1,1,3))
        # if len(opt_defconds):
        #     grad_d_z.append([grad.view(-1,1,opt_defcond.shape[-1]) for grad,opt_defcond in zip(outy[1:],opt_defconds)])
        outz=torch.autograd.grad(d[...,2],inputs,grad_outputs,retain_graph=False)
        grad_d_p.append(outz[0].view(-1,1,3))
        # if len(opt_defconds):
        #     grad_d_z.append([grad.view(-1,1,opt_defcond.shape[-1]) for grad,opt_defcond in zip(outz[1:],opt_defconds)])
        grad_d_p=torch.cat(grad_d_p,dim=1) #N,3,3
        # if len(opt_defconds):
        #     grad_d_z=[torch.cat([gradx,grady,gradz],dim=1) for gradx,grady,gradz in zip(grad_d_z[0],grad_d_z[1],grad_d_z[2])]
        v_cross=torch.zeros_like(grad_d_p)
        # cross matrix
        # [0 -a3 a2]
        # [a3 0 -a1]
        # [-a2 a1 0]


        v_cross[:,0,1]=-v[:,2]
        v_cross[:,0,2]=v[:,1]
        v_cross[:,1,0]=v[:,2]
        v_cross[:,1,2]=-v[:,0]
        v_cross[:,2,0]=-v[:,1]
        v_cross[:,2,1]=v[:,0]
        v_cross=v_cross.detach()



        # [B 4, 3]
        a1=v_cross.matmul(grad_d_p)

        b=torch.cat([grad_f_p.view(-1,1,3),a1],dim=1)
        # the gradient d_p_n need satisfied:
        # b @ d_p_n = 0

        btb=b.permute(0,2,1).matmul(b)
        # pseudo_inverse
        btb_inv,check=Fast3x3Minv(btb)

        self.info['invInfo']=(check.numel(),check.sum().item())


        # (A^t @ A) ^ -1 @ A^t

        # rhs_1 = (A^t @ A) ^ -1 @ A^t -> [N, 3, 4]
        # grad_l_p -> [N, 3] -> [N, 1, 3]
        # grad_l_p @ pesudo_inverse A^t
        rhs_1=btb_inv.matmul(b.permute(0,2,1)) #N,3,4
        rhs_1=grad_l_p.view(-1,1,3).matmul(rhs_1) #N,1,4

        loss=0.
        #for theta
        params=[param for param in self.sdf.parameters() if param.requires_grad]
        # grad_l_p @ grad_p_n
        params_grads=torch.autograd.grad(self.sdf(p,ratio),params,-rhs_1[:,:,0])
        for param,grad in zip(params,params_grads):
            loss+=(param*grad).sum()
        #for phi
        params=[param for param in self.deformer.parameters() if param.requires_grad]



        if hasattr(self.deformer.defs[0],'enableSdfcond'):
            self.sdf(p,ratio)
            d=self.deformer(p,[[defconds[0],self.sdf.rendcond],defconds[1]],self.batch_inds,ratio=ratio)
        else:
            d=self.deformer(p,defconds,self.batch_inds,ratio=ratio) #[N,3]

        temp=(rhs_1[:,:,-3:].matmul(-v_cross)).view(-1,3)

        if len(opt_defconds):
            params_grads=torch.autograd.grad(d,params,temp,retain_graph=True)
        else:
            params_grads=torch.autograd.grad(d,params,temp,retain_graph=False)
        for param,grad in zip(params,params_grads):
            loss+=(param*grad).sum()
        #for z
        if len(opt_defconds):
            params_grads=torch.autograd.grad(d,opt_defconds,temp,retain_graph=False)
            for opt_defcond,grad in zip(opt_defconds,params_grads):
                loss+=(opt_defcond*grad).sum()

        if v.requires_grad:
            dc=d.detach()-c.detach().view(1,3)
            dc_cross=torch.zeros_like(grad_d_p)
            dc_cross[:,0,1]=-dc[:,2]
            dc_cross[:,0,2]=dc[:,1]
            dc_cross[:,1,0]=dc[:,2]
            dc_cross[:,1,2]=-dc[:,0]
            dc_cross[:,2,0]=-dc[:,1]
            dc_cross[:,2,1]=dc[:,0]
            grad=rhs_1[:,:,-3:].matmul(dc_cross).view(-1,3)
            # to do: how to propagate v and c grads to camera parameters?
            loss+=(v*grad).sum()

        if c.requires_grad:
            grad=-temp.sum(0)
            loss+=(c*grad).sum()
        loss.backward()
