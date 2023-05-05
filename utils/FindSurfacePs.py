import numpy as np
import torch
from torch_scatter import scatter
import pdb

# for just a sequence with one camera
def FindSurfacePs(TmpVs,TmpFaces,frags):
    # It returns bary coordinate which belongs to intersect surface when rendering
    # and also batch_idx row_idx, col_idx and find_idx
    # NOTE that in camera, all coordinate >0 is valid
    # 3 1080 1080 1
    N,H,W,K=frags.pix_to_face.shape

    pix_to_face=frags.pix_to_face

    # NOTE that pytorch3d bary_coords represnet the weighted each vertex in current face
    bary_coords=frags.bary_coords
    innerCheck1=(bary_coords>0.0).all(-1)

    innerCheck2=pix_to_face>=0
    innerCheck=(innerCheck1*innerCheck2)


    # innerCheck=pix_to_face>=0

    rows,cols=innerCheck.view(-1,K).nonzero(as_tuple=True)
    index=torch.ones(N*H*W,dtype=torch.long,device=rows.device)*K

    # make innerCheck position equal to 0
    index=scatter(cols,rows,reduce='min',out=index).view(N,H,W)
    # index=scatter(cols,rows,reduce='min')

    innerCheck=innerCheck.any(dim=-1)


    batch_inds,row_inds,col_inds=innerCheck.nonzero(as_tuple=True)
    # batch_inds=torch.arange(N).to(index.device).view(-1,1).repeat(1,H*W).view(N,H,W)[innerCheck]
    # (innerpts , 1)

    finds=torch.gather(pix_to_face[innerCheck],1,index[innerCheck].view(-1,1)).view(-1)
    #convert finds from packed to single index
    finds=finds%TmpFaces.shape[0]
    # B H W K 3 bary_coords
    # number_innercheck, k, 3

    # 422751, 1, 3
    # print(index[innerCheck].view(-1,1,1).expand(-1,1,3).shape)

    # out[i][j][k] = input[index[i][j][k]][j][k]  # if dim == 0
    # out[i][j][k] = input[i][index[i][j][k]][k]  # if dim == 1
    # out[i][j][k] = input[i][j][index[i][j][k]]  # if dim == 2
    # print(torch.gather(bary_coords[innerCheck],1,index[innerCheck].view(-1,1,1).expand(-1,1,3)))
    ws=torch.gather(bary_coords[innerCheck],1,index[innerCheck].view(-1,1,1).expand(-1,1,3)).view(-1,3)



    initTmpPs=(TmpVs[TmpFaces[finds].view(-1)].view(-1,3,3)*ws[:,:,None]).sum(1)
    # rays=Camera(torch.cat([col_inds.view(-1,1),row_inds.view(-1,1),torch.ones_like(col_inds.view(-1,1))],dim=-1))

    return batch_inds,row_inds,col_inds,initTmpPs,finds

def FindSurfacePs_batch(TmpVs,defmeshes,frags):
    N,H,W,K=frags.pix_to_face.shape
    pix_to_face=frags.pix_to_face
    bary_coords=frags.bary_coords
    innerCheck1=(bary_coords>-0.01).all(-1)
    innerCheck2=pix_to_face>=0
    innerCheck=(innerCheck1*innerCheck2)
    rows,cols=innerCheck.view(-1,K).nonzero(as_tuple=True)
    index=torch.ones(N*H*W,dtype=torch.long,device=rows.device)*K
    index=scatter(cols,rows,reduce='min',out=index).view(N,H,W)

    innerCheck=innerCheck.any(dim=-1)
    batch_inds,row_inds,col_inds=innerCheck.nonzero(as_tuple=True)
    # batch_inds=torch.arange(N).to(index.device).view(-1,1).repeat(1,H*W).view(N,H,W)[innerCheck]
    finds=torch.gather(pix_to_face[innerCheck],1,index[innerCheck].view(-1,1)).view(-1)
    ws=torch.gather(bary_coords[innerCheck],1,index[innerCheck].view(-1,1,1).expand(-1,1,3)).view(-1,3)
    #convert packed to single index
    initTmpPs=(TmpVs[(defmeshes.faces_packed()[finds]%TmpVs.shape[0]).view(-1)].view(-1,3,3)*ws[:,:,None]).sum(1)

    # rays=Camera(torch.cat([col_inds.view(-1,1),row_inds.view(-1,1),torch.ones_like(col_inds.view(-1,1))],dim=-1))

    return batch_inds,row_inds,col_inds,initTmpPs

# def OptimizeSurfacePs(camera,rays,initTmpPs,batch_inds,tmpSdf,ratio,deformer,defconds,dthreshold=5.e-5,athreshold=0.02,w1=3.05,w2=1.,times=5):
#     cam_pos=camera.cam_pos().detach()
#     with torch.no_grad():
#         check1=tmpSdf(initTmpPs,ratio).view(-1).abs()<dthreshold
#         if hasattr(deformer,'defs') and hasattr(deformer.defs[0],'enableSdfcond'):
#             direct=deformer(initTmpPs,[[defconds[0],tmpSdf.rendcond],defconds[1]],batch_inds,ratio=ratio)-cam_pos.view(1,3)
#         else:
#             direct=deformer(initTmpPs,defconds,batch_inds,ratio=ratio)-cam_pos.view(1,3)
#         up=torch.cross(direct,rays,dim=1)
#         check2=torch.arcsin(up.norm(dim=1)/direct.norm(dim=1))*180./np.pi < athreshold
#         # print('%f:%f'%(tmpSdf(initTmpPs,ratio).view(-1).abs().mean().item(),(torch.arcsin(up.norm(dim=1)/direct.norm(dim=1))*180./np.pi).mean().item()))
#         check=check1*check2
#         unfinished=torch.ones(initTmpPs.shape[0],device=initTmpPs.device,dtype=torch.bool)
#         unfinished[check]=False
#     # initTmpPs.requires_grad_(True)
#     indices=torch.arange(unfinished.shape[0]).to(unfinished.device)
#     for ind in range(times):
#         # print('%d:%d'%(ind,unfinished.sum()))
#         curPs=initTmpPs[unfinished].detach().clone()
#         if curPs.shape[0]==0:
#             break
#         curPs.requires_grad_(True)
#         loss1=(tmpSdf(curPs,ratio).abs()).view(-1)
#         if hasattr(deformer,'defs') and hasattr(deformer.defs[0],'enableSdfcond'):
#             defPs=deformer(curPs,[[defconds[0],tmpSdf.rendcond],defconds[1]],batch_inds[unfinished],ratio=ratio)
#         else:
#             defPs=deformer(curPs,defconds,batch_inds[unfinished],ratio=ratio)
#         direct=defPs-cam_pos.view(1,3)
#         up=torch.cross(direct,rays[unfinished],dim=1)
#         # down=(direct*direct).sum(1)
#         # down[down<1.e-4]=1.e-4
#         # loss2=(up*up).sum(1)/down
#         loss2=(up.norm(dim=1)/direct.norm(dim=1)).abs()
#         loss=w1*loss1+w2*loss2
#         Loss=loss.sum()
#         grad=torch.autograd.grad(Loss,curPs,retain_graph=False,create_graph=False,only_inputs=True)[0]
#         #descent is gradient direction, later can test pinverse, maybe faster
#         t=-loss/(grad*grad).sum(1)
#         curPs=(curPs+t.view(-1,1)*grad).detach()
#         initTmpPs[unfinished]=curPs
#         with torch.no_grad():
#             check1=tmpSdf(curPs,ratio).view(-1).abs()<dthreshold
#             if hasattr(deformer,'defs') and hasattr(deformer.defs[0],'enableSdfcond'):
#                 direct=deformer(curPs,[[defconds[0],tmpSdf.rendcond],defconds[1]],batch_inds[unfinished],ratio=ratio)-cam_pos.view(1,3)
#             else:
#                 direct=deformer(curPs,defconds,batch_inds[unfinished],ratio=ratio)-cam_pos.view(1,3)
#             up=torch.cross(direct,rays[unfinished],dim=1)
#             check2=torch.arcsin(up.norm(dim=1)/direct.norm(dim=1))*180./np.pi < athreshold
#             check=check1*check2
#             unfinished[indices[unfinished][check]]=False
#     # print(torch.nn.functional.normalize(direct,dim=1))
#     initTmpPs=initTmpPs.detach()
#     with torch.no_grad():
#         if hasattr(deformer,'defs') and hasattr(deformer.defs[0],'enableSdfcond'):
#             ds=deformer(initTmpPs,[[defconds[0],tmpSdf.rendcond],defconds[1]],batch_inds,ratio=ratio)
#         else:
#             ds=deformer(initTmpPs,defconds,batch_inds,ratio=ratio)
#         [col_inds,row_inds]=camera.project(ds).unbind(1)
#     return initTmpPs,row_inds,col_inds,~unfinished

def OptimizeSurfacePs(cam_pos,rays,initTmpPs,batch_inds,tmpSdf,ratio,deformer,defconds,dthreshold=5.e-5,athreshold=0.02,w1=3.05,w2=1.,times=5):
    '''
    This function optimal surface face barycentric pts, satisfied close to surface and also arsin deform and original direction is limited to 1 pixel
    w1 is control sdf loss
    w2 is control arcsin loss between ray and deform nodes
    '''

    # this function to find the face surface satisfied with in surface, second the deform is controlled in one pixels


    with torch.no_grad():
        check1=tmpSdf(initTmpPs,ratio).view(-1).abs()<dthreshold
        if hasattr(deformer,'defs') and hasattr(deformer.defs[0],'enableSdfcond'):
            direct=deformer(initTmpPs,[[defconds[0],tmpSdf.rendcond],defconds[1]],batch_inds,ratio=ratio)-cam_pos.view(1,3)
        else:
            direct=deformer(initTmpPs,defconds,batch_inds,ratio=ratio)-cam_pos.view(1,3)
        up=torch.cross(direct,rays,dim=1)
        check2=torch.arcsin(up.norm(dim=1)/direct.norm(dim=1))*180./np.pi < athreshold
        # print('%f:%f'%(tmpSdf(initTmpPs,ratio).view(-1).abs().mean().item(),(torch.arcsin(up.norm(dim=1)/direct.norm(dim=1))*180./np.pi).mean().item()))
        check=check1*check2
        unfinished=torch.ones(initTmpPs.shape[0],device=initTmpPs.device,dtype=torch.bool)
        unfinished[check]=False
    # initTmpPs.requires_grad_(True)
    indices=torch.arange(unfinished.shape[0]).to(unfinished.device)

    # process the pts not satisfied
    for ind in range(times):
        # print('%d:%d'%(ind,unfinished.sum()))
        curPs=initTmpPs[unfinished].detach().clone()
        if curPs.shape[0]==0:
            break
        curPs.requires_grad_(True)
        loss1=(tmpSdf(curPs,ratio).abs()).view(-1)
        if hasattr(deformer,'defs') and hasattr(deformer.defs[0],'enableSdfcond'):
            defPs=deformer(curPs,[[defconds[0],tmpSdf.rendcond],defconds[1]],batch_inds[unfinished],ratio=ratio)
        else:

            defPs=deformer(curPs,defconds,batch_inds[unfinished],ratio=ratio)
        direct=defPs-cam_pos.view(1,3)
        up=torch.cross(direct,rays[unfinished],dim=1)
        # down=(direct*direct).sum(1)
        # down[down<1.e-4]=1.e-4
        # loss2=(up*up).sum(1)/down
        loss2=(up.norm(dim=1)/direct.norm(dim=1)).abs()
        loss=w1*loss1+w2*loss2
        Loss=loss.sum()
        grad=torch.autograd.grad(Loss,curPs,retain_graph=False,create_graph=False,only_inputs=True)[0]
        #descent is gradient direction, later can test pinverse, maybe faster
        # loss is larger move is larget
        t=-loss/(grad*grad).sum(1)
        curPs=(curPs+t.view(-1,1)*grad).detach()
        initTmpPs[unfinished]=curPs
        with torch.no_grad():
            check1=tmpSdf(curPs,ratio).view(-1).abs()<dthreshold
            if hasattr(deformer,'defs') and hasattr(deformer.defs[0],'enableSdfcond'):
                direct=deformer(curPs,[[defconds[0],tmpSdf.rendcond],defconds[1]],batch_inds[unfinished],ratio=ratio)-cam_pos.view(1,3)
            else:
                direct=deformer(curPs,defconds,batch_inds[unfinished],ratio=ratio)-cam_pos.view(1,3)
            up=torch.cross(direct,rays[unfinished],dim=1)
            check2=torch.arcsin(up.norm(dim=1)/direct.norm(dim=1))*180./np.pi < athreshold
            check=check1*check2
            unfinished[indices[unfinished][check]]=False
    return initTmpPs.detach(),~unfinished


def OptimizeGarmentSurfaceSinlge(cam_pos,rays,initTmpPs,batch_inds,tmpSdf,ratio,deformer,defconds,dthreshold=5.e-5,athreshold=0.02,w1=3.05,w2=1.,times=5, offset_type=None):
    '''
    This function optimal surface face barycentric pts, satisfied close to surface and also arsin deform and original direction is limited to 1 pixel
    w1 is control sdf loss
    w2 is control arcsin loss between ray and deform nodes
    '''

    # this function to find the face surface satisfied with in surface, second the deform is controlled in one pixels


    with torch.no_grad():
        check1=tmpSdf(initTmpPs,ratio).view(-1).abs()<dthreshold
        if hasattr(deformer,'defs') and hasattr(deformer.defs[0],'enableSdfcond'):
            direct=deformer(initTmpPs,[[defconds[0],tmpSdf.rendcond],defconds[1]],batch_inds,ratio=ratio, offset_type = offset_type)-cam_pos.view(1,3)
        else:
            direct=deformer(initTmpPs,defconds,batch_inds,ratio=ratio, offset_type = offset_type)-cam_pos.view(1,3)
        up=torch.cross(direct,rays,dim=1)
        check2=torch.arcsin(up.norm(dim=1)/direct.norm(dim=1))*180./np.pi < athreshold
        # print('%f:%f'%(tmpSdf(initTmpPs,ratio).view(-1).abs().mean().item(),(torch.arcsin(up.norm(dim=1)/direct.norm(dim=1))*180./np.pi).mean().item()))
        check=check1*check2
        unfinished=torch.ones(initTmpPs.shape[0],device=initTmpPs.device,dtype=torch.bool)
        unfinished[check]=False
    # initTmpPs.requires_grad_(True)
    indices=torch.arange(unfinished.shape[0]).to(unfinished.device)

    # process the pts not satisfied
    for ind in range(times):
        # print('%d:%d'%(ind,unfinished.sum()))
        curPs=initTmpPs[unfinished].detach().clone()
        if curPs.shape[0]==0:
            break
        curPs.requires_grad_(True)
        loss1=(tmpSdf(curPs,ratio).abs()).view(-1)
        if hasattr(deformer,'defs') and hasattr(deformer.defs[0],'enableSdfcond'):
            defPs=deformer(curPs,[[defconds[0],tmpSdf.rendcond],defconds[1]],batch_inds[unfinished],ratio=ratio)
        else:

            defPs=deformer(curPs,defconds,batch_inds[unfinished],ratio=ratio, offset_type = offset_type)
        direct=defPs-cam_pos.view(1,3)
        up=torch.cross(direct,rays[unfinished],dim=1)
        # down=(direct*direct).sum(1)
        # down[down<1.e-4]=1.e-4
        # loss2=(up*up).sum(1)/down
        loss2=(up.norm(dim=1)/direct.norm(dim=1)).abs()
        loss=w1*loss1+w2*loss2
        Loss=loss.sum()
        grad=torch.autograd.grad(Loss,curPs,retain_graph=False,create_graph=False,only_inputs=True)[0]
        #descent is gradient direction, later can test pinverse, maybe faster
        # loss is larger move is larget
        t=-loss/(grad*grad).sum(1)
        curPs=(curPs+t.view(-1,1)*grad).detach()
        initTmpPs[unfinished]=curPs
        with torch.no_grad():
            check1=tmpSdf(curPs,ratio).view(-1).abs()<dthreshold
            if hasattr(deformer,'defs') and hasattr(deformer.defs[0],'enableSdfcond'):
                direct=deformer(curPs,[[defconds[0],tmpSdf.rendcond],defconds[1]],batch_inds[unfinished],ratio=ratio)-cam_pos.view(1,3)
            else:
                direct=deformer(curPs,defconds,batch_inds[unfinished],ratio=ratio, offset_type = offset_type)-cam_pos.view(1,3)
            up=torch.cross(direct,rays[unfinished],dim=1)
            check2=torch.arcsin(up.norm(dim=1)/direct.norm(dim=1))*180./np.pi < athreshold
            check=check1*check2
            unfinished[indices[unfinished][check]]=False
    return initTmpPs.detach(),~unfinished
def OptimizeGarmentSurfacePs(cam_pos,rays_list,initTmpPs_list,batch_inds_list,tmpSdf_nets, ratio, deformer, defconds_list, garment_names, dthreshold=5.e-5, athreshold=0.02, w1=3.05, w2=1., times=5):
    '''
    This function optimal surface face barycentric pts, satisfied close to surface and also arsin deform and original direction is limited to 1 pixel
    w1 is control sdf loss
    w2 is control arcsin loss between ray and deform nodes

    '''

    # this function to find the face surface satisfied with in surface, second the deform is controlled in one pixels


    smpl_conds = defconds_list[1]


    optimized_init_tmp_ps_list = []
    optimized_check_list = []
    for garment_idx, (initTmpPs, batch_inds, defconds, rays, garment_name) in enumerate(zip(initTmpPs_list, batch_inds_list, defconds_list[0], rays_list, garment_names)):
        tmpSdf = tmpSdf_nets[garment_idx]
        with torch.no_grad():

            check1=tmpSdf(initTmpPs,ratio).view(-1).abs()<dthreshold
            if hasattr(deformer,'defs') and hasattr(deformer.defs[0],'enableSdfcond'):
                direct=deformer(initTmpPs,[[defconds[0],tmpSdf.rendcond],defconds[1]],batch_inds,ratio=ratio)-cam_pos.view(1,3)
            else:
                direct=deformer(initTmpPs,[defconds, smpl_conds],batch_inds,ratio=ratio, offset_type = garment_name)-cam_pos.view(1,3)
            up=torch.cross(direct,rays,dim=1)

            check2=torch.arcsin(up.norm(dim=1)/direct.norm(dim=1))*180./np.pi < athreshold
            # print('%f:%f'%(tmpSdf(initTmpPs,ratio).view(-1).abs().mean().item(),(torch.arcsin(up.norm(dim=1)/direct.norm(dim=1))*180./np.pi).mean().item()))
            check=check1*check2
            unfinished=torch.ones(initTmpPs.shape[0],device=initTmpPs.device,dtype=torch.bool)
            unfinished[check]=False
        # initTmpPs.requires_grad_(True)
        indices=torch.arange(unfinished.shape[0]).to(unfinished.device)

        # process the pts not satisfied
        for ind in range(times):
            # print('%d:%d'%(ind,unfinished.sum()))
            curPs=initTmpPs[unfinished].detach().clone()
            if curPs.shape[0]==0:
                break
            curPs.requires_grad_(True)
            loss1=(tmpSdf(curPs,ratio).abs()).view(-1)
            if hasattr(deformer,'defs') and hasattr(deformer.defs[0],'enableSdfcond'):
                defPs=deformer(curPs,[[defconds[0],tmpSdf.rendcond],defconds[1]],batch_inds[unfinished],ratio=ratio)
            else:

                defPs=deformer(curPs,[defconds, smpl_conds], batch_inds[unfinished],ratio=ratio, offset_type = garment_name)
            direct=defPs-cam_pos.view(1,3)
            up=torch.cross(direct,rays[unfinished],dim=1)
            # down=(direct*direct).sum(1)
            # down[down<1.e-4]=1.e-4
            # loss2=(up*up).sum(1)/down
            loss2=(up.norm(dim=1)/direct.norm(dim=1)).abs()


            loss=w1*loss1+w2*loss2
            Loss=loss.sum()


            grad=torch.autograd.grad(Loss,curPs,retain_graph=False,create_graph=False,only_inputs=True)[0]
            #descent is gradient direction, later can test pinverse, maybe faster
            # loss is larger move is larget
            t=-loss/(grad*grad).sum(1)
            curPs=(curPs+t.view(-1,1)*grad).detach()
            initTmpPs[unfinished]=curPs
            with torch.no_grad():
                check1=tmpSdf(curPs,ratio).view(-1).abs()<dthreshold
                if hasattr(deformer,'defs') and hasattr(deformer.defs[0],'enableSdfcond'):
                    direct=deformer(curPs,[[defconds[0],tmpSdf.rendcond],defconds[1]],batch_inds[unfinished],ratio=ratio)-cam_pos.view(1,3)
                else:
                    direct=deformer(curPs,[defconds, smpl_conds], batch_inds[unfinished],ratio=ratio, offset_type = garment_name)-cam_pos.view(1,3)
                up=torch.cross(direct,rays[unfinished],dim=1)
                check2=torch.arcsin(up.norm(dim=1)/direct.norm(dim=1))*180./np.pi < athreshold
                check=check1*check2
                unfinished[indices[unfinished][check]]=False
        optimized_init_tmp_ps_list.append(initTmpPs.detach())
        optimized_check_list.append(~unfinished)


    return optimized_init_tmp_ps_list, optimized_check_list


