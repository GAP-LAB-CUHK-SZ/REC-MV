
import torch.nn as nn
import torch
from .base_optimzier import _Base_Optimizer
from utils.common_utils import make_nograd_func
from pytorch3d.ops.knn import knn_gather, knn_points

class ICP_Optimizer(_Base_Optimizer):
    '''
    This is ICP Optimizer using to fit two different cloud setting
    '''
    def __init__(self, epoch, optimizer_setting = None):
        '''
        epoch: hook training epoch
        optimizer_setting: is optimizer setting, in the icp registry this node is set to 0
        '''
        super(ICP_Optimizer, self).__init__(optimizer_setting)

        self.name = "ICP_Optimizer"
        # hook stop time
        self.epoch = epoch
        self.energy_func = lambda x,y: torch.sum((x-y)**2)



    def __collect_data(self, inputs):
        '''the data fomat we only process to Garemnet Mesh and Garment Polygons class

        return source_pts, and target pts
        '''
        smpl_slice = inputs['smpl_slice']
        target_polygon = inputs['target_polygon']

        target_fields = target_polygon.get_fields()

        target = target_polygon.get_boundary(*target_fields)
        source = smpl_slice.get_boundary(*target_fields)

        return torch.cat(source, dim = 0), torch.cat(target, dim=0)
    def solver(self, source, target):
        '''
        Given source pts, and target pts, we solve the optimal rotation matrix, R and transform matrix T
        min ||R * source +t - tareget||2

        where t^* = mean(P_taregt) - R^* mean(P_source)


        For R^*
        we have, R^{*} = argmax(sum_{1}^{N}\hat{P_t}R\hat{P_s})
                        = trace(\hat{P_t}R\hat{P_s})

        argmax trace{\hat{P_t}R\hat{P_s}}

        R = v @ u.T

        but avoid reflection rotaiton
        we add a constrain
        let
        R = v diag[1, 1, det(v@u.t)] @ u.t
        return
            R, t
        '''
        mean_source = source.mean(dim=0, keepdim =True)
        mean_target= target.mean(dim=0, keepdim= True)

        source -=mean_source
        target -=mean_target
        # source [N,3]
        # target [N,3]
        pspt= source.T @ target
        # svd solve
        # u @ diag(s) @ v = pspt
        # NOTE v is equal to V.T in the tranditional svd
        u,s,v_t = torch.linalg.svd(pspt)
        # s = torch.diag(s)
        # print((u @ s @ v_t) - pspt)

        v = v_t.T
        s = torch.eye(3).to(source.device)
        s[2,2] = torch.det(v @ u.T)
        solve_R = v @ s @ u.T

        solve_t = mean_target - (solve_R @ mean_source.T).T

        return solve_R, solve_t

    def __encode_results(self,inputs, R,t):
        '''the data return depend on your format
        '''
        smpl_slice = inputs['smpl_slice']
        smpl_slice.transform_R_t(R,t)

    @make_nograd_func
    def fitting(self, inputs):
        '''ICP fitting functions
        Given source point clounds and target point clounds
        this funciton find global R and t to make the distance(e.g. Euclidean distance) between source points and target points match
        '''

        source, target = self.__collect_data(inputs)

        dist = knn_points(source[None],target[None])
        target_idx = dist.idx[0,...,0]
        target = target[target_idx]
        R,t = self.solver(source, target)
        new_source = (R @ source .T).T + t
        loss = (self.energy_func(new_source, target))
        self.__encode_results(inputs, R, t)

        return loss


