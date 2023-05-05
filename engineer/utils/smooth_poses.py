# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import torch
import numpy as np
import pdb

from engineer.utils.filter import OneEuroFilter
from engineer.utils.transformation import batch_rodrigues, batch_axisang2quat


def compute_angle(axisang):
    axisang_norm = torch.norm(axisang + 1e-8, p=2, dim=1)
    angle = torch.unsqueeze(axisang_norm, -1)


    return angle



def smooth_poses(pred_pose, min_cutoff=0.004, beta= 1.5, d_cutoff = 1.):
    # min_cutoff: Decreasing the minimum cutoff frequency decreases slow speed jitter
    # beta: Increasing the speed coefficient(beta) decreases speed lag.

    one_euro_filter = OneEuroFilter(
        torch.zeros_like(pred_pose[0]),
        pred_pose[0],
        min_cutoff=min_cutoff,
        beta=beta,
        d_cutoff = d_cutoff
    )
    pred_pose = pred_pose.detach().clone()
    pred_pose_hat = torch.zeros_like(pred_pose)
    # initialize
    pred_pose_hat[0] = pred_pose[0]



    for idx, pose in enumerate(pred_pose[1:]):
        idx += 1
        # Note that the original VIBE has some bug, I fix it
        if torch.abs(pred_pose[idx] - pred_pose_hat[idx-1]).sum(-1).max()>0.5:
            reverse_pose = 1+(2*torch.pi - 2*compute_angle(pose))/(compute_angle(pose))
            mask = torch.abs(pred_pose[idx] - pred_pose_hat[idx-1]).sum(-1)>0.5
            pose = pose.detach()
            pose[mask] = -pose[mask]
            pose[mask] *= reverse_pose[mask]

        t = torch.ones_like(pose) * idx
        pose = one_euro_filter(t, pose)
        pred_pose_hat[idx] = pose



    return pred_pose_hat
