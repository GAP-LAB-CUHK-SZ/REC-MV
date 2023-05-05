"""
@File: polygons.py
@Author: Lingteng Qiu
@Email: qiulingteng@link.cuhk.edu.cn
@Date: 2022-07-12
@Desc: using to sampling more points from a given polygon
"""
import numpy as np
import pdb
import torch

def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """


    device = xyz.device
    B, N, C = xyz.shape

    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10

    batch_indices = torch.arange(B, dtype=torch.long).to(device)

    #farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)

    barycenter = torch.sum((xyz), 1)
    barycenter = barycenter/xyz.shape[1]
    barycenter = barycenter.view(B, 1, 3)

    dist = torch.sum((xyz - barycenter) ** 2, -1)
    farthest = torch.max(dist,1)[1]

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]

        farthest = torch.max(distance, -1)[1]
    return centroids

def uniformsample(pgtnp_px2, newpnum):

    pgtnp_px2 = np.asarray(pgtnp_px2)
    pnum, cnum = pgtnp_px2.shape

    assert cnum == 2

    idxnext_p = (np.arange(pnum, dtype=np.int32) + 1) % pnum
    # Note that we remove the rear to head
    idxnext_p = idxnext_p[:-1]
    pgtnext_px2 = pgtnp_px2[idxnext_p]
    pgtnp_px2 = pgtnp_px2[:-1]

    pnum =  pgtnp_px2.shape[0]


    edgelen_p = np.sqrt(np.sum((pgtnext_px2 - pgtnp_px2) ** 2, axis=1))

    # two cases
    # we need to remove gt points
    # we simply remove shortest paths
    if pnum > newpnum:
        edgelen_p[0] = 0.
        edgelen_p[-1] = 0.
        # however it would remove start and end points which is not we want to think
        edgeidxsort_p = np.argsort(edgelen_p)


        edgeidxkeep_k = edgeidxsort_p[pnum - newpnum:]
        edgeidxsort_k = np.sort(edgeidxkeep_k)
        pgtnp_kx2 = pgtnp_px2[edgeidxsort_k]
        assert pgtnp_kx2.shape[0] == newpnum
        return pgtnp_kx2
    # we need to add gt points
    # we simply add it uniformly
    else:
        edgeidxsort_p = np.argsort(edgelen_p)
        edgenum = np.round(edgelen_p * newpnum / np.sum(edgelen_p)).astype(np.int32)
        for i in range(pnum):
            if edgenum[i] == 0:
                edgenum[i] = 1

        # after round, it may has 1 or 2 mismatch
        edgenumsum = np.sum(edgenum)
        if edgenumsum != newpnum:

            if edgenumsum > newpnum:

                id = -1
                passnum = edgenumsum - newpnum
                while passnum > 0:
                    edgeid = edgeidxsort_p[id]
                    if edgenum[edgeid] > passnum:
                        edgenum[edgeid] -= passnum
                        passnum -= passnum
                    else:
                        passnum -= edgenum[edgeid] - 1
                        edgenum[edgeid] -= edgenum[edgeid] - 1
                        id -= 1
            else:
                id = -1
                edgeid = edgeidxsort_p[id]
                edgenum[edgeid] += newpnum - edgenumsum

        assert np.sum(edgenum) == newpnum

        psample = []
        for i in range(pnum):
            pb_1x2 = pgtnp_px2[i:i + 1]
            pe_1x2 = pgtnext_px2[i:i + 1]

            pnewnum = edgenum[i]
            wnp_kx1 = np.arange(edgenum[i], dtype=np.float32).reshape(-1, 1) / edgenum[i]

            pmids = pb_1x2 * (1 - wnp_kx1) + pe_1x2 * wnp_kx1
            psample.append(pmids)

        psamplenp = np.concatenate(psample, axis=0)
        return psamplenp




def uniformsample3d(pgtnp_px3, newpnum):


    pgtnp_px3 = np.asarray(pgtnp_px3)
    pnum, cnum = pgtnp_px3.shape

    assert cnum == 3

    idxnext_p = (np.arange(pnum, dtype=np.int32) + 1) % pnum
    pgtnext_px3 = pgtnp_px3[idxnext_p]


    pnum =  pgtnext_px3.shape[0]

    edgelen_p = np.sqrt(np.sum((pgtnext_px3 - pgtnp_px3) ** 2, axis=1))

    # two cases
    # we need to remove gt points
    # we simply remove shortest paths
    if pnum > newpnum:


        edgelen_p[0] = 0.
        edgelen_p[-1] = 0.




        # NOTE that some artifacts from edge length choice
        # I find the farest point sample is better

        # however it would remove start and end points which is not we want to think
        # edgeidxsort_p = np.argsort(edgelen_p)
        # edgeidxkeep_k = edgeidxsort_p[pnum - newpnum:]
        # edgeidxsort_k = np.sort(edgeidxkeep_k)
        # pgtnp_kx3 = pgtnp_px3[edgeidxsort_k]

        pgtnp_px3 = torch.from_numpy(pgtnp_px3).float()

        sample_idx = farthest_point_sample(pgtnp_px3[None], newpnum)[0]
        sample_idx = sample_idx.sort()

        pgtnp_kx3 = pgtnp_px3[sample_idx.values].detach().cpu().numpy()

        assert pgtnp_kx3.shape[0] == newpnum

        return pgtnp_kx3[:-1]
    # we need to add gt points
    # we simply add it uniformly
    else:
        edgeidxsort_p = np.argsort(edgelen_p)
        edgenum = np.round(edgelen_p * newpnum / np.sum(edgelen_p)).astype(np.int32)
        for i in range(pnum):
            if edgenum[i] == 0:
                edgenum[i] = 1

        # after round, it may has 1 or 2 mismatch
        edgenumsum = np.sum(edgenum)
        if edgenumsum != newpnum:

            if edgenumsum > newpnum:

                id = -1
                passnum = edgenumsum - newpnum
                while passnum > 0:
                    edgeid = edgeidxsort_p[id]
                    if edgenum[edgeid] > passnum:
                        edgenum[edgeid] -= passnum
                        passnum -= passnum
                    else:
                        passnum -= edgenum[edgeid] - 1
                        edgenum[edgeid] -= edgenum[edgeid] - 1
                        id -= 1
            else:
                id = -1
                edgeid = edgeidxsort_p[id]
                edgenum[edgeid] += newpnum - edgenumsum

        assert np.sum(edgenum) == newpnum

        psample = []
        for i in range(pnum):
            pb_1x3 = pgtnp_px3[i:i + 1]
            pe_1x3 = pgtnext_px3[i:i + 1]

            pnewnum = edgenum[i]
            wnp_kx1 = np.arange(edgenum[i], dtype=np.float32).reshape(-1, 1) / edgenum[i]

            pmids = pb_1x3 * (1 - wnp_kx1) + pe_1x3 * wnp_kx1
            psample.append(pmids)

        psamplenp = np.concatenate(psample, axis=0)


        if sum(psamplenp[0] - psamplenp[-1]) < 1e-6:
            return psamplenp[:-1]
        else:
            return psamplenp
