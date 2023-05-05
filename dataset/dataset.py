"""
@File: dataset.
@Author: Lingteng Qiu
@Email: qiulingteng@link.cuhk.edu.cn
@Date: 2023-01-02
@Desc: REC-MV dataset
"""
import numpy as np
import torch
import os
import cv2
import os.path as osp
from glob import glob
import utils
from utils.constant import ATR_PARSING, FL_INFOS
from engineer.utils.featureline_utils import check_feature_lines, obtain_feature_lines
from pytorch3d.ops.knn import knn_gather, knn_points
from engineer.utils.polygons import uniformsample
from utils.common_utils import numpy2tensor
from engineer.utils.snug_utils import load_motion
import pdb
import joblib
from utils.common_utils import lower_bound
from engineer.utils.smooth_poses import smooth_poses

class SceneDataset(torch.utils.data.Dataset):
    # get a batch_size continuous frame sequence
    def __init__(self,data_root, conds_lens={}, garment_type = "", fl_sampling = 100, curve_sampling = 1):

        '''
        curve_sampling : how many curve be used
        '''

        assert not garment_type == ""

        self.root=data_root
        self.read_data()
        self.require_albedo=False
        self.conds=[]
        self.cond_ns=[]
        self.garment_type = garment_type

        self.conds_lens = conds_lens


        self.fl_names = FL_INFOS[garment_type]
        self.fl_sampling = fl_sampling

        frame_joints_file = os.path.join(data_root, '{}_tcmr_output.pkl'.format(garment_type))
        self.curve_sampling = curve_sampling


        if os.path.exists(frame_joints_file):
            # using to adjust smpl beta
            self.gt_joints2d_list = joblib.load(frame_joints_file)[1]['gt_joints2d']
            frame_ids =joblib.load(frame_joints_file)[1]['frame_ids']
            self.gt_joints2d = {}

            self.tcmr_poses = joblib.load(frame_joints_file)[1]['pose']
            self.tcmr_betas = joblib.load(frame_joints_file)[1]['betas']


            # only for anran as their smpl is not stable
            tcmr_poses = torch.from_numpy(self.tcmr_poses).view(-1, 24,3).float()
            # self.poses = tcmr_poses
            # self.tcmr_pose = tcmr_poses


            # self.shape = torch.from_numpy(self.tcmr_betas.mean(0)).float()
            # for big_pose we use tcmr predict to replace
            # self.poses[self.a_pose_end+1:len(tcmr_poses)] = tcmr_poses[self.a_pose_end+1:]


            for frame_id, gt_joints in zip(frame_ids, self.gt_joints2d_list):
                self.gt_joints2d[frame_id] = gt_joints
            self.joints_frame_ids  =frame_ids.tolist()

        else:
            self.gt_joints2d = None


        # cond parameter needs optimization by default
        for name,length in conds_lens.items():
            # cond=torch.zeros(self.frame_num,length,requires_grad=True)
            # torch.nn.init.normal_(cond, mean=0., std=0.001)
            #cond features: [num_frame, conds_lens]
            cond=((0.1*torch.randn(length , self.frame_num//5)).matmul(utils.DCTSpace(self.frame_num//5,self.frame_num))).transpose(0,1)

            cond.requires_grad_()
            self.conds.append(cond)
            self.cond_ns.append(name)

        self.area_size_statistic()

    def all_size(self):
        return len(self)
    def get_init_fl_datasets(self, batch_size, sampler, num_workers):

        sampler_idx = np.where(np.asarray(self.fl_supervised))[0].tolist()


        # (Init_Fl_SceneDataset, self).__init__(data_root, conds_lens, garment_type, fl_sampling, curve_sampling = curve_sampling)

        init_fl_datasets = Init_Fl_SceneDataset(self.root, self.conds_lens, self.garment_type, self.fl_sampling, self.curve_sampling, sampler_idx)
        sampler=RandomSampler(init_fl_datasets, 1, True)

        return torch.utils.data.DataLoader(init_fl_datasets, batch_size, sampler=sampler, num_workers=num_workers)

    def area_size_statistic(self):
        '''compute fl area size to obtain project weight
        NOTE that chamfer distance from pytorch3d is square.
        hence we use area scale instead of length
        '''

        def area(fl_pt):
            fl_length = fl_pt.max(0) - fl_pt.min(0)
            return max(fl_length[0],  fl_length[1])
            return fl_length[0] * fl_length[1]


        self.fl_weights = dict()


        visible = dict()

        for fl_name in self.fl_names:
            self.fl_weights[fl_name] = 0.
            visible[fl_name] = 0


        for idx in range(len(self)):
            if idx % self.curve_sampling == 0:
                fl_path = self.fl_paths[idx]
                fl_infos = obtain_feature_lines(fl_path)
                fl_pts, fl_masks = self.obtain_fl_pts(fl_infos)
                fl_pts, fl_masks = self.obtain_fl_pts(fl_infos)

                for fl_pt, fl_mask, fl_name in zip(fl_pts, fl_masks, self.fl_names):
                    if not fl_mask:
                        continue
                    mask = area(fl_pt)


                    self.fl_weights[fl_name] += mask
                    visible[fl_name] +=1

        max_area = 0.
        for fl_name in self.fl_names:
            self.fl_weights[fl_name] /= visible[fl_name]
            max_area = max(max_area, self.fl_weights[fl_name])

        for fl_name in self.fl_names:
            self.fl_weights[fl_name] = (max_area / self.fl_weights[fl_name]) **2



    def read_feature_lines(self, path):
        fl_paths = sorted(glob(os.path.join(path,'*.json')))
        self.fl_paths = []
        json_idx = 0

        for frame_id in range(self.frame_num):
            try:
                json_name = int(fl_paths[json_idx].split('/')[-1].split('.')[0])
            except:
                json_name = int(fl_paths[-1].split('/')[-1].split('.')[0])

            if frame_id == json_name:
                self.fl_paths.append(fl_paths[json_idx])
                json_idx+=1
            else:
                self.fl_paths.append(fl_paths[json_idx-1])
        # check whether has some annotation issues


        self.fl_supervised  = [True for __ in range(len(self.fl_paths))]

        for fl_path in fl_paths:
            check_feature_lines(fl_path)

    def read_data(self):
        candidate_ext=['.jpg','.png']
        imgs=[]
        for ext in candidate_ext:
            imgs.extend(glob(osp.join(self.root,'imgs/*'+ext)))
        imgs.sort(key=lambda x: int(osp.basename(x).split('.')[0]))
        self.frame_num=len(imgs)
        self.img_ns=imgs
        self.mask_ns = []
        self.parsing_mask_ns = []
        for ind,img_n in enumerate(self.img_ns):
            assert(ind==int(osp.basename(img_n).split('.')[0]))
            # self.mask_ns.append(osp.join(self.root,'masks/%d.png'%ind))
            self.mask_ns.append(osp.join(self.root,'masks/%s.png'%(osp.basename(img_n).split('.')[0])))
            self.parsing_mask_ns.append(osp.join(self.root,'parsing_SCH_ATR/%s.npy'%(osp.basename(img_n).split('.')[0])))
            assert(osp.isfile(self.mask_ns[-1]))
            assert(osp.isfile(self.parsing_mask_ns[-1]))



        self.H,self.W,_=cv2.imread(self.mask_ns[0]).shape
        # smpl_parameters []
        data=np.load(osp.join(self.root,'smpl_rec.npz'))
        #['pose', shape, trans, gender)


        self.poses=torch.from_numpy(data['poses'].astype(np.float32)).view(-1,24,3)
        self.trans=torch.from_numpy(data['trans'].astype(np.float32)).view(-1,3)
        # general beta ---> 10
        self.shape=torch.from_numpy(data['shape'].astype(np.float32)).view(-1)
        self.gender=str(data['gender']) if 'gender' in data else 'neutral'
        print('scene data use %s smpl'%self.gender)

        if 'vid_seg_indices' in data:
            if type(data['vid_seg_indices'])==np.ndarray:
                self.video_segmented_index=data['vid_seg_indices'][0:-1].tolist()
            else:
                self.video_segmented_index=data['vid_seg_indices'][0:-1]
            if len(self.video_segmented_index)>0:
                print('this dataset has %d segmented videos'%(len(self.video_segmented_index)+1))
        else:
            self.video_segmented_index=[]
        data=np.load(osp.join(self.root,'camera.npz'))
        # read_feature line
        feature_line_path = os.path.join(self.root, 'featurelines')



        assert os.path.exists(feature_line_path)



        self.read_feature_lines(feature_line_path)


        self.camera_params={'focal_length':torch.tensor(np.array([data['fx'],data['fy']]).astype(np.float32)), \
                            'princeple_points':torch.tensor(np.array([data['cx'],data['cy']]).astype(np.float32)), \
                            'cam2world_coord_quat':torch.from_numpy(data['quat'].astype(np.float32)).view(-1), \
                            'world2cam_coord_trans':torch.from_numpy(data['T'].astype(np.float32)).view(-1)}

    def opt_camera_params(self,conf):
        if type(conf)==bool:
            self.camera_params['focal_length'].requires_grad_(conf)
            self.camera_params['princeple_points'].requires_grad_(conf)
            self.camera_params['cam2world_coord_quat'].requires_grad_(conf)
            self.camera_params['world2cam_coord_trans'].requires_grad_(conf)
        else:
            self.camera_params['focal_length'].requires_grad_(conf.get_bool('focal_length'))
            self.camera_params['princeple_points'].requires_grad_(conf.get_bool('princeple_points'))
            self.camera_params['cam2world_coord_quat'].requires_grad_(conf.get_bool('quat'))
            self.camera_params['world2cam_coord_trans'].requires_grad_(conf.get_bool('T'))

    def learnable_weights(self):
        ws=[]
        ws.extend([cond for cond in self.conds if cond.requires_grad])
        ws.extend([v for k,v in self.camera_params.items() if v.requires_grad])
        ws.extend([v for v in [self.shape,self.poses,self.trans] if v.requires_grad])
        return ws

    def parsing_mask(self, idx):
        out={}
        # normalized to [-1,1]
        img=torch.from_numpy((cv2.imread(self.img_ns[idx]).astype(np.float32)/255.-0.5)*2).view(self.H,self.W,3)
        out['img']=img
        # rec_f=self.img_ns[idx].replace('.%s' % (self.img_ns[idx].split('.')[-1]), '_rect.txt')
        # if osp.isfile(rec_f):
        #     rects=np.loadtxt(rect_f, dtype=np.int32)
        #     if len(rects.shape) == 1:
        #         rects = rects[None]
        #     rect = torch.from_numpy(rects[0].astype(np.float32))
        #     out['rect']=rect
        parsing_mask = torch.from_numpy(np.load(self.parsing_mask_ns[idx])).long()
        mask=(torch.from_numpy(cv2.imread(self.mask_ns[idx]))>0).view(self.H,self.W,-1).any(-1).float()
        out['mask']=mask
        parsing_mask = self.load_parsing_mask(mask, parsing_mask)


        parsing_name = self.parsing_mask_ns[idx]
        mask_parsing_name = 'mask_parsing_' + parsing_name.split("/")[-1]
        mask_parsing_path = os.path.join('/'.join(parsing_name.split('/')[:-1]),mask_parsing_name)

        np.save(mask_parsing_path, parsing_mask)
        return mask_parsing_path



    def obtain_fl_pts(self,fls):
        '''
        obtain gt fl points for batch training
        '''
        fl_masks = []
        fl_pts = []
        # need to extract feature line

        for fl_name in self.fl_names:
            if fl_name in fls.keys():
                pts = fls[fl_name]
                dis = ((pts[:-1] - pts[1:]) **2).sum(-1)
                dis_rear_top = ((pts[-1] - pts[0]) **2).sum(-1)
                if dis_rear_top < np.max(dis):
                    max_idx = np.argmax(dis)
                    line_a = pts[:max_idx+1]
                    line_b = pts[max_idx+1:]
                    pts = np.concatenate([line_b, line_a], axis =0)

                fl = uniformsample(pts, self.fl_sampling)
                fl_masks.append(True)
                fl_pts.append(fl)
            else:
                fl_pts.append(np.zeros((self.fl_sampling, 2), np.float32))
                fl_masks.append(False)

        return fl_pts, fl_masks



    def load_parsing_mask(self,mask, parsing_logits):
        '''
        transform mask2 parsing_mask
        '''

        parsing_mask_tensor = torch.zeros_like(mask).long()

        i,j =torch.nonzero(parsing_logits,as_tuple= True)
        label = parsing_logits[i,j]
        logits_coordinate = torch.cat([i[:,None],j[:,None]], dim = -1).float().cuda()

        i,j =torch.nonzero(mask,as_tuple= True)
        mask_coordinate = torch.cat([i[:,None],j[:,None]], dim = -1)
        mask_coordinate_tensor = mask_coordinate.float().cuda()
        mask_dist = knn_points(mask_coordinate_tensor[None], logits_coordinate[None], K=1)
        knn_idx = mask_dist.idx
        parsing_mask = label[knn_idx][0,..., 0]
        parsing_mask_tensor[i,j]=parsing_mask.detach().cpu().long()
        parsing_mask_tensor.numpy().astype(np.uint8)

        return parsing_mask_tensor.numpy().astype(np.uint8)

    def obtain_parsing_mask(self, mask_parsing):
        # split to upper-clothes, bottom_pants or upper-bottom type, e.g. dress
        parsing_garment_mask = {}
        all_garment_parsing= torch.zeros_like(mask_parsing)



        for key in ATR_PARSING.keys():
            garment_mask = torch.zeros_like(mask_parsing)
            for class_id in ATR_PARSING[key]:
                garment_mask = garment_mask |(mask_parsing == class_id)
                all_garment_parsing = all_garment_parsing | (mask_parsing == class_id)
            parsing_garment_mask[key] = garment_mask
        body_parsing = mask_parsing>0
        body_parsing = body_parsing ^ all_garment_parsing
        parsing_garment_mask['body'] = body_parsing


        return parsing_garment_mask

    def __len__(self):
        return self.frame_num
    def __getitem__(self, idx):
        # convert to [-1.,1.] to keep consistent with render net tanh output
        # gallery = list(range(269,309))
        # idx = gallery[idx]

        out={}
        # normalized to [-1,1]
        img=torch.from_numpy((cv2.imread(self.img_ns[idx]).astype(np.float32)/255.-0.5)*2).view(self.H,self.W,3)
        out['img']=img
        # rec_f=self.img_ns[idx].replace('.%s' % (self.img_ns[idx].split('.')[-1]), '_rect.txt')
        # if osp.isfile(rec_f):
        #     rects=np.loadtxt(rect_f, dtype=np.int32)
        #     if len(rects.shape) == 1:
        #         rects = rects[None]
        #     rect = torch.from_numpy(rects[0].astype(np.float32))
        #     out['rect']=rect
        # parsing mask is the results of atr
        parsing_mask = self.parsing_mask_ns[idx]
        mask_parsing_name = 'mask_parsing_' + parsing_mask.split("/")[-1]
        mask_parsing_path = os.path.join('/'.join(parsing_mask.split('/')[:-1]),mask_parsing_name)
        # mask_parsing_results of mask according to knn atr mask
        mask_parsing = torch.from_numpy(np.load(mask_parsing_path)).long()
        garment_parsing = self.obtain_parsing_mask(mask_parsing)
        # return garment_parsing
        mask=(torch.from_numpy(cv2.imread(self.mask_ns[idx]))>0).view(self.H,self.W,-1).any(-1).float()
        out['mask']=mask

        # get fl infos
        fl_path = self.fl_paths[idx]
        fl_infos = obtain_feature_lines(fl_path)
        fl_pts, fl_masks = self.obtain_fl_pts(fl_infos)
        fl_pts = numpy2tensor(fl_pts)
        fl_masks = torch.Tensor(fl_masks).bool()

        if idx % self.curve_sampling != 0:
            fl_masks[...] = False

        out['fl_pts'] = torch.cat(fl_pts, dim=0)
        out['fl_masks'] = fl_masks


        norm_f=self.img_ns[idx].replace('/imgs/','/normals/')[:-3]+'png'
        if osp.isfile(norm_f):
            # bgr 2 rgb
            normals=cv2.imread(norm_f)[:,:,::-1]
            normals=2.*normals.astype(np.float32)/255.-1.
            out['normal']=normals
        # norm_e=self.img_ns[idx].replace('/imgs/','/normal_edges/')[:-3]+'png'
        # if osp.isfile(norm_e):
        #     normal_edges=cv2.imread(norm_e,cv2.IMREAD_UNCHANGED)

        #     normal_edges=normal_edges.astype(np.float32)/255.
        #     out['normal_edge']=normal_edges
        out.update(garment_parsing)
        gt_joints2d = self.gt_joints2d[idx]
        out['gt_joints2d'] = gt_joints2d


        if self.require_albedo:
            albedo=torch.from_numpy((cv2.imread(osp.join(self.root,'albedos/%d.png'%idx)).astype(np.float32)/255.-0.5)*2.).view(self.H,self.W,3)
            out['albedo']=albedo
        # return idx,img,mask,albedo
        return idx, out
    # this function is a patch for __getitem__, it seems that dataloader cannot fetch data with requires_grad=True, because of stack(out=out)
    def get_grad_parameters(self,idxs,device):
        conds=[cond[idxs].to(device) for cond in self.conds]


        if len(conds)>1:
            return self.poses[idxs].to(device),self.trans[idxs].to(device),*conds
        else:
            return self.poses[idxs].to(device),self.trans[idxs].to(device),*conds, None


    def get_camera_parameters(self,N,device):
        return (self.camera_params['focal_length'].to(device).view(1,2).expand(N,2),self.camera_params['princeple_points'].to(device).view(1,2).expand(N,2), \
            utils.quat2mat(self.camera_params['cam2world_coord_quat'].to(device).view(1,4)).expand(N,3,3),self.camera_params['world2cam_coord_trans'].to(device).view(1,3).expand(N,3),self.H,self.W)
    def get_batchframe_data(self,name,fids,batchsize):


        if len(self.video_segmented_index)==0:
            assert(batchsize<self.frame_num)
            assert(hasattr(self,name))
            videodata=getattr(self,name)
            assert(videodata.shape[0]>=self.frame_num)
            videodata=videodata[:self.frame_num].to(fids.device)
            starts=fids-batchsize//2
            ends=starts+batchsize
            offset=starts-0
            sel=offset<0
            starts[sel]-=offset[sel]
            ends[sel]-=offset[sel]
            offset=ends-self.frame_num
            sel=offset>0
            starts[sel]-=offset[sel]
            ends[sel]-=offset[sel]
            return videodata[starts.view(-1,1)+torch.arange(0,batchsize).view(1,batchsize).to(fids.device)], fids-starts
        elif len(self.video_segmented_index)==1:
            def extract_batch(start,end,fids,batchsize):
                starts=fids-batchsize//2
                ends=starts+batchsize
                offset=starts-start
                sel=offset<0
                starts[sel]-=offset[sel]
                ends[sel]-=offset[sel]

                offset=ends-end
                sel=offset>0
                starts[sel]-=offset[sel]
                ends[sel]-=offset[sel]
                return starts,ends

            assert(hasattr(self,name))
            videodata=getattr(self,name)
            assert(videodata.shape[0]>=self.frame_num)
            videodata=videodata[:self.frame_num].to(fids.device)
            starts=torch.zeros_like(fids)-1
            ends=torch.zeros_like(fids)-1

            start=0
            end=self.video_segmented_index[0]
            sel=(fids>=start) * (fids<end)
            assert(batchsize<end-start)
            ss,es=extract_batch(start,end,fids[sel],batchsize)
            starts[sel]=ss
            ends[sel]=es

            start=self.video_segmented_index[0]
            end=self.frame_num
            sel=(fids>=start) * (fids<end)
            assert(batchsize<end-start)
            ss,es=extract_batch(start,end,fids[sel],batchsize)
            starts[sel]=ss
            ends[sel]=es
            assert((starts>=0).all().item())
            assert((ends>=0).all().item())
            return videodata[starts.view(-1,1)+torch.arange(0,batchsize).view(1,batchsize).to(fids.device)], fids-starts


        else:
            raise NotImplementedError

class People_Snapshot_SceneDataset(SceneDataset):
    def __init__(self,data_root, conds_lens={}, garment_type = "", fl_sampling = 100, curve_sampling = 1, a_pose = False):
        super(People_Snapshot_SceneDataset, self).__init__(data_root, conds_lens, garment_type, fl_sampling, curve_sampling = curve_sampling)
        self.a_pose = a_pose
        feature_lines_path = osp.join(data_root,'mask2fl')

        # smooth depth
        # self.trans[self.a_pose_end:, -1] = self.trans[self.a_pose_end, -1]
        # self.trans = smooth_poses(self.trans.detach().cpu(), min_cutoff=0.004, beta= 0.7, d_cutoff = 1.)
        # self.poses = smooth_poses(self.poses.detach().cpu(), min_cutoff=0.004, beta= 0.7, d_cutoff = 1.)
        if os.path.exists(feature_lines_path):
            self.read_feature_lines(feature_lines_path)
            self.area_size_statistic()

            if self.a_pose:
                self.frame_num = self.a_pose_end - self.a_pose_start +1
                self.start_idx = self.a_pose_start
            else:
                self.frame_num = len(self) - self.a_pose_end -1
                self.start_idx = self.a_pose_end +1
        else:
            self.start_idx = 0






    def get_init_fl_datasets(self, batch_size, sampler, num_workers):

        sampler_idx = np.where(np.asarray(self.fl_supervised))[0].tolist()
        # (Init_Fl_SceneDataset, self).__init__(data_root, conds_lens, garment_type, fl_sampling, curve_sampling = curve_sampling)
        init_fl_datasets = Init_Fl_SceneDataset(self.root, self.conds_lens, self.garment_type, self.fl_sampling, self.curve_sampling, sampler_idx)
        sampler=RandomSampler(init_fl_datasets, 1, True)

        return torch.utils.data.DataLoader(init_fl_datasets, batch_size, sampler=sampler, num_workers=num_workers)



    def area_size_statistic(self):
        '''compute fl area size to obtain project weight
        NOTE that chamfer distance from pytorch3d is square.
        hence we use area scale instead of length
        '''

        def area(fl_pt):
            fl_length = fl_pt.max(0) - fl_pt.min(0)
            return max(fl_length[0],  fl_length[1])

        self.fl_weights = dict()


        visible = dict()

        for fl_name in self.fl_names:
            self.fl_weights[fl_name] = 0.
            visible[fl_name] = 0

        for idx in range(len(self.fl_paths)):
            if self.fl_supervised[idx] == True:
                fl_path = self.fl_paths[idx]
                fl_infos = obtain_feature_lines(fl_path)
                fl_pts, fl_masks = self.obtain_fl_pts(fl_infos)

                for fl_pt, fl_mask, fl_name in zip(fl_pts, fl_masks, self.fl_names):
                    if not fl_mask:
                        continue
                    mask = area(fl_pt)

                    self.fl_weights[fl_name] += mask
                    visible[fl_name] +=1


        max_area = 0.
        for fl_name in self.fl_names:
            self.fl_weights[fl_name] /= visible[fl_name]
            max_area = max(max_area, self.fl_weights[fl_name])


        for fl_name in self.fl_names:
            self.fl_weights[fl_name] = (max_area / self.fl_weights[fl_name]) **2




    def read_feature_lines(self, path):
        fl_paths = sorted(glob(os.path.join(path,'*.json')))
        self.a_pose_start, self.a_pose_end = int(fl_paths[0].split('/')[-1].replace('.json','')), int(fl_paths[-1].split('/')[-1].replace('.json',''))

        self.fl_paths = []
        self.fl_supervised =[]
        json_idx = 0

        for frame_id in range(self.frame_num):
            try:
                json_name = int(fl_paths[json_idx].split('/')[-1].split('.')[0])
            except:
                json_name = int(fl_paths[-1].split('/')[-1].split('.')[0])

            if frame_id == json_name:
                self.fl_paths.append(fl_paths[json_idx])
                json_idx+=1
                self.fl_supervised.append(True)

            else:
                self.fl_paths.append(fl_paths[json_idx-1])
                self.fl_supervised.append(False)
        # check whether has some annotation issues

        for fl_path in fl_paths:
            check_feature_lines(fl_path)



    def __getitem__(self, idx):

        # idx = self.start_idx + idx
        # gallery_idx = [90]
        # idx = gallery_idx[idx]

        # convert to [-1.,1.] to keep consistent with render net tanh output
        out={}
        # normalized to [-1,1]
        img=torch.from_numpy((cv2.imread(self.img_ns[idx]).astype(np.float32)/255.-0.5)*2).view(self.H,self.W,3)
        out['img']=img
        # rec_f=self.img_ns[idx].replace('.%s' % (self.img_ns[idx].split('.')[-1]), '_rect.txt')
        # if osp.isfile(rec_f):
        #     rects=np.loadtxt(rect_f, dtype=np.int32)
        #     if len(rects.shape) == 1:
        #         rects = rects[None]
        #     rect = torch.from_numpy(rects[0].astype(np.float32))
        #     out['rect']=rect
        # parsing mask is the results of atr
        parsing_mask = self.parsing_mask_ns[idx]
        mask_parsing_name = 'mask_parsing_' + parsing_mask.split("/")[-1]
        mask_parsing_path = os.path.join('/'.join(parsing_mask.split('/')[:-1]),mask_parsing_name)
        # mask_parsing_results of mask according to knn atr mask
        mask_parsing = torch.from_numpy(np.load(mask_parsing_path)).long()
        garment_parsing = self.obtain_parsing_mask(mask_parsing)
        # return garment_parsing
        mask=(torch.from_numpy(cv2.imread(self.mask_ns[idx]))>0).view(self.H,self.W,-1).any(-1).float()
        out['mask']=mask

        # get fl infos
        fl_path = self.fl_paths[idx]
        fl_infos = obtain_feature_lines(fl_path)
        fl_pts, fl_masks = self.obtain_fl_pts(fl_infos)
        fl_pts = numpy2tensor(fl_pts)
        fl_masks = torch.Tensor(fl_masks).bool()

        if not self.fl_supervised[idx]:
            fl_masks[...] = False

        out['fl_pts'] = torch.cat(fl_pts, dim=0)
        out['fl_masks'] = fl_masks

        norm_f=self.img_ns[idx].replace('/imgs/','/normals/')[:-3]+'png'
        if osp.isfile(norm_f):
            # bgr 2 rgb
            normals=cv2.imread(norm_f)[:,:,::-1]
            normals=2.*normals.astype(np.float32)/255.-1.
            out['normal']=normals
        # norm_e=self.img_ns[idx].replace('/imgs/','/normal_edges/')[:-3]+'png'
        # if osp.isfile(norm_e):
        #     normal_edges=cv2.imread(norm_e,cv2.IMREAD_UNCHANGED)

        #     normal_edges=normal_edges.astype(np.float32)/255.
        #     out['normal_edge']=normal_edges
        out.update(garment_parsing)



        if self.require_albedo:
            albedo=torch.from_numpy((cv2.imread(osp.join(self.root,'albedos/%d.png'%idx)).astype(np.float32)/255.-0.5)*2.).view(self.H,self.W,3)
            out['albedo']=albedo
        # return idx,img,mask,albedo
        return idx, out

class Large_Pose_SceneDataset(SceneDataset):
    def __init__(self,data_root, conds_lens={}, garment_type = "", fl_sampling = 100, curve_sampling = 1, a_pose = False):
        '''As the large pose has more noise, the videoavatar has not consistency in translation, we need smooth

        '''

        super(Large_Pose_SceneDataset, self).__init__(data_root, conds_lens, garment_type, fl_sampling, curve_sampling = curve_sampling)

        self.a_pose = a_pose
        feature_lines_path = osp.join(data_root,'mask2fl')

        # smooth depth

        joints_idx = [lower_bound(self.joints_frame_ids,0,len(self.joints_frame_ids), idx) for idx in range(len(self))]

        self.trans[self.a_pose_end:, -1] = self.trans[self.a_pose_end, -1]
        self.trans = smooth_poses(self.trans.detach().cpu(), min_cutoff=0.004, beta= 0.7, d_cutoff = 1.)


        # only for leyang jump
        self.shape = torch.from_numpy(self.tcmr_betas[self.a_pose_start: self.a_pose_end+1].mean(0)).float()


        # using
        tcmr_poses = torch.from_numpy(self.tcmr_poses[joints_idx]).view(-1, 24,3).float()
        # self.shape = torch.from_numpy(self.tcmr_betas.mean(0)).float()
        # for big_pose we use tcmr predict to replace
        self.poses[self.a_pose_end+1:len(tcmr_poses)] = tcmr_poses[self.a_pose_end+1:]
        # self.poses = smooth_poses(self.poses.detach().cpu(), min_cutoff=0.004, beta= 0.7, d_cutoff = 1.)

        # data=np.load(osp.join(self.root,'smpl_rec.npz'))


        # pymaf_poses = joblib.load('./female_large_pose_process_new/leyang_steps/pymaf_output.pkl')[1]['pose']
        # pymaf_poses = pymaf_poses.reshape(-1, 24, 3)
        # poses = data['poses']

        # pdb.set_trace()










        if os.path.exists(feature_lines_path):
            self.read_feature_lines(feature_lines_path)
            self.area_size_statistic()

            if self.a_pose:
                self.frame_num = self.a_pose_end - self.a_pose_start +1
                self.start_idx = 0
            else:
                self.frame_num = len(self) - self.a_pose_end -1
                self.start_idx = self.a_pose_end +1
        else:
            self.start_idx = 0





    def all_size(self):
        return self.poses.shape[0]

    def get_init_fl_datasets(self, batch_size, sampler, num_workers):

        sampler_idx = np.where(np.asarray(self.fl_supervised))[0].tolist()
        # (Init_Fl_SceneDataset, self).__init__(data_root, conds_lens, garment_type, fl_sampling, curve_sampling = curve_sampling)
        init_fl_datasets = Init_Fl_SceneDataset(self.root, self.conds_lens, self.garment_type, self.fl_sampling, self.curve_sampling, sampler_idx)
        sampler=RandomSampler(init_fl_datasets, 1, True)

        return torch.utils.data.DataLoader(init_fl_datasets, batch_size, sampler=sampler, num_workers=num_workers)



    def area_size_statistic(self):
        '''compute fl area size to obtain project weight
        NOTE that chamfer distance from pytorch3d is square.
        hence we use area scale instead of length
        '''

        def area(fl_pt):
            fl_length = fl_pt.max(0) - fl_pt.min(0)
            return max(fl_length[0],  fl_length[1])

        self.fl_weights = dict()


        visible = dict()

        for fl_name in self.fl_names:
            self.fl_weights[fl_name] = 0.
            visible[fl_name] = 0

        for idx in range(len(self.fl_paths)):
            if self.fl_supervised[idx] == True:
                fl_path = self.fl_paths[idx]
                fl_infos = obtain_feature_lines(fl_path)
                fl_pts, fl_masks = self.obtain_fl_pts(fl_infos)

                for fl_pt, fl_mask, fl_name in zip(fl_pts, fl_masks, self.fl_names):
                    if not fl_mask:
                        continue
                    mask = area(fl_pt)

                    self.fl_weights[fl_name] += mask
                    visible[fl_name] +=1


        max_area = 0.
        for fl_name in self.fl_names:
            self.fl_weights[fl_name] /= visible[fl_name]
            max_area = max(max_area, self.fl_weights[fl_name])

        for fl_name in self.fl_names:
            self.fl_weights[fl_name] = (max_area / self.fl_weights[fl_name]) **2




    def read_feature_lines(self, path):
        fl_paths = sorted(glob(os.path.join(path,'*.json')))
        self.a_pose_start, self.a_pose_end = int(fl_paths[0].split('/')[-1].replace('.json','')), int(fl_paths[-1].split('/')[-1].replace('.json',''))

        self.fl_paths = []
        self.fl_supervised =[]
        json_idx = 0

        for frame_id in range(self.frame_num):
            try:
                json_name = int(fl_paths[json_idx].split('/')[-1].split('.')[0])
            except:
                json_name = int(fl_paths[-1].split('/')[-1].split('.')[0])

            if frame_id == json_name:
                self.fl_paths.append(fl_paths[json_idx])
                json_idx+=1
                self.fl_supervised.append(True)

            else:
                self.fl_paths.append(fl_paths[json_idx-1])
                self.fl_supervised.append(False)
        # check whether has some annotation issues

        for fl_path in fl_paths:
            check_feature_lines(fl_path)



    def __getitem__(self, idx):
        idx = self.start_idx + idx

        # convert to [-1.,1.] to keep consistent with render net tanh output
        out={}
        # normalized to [-1,1]
        img=torch.from_numpy((cv2.imread(self.img_ns[idx]).astype(np.float32)/255.-0.5)*2).view(self.H,self.W,3)
        out['img']=img

        parsing_mask = self.parsing_mask_ns[idx]
        mask_parsing_name = 'mask_parsing_' + parsing_mask.split("/")[-1]
        mask_parsing_path = os.path.join('/'.join(parsing_mask.split('/')[:-1]),mask_parsing_name)
        # mask_parsing_results of mask according to knn atr mask
        mask_parsing = torch.from_numpy(np.load(mask_parsing_path)).long()
        garment_parsing = self.obtain_parsing_mask(mask_parsing)
        # return garment_parsing
        mask=(torch.from_numpy(cv2.imread(self.mask_ns[idx]))>0).view(self.H,self.W,-1).any(-1).float()
        out['mask']=mask

        # get fl infos
        fl_path = self.fl_paths[idx]
        fl_infos = obtain_feature_lines(fl_path)
        fl_pts, fl_masks = self.obtain_fl_pts(fl_infos)
        fl_pts = numpy2tensor(fl_pts)
        fl_masks = torch.Tensor(fl_masks).bool()

        if not self.fl_supervised[idx]:
            fl_masks[...] = False

        out['fl_pts'] = torch.cat(fl_pts, dim=0)
        out['fl_masks'] = fl_masks

        norm_f=self.img_ns[idx].replace('/imgs/','/normals/')[:-3]+'png'
        if osp.isfile(norm_f):
            # bgr 2 rgb
            normals=cv2.imread(norm_f)[:,:,::-1]
            normals=2.*normals.astype(np.float32)/255.-1.
            out['normal']=normals
        # norm_e=self.img_ns[idx].replace('/imgs/','/normal_edges/')[:-3]+'png'
        # if osp.isfile(norm_e):
        #     normal_edges=cv2.imread(norm_e,cv2.IMREAD_UNCHANGED)

        #     normal_edges=normal_edges.astype(np.float32)/255.
        #     out['normal_edge']=normal_edges
        out.update(garment_parsing)

        joints_idx = self.joints_frame_ids[lower_bound(self.joints_frame_ids,0,len(self.joints_frame_ids), idx)]
        gt_joints2d = self.gt_joints2d[joints_idx]
        out['gt_joints2d'] = gt_joints2d




        if self.require_albedo:
            albedo=torch.from_numpy((cv2.imread(osp.join(self.root,'albedos/%d.png'%idx)).astype(np.float32)/255.-0.5)*2.).view(self.H,self.W,3)
            out['albedo']=albedo
        # return idx,img,mask,albedo

        return idx, out

class Init_Fl_SceneDataset(SceneDataset):
    def __init__(self,data_root, conds_lens={}, garment_type = "", fl_sampling = 100, curve_sampling = 1, sample_idx = []):
        super(Init_Fl_SceneDataset, self).__init__(data_root, conds_lens, garment_type, fl_sampling, curve_sampling = curve_sampling)

        feature_lines_path = osp.join(data_root,'mask2fl')
        if not os.path.exists(feature_lines_path):
            feature_lines_path = osp.join(data_root,'featurelines')

        self.read_feature_lines(feature_lines_path)

        self.frame_num = len(sample_idx)
        self.idx = sample_idx



    def read_feature_lines(self, path):
        fl_paths = sorted(glob(os.path.join(path,'*.json')))

        self.a_pose_start, self.a_pose_end = int(fl_paths[0].split('/')[-1].replace('.json','')), int(fl_paths[-1].split('/')[-1].replace('.json',''))

        self.fl_paths = []
        self.fl_supervised =[]
        json_idx = 0

        for frame_id in range(self.frame_num):
            try:
                json_name = int(fl_paths[json_idx].split('/')[-1].split('.')[0])
            except:
                json_name = int(fl_paths[-1].split('/')[-1].split('.')[0])

            if frame_id == json_name:
                self.fl_paths.append(fl_paths[json_idx])
                json_idx+=1
                self.fl_supervised.append(True)

            else:
                self.fl_paths.append(fl_paths[json_idx-1])
                self.fl_supervised.append(False)
        # check whether has some annotation issues

        for fl_path in fl_paths:
            check_feature_lines(fl_path)



    def __getitem__(self, idx):


        idx = self.idx[idx]

        # convert to [-1.,1.] to keep consistent with render net tanh output
        out={}
        # normalized to [-1,1]
        img=torch.from_numpy((cv2.imread(self.img_ns[idx]).astype(np.float32)/255.-0.5)*2).view(self.H,self.W,3)
        out['img']=img
        # rec_f=self.img_ns[idx].replace('.%s' % (self.img_ns[idx].split('.')[-1]), '_rect.txt')
        # if osp.isfile(rec_f):
        #     rects=np.loadtxt(rect_f, dtype=np.int32)
        #     if len(rects.shape) == 1:
        #         rects = rects[None]
        #     rect = torch.from_numpy(rects[0].astype(np.float32))
        #     out['rect']=rect
        # parsing mask is the results of atr
        parsing_mask = self.parsing_mask_ns[idx]
        mask_parsing_name = 'mask_parsing_' + parsing_mask.split("/")[-1]
        mask_parsing_path = os.path.join('/'.join(parsing_mask.split('/')[:-1]),mask_parsing_name)
        # mask_parsing_results of mask according to knn atr mask
        mask_parsing = torch.from_numpy(np.load(mask_parsing_path)).long()
        garment_parsing = self.obtain_parsing_mask(mask_parsing)
        # return garment_parsing
        mask=(torch.from_numpy(cv2.imread(self.mask_ns[idx]))>0).view(self.H,self.W,-1).any(-1).float()
        out['mask']=mask

        # get fl infos
        fl_path = self.fl_paths[idx]
        fl_infos = obtain_feature_lines(fl_path)
        fl_pts, fl_masks = self.obtain_fl_pts(fl_infos)
        fl_pts = numpy2tensor(fl_pts)
        fl_masks = torch.Tensor(fl_masks).bool()

        if not self.fl_supervised[idx]:
            fl_masks[...] = False

        out['fl_pts'] = torch.cat(fl_pts, dim=0)
        out['fl_masks'] = fl_masks

        norm_f=self.img_ns[idx].replace('/imgs/','/normals/')[:-3]+'png'
        if osp.isfile(norm_f):
            # bgr 2 rgb
            normals=cv2.imread(norm_f)[:,:,::-1]
            normals=2.*normals.astype(np.float32)/255.-1.
            out['normal']=normals
        # norm_e=self.img_ns[idx].replace('/imgs/','/normal_edges/')[:-3]+'png'
        # if osp.isfile(norm_e):
        #     normal_edges=cv2.imread(norm_e,cv2.IMREAD_UNCHANGED)

        #     normal_edges=normal_edges.astype(np.float32)/255.
        #     out['normal_edge']=normal_edges
        out.update(garment_parsing)

        if self.gt_joints2d is not None:
            gt_joints2d = self.gt_joints2d[idx]
            out['gt_joints2d'] = gt_joints2d


        if self.require_albedo:
            albedo=torch.from_numpy((cv2.imread(osp.join(self.root,'albedos/%d.png'%idx)).astype(np.float32)/255.-0.5)*2.).view(self.H,self.W,3)
            out['albedo']=albedo
        # return idx,img,mask,albedo
        return idx, out
class Synthe_SceneDataset(SceneDataset):
    '''
    Synthetic Dataset
    it DO NOT need joints refine beta
    '''
    def __init__(self,data_root, conds_lens={}, garment_type = "", fl_sampling = 100, curve_sampling = 1):
        super(Synthe_SceneDataset, self).__init__(data_root, conds_lens, garment_type, fl_sampling, curve_sampling = curve_sampling)
    def __getitem__(self, idx):
        # convert to [-1.,1.] to keep consistent with render net tanh output
        out={}
        # normalized to [-1,1]
        img=torch.from_numpy((cv2.imread(self.img_ns[idx]).astype(np.float32)/255.-0.5)*2).view(self.H,self.W,3)
        out['img']=img
        # rec_f=self.img_ns[idx].replace('.%s' % (self.img_ns[idx].split('.')[-1]), '_rect.txt')
        # if osp.isfile(rec_f):
        #     rects=np.loadtxt(rect_f, dtype=np.int32)
        #     if len(rects.shape) == 1:
        #         rects = rects[None]
        #     rect = torch.from_numpy(rects[0].astype(np.float32))
        #     out['rect']=rect
        # parsing mask is the results of atr
        parsing_mask = self.parsing_mask_ns[idx]
        mask_parsing_name = 'mask_parsing_' + parsing_mask.split("/")[-1]
        mask_parsing_path = os.path.join('/'.join(parsing_mask.split('/')[:-1]),mask_parsing_name)
        # mask_parsing_results of mask according to knn atr mask
        mask_parsing = torch.from_numpy(np.load(mask_parsing_path)).long()
        garment_parsing = self.obtain_parsing_mask(mask_parsing)
        # return garment_parsing
        mask=(torch.from_numpy(cv2.imread(self.mask_ns[idx]))>0).view(self.H,self.W,-1).any(-1).float()
        out['mask']=mask

        # get fl infos


        fl_path = self.fl_paths[idx]
        fl_infos = obtain_feature_lines(fl_path)
        fl_pts, fl_masks = self.obtain_fl_pts(fl_infos)
        fl_pts = numpy2tensor(fl_pts)
        fl_masks = torch.Tensor(fl_masks).bool()

        out['fl_pts'] = torch.cat(fl_pts, dim=0)
        out['fl_masks'] = fl_masks
        norm_f=self.img_ns[idx].replace('/imgs/','/normals/')[:-3]+'png'
        if osp.isfile(norm_f):
            # bgr 2 rgb
            normals=cv2.imread(norm_f)[:,:,::-1]
            normals=2.*normals.astype(np.float32)/255.-1.
            out['normal']=normals
        # norm_e=self.img_ns[idx].replace('/imgs/','/normal_edges/')[:-3]+'png'
        # if osp.isfile(norm_e):
        #     normal_edges=cv2.imread(norm_e,cv2.IMREAD_UNCHANGED)

        #     normal_edges=normal_edges.astype(np.float32)/255.
        #     out['normal_edge']=normal_edges
        out.update(garment_parsing)

        if self.require_albedo:
            albedo=torch.from_numpy((cv2.imread(osp.join(self.root,'albedos/%d.png'%idx)).astype(np.float32)/255.-0.5)*2.).view(self.H,self.W,3)
            out['albedo']=albedo
        # return idx,img,mask,albedo
        return idx, out


class Snug_SceneDataset(SceneDataset):

    def __init__(self,data_root, conds_lens={}, garment_type = "", fl_sampling = 100, class_name = '131_11', curve_sampling = 1):
        self.anim_poses, __, __ = load_motion(os.path.join('../snug/assets/CMU', class_name.split('_')[0], class_name+'_poses.npz'))
        super(Snug_SceneDataset, self).__init__(data_root, conds_lens, garment_type, fl_sampling, curve_sampling= curve_sampling)
        self.get_grad_parameters([0], 'cuda:0')




    def get_grad_parameters(self,idxs,device):
        conds=[cond[idxs].to(device) for cond in self.conds]

        trans = self.trans[idxs].clone()
        trans[...,-1] = -trans[...,-1]


        if len(conds)>1:
            return self.poses[idxs].to(device), trans.to(device),*conds
        else:
            return self.poses[idxs].to(device), trans.to(device),*conds, None

    def __getitem__(self, idx):

        out = dict()

        out['poses_y'] = torch.from_numpy(self.anim_poses[idx]).float()
        return idx, out
    def __len__(self):
        return self.anim_poses.shape[0]

    def origin_size(self):
        return self.frame_num

    def get_camera_parameters(self,N,device):
        # return (self.camera_params['focal_length'].to(device).view(1,2).expand(N,2),self.camera_params['princeple_points'].to(device).view(1,2).expand(N,2), \
        #     utils.quat2mat(self.camera_params['cam2world_coord_quat'].to(device).view(1,4)).expand(N,3,3),
        #     self.camera_params['world2cam_coord_trans'].to(device).view(1,3).expand(N,3),self.H,self.W)


        return (self.camera_params['focal_length'].to(device).view(1,2).expand(N,2),self.camera_params['princeple_points'].to(device).view(1,2).expand(N,2), \
            utils.quat2mat(self.camera_params['cam2world_coord_quat'].to(device).view(1,4)).expand(N,3,3) @ torch.tensor([[[1.,0.,0.],[0.,-1.,0.],[0.,0., -1.]]],device=device).repeat(N,1,1),
            self.camera_params['world2cam_coord_trans'].to(device).view(1,3).expand(N,3),self.H,self.W)


import random
class ClipSampler(torch.utils.data.Sampler):
    def __init__(self,data_source,clip_size,shuffle):
        self.data_source=data_source
        self.clip_size=clip_size
        self.shuffle=shuffle
        self.n=len(self.data_source)//self.clip_size
        if len(self.data_source)==self.n*self.clip_size:
            self.n=self.n-1
        self.start=len(self.data_source)-self.n*self.clip_size
    def __iter__(self):
        if self.shuffle:
            start=random.sample(list(range(0,self.start+1)),1)[0]
        else:
            start=0
        assert(start+self.n*self.clip_size<=len(self.data_source))
        out=torch.arange(start,start+self.n*self.clip_size).view(self.n,self.clip_size)
        if self.shuffle:
            out=out[torch.randperm(self.n)]
        return iter(out.view(-1).tolist())
    def __len__(self):
        return self.n*self.clip_size

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


def getDatasetAndLoader(root,conds_lens,batch_size,shuffle,num_workers,opt_pose,opt_trans,opt_camera, garment_type, data_type = None, curve_sampling = 1, a_pose = False):
    if data_type == 'scene':
        dataset=SceneDataset(root,conds_lens, garment_type, curve_sampling = curve_sampling)
    elif data_type == 'snug':
        dataset=Snug_SceneDataset(root, conds_lens, garment_type, curve_sampling = curve_sampling)
    elif data_type == 'synthe':
        dataset=Synthe_SceneDataset(root, conds_lens, garment_type, curve_sampling = curve_sampling)
    elif data_type == 'large_pose':
        dataset=Large_Pose_SceneDataset(root, conds_lens, garment_type, curve_sampling = curve_sampling, a_pose = a_pose)
    elif data_type == 'people_snap':
        dataset=People_Snapshot_SceneDataset(root, conds_lens, garment_type, curve_sampling = curve_sampling, a_pose = a_pose)
    else:
        raise NotImplemented('data type {} is not implemented'.format(data_type))
    #opt_pose ---> True
    if opt_pose:
        dataset.poses.requires_grad_(True)
    #opt_trans --->True
    if opt_trans:
        dataset.trans.requires_grad_(True)
    dataset.opt_camera_params(opt_camera)
    # sampler=ClipSampler(dataset,batch_size,shuffle)
    # intesect sampling strategy
    sampler=RandomSampler(dataset,1,shuffle)
    dataloader=torch.utils.data.DataLoader(dataset,batch_size,sampler=sampler,num_workers=num_workers)
    return dataset,dataloader
