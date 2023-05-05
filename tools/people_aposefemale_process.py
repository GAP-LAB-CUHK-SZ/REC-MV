import h5py
import pickle
import numpy as np
import torch
import os
import os.path as osp
import cv2
from tqdm import tqdm
from glob import glob
import shutil
import argparse
parser = argparse.ArgumentParser(description='neu video body rec')
parser.add_argument('--root',default=None,metavar='M',
                        help='data root')
parser.add_argument('--sid',default=0,type=int,metavar='IDs',
					help='start frame index')
parser.add_argument('--save_root',default=None,metavar='M',
	help='save data root')
args = parser.parse_args()

root=args.root
sid=args.sid
gender='female'
save_root=args.save_root
os.makedirs(save_root,exist_ok=True)

rgb_root=osp.join(save_root,'imgs')
if not osp.isdir(rgb_root):
	os.makedirs(rgb_root)

img_files = sorted(glob(os.path.join(root,'imgs/*.jpg')))
frame_num = len(img_files)

# for ind in tqdm(range(frame_num),desc='rgbs'):
for ind in tqdm(range(len(img_files)),desc='rgbs'):
    src_img = img_files[ind]
    target_img = osp.join(rgb_root,'%06d.jpg'%(ind-sid))
    shutil.copy2(src_img, target_img)


with h5py.File(osp.join(root,'reconstructed_poses.hdf5'),'r') as ff:
    shape=ff['betas'][:].reshape(10)
    poses=ff['pose'][:].reshape(-1,24,3)[sid:,:,:]
    trans=ff['trans'][:].reshape(-1,3)[sid:,:]
    assert(poses.shape[0]>=frame_num-sid and trans.shape[0]>=frame_num-sid)
    np.savez(osp.join(save_root,'smpl_rec.npz'),poses=poses,shape=shape,trans=trans,gender=gender)

with h5py.File(osp.join(root,'masks.hdf5'),'r') as ff:
	fnum=ff['masks'].shape[0]
	assert fnum>sid
	mask_root=osp.join(save_root,'masks')
	os.makedirs(mask_root,exist_ok=True)
	for ind in tqdm(range(sid,fnum),desc='masks'):
		cv2.imwrite(osp.join(mask_root,'%06d.png'%(ind-sid)),ff['masks'][ind]*255)


with open(osp.join(root,'camera.pkl'),'rb') as ff:
    cam_data=pickle.load(ff,encoding='latin1')
    ps=cam_data['camera_c']
    fs=cam_data['camera_f']
    trans=cam_data['camera_t']
    rt=cam_data['camera_rt']
    assert(np.linalg.norm(rt)<0.0001) # The cameras of snapshot dataset seems no rotation and translation
    H=cam_data['height']
    W=cam_data['width']

    quat=np.array([np.cos(np.pi/2.),0.,0.,np.sin(np.pi/2.)])
    T=trans
    fx=fs[0]
    fy=fs[1]
    cx=ps[0]
    cy=ps[1]
    print(fx,fy,cx,cy,quat,T)

    np.savez(osp.join(save_root,'camera.npz'),fx=fx,fy=fy,cx=cx,cy=cy,quat=quat,T=T)





