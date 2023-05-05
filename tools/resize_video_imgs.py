import fire
import os
import glob
import cv2
import numpy as np
from tqdm import tqdm
import json



def stem(name):
    return name.split("/")[-1]
def obj_name(name):
    return name.split("/")[-3]

def parent(name, idx=-1):
    return '/'.join(name.split("/")[:idx])
def resize_imgs(input_imgs, input_mask, joints, save_path, size = 1080,  visualized = True):

    imgs = sorted(glob.glob(os.path.join(input_imgs,'*.jpg')))
    if len(imgs)==0:
        imgs = sorted(glob.glob(os.path.join(input_imgs,'*.png')))
    # masks = sorted(glob.glob(os.path.join(input_mask,'*alpha.png')))
    masks = sorted(glob.glob(os.path.join(input_mask,'*.png')))
    with open(joints) as fp:
        pose_seq = json.load(fp)
    obj_dirs = save_path
    os.makedirs(obj_dirs, exist_ok = True)
    img_dirs = os.path.join(obj_dirs, 'imgs')
    mask_dirs = os.path.join(obj_dirs, 'masks')
    vis_dirs = os.path.join(obj_dirs, 'visualized')
    json_name = os.path.join(obj_dirs, '2Djoints.json')
    scale_info = os.path.join(obj_dirs, 'scale_info.txt')





    scale_info_writer = open(scale_info,'w')

    os.makedirs(obj_dirs, exist_ok= True)
    os.makedirs(img_dirs, exist_ok= True)
    os.makedirs(mask_dirs, exist_ok= True)
    os.makedirs(vis_dirs, exist_ok= True)

    pose_seq_key = pose_seq.keys()
    pose_seq_key = sorted(pose_seq_key)


    h0 = float('inf')
    w0 = float('inf')
    h1 = 0
    w1 = 0

    for mask in tqdm(masks):
        mask = cv2.imread(mask, cv2.IMREAD_GRAYSCALE)
        _, silh = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
        silh = silh.astype(np.bool)
        silh = silh.astype(np.uint8) *255

        kernel = np.ones((15, 15), dtype=np.uint8)
        silh = cv2.erode(silh, kernel)
        silh = cv2.dilate(silh, kernel)
        y,x = np.where(silh>0)
        _h0, _w0, _h1, _w1 = y.min(), x.min(), y.max(), x.max()
        h0 = min(_h0, h0)
        w0 = min(_w0, w0)
        h1 = max(_h1, h1)
        w1 = max(_w1, w1)



    new_pose_seq = {}


    for img_name, mask_name, pose_key in zip(imgs, masks, pose_seq_key):

        info = ""
        print(img_name)
        # parent_dir = parent(img_name,-2)
        # os.path.join(img_name, 'visualized')

        img = cv2.imread(img_name)
        h,w, __ =img.shape
        mask = cv2.imread(mask_name, cv2.IMREAD_GRAYSCALE)


        h_l = (h1-h0)//2
        w_l = (w1-w0)//2
        h_l = h_l*1.1
        w_l = w_l*1.1
        h_c = (h1+h0) //2
        w_c = (w1+w0) //2

        y0,x0,y1,x1 = h_c-h_l, w_c-w_l, h_c+h_l, w_c+w_l
        y0 = int(max(0,y0))
        x0 = int(max(0,x0))
        x1 = int(min(x1, w))
        y1 = int(min(y1, h))

        pose = pose_seq[pose_key]

        for key in pose.keys():
            part_pose = np.asarray(pose[key])
            if part_pose.shape[0] == 0:
                continue
            else:
                part_pose[...,0]-=x0
                part_pose[...,1]-=y0
            pose[key] = part_pose.tolist()


        info +="{:.4f}\t{:.4f}\t".format(x0, y0)

        new_img = img[y0:y1,x0:x1]
        mask = mask[y0:y1,x0:x1]

        h,w,_ = new_img.shape
        h_pad = (max(h,w) - h)//2
        w_pad = (max(h,w) - w)//2
        new_img = np.pad(new_img,((h_pad,h_pad),(w_pad,w_pad),(0,0)),mode='constant', constant_values = 0)
        new_mask = np.pad(mask,((h_pad,h_pad),(w_pad,w_pad)),mode='constant', constant_values = 0)

        # padding
        for key in pose.keys():
            part_pose = np.asarray(pose[key])
            if part_pose.shape[0] == 0:
                continue
            else:
                part_pose[...,0]+=w_pad
                part_pose[...,1]+=h_pad
            pose[key] = part_pose.tolist()


        info +="{:.4f}\t{:.4f}\t".format(w_pad, h_pad)
        scale_ratio = size / new_img.shape[0]

        # resize pose
        for key in pose.keys():
            part_pose = np.asarray(pose[key])
            if part_pose.shape[0] == 0:
                continue
            else:
                part_pose[...,0]*=scale_ratio
                part_pose[...,1]*=scale_ratio
            pose[key] = part_pose.tolist()


        info +="{:.4f}\n".format(scale_ratio)
        scale_info_writer.write(info)



        new_img = cv2.resize(new_img, (size,size))
        new_mask = cv2.resize(new_mask, (size,size))


        stem_name = stem(img_name)

        cv2.imwrite(os.path.join(img_dirs,stem_name), new_img)
        cv2.imwrite(os.path.join(mask_dirs,stem(mask_name)), new_mask)

        if visualized:
            _, silh = cv2.threshold(new_mask, 100, 255, cv2.THRESH_BINARY)
            silh = silh.astype(np.bool)

            new_img[silh == False] = [0,0,0]
            for uv in np.asarray(pose['pose_keypoints_2d']).astype(np.int):
                new_img = cv2.circle(new_img,(uv[0], uv[1]) ,2, (0,0, 255),2)


            visualzied_file = os.path.join(vis_dirs,stem_name)

            cv2.imwrite(visualzied_file,new_img)
        #cv2.imwrite(img_name, new_img)
        #cv2.imwrite(mask_name, new_mask)

    with open(json_name, 'w') as writer:
        json.dump(pose_seq,writer)









if __name__ == '__main__':
    fire.Fire(resize_imgs)

