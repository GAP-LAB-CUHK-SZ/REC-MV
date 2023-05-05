import sys
sys.path.append('./')
import argparse
import os
import glob
import cv2
import os.path as osp
from utils.constant import TEMPLATE_GARMENT
import numpy as np
path = '/data4/lingtengqiu/show/gallery/'

def get_parse():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--input', default='input_path')


    args = parser.parse_args()
    return args
def main(args):
    input_path = args.input
    if input_path[-1] =='/':
        input_path = input_path[:-1]

    subject = input_path.split('/')[-2]
    garment_types = TEMPLATE_GARMENT[subject]
    main_subject = osp.join(*input_path.split('/')[:-1])

    save_path = osp.join(path, subject)
    os.makedirs(save_path, exist_ok= True)
    imgs = sorted(glob.glob(os.path.join(main_subject, 'imgs/*.png')))
    imgs.extend(sorted(glob.glob(os.path.join(main_subject, 'imgs/*.jpg'))))
    meshs_path = sorted(glob.glob(osp.join(input_path,'meshs/*.png')))
    fl_path = sorted(glob.glob(osp.join(main_subject,'featurelines/*.json')))
    start = int(fl_path[0].split('/')[-1].replace('.json', ''))
    end = int(fl_path[-1].split('/')[-1].replace('.json', ''))



    imgs = imgs[start:end+1]



    if len(garment_types) ==2:

        half = len(meshs_path) // 2

        upper_idx = list(range(half))
        bottom_idx = list(range(half, len(meshs_path)))

        meshs_path = [[meshs_path[i], meshs_path[j]] for i,j in zip(upper_idx, bottom_idx)]

    else:
        meshs_path =[ [mesh] for mesh in meshs_path]


    for idx, (img, meshs) in enumerate(zip(imgs, meshs_path)):
        idx = start+idx


        img = cv2.imread(img)
        meshes=  [img]
        for mesh in meshs:
            mesh_img = cv2.imread(mesh)
            meshes.append(mesh_img)


        mesh_img = np.concatenate(meshes, axis = 1)


        save_img = os.path.join(save_path, '{:06d}.png'.format(idx))
        cv2.imwrite(save_img, mesh_img)


        print(save_img)






if __name__ == '__main__':
    args = get_parse()
    main(args)

