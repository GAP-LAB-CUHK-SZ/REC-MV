"""
@File: parsing_mask_to_fl.py
@Author: Lingteng Qiu
@Email: qiulingteng@link.cuhk.edu.cn
@Date: 2022-10-19
@Desc: parsing mask to polygons, given a series of mask infos
"""
import sys
sys.path.extend('./')
import argparse
import os
import glob
import os.path as osp
import numpy as np
import cv2
import pdb
import cv2
import json
from pytorch3d.ops.knn import knn_points
import torch
from utils.constant import FL_EXTRACT, TEMPLATE_GARMENT


FL_COLOR = {
    'neck':(0, 0, 255),
    'right_cuff': (0, 255, 0),
    'left_cuff':(255, 0, 0),
    'left_pant': (127, 127, 0),
    'right_pant':(0, 127, 127),
    'upper_bottom': (127, 0, 127),
    'bottom_curve':(0, 127, 127),
}

class PolyMask(object):
    def __init__(self, mask):
        self.mask = mask

    def query(self,query_sets ,labels, garment_key):

        mask = np.zeros_like(self.mask, dtype= np.bool)

        for label in labels:
            label_mask = np.zeros_like(self.mask, dtype =np.bool)
            i,j = np.where(self.mask == label)
            label_mask[i,j] = True

            mask |= label_mask

        mask = mask.astype(np.uint8)*255
        mask = self.smooth_noise(mask)

        mask_polygons, mask_area = self.mask2polygon(mask)




        length_dp = []
        for mask_polygon in mask_polygons:
            dis = [0]
            dis.extend([abs(mask_polygon[p_i][0]- mask_polygon[p_i+1][0]) + abs(mask_polygon[p_i][1]- mask_polygon[p_i+1][1]) for p_i in range(mask_polygon.shape[0]-1)])
            dis.append(abs(mask_polygon[0][0]- mask_polygon[-1][0]) + abs(mask_polygon[0][1]- mask_polygon[-1][1]))
            dp = np.cumsum(dis)
            length_dp.append(dp)
        new_query_sets = {}



        reply_pts = np.concatenate(mask_polygons, axis=0)
        reply_pts = torch.from_numpy(reply_pts).float().cuda()



        for key in query_sets.keys():

            polygon = query_sets[key]



            assert polygon.shape[0] %2 == 0
            polygons = polygon.reshape(-1, 2, 2)

            group = []
            for group_id, mask_polygon in enumerate(mask_polygons):
                group.extend([group_id for i in range(mask_polygon.shape[0])])



            group =  torch.tensor(group).long()

            new_polygons=[]

            pre_polygon = None



            for polygon in polygons:

                polygon = torch.from_numpy(polygon).float().cuda()





                if pre_polygon is not None:
                    dis = torch.sqrt(((polygon[0] - pre_polygon[-1]) **2).sum())
                    if dis < 10:
                        new_polygons.append(polygon.detach().cpu().numpy())
                        pre_polygon = None
                        continue


                pre_polygon = polygon.detach().clone()

                dist = knn_points(polygon[None], reply_pts[None])
                idx = dist.idx[0, ...,0]
                group_id = group[idx]

                if dist.dists.max()>1000:
                    new_polygons.append(polygon.detach().cpu().numpy())
                    continue


                prefer_id = group_id[0] if mask_area[group_id[0]] > mask_area[group_id[1]] else group_id[1]
                prefer_pts = torch.from_numpy(mask_polygons[prefer_id]).float().cuda()
                dist = knn_points(polygon[None], prefer_pts[None])
                idx = dist.idx[0, ...,0].sort()


                polygon= polygon[idx.indices]
                idx=idx.values

                reverse_flag = (not idx[0] == dist.idx[0, 0, 0])


                # obtain slice_curve

                dp = length_dp[prefer_id]
                slice_a = dp[idx[1]] - dp[idx[0]]
                slice_b = dp[-1] - slice_a

                #obtain slice_b
                if slice_a>slice_b:
                    segment = torch.cat([polygon[1:],prefer_pts[idx[1]:], prefer_pts[:idx[0]+1], polygon[0:1]], dim = 0)
                    reverse_flag = (not reverse_flag)

                else:
                    segment = torch.cat([polygon[0:1], prefer_pts[idx[0]:idx[1]+1], polygon[1:]], dim=0)

                segment = segment.detach().cpu().numpy()
                if reverse_flag:
                    segment = segment[::-1]

                new_polygons.append(segment)


            new_polygons = np.concatenate(new_polygons, axis = 0)



            new_query_sets[key] = new_polygons
        return new_query_sets, mask





    def smooth_noise(self, mask):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5, 5))
        mask = cv2.erode(mask, kernel, iterations=2)
        mask = cv2.dilate(mask, kernel, iterations=2)

        return mask

    def mask2polygon(self, mask):
        contours, hierarchy = cv2.findContours((mask).astype(np.uint8), cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        # mask_new, contours, hierarchy = cv2.findContours((mask).astype(np.uint8), cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        segmentation = []
        polygon_size = []
        for contour in contours:
            contour_list = contour.flatten().tolist()
            if len(contour_list) > 4:# and cv2.contourArea(contour)>10000

                area = self.polygons_to_mask(mask.shape, contour_list).sum()
                polygon_size.append(area)

                contour_numpy = np.asarray(contour_list).reshape(-1, 2)
                segmentation.append(contour_numpy)


        return segmentation, polygon_size

    def polygons_to_mask(self, img_shape, polygons):
        mask = np.zeros(img_shape, dtype=np.uint8)

        polygons = np.asarray(polygons, np.int32) # 这里必须是int32，其他类型使用fillPoly会报错
        shape=polygons.shape

        polygons=polygons.reshape(-1,2)
        cv2.fillPoly(mask, [polygons],color=1) # 非int32 会报错
        return mask



def get_upper_bttom_type(parsing_type, key):

    # 'ATR': ['Background', 'Hat', 'Hair', 'Sunglasses', 'Upper-clothes', 'Skirt', 'Pants', 'Dress', 'Belt',
    #               'Left-shoe', 'Right-shoe', 'Face', 'Left-leg', 'Right-leg', 'Left-arm', 'Right-arm', 'Bag', 'Scarf'],
    ATR_PARSING = {
        # with head and hand
        'upper':[4, 16, 17],
        #without head
        #  'upper':[1, 2, 3, 4, 11, 16, 17],
        'bottom':[5, 6, 8],
        #  with head and hand
        'upper_bottom':[4, 5, 6, 7, 8, 16, 17]
    }
    CLO_PARSING = {
        # with head and hand
        'upper':[1,2,3],
        #without head
        #  'upper':[1, 2, 3, 4, 11, 16, 17],
        'bottom':[1,2,3],
        #  with head and hand
        'upper_bottom':[1,2,3]
        # w/o hand
        # 'upper_bottom':[4, 5, 7, 16, 17]
    }

    if parsing_type =='ATR':
        return ATR_PARSING[key]
    else:
        return CLO_PARSING[key]


def get_parsing_label(parsing_type):
    parsing_table ={
        'ATR': ['Background', 'Hat', 'Hair', 'Sunglasses', 'Upper-clothes', 'Skirt', 'Pants', 'Dress', 'Belt',
                  'Left-shoe', 'Right-shoe', 'Face', 'Left-leg', 'Right-leg', 'Left-arm', 'Right-arm', 'Bag', 'Scarf'],
        'CLO':['background', 'upper', 'bottom', 'upper-bottom']
    }

    return parsing_table[parsing_type]

def get_parse():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--parsing_type', default='ATR', help='garment_parsing type', choices=['ATR', 'CLO'])
    parser.add_argument('--input_path', default='', help='select model')
    parser.add_argument('--output_path', default='', help='polygons output')

    args = parser.parse_args()
    return args

def parsing_curve(query_file, parsing_file, parsing_type,  class_type, debug_path, name):

    query_sets = {}
    with open(query_file) as reader:
        fl_infos = json.load(reader)
        shapes = fl_infos['shapes']
        for fl in shapes:
            query_sets[fl['label']] = np.asarray(fl['points']).astype(np.float32)

    class_table = dict(
            female_outfit3=['upper_bottom'],
            female_outfit1=['upper_bottom'],
            anran_run = ['short_sleeve_upper', 'skirt'],
            anran_tic = ['short_sleeve_upper', 'skirt'],
            leyang_jump = ['dress'],
            leyang_steps = ['dress'],
            )

    garment_table = dict(
            short_sleeve_upper='upper',
            skirt='bottom',
            dress='upper_bottom',
            long_sleeve_upper='upper',
            long_pants='bottom',
            short_pants='bottom',
            )

    masks = np.load(parsing_file, allow_pickle= True)




    parsing_name = parsing_file.split('/')[-1]

    poly_mask = PolyMask(masks)
    new_query_sets = {}



    for garment_key in  TEMPLATE_GARMENT[class_type]:

        garment_class = get_upper_bttom_type(parsing_type, garment_table[garment_key])
        fl_names = FL_EXTRACT[garment_key]
        fl_query_sets = {}

        for fl_name in fl_names:
            if fl_name in query_sets.keys():
                fl_query_sets[fl_name] = query_sets[fl_name]



        new_fl_query_sets, mask = poly_mask.query(fl_query_sets, garment_class, garment_key)
        new_query_sets.update(new_fl_query_sets)

        cv2.imwrite(osp.join(debug_path, 'mask_{}_'.format(garment_key)+name), mask)


    return new_query_sets, mask


def main(args):
    parsing_type = args.parsing_type
    parsing_label = get_parsing_label(parsing_type)

    parsing_dir = osp.join(args.input_path, 'parsing_SCH_{}'.format(parsing_type))
    img_dir = osp.join(args.input_path, 'imgs/')
    json_files = sorted(glob.glob(osp.join(args.input_path, 'featurelines/*.json')))
    img_files = sorted(glob.glob(osp.join(img_dir, '*.jpg')))
    json_key = [json_file.split('/')[-1][:-5] for json_file in json_files]

    parsing_files = sorted(glob.glob(osp.join(parsing_dir,'mask_parsing_*.npy')))
    filter_parsing_files = list(filter(lambda x: x.split('/')[-1].split('_')[-1][:-4] in json_key, parsing_files))
    filter_img_files = list(filter(lambda x: x.split('/')[-1][:-4] in json_key, img_files))








    if args.input_path[-1] =='/':
        input_path = args.input_path[:-1]

    class_type = input_path.split('/')[-1]
    debug_path =  osp.join('./debug/{}/polymask'.format(class_type))
    output_path = args.output_path
    os.makedirs(output_path, exist_ok = True)
    os.makedirs(debug_path, exist_ok= True)

    # filter_parsing_files = filter_parsing_files[12:13]
    # json_files = json_files[12:13]
    # filter_img_files = filter_img_files[12:13]



    for parsing_file, json_file, filter_img_file in zip(filter_parsing_files, json_files, filter_img_files):
        print('processing: {}'.format(filter_img_file))
        img = cv2.imread(filter_img_file)

        name = filter_img_file.split('/')[-1]

        new_query_sets, mask = parsing_curve(json_file, parsing_file, args.parsing_type, class_type, debug_path, name)

        with open(json_file) as reader:
            fl_infos = json.load(reader)
            shapes = fl_infos['shapes']
            for fl in shapes:
                # query_sets[fl['label']] = np.asarray(fl['points']).astype(np.float32)
                fl['points']= new_query_sets[fl['label']].tolist()


        json_name = json_file.split('/')[-1]
        new_json_file = os.path.join(output_path, json_name)
        with open(new_json_file, 'w') as writer:
            json.dump(fl_infos, writer)


        for key in new_query_sets.keys():
            color = FL_COLOR[key]
            pt_list = new_query_sets[key].astype(np.int)
            for pt in new_query_sets[key].astype(np.int):
                img = cv2.circle(img, (pt[0], pt[1]),2, color,2)

            for pt_idx in range(pt_list.shape[0]-1):
                img = cv2.line(img, (pt_list[pt_idx][0],pt_list[pt_idx][1]), (pt_list[pt_idx+1][0],pt_list[pt_idx+1][1]), color, 2)


        cv2.imwrite(osp.join(debug_path, name), img)







if __name__ == '__main__':

    args = get_parse()
    main(args)


