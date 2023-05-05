"""
@File: featureline_utils.
@Author: Lingteng Qiu
@Email: qiulingteng@link.cuhk.edu.cn
@Date: 2022-07-02
@Desc:
"""
import json
import numpy as np
import torch
'''
annotation of feature lines
where
shapes: list[]

for each shape
we have 'label', 'shape_type', 'line_color', 'fill_color' and  'points' keys
'''
def check_feature_lines(name):
    fl_name = set()
    with open(name) as reader:

        fl_infos = json.load(reader)
        shapes = fl_infos['shapes']
        for fl in shapes:

            assert not fl['label'] in fl_name, "label conflict"
            fl_name.add(fl['label'])
def obtain_feature_lines(name):
    gt_fls = dict()
    with open(name) as reader:
        fl_infos = json.load(reader)
        shapes = fl_infos['shapes']
        for shape in shapes:
            label = shape['label']
            points = shape['points']
            points = np.asarray(points).astype(np.float32)
            gt_fls[label] = points

    return gt_fls




def get_curve_faces(curve):

    curve.shape[0]
    curve_id = torch.arange(curve.shape[0])[..., None]
    curve_id = torch.cat([curve_id, curve_id[0:1]], dim = 0)


    curve_faces = torch.cat([curve_id[:-1], curve_id[1:], curve_id[:-1]], dim = -1).to(curve).long()


    return curve_faces

