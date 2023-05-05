"""
@File: fitting_garment_meshes.
@Author: Lingteng Qiu
@Email: qiulingteng@link.cuhk.edu.cn
@Date: 2022-10-15
@Desc: fitting self-recon synthetic dataset to gt garment meshes
"""
import os
import sys
import glob
sys.path.append('./')
from utils.constant import *
import os.path as osp
import numpy as np
import pdb
import utils
import torch
from pytorch3d.io import save_obj, save_ply
from pathlib import Path
import torch.nn as nn
import trimesh
from engineer.networks.OptimGarmentNetwork import bi_search, decode_color, encode_color_map
from utils.common_utils import trimesh2torch3d, numpy2tensor, tocuda, torch3dverts
from  pytorch3d.ops.knn import knn_points
from engineer.utils.mesh_utils import slice_garment_mesh, merge_meshes
from engineer.optimizer.lap_deform_optimizer import Laplacian_Optimizer
from engineer.optimizer.nricp_optimizer import NRICP_Optimizer_AdamW
from engineer.optimizer.surface_intesection import Surface_Intesection
from engineer.utils.featureline_utils import get_curve_faces
from model.network import initialLBSkinner,getTranslatorNet,CompositeDeformer,LBSkinner
from pytorch3d.structures import Meshes

def reorder_face(vertices_id, faces_id):
    new_faces_id = np.zeros_like(faces_id)
    vert_id ={}
    for new_id, ver_id in enumerate(vertices_id):
        vert_id[ver_id] = new_id
    for i in range(faces_id.shape[0]):
        for j in range(faces_id.shape[1]):
            new_faces_id[i][j] = vert_id[faces_id[i][j]]

    return new_faces_id
def find_face(vid, faces):

    faces = faces.reshape(-1)
    vertex_set = set(vid)

    faces_belongs =np.asarray([face in vertex_set for face in faces])
    faces_belongs = faces_belongs.reshape(-1,3)
    faces_sum = np.sum(faces_belongs, axis=1)
    face_idx = np.where(faces_sum == 3)[0]


    return face_idx
def prepare_dataset(meshes):

    data_info = {}

    for mesh in meshes:
        mesh_name = mesh.split('/')[-1]

        files = sorted(glob.glob(os.path.join(mesh,'*.obj')))
        mesh_property = dict()
        for file in files:
            key = file.split('/')[-1].replace('.obj','')
            mesh_property[key] = file

        files = sorted(glob.glob(os.path.join(mesh,'*.ply')))
        mesh_property['gt_mesh'] = files[0]

        data_info[mesh_name] = mesh_property

    return data_info

class fitting_net(nn.Module):
    def __init__(self, tmpBodyVs = None, tmpBodyFs = None, garment_type = None, is_upper_bottom = False):

        super().__init__()
        self.register_buffer('tmpBodyVs',tmpBodyVs)
        self.register_buffer('tmpBodyFs',tmpBodyFs)

        # add head infos
        head_ply = trimesh.load('../smpl_clothes_template/clothes_template/head.obj', process= False)
        head_colors = head_ply.visual.vertex_colors[..., 0]
        self.head_pts_idx = np.where(head_colors == 255)[0]

        self.garment_template_path = Path('../smpl_clothes_template')
        self.garment_type = garment_type

        try:
            self.is_upper_bottom = conf.get_bool('train.is_upper_bottom')
        except:
            self.is_upper_bottom = False

        self.init_template(add_head = False)
        self.garment_names = TEMPLATE_GARMENT[garment_type]


        self.fl_names = FL_INFOS[garment_type]


        self.garment_size = len(TEMPLATE_GARMENT[garment_type])
        # body vertices and garment vertices define

        # garment_engine setting

    def __load_smpl_garment_tempalte(self, template_path, add_head = True):
        '''
        load_smpl garment indx
        '''

        def load_template(temp):
            smpl_id = np.load(temp, allow_pickle= True)['smpl_id']
            return smpl_id

        def find_face(vid):

            faces = self.tmpBodyFs.detach().cpu().numpy()
            faces = faces.reshape(-1)
            vertex_set = set(vid)


            faces_belongs =np.asarray([face in vertex_set for face in faces])
            faces_belongs = faces_belongs.reshape(-1,3)
            faces_sum = np.sum(faces_belongs, axis=1)
            face_idx = np.where(faces_sum == 3)[0]


            return face_idx

        # NOTE that the deepfashion3d model has two strange pts, we remove it at the beginning
        remove_id = [5898,2568]
        need_remove_strange_pts = ['long_pants','long_skirt','short_pants','skirt', 'tube']
        self.template_id = sorted(template_path.glob('smpl_clothes_map/*.pkl'))
        self.template_v_id = {}
        self.template_f_id = {}
        self.template_add_v_id ={}
        head_pts_set = set(self.head_pts_idx.tolist())

        for template_id in self.template_id:
            add_v_set = []
            template_name = str(template_id.stem)
            if template_name in need_remove_strange_pts:
                v_id = load_template(template_id)
                for target in remove_id:
                    idx = bi_search(v_id, 0, len(v_id), target)
                    v_id.pop(idx)
            else:
                v_id = load_template(template_id)
                template_set = set(v_id)
                overlap_set = template_set & head_pts_set
                add_v_set = list(head_pts_set - overlap_set)


                if add_head:
                    v_id.extend(self.head_pts_idx.tolist())
                v_id = np.unique(v_id).tolist()

            self.template_v_id[template_name] = v_id
            self.template_f_id[template_name] = find_face(v_id)
            self.template_add_v_id[template_name] = add_v_set




    def __load_deepfashion3d_template(self, template_path):
        # load_deepfashion3d_color infos and also help us to find boudary infos
        dp3d_template = sorted(template_path.glob('clothes_template/*.ply'))


        self.dp3d_template = {}
        self.dp3d_template_color = {}
        color_map  = lambda x: x[...,0]+ (x[...,1]<<8) + (x[...,2] <<16)
        for template_id in dp3d_template:

            template_name = str(template_id.stem)
            trimesh_ply = trimesh.load_mesh(template_id, process =False)
            # color_map = trimesh_ply.visual.vertex_colors
            self.dp3d_template[template_name] =trimesh_ply
            colors = (trimesh_ply.visual.vertex_colors[..., :3]).astype(np.int32)
            colors = color_map(colors)
            self.dp3d_template_color[template_name] = colors
            uni_colors = np.unique(colors)
            uni_colors = [decode_color(x) for x in uni_colors]

        self.dp3d_template = trimesh2torch3d(self.dp3d_template)
        self.dp3d_template_color = numpy2tensor(self.dp3d_template_color)

    def init_template(self, add_head = True):
        # to compute boudnary_field
        self.__load_smpl_garment_tempalte(self.garment_template_path, add_head = add_head)
        self.__load_deepfashion3d_template(self.garment_template_path)
        align_smpl_tmp = trimesh.load('../smpl_clothes_template/aligned_smpl.obj', process=False)

        align_smpl_verts = torch.from_numpy(align_smpl_tmp.vertices).float()
        vertices = tocuda(align_smpl_verts)[None]
        dp3d_verts = torch3dverts(self.dp3d_template)
        smpl_boundary_colors = dict()
        bg_color = encode_color_map([125,125,125])

        # obtain boundary_infos
        for garment_type in TEMPLATE_GARMENT[self.garment_type]:
            add_v_id = self.template_add_v_id[garment_type]
            dp3d_vert = dp3d_verts[garment_type]
            dp3d_color = self.dp3d_template_color[garment_type]
            dp3d_vert=  tocuda(dp3d_vert)
            dist = knn_points(vertices, dp3d_vert)
            knearest = dist.idx
            smpl_v_color = dp3d_color[knearest]
            smpl_v_color[0 ,add_v_id, ...] = encode_color_map([125,125,125])
            smpl_boundary_colors[garment_type] = smpl_v_color

            # NOTE that the following initialize boundary method
            # is hard to converge.

            # add_v_id = self.template_add_v_id[garment_type]
            # dp3d_vert = dp3d_verts[garment_type]
            # dp3d_color = self.dp3d_template_color[garment_type]
            # dp3d_vert=  tocuda(dp3d_vert)
            # # dist = knn_points(vertices, dp3d_vert)
            # # knearest = dist.idx
            # dist = knn_points(dp3d_vert, vertices)
            # knearest = dist.idx[0,...,0]
            # bg_idx = (dp3d_color == bg_color)
            # knearest = knearest[torch.logical_not(bg_idx)]
            # # smpl is 6890 vertices
            # smpl_v_color = scatter(dp3d_color[torch.logical_not(bg_idx)].cuda(), knearest,reduce='mean', dim_size=6890).detach().cpu()
            # smpl_v_color[smpl_v_color ==0.] = bg_color
            # smpl_v_color = smpl_v_color[None, ..., None]
            # smpl_v_color[0 ,add_v_id, ...] = bg_color
            # smpl_boundary_colors[garment_type] = smpl_v_color


        self.smpl_boundary_colors = smpl_boundary_colors



    def obtain_body_pts_id(self, body_name, need_joints = True):

        '''
        return body_idx and body face
        '''
        body_id = [ ]
        face_id = []

        for body in body_name:
            if need_joints:
                body_id.extend((self.template_v_id[body]))
                face_id.extend((self.template_f_id[body]))
            else:
                body_id.append(self.template_v_id[body])
                face_id.append(self.template_f_id[body])
        return (np.unique(body_id), np.unique(face_id)) if need_joints else (body_id, face_id)
    def garment_by_init_smpl(self):
        '''
        obtain init smpl given canonical smpl
        '''

        # save_obj('predict_smpl.obj',vertices[0],torch.from_numpy(self.faces.astype(np.int32)))
        # xxx
        #smpl_idx, smpl_face
        body_idx, face_idx = self.obtain_body_pts_id(TEMPLATE_GARMENT[self.garment_type], False)


        slice_mesh_list = []
        garment_color_map = encode_color_map(GARMENT_COLOR_MAP)
        vertices = self.tmpBodyVs.detach().cpu()[None]
        faces = self.tmpBodyFs.detach().cpu().numpy()
        for body_id, face_id, garment_name in zip(body_idx, face_idx, TEMPLATE_GARMENT[self.garment_type]):
            # 6890 -> colors
            smpl_colors = self.smpl_boundary_colors[garment_name]
            vertices_colors= smpl_colors[:, body_id]
            smpl_verts = vertices[:, body_id]
            smpl_faces = faces[face_id, :]
            # vert, face, color
            slice_mesh_list.append(slice_garment_mesh(body_id, smpl_faces, smpl_verts, vertices_colors, decode_color, boundary_color_map = garment_color_map[garment_name], garment_type= garment_name))
        # for slice_mesh, slice_name in zip(slice_mesh_list,TEMPLATE_GARMENT[self.garment_type]):
        #     slice_mesh.save_obj(os.path.join('./debug/', 'tmp_{}.obj'.format(slice_name)))

        return slice_mesh_list


    def fitting(self, curves, gt_garment_path, root, save_path):
        garment_templates = self.garment_by_init_smpl()

        gt_mesh = trimesh.load(gt_garment_path,process =False)
        target_garment =Meshes([torch.from_numpy(gt_mesh.vertices).float()], [torch.from_numpy(gt_mesh.faces).long()])

        dense_garment_templates = []
        for garment_idx, garment_template in enumerate(garment_templates):
            # edge dense pc times
            for __ in range(3):
                garment_template = garment_template.dense_boundary()
            dense_garment_templates.append(garment_template)

        curve_sets = {}
        for curve, fl_name in zip(curves, self.fl_names):
            curve_sets[fl_name] = curve
        curve_sets = curve_sets



        garment_engine = dict(
            fl_init_registry = Laplacian_Optimizer(),
            fl_fit_surface_registry = Surface_Intesection(),
            fl_fit_registry = NRICP_Optimizer_AdamW(epoch = 250, dense_pcl=5e4, stiffness_weight = [50, 20, 5, 2, 0.8,0.5,0.35, 0.2, 0.1],
                use_normal = True, inner_iter = 50,mile_stone =[50, 80, 100, 110, 120 , 130, 140, 200], laplacian_weight = [250, 250, 250, 250, 250, 250, 250 , 250, 250]),
            )


        garment_handler = os.path.join(save_path, 'registry_{}.obj')

        registry_meshes = []
        for garment_name, garment_template, in zip(self.garment_names,dense_garment_templates):



            if os.path.exists(garment_handler.format(garment_name)):
                continue

            garment_fl_names = GARMENT_FL_MATCH[garment_name]
            curves = [curve_sets[fl_name] for fl_name in garment_fl_names]
            curve_meshes = [Meshes([curve], [get_curve_faces(curve)]) for curve in curves]

            # curve_mesh = garment_engine['fl_fit_surface_registry'](smpl_slice = curve_meshes, cano_meshes = target_garment, curve_types = )
            # 1.template laplacian deform

            lap_curve_mesh = garment_engine['fl_init_registry'](source_fl_meshes = [garment_template], target_meshes = curve_meshes, source_type = [garment_name], target_fl_type = garment_fl_names, outlayer=True)
            garment_template = lap_curve_mesh['source_fl_meshes'][0]

            garment_template.save_obj(osp.join(root, 'lap_{}.obj'.format(garment_name)))


            loss, registry_mesh = garment_engine['fl_fit_registry'](smpl_slice = garment_template, cano_meshes = target_garment, save_path = root, garment_name = garment_name, static_pts_type = 'neck')

            registry_mesh.save_obj(garment_handler.format(garment_name))



            registry_meshes.append(Meshes([registry_mesh.vertices], [registry_mesh.faces]))



        return registry_meshes




def obtain_gt_curve(data, category):

    gt_curves_list = []


    for key in FL_INFOS[category]:
        curve = trimesh.load(data[key], process = True)
        boundaries = curve.outline()
        b_pts_list = [entity.points for entity in boundaries.entities]
        b_face_ids_list = [entity.nodes for entity in boundaries.entities]
        b_pts_list =[b_pts[:-1] for b_pts in b_pts_list]

        fl_curve_list = [curve.vertices[b_pts].mean(0)[None] for b_pts in b_pts_list]
        fl_curve = torch.from_numpy(np.concatenate(fl_curve_list,axis= 0)).float()

        gt_curves_list.append(fl_curve)





    return gt_curves_list













def obtain_template(data, gender, shape, category, device = 'cuda:0'):

    gt_mesh_path  = data['gt_mesh']

    data.pop('gt_mesh')
    gt_curve_list = obtain_gt_curve(data, category)



    if os.path.exists('tmp/{}_skin.pth'.format(category)):
        data=torch.load('tmp/{}_skin.pth'.format(category), map_location='cpu')
        skinner=LBSkinner(data['ws'],data['bmins'],data['bmaxs'],data['Js'],data['parents'],init_pose=data['init_pose'],align_corners=False, extra_trans = data['extra_trans'])
        tmpBodyVs=data['tmpBodyVs']
        tmpBodyFs=data['tmpBodyFs']
    else:
        initPose=torch.from_numpy(utils.smpl_tmp_Apose(1)).view(1,24,3).to(device)
        skinner,tmpBodyVs,tmpBodyFs=initialLBSkinner(gender, shape.to(device),initPose,(128+1, 224+1, 64+1),None,None, None)
        torch.save({'ws':skinner.ws,'bmins':skinner.b_min,'bmaxs':skinner.b_max,'Js':skinner.Js,
                    'parents':skinner.parents,'init_pose':skinner.init_pose,
                    'tmpBodyVs':tmpBodyVs,'tmpBodyFs':tmpBodyFs, 'betas':shape, 'extra_trans': None },
                    'tmp/{}_skin.pth'.format(category))

    debug_path = os.path.join('./debug/gt_garment',category)
    save_path ='./selfrecon_sythe/{}'.format(category)
    os.makedirs(debug_path, exist_ok = True)

    fitnet = fitting_net(tmpBodyVs,tmpBodyFs, category)

    fitnet.fitting(gt_curve_list, gt_mesh_path, root = debug_path, save_path = save_path)


def extract_gt_mesh_points():
    tube_path = './selfrecon_sythe/gt_meshes/female_outfit3/female_outfit3_tube.ply'
    skirt_path = './selfrecon_sythe/gt_meshes/female_outfit3/female_outfit3_skirt.ply'

    template_path ={'skirt':skirt_path, 'tube':tube_path}


    color_map  = lambda x: x[...,0]+ (x[...,1]<<8) + (x[...,2] <<16)
    for key in template_path.keys():

        path = template_path[key]
        trimesh_ply = trimesh.load_mesh(path, process =False)
        # color_map = trimesh_ply.visual.vertex_colors
        colors = (trimesh_ply.visual.vertex_colors[..., :3]).astype(np.int32)
        colors = color_map(colors)


        faces = trimesh_ply.faces
        verts_idx = np.where(colors == 255)[0]

        face_idx = find_face(verts_idx, faces)
        slice_faces = faces[face_idx]


        new_face_idx = reorder_face(verts_idx, slice_faces)

        new_verts = trimesh_ply.vertices[verts_idx]
        slice_mesh = trimesh.Trimesh(new_verts, new_face_idx, process = False)
        slice_mesh.export('./{}.obj'.format(key))







def main():
    extract_gt_mesh_points()
    meshes = sorted(glob.glob('./selfrecon_sythe/gt_meshes/*'))

    datasets = prepare_dataset(meshes)
    smpl_dir_tmp = './selfrecon_sythe/{}/smpl_rec.npz'

    for key in datasets.keys():
        fl_data = datasets[key]

        smpls_path = smpl_dir_tmp.format(key)

        data=np.load(smpls_path)



        #['pose', shape, trans, gender)

        poses=torch.from_numpy(data['poses'].astype(np.float32)).view(-1,24,3)
        trans=torch.from_numpy(data['trans'].astype(np.float32)).view(-1,3)
        # general beta ---> 10
        shape=torch.from_numpy(data['shape'].astype(np.float32)).view(-1)
        gender=str(data['gender']) if 'gender' in data else 'neutral'


        obtain_template(fl_data, gender, shape, key,)






if __name__ == '__main__':
    main()
