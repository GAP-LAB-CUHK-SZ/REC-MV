import sys
sys.path.append('./')
import torch
import trimesh
import numpy as np
import argparse
from pytorch3d.renderer import (
    RasterizationSettings,
    MeshRasterizer,
    SoftSilhouetteShader,
    TexturesVertex,
    BlendParams,
    PointsRasterizationSettings,
    # PointsRenderer,
    PointsRasterizer,
    PointLights,
    HardPhongShader,
    AlphaCompositor
)
from pytorch3d.renderer.mesh.renderer import MeshRendererWithFragments
from model.CameraMine import RectifiedPerspectiveCameras
import utils
import os

device='cpu'

camera_path ={
        'xiaolin':'./a_pose_female_process/xiaolin/camera.npz',
        'anran': None,
        'anran_tic': None,
        'lingteng_dance': None,
        }


frame_range_dict={

        'xiaolin': list(range(100,130)),
        'anran': None,
        'anran_tic': None,
        'lingteng_dance': None,
        }
file_path_dict ={
        'xiaolin':'./comparison_results/reff/xiaolin/',
        'anran': None,
        'anran_tic': None,
        'lingteng_dance': None,
        }



def get_parse():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model', default='reff', type=str)
    parser.add_argument('--subject', default='xiaolin', type=str)


    args = parser.parse_args()
    return args


def read_camera(subject):
    N = 1
    camera_file = camera_path[subject]
    data=np.load(camera_file)
    cam_data={'focal_length':torch.tensor(np.array([data['fx'],data['fy']]).astype(np.float32)), \
                        'princeple_points':torch.tensor(np.array([data['cx'],data['cy']]).astype(np.float32)), \
                        'cam2world_coord_quat':torch.from_numpy(data['quat'].astype(np.float32)).view(-1), \
                        'world2cam_coord_trans':torch.from_numpy(data['T'].astype(np.float32)).view(-1)}
    cameras=RectifiedPerspectiveCameras(cam_data['focal_length'].view(1,2).expand(N,2),cam_data['princeple_points'].view(1,2).expand(1,2),
                utils.quat2mat(cam_data['cam2world_coord_quat'].view(1,4)).expand(N,3,3),cam_data['world2cam_coord_trans'].view(1,3).expand(1,3),
                image_size=[(1080, 1080)]).to(device)

    return cameras








def main():

    args = get_parse()
    cameras = read_camera(args.subject)



    raster_settings_silhouette = RasterizationSettings(
        image_size=(1080, 1080),
        blur_radius=0.,
        bin_size=int(2 ** max(np.ceil(np.log2(max(1080, 1080))) - 4, 4)),
        faces_per_pixel=1,
        perspective_correct=True,
        clip_barycentric_coords=False,
        cull_backfaces=False
    )
    renderer = MeshRendererWithFragments(
    rasterizer=MeshRasterizer(
        cameras=cameras,
        raster_settings=raster_settings_silhouette
    ),
    shader=SoftSilhouetteShader()
    )

    raster_settings=RasterizationSettings(
            image_size=(1080,1080),
            blur_radius=0.,
            # blur_radius=np.log(1. / 1e-4 - 1.)*3.e-6,
            bin_size=(92 if max(1080,1080)>1024 and max(1080,1080)<=2048 else None),
            faces_per_pixel=1,
            perspective_correct=True,
            clip_barycentric_coords=False,
            cull_backfaces= False
        )
    renderer.rasterizer.raster_settings=raster_settings
    renderer.shader=HardPhongShader(device,renderer.rasterizer.cameras)

    frame_range = list(frame_range_dict[args.subject])

    file_path = file_path_dict[args.subject]
    for idx in frame_range:
        subj_name = os.path.join(file_path, 'nicp_{:06d}.obj'.format(idx))
        meshes = trimesh.load(subj_name, process =False)

















if __name__ == '__main__':
    main()

