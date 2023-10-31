#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from scene.cameras import Camera
import numpy as np
from utils.graphics_utils import focal2fov, fov2focal
from scene.colmap_loader import read_extrinsics_binary,read_intrinsics_binary
from scene.dataset_readers import readColmapCameras

def initial_render(dataset : ModelParams, iteration : int, pipeline : PipelineParams):
  with torch.no_grad():
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

  return gaussians, pipeline, background

def gen_set(gaussians,
      pipeline, 
      background,
      center_view={'R':np.array([[0.99988134,0.01374238,-0.00696055],[-0.01426837,0.99651294,-0.08220929],[0.00580653,0.08229885,0.99659078]],dtype=np.float32),
      'T':np.array([0.78590866,0.58750821,2.89284463],dtype=np.float32),'fx':1.5707963267948966,'fy':0.7853981633974483,'width':1000,'height':600},
      cam_range=np.array([[-0.1,0.1],[-0.1,0.1]],dtype=np.float32),
      cam_num=np.array([11,11],dtype=np.int32),
      gen_path=r'D:\LightFieldData\Homework1\LightFieldData\base_test'):
    makedirs(gen_path, exist_ok=True)
    Dx=np.linspace(cam_range[0,0],cam_range[0,1],cam_num[0])
    Dy=np.linspace(cam_range[1,0],cam_range[1,1],cam_num[1])
    for i,dx in enumerate(Dx):
      for j,dy in enumerate(Dy):
        FovX=focal2fov(center_view['fx'],center_view['width'])
        FovY=focal2fov(center_view['fy'],center_view['height'])
        cam_dict={'id':i*cam_num[0]+j,'R':center_view['R'],'T':center_view['T']+np.array([dx,dy,0],dtype=np.float32),'FovX':FovX,'FovY':FovY,'width':center_view['width'],'height':center_view['height']}
        np.save(os.path.join(gen_path,'{0:05d}'.format(cam_dict['id'])),cam_dict)
        view=Camera(colmap_id=0, 
              R=cam_dict['R'], 
              T=cam_dict['T'],
              FoVx=cam_dict['FovX'], 
              FoVy=cam_dict['FovY'], 
              image=torch.randn((3,cam_dict['height'],cam_dict['width'])), 
              gt_alpha_mask=None,
              image_name='{}'.format(cam_dict['id']), 
              uid=id)
        rendering = render(view, gaussians, pipeline, background)["render"]
        #print(rendering)
        torchvision.utils.save_image(rendering, os.path.join(gen_path, '{0:05d}'.format(cam_dict['id']) + ".png"))
    cam_range=np.array([[center_view['T'][0]+cam_range[0,0],center_view['T'][0]+cam_range[0,1]],[center_view['T'][1]+cam_range[1,0],center_view['T'][1]+cam_range[1,1]]],dtype=np.float32)
    field_param={'cam_range':cam_range,'cam_num':cam_num,'fx':center_view['fx'],'fy':center_view['fy']}
    np.save(os.path.join(gen_path,'parameters.npy'),field_param)

def gen_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        gen_set(gaussians, pipeline, background)


if __name__ == "__main__":

    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("-l", type=str)                # Save Path for Generated Traditional Light Field Data by Gaussian Splatting Rendering
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    #print(args.model_path)
    dense_sample=np.load(os.path.join(args.model_path,'dense_sample.npy'),allow_pickle=True).item()
    # Load Dense Sampled Traditional Light Field Virtual Cameras' Parameters
    normal_sample=np.load(os.path.join(args.model_path,'normal_sample.npy'),allow_pickle=True).item()
    # Load Normal Sampled Traditional Light Field Virtual Cameras' Parameters
    under_sample=np.load(os.path.join(args.model_path,'sparse_sample.npy'),allow_pickle=True).item()
    # print(under_sample['cam_num'])
    # Load Under Sampled Traditional Light Field Virtual Cameras' Parameters
    makedirs(os.path.join(args.l, 'dense_sample'), exist_ok=True)  # Dense Sampled Data Save Path
    makedirs(os.path.join(args.l,'normal_sample'),exist_ok=True) #Normal Sampled Data Save Path
    makedirs(os.path.join(args.l,'under_sample'),exist_ok=True)  #Under Sampled Data Save Path
    # Initialize system state (RNG)
    safe_state(args.quiet)
    gaussians, pipeline, background=initial_render(model.extract(args), args.iteration, pipeline.extract(args))
    gen_set(gaussians,
            pipeline,
            background,
            center_view=normal_sample['center_view'],
            cam_range=normal_sample['cam_range'],
            cam_num=normal_sample['cam_num'],
            gen_path=os.path.join(args.l,'normal_sample'))

    gen_set(gaussians,
            pipeline,
            background,
            center_view=under_sample['center_view'],
            cam_range=under_sample['cam_range'],
            cam_num=under_sample['cam_num'],
            gen_path=os.path.join(args.l, 'under_sample'))

    gen_set(gaussians,
            pipeline,
            background,
            center_view=dense_sample['center_view'],
            cam_range=dense_sample['cam_range'],
            cam_num=dense_sample['cam_num'],
            gen_path=os.path.join(args.l, 'dense_sample'))
    '''
    safe_state(args.quiet)
    gaussians, pipeline, background = initial_render(model.extract(args), args.iteration, pipeline.extract(args))
    # gen_set(gaussians,
    #         pipeline,
    #         background,
    #         center_view={'R': np.array([[0.03784449,-0.86016659,0.50860716],
    #                                     [0.99566015,-0.01084677,-0.09242946],
    #                                     [0.08502148,0.50989782,0.85602311]], dtype=np.float32),
    #                      'T': np.array([-0.0299258-0.5,-0.89334162+0.4,0.38188179-0.5], dtype=np.float32),
    #                      'FovX': 1.1431171654164658, 'FovY': 0.8006899358019289, 'width': 800, 'height': 500},
    #         cam_range=np.array([[-0.1, 0.1], [-0.1, 0.1]], dtype=np.float32),
    #         cam_num=np.array([3, 3], dtype=np.int32),
    #         gen_path=r'D:\LightFieldData\Homework1\LightFieldData\drjohnson_test')
    '''
    # #gen_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)

    # extrinsics=read_extrinsics_binary(r'D:\LightFieldData\Homework1\tandt\truck\sparse\0\cameras.bin')
    # intrinsics=read_intrinsics_binary(r'D:\LightFieldData\Homework1\tandt\truck\sparse\0\images.bin')

    # print(extrinsics)
