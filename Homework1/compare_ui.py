import PySimpleGUI as sg
from collections import  namedtuple
import numpy as np
from traditional_rendering import (LightFieldData,traditional_rendering)
import torch
from gaussian_rendering import (initial_render,gaussian_rendering)
from scene import Scene
from tqdm import tqdm
import os
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from scene.cameras import Camera
import torchvision

if __name__=='__main__':
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("-l", type=str)          #traditional field data path
    parser.add_argument("-t", type=str)          #tmp path to store synthetic images to show in UI
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)
  # Initialize system state (RNG)
    safe_state(args.quiet)

    sg.theme('LightGreen')

    dense_data=LightFieldData(path=os.path.join(args.l,'dense_sample'))
    normal_data=LightFieldData(path=os.path.join(args.l,'normal_sample'))
    sparse_data=LightFieldData(path=os.path.join(args.l,'under_sample'))
    data={'Dense':dense_data,'Normal':normal_data,'Sparse':sparse_data}
    #data={'Normal':normal_data}
    makedirs(os.path.join(args.l, 'dense_sample', 'synthetic'), exist_ok=True)
    makedirs(os.path.join(args.l, 'normal_sample', 'synthetic'), exist_ok=True)
    makedirs(os.path.join(args.l, 'under_sample', 'synthetic'), exist_ok=True)
    save_path={'Dense':[os.path.join(args.l,'dense_sample','synthetic','traditional_syn.png'), os.path.join(args.l,'dense_sample','synthetic','gaussian_syn.png')],
               'Normal':[os.path.join(args.l,'normal_sample','synthetic','traditional_syn.png'), os.path.join(args.l,'normal_sample','synthetic','gaussian_syn.png')],
               'Sparse':[os.path.join(args.l,'under_sample','synthetic','traditional_syn.png'), os.path.join(args.l,'under_sample','synthetic','gaussian_syn.png')]}
    cam_xyz=torch.from_numpy(data['Normal'].get_init_view()).float()                   # initialize view as the center camera in Traditional Light Field
    rotation=data['Normal'].get_cam_rotation()
    z_cam=cam_xyz[-1].item()
    z_focal=0
    FovX,FovY=data['Normal'].get_fov()
    aperture_size=[2,2]
    sigma=0.001
    sample_num=aperture_size[0]*aperture_size[1]
    gaussian_weight=False
    theta=1
    #print(cam_xyz)
    h_range,w_range,d_range=data['Normal'].get_view_range()
    #print(h_range,w_range,d_range)
    h_res,w_res,d_res=data['Normal'].get_view_resolution(200,200,200)
    focal_range=data['Normal'].get_focal_range()
    focal_res=data['Normal'].get_focal_resolution()
    height,width=data['Normal'].get_img_res()

    set_layout=[[sg.Text('X coordinate:'),sg.Slider(range=h_range,default_value=cam_xyz[0].item(),resolution=h_res,orientation='horizontal',key='x',enable_events=True)],
            [sg.Text('Y coordinate:'),sg.Slider(range=w_range,default_value=cam_xyz[1].item(),resolution=w_res,orientation='horizontal',key='y',enable_events=True)],
            [sg.Text('Z coordinate:'),sg.Slider(range=d_range,default_value=cam_xyz[2].item(),resolution=d_res,orientation='horizontal',key='z',enable_events=True)],
            [sg.Text('Focal Plane :'),sg.Slider(range=focal_range,default_value=z_focal,resolution=focal_res,orientation='horizontal',key='z_focal',enable_events=True)],
            [sg.Text('Aperture Size:'),sg.Input(size=(5,1),default_text=str(aperture_size[0]),key='ApertureX',enable_events=True),
             sg.Input(size=(5,1),default_text=str(aperture_size[1]),key='ApertureY',enable_events=True)],
            [sg.Text('Sigma of Gaussian Distribution to sample neighbor cameras:'),sg.Input(size=(5,1),default_text=str(sigma),key='sigma',enable_events=True)],
            [sg.Text('Choose Sample Density of Traditional Light Field Data rendered by Gaussian Splatting:'),sg.Combo(['Dense', 'Normal', 'Sparse'], 'Normal', key='DataType',enable_events=True)],
            [sg.Text('Camera Interpolation Method:'),sg.Combo(['Bilinear','Quadra-linear','Gaussian'],'Bilinear',key='Interpolation',enable_events=True)],
            [sg.Text('Theta of Gaussian Distance Weighted Func:'),sg.Input(size=(5,1),default_text=str(theta),key='theta',enable_events=True)]]

    set_window = sg.Window('Set View', set_layout)

    sep=traditional_rendering(data=data['Normal'],
                 cam_xyz=cam_xyz,
                 z_focal=z_focal,
                 aperture_size=aperture_size,#the aperture size along x and y directions
                 save_path=save_path['Normal'][0],
                 gaussian_weight=gaussian_weight,
                 theta=theta)

    show_layout=[[sg.Image(filename=save_path['Normal'][0],size=(width,height),key='Traditional_Rendering')]]
    show_window = sg.Window('Traditional Rendering', show_layout, location=(2,2))

    gaussians, pipeline, background = initial_render(model.extract(args), args.iteration, pipeline.extract(args))
    gaussian_rendering(gaussians,
                pipeline,
                background,
                cam_xyz=cam_xyz,
                rotation=rotation,
                z_cam=z_cam,
                z_focal=z_focal,
                sigma=sigma,
                sample_num=sample_num,
                height=height,
                width=width,
                FovX=FovX,
                FovY=FovY,
                sep=sep,
                save_path=save_path['Normal'][1],
                gaussian_weight=gaussian_weight,
                theta=theta)
    render_layout=[[sg.Image(filename=save_path['Normal'][1],size=(width,height),key='Gaussian_Rendering')]]

    render_window = sg.Window('Gaussian Rendering', render_layout, location=(2,2))

    while True:
        set_event, set_values = set_window.read(timeout=2000)
        if set_event!='__TIMEOUT__':
            #print(set_values)
            if set_values['Interpolation']=='Bilinear':
                set_window['ApertureX'].update(value=str(2))
                set_window['ApertureY'].update(value=str(2))
                set_values['ApertureX']=str(2)
                set_values['ApertureY']=str(2)
            elif set_values['Interpolation']=='Quadra-linear':
                set_window['ApertureX'].update(value=str(4))
                set_window['ApertureY'].update(value=str(4))
                set_values['ApertureX']=str(4)
                set_values['ApertureY']=str(4)
            set_values['ApertureX']=str(min(int(set_values['ApertureX']),data[set_values['DataType']].get_field_param()['cam_num'][0]))
            set_values['ApertureY']=str(min(int(set_values['ApertureY']),data[set_values['DataType']].get_field_param()['cam_num'][1]))
            set_window['ApertureX'].update(value=set_values['ApertureX'])
            set_window['ApertureY'].update(value=set_values['ApertureY'])
            sep=traditional_rendering(data=data[set_values['DataType']],
                                  cam_xyz=torch.tensor([set_values['x'],set_values['y'],set_values['z']]).float(),
                                  z_focal=set_values['z_focal'],
                                  aperture_size=[int(set_values['ApertureX']),int(set_values['ApertureY'])],  # the aperture size along x and y directions
                                  save_path=save_path[set_values['DataType']][0],
                                  gaussian_weight=True if set_values['Interpolation']=='Gaussian' else False,
                                  theta=float(set_values['theta']))
            #print(sample_range)
            gaussian_rendering(gaussians,
                               pipeline,
                               background,
                               cam_xyz=torch.tensor([set_values['x'],set_values['y'],set_values['z']]).float(),
                               rotation=rotation,
                               z_cam=z_cam,
                               z_focal=set_values['z_focal'],
                               sigma=float(set_values['sigma']),
                               sample_num=int(set_values['ApertureX'])*int(set_values['ApertureY']),
                               height=height,
                               width=width,
                               FovX=FovX,
                               FovY=FovY,
                               sep=sep,
                               save_path=save_path[set_values['DataType']][1],
                               gaussian_weight=True if set_values['Interpolation']=='Gaussian' else False,
                               theta=float(set_values['theta']))
            print('Done!!!')
            show_window['Traditional_Rendering'].update(source=save_path[set_values['DataType']][0])
            render_window['Gaussian_Rendering'].update(source=save_path[set_values['DataType']][1])
        show_event, show_values = show_window.read(timeout=100)
        render_event, render_values = render_window.read(timeout=100)
        # print(event,values)
        if show_event == 'Exit' or show_event == sg.WIN_CLOSED or set_event == 'Exit' or set_event == sg.WIN_CLOSED:
            break
