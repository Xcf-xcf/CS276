import numpy as np
import torchvision
import torch.nn as nn
import torch
import torch.nn.functional as F
import os
import torch
from scene import Scene
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from scene.cameras import Camera
from utils.graphics_utils import focal2fov, fov2focal

class Gaussian(nn.Module):
  def __init__(self,theta):
    super(Gaussian,self).__init__()
    self.theta=theta

  def forward(self,dis):
    dis=torch.norm(dis,dim=1)
    rbf=torch.exp(-dis/2/self.theta**2)
    rbf/=rbf.sum()
    return rbf

class ImageMap(nn.Module):#(u,v) --> (u',v')
  def __init__(self,z_focal,center_pixel,f,sep,eps=1e-9):
    super(ImageMap,self).__init__()
    self.z_focal=z_focal
    self.center_pixel=center_pixel
    self.disparity=f*sep/z_focal
    self.eps=eps

  def forward(self,img,cam,dis):
    #img:    [H,W,2]
    #cam:    [3]  s,t,z
    #neighbors: [Aperture_size,2]
    img_uv = img[None].expand(dis.shape[0], *img.shape).clone()
    #print(img_uv[0, 0, 999, 1])
    if self.z_focal:
        #print(dis)
        d=dis*self.disparity[None,:].expand(*dis.shape)
        #print(d)
        # print(self.disparity)
        # print(cam[:2])
        # print(neighbors)
        #print(d)
        img_uv+=d[:,None,None,:].expand(d.shape[0],img.shape[0],img.shape[1],d.shape[1])
        #print(dis.shape)
        #img_uv+=dis*self.disparity[None,:].expand(*dis.shape)
        scale = (self.z_focal - cam[-1]) / self.z_focal
        # print(scale)
        if scale <= self.eps:
          scale = 0
        # print(self.center_pixel)
        # print(img_uv[0,0,999,1])
        #img_uv = (img_uv - self.center_pixel[None, None, None, :].expand(*img_uv.shape)) * (scale) + self.center_pixel[None, None, None, :].expand(*img_uv.shape)
    #print(img_uv.round().int()[0,0,999,1])
    return img_uv.round().int()

class Interpolate(nn.Module):
  def __init__(self,data,uv_map,rbf=None,eps=1e-9):
    super(Interpolate,self).__init__()
    self.uv_map=uv_map
    self.rbf=rbf
    self.data=data
    self.eps=eps

  def forward(self,pixel,cam_xyz,neighbors,dis):
    #pixel:  [H,W,2]
    #cam_xyz: [3]  s,t,z
    #neighbors:[Aperture_size,2]
    uv = self.uv_map(pixel, cam_xyz, dis)
    if self.rbf==None:
        '''
        tmp=cam_xyz[None,:2].expand(*neighbors.shape)-neighbors
        '''
        tmp=torch.abs(dis[:,0]*dis[:,1]).flip(-1)
        tmp[torch.abs(tmp)<=self.eps]=0
        if torch.abs(tmp.sum()) <= self.eps:
            u = uv[0, :, :, 0].flatten().clamp(0, pixel.shape[0] - 1)
            v = uv[0, :, :, 1].flatten().clamp(0, pixel.shape[1] - 1)
            return self.data[0, :, u, v].reshape(3, pixel.shape[0], pixel.shape[1])
        wgt=tmp/tmp.sum()
    else:
        #print(dis)
        wgt=self.rbf(dis)
    #print(wgt)
    wgt[torch.abs(wgt)<=self.eps]=0
    #print((uv[0,:,:,0]-pixel[:,:,0]).sum())
    #print((uv[0,:,:,1]-pixel[:,:,1]).sum())
    rgb=torch.zeros((3,pixel.shape[0],pixel.shape[1])).float()
    #wgt=wgt[:,None,None,None].expand(*self.data.shape)
    for i in range(neighbors.shape[0]):
      if wgt[i]==0:
        continue
      u=uv[i,:,:,0].flatten().clamp(0,pixel.shape[0]-1)
      v=uv[i,:,:,1].flatten().clamp(0,pixel.shape[1]-1)
      rgb+=(wgt[i]*self.data[i,:,u,v]).reshape(3,pixel.shape[0],pixel.shape[1])
    return rgb

def initial_render(dataset : ModelParams, iteration : int, pipeline : PipelineParams):
  with torch.no_grad():
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

  return gaussians, pipeline, background

def gaussian_render(gaussians, pipeline, background, cam_xyz, neighbors, rotation, FovX, FovY, height, width, sample_density):
  #with torch.no_grad():
    rendering=None
    for i in range(sample_density):
      # print(rotation.numpy().shape)
      # print(torch.cat([neighbors[i], cam_xyz[-1][None]]).numpy().shape)
      view = Camera(colmap_id=0,
                    R=rotation,
                    T=np.array([neighbors[i,0].item(), neighbors[i,1].item(), cam_xyz[-1].item()]),
                    FoVx=FovX,
                    FoVy=FovY,
                    image=torch.randn((3, height, width)),
                    gt_alpha_mask=None,
                    image_name='{}'.format(i),
                    uid=id)
      img=render(view, gaussians, pipeline, background)["render"][None]
      #print(img)
      #torchvision.io.write_png(img.cpu().to(dtype=torch.uint8), r'D:\LightFieldData\Homework1\LightFieldData\base\synthetic\{0:05d}.png'.format(i))
      if rendering==None:
        rendering = img
      else:
        rendering = torch.cat([rendering,img])

    return rendering


# Assume that all cameras in Light Field head towards -z direction
# which are located in xy plane where the original point is the center camera.
def gaussian_rendering(gaussians,
            pipeline,
            background,
            cam_xyz,   #the position of the camera of a new synthetic view
            rotation,             #Rotation Matrix of the new view
            z_cam,                #z coordinate of cameras in Traditional Light Field
            z_focal,              #the focal plane's z coordinate
            sigma,                #sigma controls the distribution of distance between the target camera and sampled cameras
            sample_num,           #num of sampling points in sample range
            height,               #resolution of x axis
            width,                #resolution of y axis
            FovX,                 #same as that in cameras of Traditional Light Field
            FovY,                 #same as that in cameras of Traditional Light Field
            sep,                  #the interval of cameras in Traditional Light Field to normalize the subtract of neighbors and virtual camera's position vector
            save_path,            #path of the new view's image
            gaussian_weight=False, #Gaussian Weighted Interpolation(RBF kernal)
            theta=1,
            eps=1e-9):             #param of RBF kernal

  # neighbors=torch.rand(sample_num,2)
  # neighbors[:,0]=neighbors[:,0]*(sample_range[0,1]-sample_range[0,0])+sample_range[0,0]
  # neighbors[:,1]=neighbors[:,1]*(sample_range[1,1]-sample_range[1,0])+sample_range[1,0]
  neighbors = torch.randn(sample_num, 2)
  neighbors = neighbors * sigma + cam_xyz[None,:2].expand(*neighbors.shape)
  # print(neighbors)
  # print(cam_xyz)
  f=torch.tensor([fov2focal(FovY,height), fov2focal(FovX,width)]).float()
  #print(neighbors)
  dis=(neighbors-cam_xyz[None,:2].expand(*neighbors.shape))/(neighbors.max(dim=0)[0]-neighbors.min(dim=0)[0])[None,:].expand(*neighbors.shape)
  dis = torch.where(torch.isnan(dis), torch.full_like(dis, 0), dis)
  #print(dis)
  center_pixel = torch.tensor([(height - 1) / 2, (width - 1) / 2]).float()
  uv_map=ImageMap(z_focal,center_pixel,f,sep,eps=eps)
  rbf=None
  if gaussian_weight:
    rbf=Gaussian(theta)

  data=gaussian_render(gaussians, pipeline, background, cam_xyz, neighbors, rotation, FovX, FovY, height, width, sample_num).cpu()

  render=Interpolate(data,uv_map,rbf,eps=eps)
  #print(cam_xy)
  #print(neighbors)
  pixel=torch.zeros(height,width,2).float()
  pixel[:,:,0]=torch.arange(height)[:,None].expand(height,width)
  pixel[:,:,1]=torch.arange(width)[None,:].expand(height,width)
  rendering=render(pixel=pixel,
              cam_xyz=torch.cat([cam_xyz[:2],(cam_xyz[-1]-z_cam)[None]]),
              neighbors=neighbors,
              dis=dis)
  torchvision.utils.save_image(rendering,save_path)

if __name__=='__main__':
  parser = ArgumentParser(description="Testing script parameters")
  model = ModelParams(parser, sentinel=True)
  pipeline = PipelineParams(parser)
  parser.add_argument("--iteration", default=-1, type=int)
  parser.add_argument("--skip_train", action="store_true")
  parser.add_argument("--skip_test", action="store_true")
  parser.add_argument("--quiet", action="store_true")
  parser.add_argument("-l", type=str)
  args = get_combined_args(parser)
  print("Rendering " + args.model_path)
  #print(args.l)
  # Initialize system state (RNG)
  safe_state(args.quiet)

  gaussians, pipeline, background = initial_render(model.extract(args), args.iteration, pipeline.extract(args))

  gaussian_rendering(gaussians,
            pipeline,
            background,
            cam_xyz=torch.tensor([0.78590866+0.01,0.58750821+0.01,2.89284463]).float(),
            rotation=torch.tensor([[0.99988134,0.01374238,-0.00696055],[-0.01426837,0.99651294,-0.08220929],[0.00580653,0.08229885,0.99659078]]).float().numpy(),
            z_cam=2.89284463,
            z_focal=0,
            sample_range=torch.tensor([[0.78590866,0.78590866+0.02],[0.58750821,0.58750821+0.02]]).float(),
            sample_density=4,
            height=600,
            width=1000,
            FovX=1.5707963267948966,
            FovY=0.7853981633974483,
            sep=torch.tensor([0.02,0.02]).float(),
            save_path=r'D:\LightFieldData\Homework1\LightFieldData\base\synthetic\gaussian_synthetic.png',
            gaussian_weight=True,
            theta=1)

  '''
  data=LightFieldData(path=r'D:\LightFieldData\Homework1\LightFieldData\base')
  num=0
  for z_focal in np.linspace(1,1.001,100):
    traditional_rendering(data=data,
               cam_xyz=torch.tensor([0.78590866+0.01,0.58750821+0.01,2.89284463]).float(),
               z_focal=1,
               aperture_size=[2,2],#the aperture size along x and y directions
               save_path=r'D:\LightFieldData\Homework1\LightFieldData\base\synthetic\{}.png'.format(1),
               quadra=True,
               gaussian_weight=False,
               theta=1)
    num+=1
    break
    '''