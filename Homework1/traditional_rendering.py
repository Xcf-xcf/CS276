import numpy as np
import torchvision
import torch.nn as nn
import torch
import torch.nn.functional as F
import os

class LightFieldData:
  def __init__(self,path,background_color=torch.tensor([1,1,1])):
    super(LightFieldData,self).__init__()
    files=os.listdir(path)
    imgs=[]
    cams=[]
    for f in files:
      if f=='parameters.npy':
        continue
      if 'png' in f:
        imgs.append(f)
      if 'npy' in f:
        cams.append(f)
    cams.sort()
    imgs.sort()
    self.num=len(imgs)
    self.cameras=[]
    self.background_color=background_color
    for cam in cams:
      self.cameras.append(np.load(os.path.join(path,cam),allow_pickle=True).item())
    self.images=torch.zeros(self.num,3,self.cameras[0]['height'],self.cameras[0]['width'])
    #print(self.cameras[0])
    for i,img in enumerate(imgs):
      self.images[i]=torchvision.io.read_image(os.path.join(path,img))
    self.param=np.load(os.path.join(path,'parameters.npy'),allow_pickle=True).item()

  def __getitem__(self,index):
    s,t,u,v=index[0],index[1],index[2],index[3]
    cam_id=s*self.param['cam_num'][1]+t
    # or u < 0 or u >= self.cameras[cam_id]['height'] or v < 0 or v >= self.cameras[cam_id]['width']:
    if cam_id < 0 or cam_id >= self.images.shape[0]:
      return self.background_color[:,None].expand(3,u.shape[0])
    else:
      #print(u[u<0])
      #print(u[u>=self.cameras[cam_id]['height']])
      '''
      idx=torch.bitwise_and(torch.bitwise_and(u>=0,u<self.cameras[cam_id]['height']),torch.bitwise_and(v>=0,v<self.cameras[cam_id]['width']))
      img=(-1)*torch.ones(3,self.cameras[cam_id]['height'],self.cameras[cam_id]['width'])
      img[:,u[idx],v[idx]]=self.images[cam_id,:,u[idx],v[idx]]
      img[img==-1]=self.background_color[0]
      '''
      return self.images[cam_id,:,u.clamp(0,self.cameras[cam_id]['height']-1),v.clamp(0,self.cameras[cam_id]['width']-1)]

  def __len__(self):
    return len(self.cameras)

  def get_field_param(self):
    return self.param

  def get_cam_param(self,id):
    return self.cameras[id]

  def get_cam_rotation(self):
    return self.cameras[0]['R']

  def get_fov(self):
    return self.cameras[0]['FovX'],self.cameras[0]['FovY']

  def get_all_cam_xyz(self):
    xyz=np.array([])
    for cam in self.cameras:
      if xyz.shape[0]==0:
        xyz=cam['T'][None]
      else:
        xyz=np.concatenate([xyz,cam['T'][None]])
    return xyz

  def get_init_view(self):
    return self.cameras[len(self.cameras)//2]['T']

  def get_view_resolution(self,h,w,d):
    return (self.param['cam_range'][0,1]-self.param['cam_range'][0,0])/h,(self.param['cam_range'][1,1]-self.param['cam_range'][1,0])/w,1/d

  def get_view_range(self):
    center=self.get_init_view()
    return ((self.param['cam_range'][0,0],self.param['cam_range'][0,1]),
            (self.param['cam_range'][1,0],self.param['cam_range'][1,1]),
            (center[2]-5,center[2]+5))

  def get_focal_resolution(self):
    return 0.01

  def get_focal_range(self):
    return (0,20)

  def get_img_res(self):
    return self.cameras[0]['height'],self.cameras[0]['width']

class Gaussian(nn.Module):
  def __init__(self,theta):
    super(Gaussian,self).__init__()
    self.theta=theta

  def forward(self,x,X):
    dis=torch.norm(x[None].expand(X.shape[0],2)-X,dim=1)**2
    rbf=torch.exp(-dis/2/self.theta**2)
    rbf/=rbf.sum()
    return rbf

class ImageMap(nn.Module):#(u,v) --> (u',v')
  def __init__(self,z_focal,center_pixel,f,sep,eps=1e-6):
    super(ImageMap,self).__init__()
    self.z_focal=z_focal
    self.center_pixel=center_pixel
    self.f=f
    self.sep=sep
    self.disparity=f*sep/z_focal
    #self.equal=equal
    self.eps=eps

  def forward(self,img,cam,neighbors):
    #img:    [H,W,2]
    #cam:    [3]  s,t,z
    #neighbors: [Aperture_size,2]
    img_uv = img[None].expand(neighbors.shape[0], *img.shape)
    #print(img_uv[0, 0, 999, 1])
    if self.z_focal:
      d=(neighbors-cam[None,:2].expand(*neighbors.shape))
      #d=d*d
      #disparity=1/self.z_focal
      #disparity=torch.tensor([1156.2804049882861*0.02/self.z_focal,1163.2547280302354*0.02/self.z_focal])[None,:].expand(*neighbors.shape)
      #disparity=disparity*d.norm(dim=1)[:,None].expand(*disparity.shape)
      d*=self.disparity[None,:].expand(*neighbors.shape)
      # print(self.disparity)
      # print(cam[:2])
      # print(neighbors)
      #print(d)
      img_uv=img[None].expand(neighbors.shape[0],*img.shape)+d[:,None,None,:].expand(d.shape[0],img.shape[0],img.shape[1],d.shape[1])

      if torch.abs(self.z_focal - cam[-1]) <= 1e-6:
        raise ValueError('Distance Between Image Plane and Focal Plane can not be 0')
      scale = self.z_focal / (self.z_focal - cam[-1])
      # print(scale)
      if scale <= self.eps:
        scale = 0
      # print(self.center_pixel)
      # print(img_uv[0,0,999,1])
      img_uv = (img_uv - self.center_pixel[None, None, None, :].expand(*img_uv.shape)) * (scale) + self.center_pixel[None, None, None, :].expand(*img_uv.shape)
    #print(img_uv.round().int()[0,0,999,1])
    return img_uv.round().int()

class CamNeighbor(nn.Module):# Find neighbors to interpolate
  def __init__(self,aperture_size,cam_range,cam_num,center_cam=None,eps=1e-6):
    super(CamNeighbor,self).__init__()
    self.aperture_size=aperture_size
    self.cam_range=cam_range
    self.cam_num=cam_num
    self.center_cam=center_cam
    #self.cam_xy=np.array(np.meshgrid(np.linspace(*cam_range[0],cam_num[0]),np.meshgrid(np.linspace(*cam_range[1],cam_num[1])))).reshape(2,-1).T
    self.eps=eps

  def forward(self,cam_xyz):
    cam_z=cam_xyz[None,-1]
    sep=((self.cam_range[:,1]-self.cam_range[:,0])/(self.cam_num-1)).float()
    cam_xy=(cam_xyz[:2]-self.cam_range[:,0])/sep

    # print(sep)
    #print(cam_xy)
    cams=torch.zeros(1,2,self.cam_num[0],self.cam_num[1]).float()
    cams[0,0,:,:]=torch.arange(self.cam_num[0])[:,None].expand(self.cam_num[0],self.cam_num[1])
    cams[0,1,:,:]=torch.arange(self.cam_num[1])[None,:].expand(self.cam_num[0],self.cam_num[1])

    # print(cam_xy)
    # print((2*cam_xy/(torch.tensor(self.cam_num)-1).float()-1))
    neighbor=F.grid_sample(cams,(2*cam_xy/(self.cam_num-1).float()-1).flip(-1)[None,None,None],mode='nearest',align_corners=True)[0,:,0,0]
    #print(neighbor)
    neighbor=neighbor.int()

    dct=cam_xy-neighbor
    dct[torch.abs(dct)<=self.eps]=1
    dct=torch.sign(dct)
    #print(dct)
    if self.aperture_size[0]%2==0:
      if dct[0]==-1:
        x0=neighbor[0]-((self.aperture_size[0]-1)//2+1)
      else:
        x0=neighbor[0]-((self.aperture_size[0]-1)//2)
    else:
      x0=neighbor[0]-((self.aperture_size[0]-1)//2)

    if self.aperture_size[1]%2==0:
      if dct[1]==-1:
        y0=neighbor[1]-((self.aperture_size[1]-1)//2+1)
      else:
        y0=neighbor[1]-((self.aperture_size[1]-1)//2)
    else:
      y0=neighbor[1]-((self.aperture_size[1]-1)//2)

    # print(neighbor)
    x0=max(0,x0)
    y0=max(0,y0)
    x1=x0+self.aperture_size[0]
    y1=y0+self.aperture_size[1]
    dx=max(0,x1-self.cam_num[0])
    dy=max(0,y1-self.cam_num[1])
    x0-=dx
    y0-=dy
    x1-=dx
    y1-=dy
    #print(x0,y0,x1,y1)
    neighbors=cams[0,:,x0:x1,y0:y1].permute(1,2,0).reshape(self.aperture_size[0]*self.aperture_size[1],2).int()

    return cam_xy,neighbors


class Interpolate(nn.Module):
  def __init__(self,data,uv_map,rbf,eps=1e-6):
    super(Interpolate,self).__init__()
    self.uv_map=uv_map
    self.rbf=rbf
    self.data=data
    self.eps=eps

  def forward(self,pixel,cam_xyz,neighbors):
    #pixel:  [H,W,2]
    #cam_xyz: [3]  s,t,z
    #neighbors:[Aperture_size,2]
    if self.rbf==None:
        tmp=cam_xyz[None,:2].expand(*neighbors.shape)-neighbors
        tmp=torch.abs(tmp[:,0]*tmp[:,1]).flip(-1)
        tmp[torch.abs(tmp)<=1e-6]=0
        wgt=tmp/tmp.sum()
    else:
      wgt=self.rbf(cam_xyz[:2],neighbors)

    # print(wgt)
    wgt[torch.abs(wgt)<=1e-6]=0
    uv=self.uv_map(pixel,cam_xyz,neighbors)

    #print((uv[0,:,:,0]-pixel[:,:,0]).sum())
    #print((uv[0,:,:,1]-pixel[:,:,1]).sum())
    rgb=torch.zeros((3,pixel.shape[0],pixel.shape[1])).float()
    for i in range(neighbors.shape[0]):
      if wgt[i]==0:
        continue
      index=[neighbors[i,0],neighbors[i,1],uv[i,:,:,0].flatten(),uv[i,:,:,1].flatten()]
      rgb+=(wgt[i]*self.data[index]).reshape(3,pixel.shape[0],pixel.shape[1])
    return rgb



# Assume that all cameras in Light Field head towards -z direction
# which are located in xy plane where the original point is the center camera.
def traditional_rendering(data,   #Light Field Dataset
            cam_xyz,   #the position of the camera of a new synthetic view
            z_focal,   #the focal plane's z coordinate
            aperture_size,#the aperture size along x and y directions
            save_path,
            gaussian_weight=False,# True if use gaussian weighted interpolation
            theta=1,
            eps=1e-9):
  field_param=data.get_field_param()
  cam_range=torch.tensor(field_param['cam_range']).float()
  cam_num=torch.tensor(field_param['cam_num']).int()
  cam_param=data.get_cam_param(0)
  sep = ((cam_range[:, 1] - cam_range[:, 0]) / (cam_num - 1)).float()
  center_pixel=torch.tensor([(cam_param['height']-1)/2,(cam_param['width']-1)/2]).float()
  find_cam_neighbor=CamNeighbor(aperture_size,cam_range,cam_num,eps=eps)
  uv_map=ImageMap(z_focal,center_pixel,f=torch.tensor([field_param['fy'],field_param['fx']]).float(),sep=sep,eps=eps)
  rbf=None
  if gaussian_weight:
    rbf=Gaussian(theta)
  render=Interpolate(data,uv_map,rbf,eps=eps)
  cam_xy,neighbors=find_cam_neighbor(cam_xyz)
  #print(cam_xy)
  #print(neighbors)
  pixel=torch.zeros(cam_param['height'],cam_param['width'],2).float()
  pixel[:,:,0]=torch.arange(cam_param['height'])[:,None].expand(cam_param['height'],cam_param['width'])
  pixel[:,:,1]=torch.arange(cam_param['width'])[None,:].expand(cam_param['height'],cam_param['width'])
  rendering=render(pixel=pixel,
              cam_xyz=torch.cat([cam_xy,(cam_xyz[-1]-cam_param['T'][-1])[None]]),
              neighbors=neighbors)
  torchvision.io.write_png(rendering.to(dtype=torch.uint8),save_path)
  x0,y0,x1,y1=neighbors[:,0].min().item(),neighbors[:,1].min().item(),neighbors[:,0].max().item(),neighbors[:,1].max().item()
  x,y=cam_param['T'][0],cam_param['T'][1]
  #sample_range=torch.tensor([[x0*sep[0]+x,x1*sep[0]+x],[y0*sep[1]+y,y1*sep[1]+y]]).float()
  return sep

if __name__=='__main__':
  data=LightFieldData(path=r'D:\LightFieldData\Homework1\LightFieldData\truck\dense_sample')
  num=0
  for z_focal in np.linspace(0,10,401):
    traditional_rendering(data=data,
               cam_xyz=torch.tensor([-0.4148,-0.2764,2.29]).float(),
               z_focal=z_focal,
               aperture_size=[2,2],#the aperture size along x and y directions
               save_path=r'D:\LightFieldData\Homework1\LightFieldData\truck\dense_sample\synthetic\{}.png'.format(num),
               gaussian_weight=False,
               theta=1)
    num+=1