# CS276
Homeworks of Computational Photography 

## Homework 1
### Usage
1. Configure Environment
In my project, I have used
- CUDA Toolkit 11.8
- Python 3.10.3
- Pytorch 2.1.0
- Visual Studio 2022
- Cmake 3.28.0
in Windows x64 platform with RTX3060 GPU. 
```sh
git clone --recursive https://github.com/graphdeco-inria/gaussian-splatting.git
cd gaussian_splatting
conda create -n gaussian_splatting python=3.10
conda activate gaussian_splatting
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install PySimpleGUI==4.60.5
pip install tqdm
pip install plyfile==0.8.1
pip install -e submodules/diff-gaussian-rasterization/.
pip install -e submodules/simple-knn/.
```
Or you can configure the environment following the introduction in Gaussian Splatting and
```sh
pip install PySimpleGUI
```
Then you should download my scripts and move them to your Gaussian Splatting path like this,
```sh
cd ..
git clone https://github.com/Xcf-xcf/CS276.git
mv /CS276/Homework1/* /gaussian_splatting/
cd gaussian_splatting
```
2. Generate DataSets
You can use my models to generate light field dataset like these,
```sh
python gen.py -m /path/to/the/model -s /path/to/the/train/data/of/Gaussian/Splatting -l /path/to/save/the/light/field/data
```
Be careful that `-m` and `-s` args are same as those in `render.py` script of the original Gaussian Splatting project.
If you would like to use you own models to generate datasets, make sure that you have created three files named `dense_sample.npy`, `normal_sample.npy` and `sparse_sample.npy` which have different sampling numbers of views in the model path.
These files all include a `dict` which has these elemets.
- 'center_view': a 'dict' <br>
   'R': a `numpy.ndarray` with `(3,3)` shape and `float32` dtype which is rotation matrix for all views of light field,<br>
   'T': a `numpy.ndarray` with `(3,)` shape and `float32` dtype which is translation matrix for the center view of light field,<br>
   'width': an `int32` number of images' resolution for all views of light field along x direction,<br>
   'height': an `int32` number of images' resolution for all views of light field along y direction,<br>
   'fx': a `float32` number which is focal length along x direction corresponding to width of images,<br>
   'fy': a `float32` number which is focal length along y direction corresponding to height of images
- 'cam_range': a `numpy.ndarray` with `(2,2)` shape and `float32` dtype which have the offsets from the center virtual camera to the marginal virtual cameras along x,y directions
- 'can_num':   a `numpy.ndarray` with `(2,)` shape and `int32` dtype which have the numbers of rendering(sampling) views in Gaussian Splatting for lighjt field along x,y directions
3.  Run Compare UI
After generation of datasets, you can run the UI to compare traditional light field and gaussian splatting light field like this.
```sh
python compare_ui.py -m /path/to/the/model -s /path/to/the/train/data/of/Gaussian/Splatting -l /path/to/the/light/field/data
```
Then you will see a small setting UI to change parameters and two showing UI to display the renderings of traditional light field and Guasian Splatting light field like these.
- Set Window
![image](main/Homework1/set_window.png)
- Traditional Rendering Window
![image](main/Homework1/traditional_window.png)
- Gaussian Rendering Window
![image](main/Homework1/gaussian_window.png)
There are several parameters for traditional and gaussian splatting light field rendering such as,
- `X coordinate`
- `Y coordinate`
- `Z coordinate`
- `Focal Plane` which is depth of focal plane
- `Aperture Size` which is the number of selected virtual cameras to interpolate
- `Sigma of Gaussian Distribution to sample neighbor cameras` which controls the dispersion of sampling cameras in Gaussian Splatting light field
- `Sample Density of Traditional Light Field Data rendered by Gaussian Splatting` which includes dense, normal and sparse three modes
- `Camera Interpolation Method` which include bilinear, quadra-linear and Gaussian three modes
- `Theta of Gaussian Distance Weighted Func` which controls the relationship between the distance changing between the target camera and its neighbors and weight changing of its neighbors 
 
