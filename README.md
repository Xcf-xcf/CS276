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
git clone
cd CS276/Homework1
conda env create -file environment.yml
conda activate gaussian_splatting
```
Or you can configure the environment following the introduction in Gaussian Splatting and
```sh
pip install PySimpleGUI
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
