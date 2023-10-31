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
python gen.py -m /path/to/the/model -s /path/to/the/train/data -l /path/to/save/the/light/field/data
```
Be careful that `-m` and `-s` args are same as those in `render.py` script of the original Gaussian Splatting project.
If you would like to use you own models to generate datasets, you need creates three files named `dense_sample.npy`, `normal_sample.npy` and `sparse_sample.npy` in the model path.
These files all include a `dict` which has these elemets.
- center_view: {R: numpy.ndarray with 3*3 shape and float32 dtype which is rotation matrix,
                T: numpy.ndarray with 3*3 shape and float32 dtype which is rotation matrix,
4. 
