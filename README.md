# Facial De-morphing: Extracting Component Faces from a Single Morph (IJCB 2022)
>A face morph is created by strategically combining two or more face images corresponding to multiple identities. The intention is for the morphed image to match with multiple identities. Current morph attack detection strategies can detect morphs but cannot recover the images or identities used in creating them. The task of deducing the individual face images from a morphed face image is known as \textit{de-morphing}. Existing work in de-morphing assume the availability of a reference image pertaining to one identity in order to recover the image of the accomplice - i.e., the other identity. In this work, we propose a novel de-morphing method that can recover images of both identities simultaneously from a single morphed face image without needing a reference image or prior information about the morphing process. We propose a generative adversarial network that achieves single image-based de-morphing with a surprisingly high degree of visual realism and biometric similarity with the original face images. We demonstrate the performance of our method on landmark-based morphs and generative model-based morphs with promising results.

<a href="https://arxiv.org/abs/2209.02933"><img src="https://img.shields.io/badge/arXiv-2209.02933-b31b1b.svg" height=22.5></a>


<img src="https://github.com/prateekj7777/Morphing/blob/main/arch.jpg" width="500px" height="300px" alt="architecture.jpg" align=center />
</div>
</br>

## Table of Contents
  * [Getting Started](#getting-started)
    + [Prerequisites](#prerequisites)
    + [Installation](#installation)
  * [Requirements](#requirements)
  * [Data Preprocessing](#data-preprocessing) 
  * [Directory Structure for Datasets](#directory-structure-for-datasets) 
  * [Directory Structure for test outputs](#directory-structure-for-test-outputs)
  * [How to Run](#how-to-run)
  * [Training](#training)
    + [AMSL](#amsl)
    + [MORGAN](#morgan)
    + [EMORGAN](#emorgan)
    + [REGEN](#regen)
  * [Testing](#testing)
  * [Dataset](#dataset)
    + [Dataset specification](#dataset-specification)
  * [Score Computation](#score-computation)
  * [Credits](#credits)
  * [Acknowledgments](#acknowledgments)
  * [Citation](#citation)


## Getting Started
### Prerequisites
- Linux or macOS
- NVIDIA GPU + CUDA CuDNN (CPU may be possible with some modifications, but is not inherently supported)
- Python 3

### Installation
- Dependencies: 
We recommend running this repository using [Anaconda](https://docs.anaconda.com/anaconda/install/). 
All dependencies for defining the environment

## Requirements
Please install following packages 
- numpy= 1.11.0
- torch=1.9.-
- torchvision=0.10.0
- deepface=0.0.74
- Pillow==3.1.2
- psutil==3.4.2
- matplotlib

## Data preprocessing
<img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRFLdAmzXI0emjIfStSYtGBWCS3oLz_n_0YGQ&usqp=CAU" width="800px" height="250px" alt="architecture.jpg" align=center />

</br>
Collect the dataset for AMSL or MORGAN then split the data for train test. Here we took 1.2K morphed images in which we used 73 subjects ID for training and 19 subject for testing. First you need to use split.py and split the dataset into train and test so that you train and test images. Now for pairing mostly both AMSL morphed and there GT(Ground Truths) are same so its not an issue while pairing both set of images. 

For loading data you need check wheather the respective class has been called or passed while doing training. Same applied for all the dataset create a class check wheather naming conventions are matching or not otherwise try changing the names of files.

## Directory Structure for Datasets

```
.
└── datasets/
    ├── AMSL/
    │   ├── Morph
    |   ├── Nonmorph
    │   └── orig
    ├── MOR/
    |   ├── Morphed_Train
    |   ├── Nonmorph
    |   └── Morphed_Test
    ├── EMOR/
    |   ├── Demo
    |   ├── Morphed_Train
    |   ├── Nonmorph
    |   ├── Set1
    |   └── Set2
    ├── REGEN/
        ├── test
        ├── train
        ├── Nonmorph
        └── gt
        
```
## Directory Structure for test outputs

```
.
└── eval_output/
    ├── AMSL/
    │   ├── Morph
    |   ├── Nonmorph
    ├── MOR/
    |   ├── Morph
    |   ├── Nonmorph
    ├── EMOR/
    |   ├── Morph
    |   ├── Nonmorph
    ├── REGEN/
        ├── Morph
        ├── Nonmorph


```

# How to RUN
1. Download the data files for the required model. Extract and place it in the data folder.
2. Create a new virtual environment

using conda:
```
>>> conda create -n <env-name>
```
using pip
```
>>> python -m pip install --user virtualenv
```
4. Install the required packages using the requirements.txt file with the following command
```
>>> pip install -r requirements.txt
```

Once you have all the above requirements now you can start running the program.

# Training

For training the model you need to change directory `$ cd Morphing`. Now you can run following command:

## AMSL
>  python train.py --dataset AMSL --net_G unet_128 --checkpoint_dir checkpoints --vis_dir val_out --max_num_epochs 200 --batch_size 2 --enable_d1d2 --enable_d3 --enable_synfake --enable_biometric_loss --output_auto_enhance --gpu_devices 0

## MORGAN
>  python train.py --dataset MOR --net_G unet_128 --checkpoint_dir checkpoints --vis_dir val_out --max_num_epochs 200 --batch_size 2 --enable_d1d2 --enable_d3 --enable_synfake --enable_biometric_loss --output_auto_enhance --gpu_devices 0

## EMORGAN
>  python train.py --dataset EMOR --net_G unet_128 --checkpoint_dir checkpoints --vis_dir val_out --max_num_epochs 200 --batch_size 2 --enable_d1d2 --enable_d3 --enable_synfake --enable_biometric_loss --output_auto_enhance --gpu_devices 0

## REGEN
> python train.py --dataset REGEN --net_G unet_128 --checkpoint_dir checkpoints --vis_dir val_out --max_num_epochs 200 --batch_size 2 --enable_d1d2 --enable_d3 --enable_synfake --enable_biometric_loss --output_auto_enhance --gpu_devices 0

# Testing

For testing the datasets you have to run the following command
> #python eval_unmix.py --dataset REGEN --ckptdir checkpoint --in_size 128 --net_G unet_128 --save_output


Here also you have to mention the dataset specifically when you are running like above.

For Morphed and non morphed images all changes you have to do in `eval_unmix.py` while calling the the specific data loader. You can run same testing command for both.  

# Dataset

We have used two face morph datasets.

### AMSL Dataset

This dataset contains morph images from 102 subjects captured with neutral as well as smiling expressions. Thare are 2175 morphed images corresponding to 92 subjects created using a landmark-based approach. 

### E-MorGAN Dataset


This dataset created using GAN's and all the morphed images are generated with the GAN architecture called MorGAN. Dataset is having at 1000 images which are splittes as train and test sets of 500 images in each set.

Dataset specification
---------------------


| Dataset 	| split 	| Morphed/Non-morphed 	| no. of species | No. of images
|---	|---	|---	|---  |---
| AMSL | Train | Morphed | 73 | 946
| AMSL | Test | Morphed | 16 | 56
| AMSL | Test | Non-morphed | 29 | 57
| E-MorGAN | Train | Morphed | 251 | 499
| E-MorGAN | Test | Morphed | 90 | 100
| E-MorGAN | Test | Non-morphed | 50 | 100

Here, the required details about each dataset presented in the below table.

## Score Computation 

For computing scores you have to run `score_computation.py` and for normalization of score you have to run `score_norm.py once` this thing is done you can run `score_norm2.py` so that the scores rounded off upto 3 places of decimal. While running this files check if the paths for all kind images are clearly specify or not.


## References

The baseline code taken from the git repo [Deep-Adversarial-Decomposition](https://github.com/jiupinjia/Deep-adversarial-decomposition) mentioned in the reference paper

## Citation

If you use this code for your research, please cite our paper:

<!-- ``````
@inproceedings{zou2020deep,
  title={Deep Adversarial Decomposition: A Unified Framework for Separating Superimposed Images},
  author={Zou, Zhengxia and Lei, Sen and Shi, Tianyang and Shi, Zhenwei and Ye, Jieping},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={12806--12816},
  year={2020}
}
`````` -->

