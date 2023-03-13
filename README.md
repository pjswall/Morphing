# Facial De-morphing: Extracting Component Faces from a Single Morph (IJCB 2022)
>A face morph is created by strategically combining two or more face images corresponding to multiple identities. The intention is for the morphed image to match with multiple identities. Current morph attack detection strategies can detect morphs but cannot recover the images or identities used in creating them. The task of deducing the individual face images from a morphed face image is known as \textit{de-morphing}. Existing work in de-morphing assume the availability of a reference image pertaining to one identity in order to recover the image of the accomplice - i.e., the other identity. In this work, we propose a novel de-morphing method that can recover images of both identities simultaneously from a single morphed face image without needing a reference image or prior information about the morphing process. We propose a generative adversarial network that achieves single image-based de-morphing with a surprisingly high degree of visual realism and biometric similarity with the original face images. We demonstrate the performance of our method on landmark-based morphs and generative model-based morphs with promising results.

<a href="https://arxiv.org/abs/2209.02933"><img src="https://img.shields.io/badge/arXiv-2209.02933-b31b1b.svg" height=22.5></a>


<img src="https://github.com/sudban3089/De-Morphing-IJCB2022/blob/main/arch.jpg" style="display:block; margin:auto"/>
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
  * [Testing](#testing)
  * [Acknowledgments](#acknowledgments)
  * [Citation](#citation)
  * [Note](#note)


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
<img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRFLdAmzXI0emjIfStSYtGBWCS3oLz_n_0YGQ&usqp=CAU" width="800px" height="250px" alt="architecture.jpg" style="display:block; margin:auto">

</br>
Collect the morphed dataset from the original authors and then split into train and test sets using `split.py`. Please refer to the paper for training and test distributions. In this repository, we have provided supporting code for the AMSL dataset, you can follow the same process for other datasets. We followed the filenaming convention provided in [github repo](https://github.com/jiupinjia/Deep-adversarial-decomposition). Ensure that when image pairs are provided for training, both morphed and ground-truth images used in creating the morph have the same filename.  


## Directory Structure for Datasets

```
.
└── datasets/
    ├── AMSL/
       ├── Morph
       ├── Nonmorph
       └── orig
   
        
```
## Directory Structure for test outputs

```
.
└── eval_output/
    ├── AMSL/
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

Please staisfy all the requirements before moving to the next step.

# Training

For training the model change the directory `$ cd Morphing`. Now you can run following command:

## AMSL
>  python train.py --dataset AMSL --net_G unet_128 --checkpoint_dir checkpoints --vis_dir val_out --max_num_epochs 200 --batch_size 2 --enable_d1d2 --enable_d3 --enable_synfake --enable_biometric_loss --output_auto_enhance --gpu_devices 0

You can follow the same procedure for training using other datasets. Note we are training **only** using morphed images

# Testing

For testing run the following command:

> python eval_unmix.py --dataset AMSL --test_mode morph --ckptdir checkpoint --in_size 128 --net_G unet_128 --save_output


Mention the dataset name when you are running like above.

For Morphed and non morphed images all changes you have to do in `eval_unmix.py` while calling the the specific data loader. You can run same testing command for both. 


## Acknowledgment

The code is heavily dependent on the [github repo](https://github.com/jiupinjia/Deep-adversarial-decomposition) from the paper "Deep Adversarial Decomposition: A Unified Framework for Separating Superimposed Images" published in CVPR 2020. We used the pretrained ArcFace model provided by [deepface](https://pypi.org/project/deepface/).

## Citation

If you use this code for your research, please cite our paper:

``````
@inproceedings{fdmorph,
  title={Facial De-morphing: Extracting Component Faces from a Single Morph (IJCB 2022)},
  author={Sudipta Banerjee, Prateek Jaiswal, Arun Ross},
  booktitle={Proceedings of the International Joint Conference on Biometrics},
  year={2022}
}
``````

## Note

If you find any bugs in this code or experience issues please report it to Issues page on the Github repo or email parteek.jaiswal@research.iiit.ac.in
