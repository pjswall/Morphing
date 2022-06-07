# Morphing
Morphing is a process of creating a single identity from two different identities(but follow some similarity constraints). The morph image can match with both the identities which are used for generating that image. The work presented here is related with morph attack detection and tries to detect the morphed image with single image only without any reference image as aid to detection which is the first work in this direction. Existing methods only work with a reference image. It will boost secutrity at border control systems. We do separation of a morph image into its constituents ID's using GAN based approach with biometric loss function as a separation critic along with GAN loss and Pixel loss.

<img src="https://github.com/prateekj7777/Morphing/blob/main/arch..jpg" width="500px" height="300px" alt="architecture.jpg" align=center />
</div>
</br>

# Requirements
Please install following packages 
- numpy= 1.11.0
- torch=1.9.-
- torchvision=0.10.0
- deepface=0.0.74
- Pillow==3.1.2
- psutil==3.4.2
- matplotlib

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

# References

The baseline code taken from the git repo [Deep-Adversarial-Decomposition](https://github.com/jiupinjia/Deep-adversarial-decomposition) mentioned in the reference paper

