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

# Data preprocessing
<img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRFLdAmzXI0emjIfStSYtGBWCS3oLz_n_0YGQ&usqp=CAU" width="800px" height="250px" alt="architecture.jpg" align=center />
</div>
</br>
Collect the dataset for AMSL or MORGAN then split the data for train test. Here we took 1.2K morphed images in which we used 73 subjects ID for training and 19 subject for testing. First you need to use split.py and split the dataset into train and test so that you train and test images. Now for pairing mostly both AMSL morphed and there GT(Ground Truths) are same so its not an issue while pairing both set of images. 

For loading data you need check wheather the respective class has been called or passed while doing training. Same applied for all the dataset create a class check wheather naming conventions are matching or not otherwise try changing the names of files.

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

# Score Computation 

For computing scores you have to run score_computation.py and for normalization of score you have to run score_norm.py once this thing is done you can run score_norm2.py so that the scores rounded off upto 3 places of decimal. While running this files check if the paths for all kind images are clearly specify or not.

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

