import time
import sys
import os.path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cv2
import argparse
from glob import glob
import torch
import torchvision.transforms.functional as TF

def read_img(image_paths, DESTDIR):
    #if not os.path(image_paths):
        #print("Not specified directory location")
        #return none
    i = 0
    for path in image_paths:
        pic = cv2.imread(path)
        min_w, min_h = sys.maxsize, sys.maxsize
        h, w, channels = pic.shape
        min_w = min(min_w, w)
        min_h = min(min_h, h)
        pic = pic[:min_h,:min_w,:]
        
        img = pic/255.0
        
#         cv2.imshow("I",img)
#         cv2.waitKey(3000)
        
        i += 1
        cv2.imwrite(DESTDIR+"image_"+str(i)+".jpg",img)
        
    print("Done")
        
if __name__ == '__main__':
    
    SOURCEDIR = '../AMSL_FaceMorphImageDataSet/londondb_morph_combined_alpha0.5_passport-scale_15kb/'
    DESTDIR = '../AMSL_FaceMorphImageDataSet/IMG_CROP/'

    image_paths = glob(SOURCEDIR+'*.jpg')
    read_img(image_paths, DESTDIR)
    
    