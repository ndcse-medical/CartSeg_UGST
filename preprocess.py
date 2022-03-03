from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import math
import numpy as np
import os
import random
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import yaml
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import skimage
from PIL import Image 
from skimage.io import imread
from skimage.io import imsave
from skimage.transform import rotate
from skimage.transform import rescale

cnf = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)
stack_name = cnf["PREPROCESS"]["stack_name"]

img_source = cnf["PREPROCESS"]["img_source"]
img_destination = cnf["PREPROCESS"]["img_destination"]

mask_source = cnf["PREPROCESS"]["mask_source"]
mask_destination = cnf["PREPROCESS"]["mask_destination"]


if not os.path.isdir(os.path.join(img_destination)):
        os.makedirs(img_destination)

print("-"*35)
print("Converting tiff images to png . . .")
print("-"*35)  
imgs = []
for file in os.listdir(img_source):
    if file.endswith(".tif"):
        imgs.append(file)
imgs.sort()
print(f'Saving {img_source} to {img_destination}')
print(f'total # images {len(imgs)}')
print("-"*35)  

for i in imgs:
    new_name = stack_name+'_'+i[-8:-4]+'.png'
    print(new_name)
    tempI = imread(os.path.join(img_source,i))
    imsave(os.path.join(img_destination, new_name), ((tempI/65535.0)*255).astype('uint8'))

if len(mask_source) != 0:
    print("-"*35)
    print("Converting tiff annotations to png masks . . .")
    print("-"*35)

    imgs = []
    for file in os.listdir(mask_source):
        if file.endswith(".tif"):
            imgs.append(file)      
    imgs.sort()

    if not os.path.isdir(mask_destination):
            os.makedirs(mask_destination)

    print(f'Saving {mask_source} to {mask_destination}')
    print(f'total # images {len(imgs)}')
    print("-"*35)  

    for i in imgs:
        new_name = stack_name+'_'+i[-8:-4]+'.png'
        print(new_name)
        tempI = imread(os.path.join(mask_source,i),as_gray=True)
        tempI = (tempI*255).astype('uint8')
        mask = np.zeros(tempI.shape)
        maxv = tempI.max()
        minv = tempI.min()
        if maxv == minv:
            mask[tempI==maxv] = 0
        else:
            mask[tempI==maxv] =0
            mask[tempI==minv] =200
        pil_mask = Image.fromarray(mask).convert('L')
        pil_mask.save(os.path.join(mask_destination, new_name))
    print("--"*35)
    print(f'Done converting {stack_name} to PNG Images')
    print(f'check folder: {img_destination} for images')
    print(f'check folder: {mask_destination} for masks')
    print("--"*35)

else:
    print("--"*35)
    print(f'Done converting {stack_name} to PNG Images')
    print(f'check folder: {img_destination} for images')
    print("--"*35)