#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 18:37:02 2018

@author: aruroxx31 and palak
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import PIL.Image as Image
import matplotlib.image as mpimg
from skimage import io, transform
from skimage import data
from skimage import color
from skimage.util import view_as_blocks
from scipy import fftpack

%matplotlib inline

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def get_2D_dct(img):
    """ Get 2D Cosine Transform of Image
    """
    return fftpack.dct(fftpack.dct(img.T, norm='ortho').T, norm='ortho')


img = mpimg.imread('test.jpg')     
gray = rgb2gray(img)
gray = gray[0: 352, 0: 624]
plt.imshow(gray, cmap = plt.get_cmap('gray'))
gray = np.array(gray)


bw, bh = 8, 8 # block size
w = gray.shape[1] # width, height of image
h = gray.shape[0]

"""
if (sz==8) it tells us that it is an 8 bit grayscale image 
"""

# size of blocks
block_shape = (8, 8)
imagematrix = view_as_blocks(gray, block_shape)

dct = get_2D_dct(imagematrix)
"""
Till here We get the Fourier Cofficients of the whole Image
"""


"""
We do use a Standard Quantisation matrix for Every Fourier 8*8 Block  
"""
QUANTIZATION_MAT = np.array([[16,11,10,16,24,40,51,61],[12,12,14,19,26,58,60,55],[14,13,16,24,40,57,69,56 ],[14,17,22,29,51,87,80,62],[18,22,37,56,68,109,103,77],[24,35,55,64,81,104,113,92],[49,64,78,87,103,121,120,101],[72,92,95,98,112,100,103,99]])

quantized_block = np.divide(imagematrix[0, 0, 0:, 0: ], QUANTIZATION_MAT).astype(int)

quantizedmatrix = []

for i in range(0, 44):
    for j in range(0, 78):
        quantizedmatrix.append(np.divide(imagematrix[i, j, 0:, 0: ], QUANTIZATION_MAT).astype(int))












