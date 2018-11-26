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
from skimage.measure import block_reduce
import math

%matplotlib inline

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def get_2D_dct(img):
    """ Get 2D Cosine Transform of Image
    """
    return fftpack.dct(fftpack.dct(img.T, norm='ortho').T, norm='ortho')


def get_2d_idct(coefficients):
    """ Get 2D Inverse Cosine Transform of Image
    """
    return fftpack.idct(fftpack.idct(coefficients.T, norm='ortho').T, norm='ortho')


def rgb2ycbcr(im):
    xform = np.array([[.299, .587, .114], [-.1687, -.3313, .5], [.5, -.4187, -.0813]])
    ycbcr = im.dot(xform.T)
    ycbcr[:,:,[1,2]] += 128
    return np.uint8(ycbcr)


def ycbcr2rgb(im):
    xform = np.array([[1, 0, 1.402], [1, -0.34414, -.71414], [1, 1.772, 0]])
    rgb = im.astype(np.float)
    rgb[:,:,[1,2]] -= 128
    rgb = rgb.dot(xform.T)
    np.putmask(rgb, rgb > 255, 255)
    np.putmask(rgb, rgb < 0, 0)
    return np.uint8(rgb)

img = mpimg.imread('test.jpg')     

"""
Converted it into GrayScale Image
"""
gray = rgb2gray(img)
gray = gray[0: 352, 0: 624]
plt.imshow(gray, cmap = plt.get_cmap('gray'))
gray = np.array(gray)


bw, bh = 8, 8 # block size
w = gray.shape[1] # width, height of image
h = gray.shape[0]

"""
Converted it into YCbCr Image
"""
ycbcr = rgb2ycbcr(img)
ycbcr = ycbcr[0: 352, 0: 624]
plt.imshow(ycbcr)
ycbcr = np.array(ycbcr)

"""
Converting 3 Dimensional Ycbcr array into three 2D arrays
"""

yplane = []
cbplane = []
crplane = []

yplane.append(ycbcr[0: , 0:, 0])
cbplane.append(ycbcr[0: , 0:, 1])
crplane.append(ycbcr[0: , 0:, 2])

yplane = np.asarray(yplane)
cbplane = np.asarray(cbplane)
crplane = np.asarray(crplane)

yplane = yplane.reshape(352, 624)
cbplane = cbplane.reshape(352, 624)
crplane = crplane.reshape(352, 624)

"""
SubSampling the Cb Cr planes only by a factor of 2
"""
cbplanesampled = block_reduce(cbplane, block_size=(2, 2), func=np.mean)
crplanesampled = block_reduce(crplane, block_size=(2, 2), func=np.mean)


"""
if (sz==8) it tells us that it is an 8 bit grayscale image 
"""

# size of blocks
block_shape = (8, 8)

"""
Fourier for GrayScale Image
"""
gray = get_2D_dct(gray)
dct = view_as_blocks(gray, block_shape)

"""
Fourier for RGB Image
"""
ymatrix = get_2D_dct(yplane)
dctymatrix = view_as_blocks(ymatrix, block_shape)

cbmatrix = get_2D_dct(cbplanesampled)
dctcbmatrix = view_as_blocks(cbmatrix, block_shape)

crmatrix = get_2D_dct(crplanesampled)
dctcrmatrix = view_as_blocks(crmatrix, block_shape)

"""
Till here We get the Fourier Cofficients of the whole Image
"""

"""
We do use a Standard Quantisation matrix for Every Fourier 8*8 Block  
"""
QUANTIZATION_MAT = np.array([[16,11,10,16,24,40,51,61],[12,12,14,19,26,58,60,55],[14,13,16,24,40,57,69,56 ],[14,17,22,29,51,87,80,62],[18,22,37,56,68,109,103,77],[24,35,55,64,81,104,113,92],[49,64,78,87,103,121,120,101],[72,92,95,98,112,100,103,99]])

quantized_block = np.round(np.divide(dct[0, 0, 0:, 0: ], QUANTIZATION_MAT))

quantized_block_y = np.divide(dctymatrix[0, 0, 0:, 0: ], QUANTIZATION_MAT).astype(int)

quantizedgraymatrix = []
quantizedymatrix = []
quantizedcbmatrix = []
quantizedcrmatrix = []

for i in range(0, 44):
    for j in range(0, 78):
        quantizedgraymatrix.append(np.divide(dct[i, j, 0:, 0: ], QUANTIZATION_MAT).astype(int))


for i in range(0, 44):
    for j in range(0, 78):
        quantizedymatrix.append(np.divide(dctymatrix[i, j, 0:, 0: ], QUANTIZATION_MAT).astype(int))


for i in range(0, 22):
    for j in range(0, 39):
        quantizedcbmatrix.append(np.divide(dctcbmatrix[i, j, 0:, 0: ], QUANTIZATION_MAT).astype(int))


for i in range(0, 22):
    for j in range(0, 39):
        quantizedcrmatrix.append(np.divide(dctcrmatrix[i, j, 0:, 0: ], QUANTIZATION_MAT).astype(int))


quantizedgraymatrix = np.asarray(quantizedgraymatrix)
quantizedgraymatrix = quantizedgraymatrix.reshape(44, 78, 8, 8)

quantizedymatrix = np.asarray(quantizedymatrix)
quantizedymatrix = quantizedymatrix.reshape(44, 78, 8, 8)

quantizedcbmatrix = np.asarray(quantizedcbmatrix)
quantizedcbmatrix = quantizedcbmatrix.reshape(22, 39, 8, 8)

quantizedcrmatrix = np.asarray(quantizedcrmatrix)
quantizedcrmatrix = quantizedcrmatrix.reshape(22, 39, 8, 8)


"""
Dequantisation Done hereS
"""

dequantizedgraymatrix = []
dequantizedymatrix = []
dequantizedcbmatrix = []
dequantizedcrmatrix = []


for i in range(0, 44):
    for j in range(0, 78):
        dequantizedgraymatrix.append(np.multiply(quantizedgraymatrix[i, j, 0:, 0: ], QUANTIZATION_MAT))

for i in range(0, 44):
    for j in range(0, 78):
        dequantizedymatrix.append(np.multiply(quantizedymatrix[i, j, 0:, 0: ], QUANTIZATION_MAT))

for i in range(0, 22):
    for j in range(0, 39):
        dequantizedcbmatrix.append(np.multiply(quantizedcbmatrix[i, j, 0:, 0: ], QUANTIZATION_MAT))

for i in range(0, 22):
    for j in range(0, 39):
        dequantizedcrmatrix.append(np.multiply(quantizedcrmatrix[i, j, 0:, 0: ], QUANTIZATION_MAT))


dequantizedgraymatrix = np.asarray(dequantizedgraymatrix)
dequantizedgraymatrix = dequantizedgraymatrix.reshape(44, 78, 8, 8)

dequantizedymatrix = np.asarray(dequantizedymatrix)
dequantizedymatrix = dequantizedymatrix.reshape(44, 78, 8, 8)

dequantizedcbmatrix = np.asarray(dequantizedcbmatrix)
dequantizedcbmatrix = dequantizedcbmatrix.reshape(22, 39, 8, 8)

dequantizedcrmatrix = np.asarray(dequantizedcrmatrix)
dequantizedcrmatrix = dequantizedcrmatrix.reshape(22, 39, 8, 8)

grayreshaped = []
yreshaped = []

yinitialreshaped = dctymatrix.transpose(0, 2, 1, 3).reshape(352, 624)


grayreshaped = dequantizedgraymatrix.transpose(0, 2, 1, 3).reshape(352, 624)
yreshaped = dequantizedymatrix.transpose(0, 2, 1, 3).reshape(352, 624)

grayfromdct = get_2d_idct(grayreshaped);
yfromdct = get_2d_idct(yreshaped);

plt.imshow(grayfromdct)
plt.imshow(yfromdct)


trunction_error = np.absolute(np.array(yinitialreshaped) - np.array(yreshaped))

for i in range(0, 352):
    for j in range(0 ,624):
        if trunction_error[i, j] > 2:
            trunction_error[i, j] = 1
        else:
            trunction_error[i, j] = 0


















