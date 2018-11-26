# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import matplotlib.image as mpimg

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

img = mpimg.imread('/home/palakgoenka/Pictures/test.jpg')     
gray = rgb2gray(img) 
#img.save('/home/palakgoenka/Pictures/test/gray')