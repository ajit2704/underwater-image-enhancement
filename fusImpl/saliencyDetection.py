#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 18:10:21 2017

@author: ajit
"""
import cv2
import numpy as np
from skimage import color
def matlab_style_gauss2D(shape=(3,3),sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h
def saliencyDetection(img):
    
    kernel= np.array(matlab_style_gauss2D((3,3),1))
    gfrgb= cv2.filter2D(img,-1,kernel,cv2.BORDER_WRAP)
    lab= color.rgb2lab(gfrgb)
    l = np.double(lab[:,:,0])
    a = np.double(lab[:,:,1])
    b = np.double(lab[:,:,2])
    lm = np.mean(np.mean(l))
    am = np.mean(np.mean(a))
    bm = np.mean(np.mean(b))
    sm = np.square(l-lm)+ np.square(a-am) + np.square((b-bm))
    return sm
def main():
    im=cv2.imread("org-1.jpg")
    sm=saliencyDetection(im)
    print ("nm",sm)
if __name__ == '__main__':
    main()
