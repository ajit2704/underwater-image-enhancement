#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 11:36:16 2017

@author: ajit
"""
import cv2
import numpy as np
def adjust_gamma(image, gamma=1.0):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
 
	# apply gamma correction using the lookup table
	return cv2.LUT(image, table)
def main():
    img=cv2.imread('2.png')
    
 
	# apply gamma correction and show the images
    gamma = 0.1
    adjusted = adjust_gamma(img, gamma=gamma)
    median = cv2.medianBlur(adjusted,5)
    cv2.imshow("Images", np.hstack([img, median]))
    cv2.waitKey(0)
if __name__ == '__main__':
    main()
