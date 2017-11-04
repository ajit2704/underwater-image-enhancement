#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 09:46:07 2017

@author: ajit
"""
import numpy as np
import math

import cv2

from simplestColorBalance import simplest_cb
from saliencyDetection import saliencyDetection

import math
'''split rgb image to its channels'''
def split_rgb(image):
  red = None
  green = None
  blue = None
  (blue, green, red) = cv2.split(image)
  return red, green, blue
 
'''generate a 5x5 kernel'''

 
'''reduce image by 1/2'''
def ireduce(image):
  out = None
  h= np.array([1,4,6,4,1])/16
  filt= (h.T).dot(h)
  outimage = cv2.filter2D(image,cv2.CV_64F,filt)
  out = outimage[::2,::2]
  return out
 
'''expand image by factor of 2'''
def iexpand(image):
  out = None
  h= np.array([1,4,6,4,1])/16
  filt= (h.T).dot(h)
  outimage = np.zeros((image.shape[0]*2, image.shape[1]*2), dtype=np.float64)
  outimage[::2,::2]=image[:,:]
  out = cv2.filter2D(outimage,cv2.CV_64F,filt)
  return out
 
'''create a gaussain pyramid of a given image'''
def gauss_pyramid(image, levels):
  output = []
  output.append(image)
  tmp = image
  for i in range(0,levels):
    tmp = ireduce(tmp)
    output.append(tmp)
  return output
 
'''build a laplacian pyramid'''
def lapl_pyramid(gauss_pyr):
  output = []
  k = len(gauss_pyr)
  for i in range(0,k-1):
    gu = gauss_pyr[i]
    egu = iexpand(gauss_pyr[i+1])
    if egu.shape[0] > gu.shape[0]:
       egu = np.delete(egu,(-1),axis=0)
    if egu.shape[1] > gu.shape[1]:
      egu = np.delete(egu,(-1),axis=1)
    output.append(gu - egu)
  output.append(gauss_pyr.pop())
  return output
'''Blend the two laplacian pyramids by weighting them according to the mask.'''
def blend(lapl_pyr_white, lapl_pyr_black, gauss_pyr_mask):
  blended_pyr = []
  k= len(gauss_pyr_mask)
  for i in range(0,k):
   p1= gauss_pyr_mask[i]*lapl_pyr_white[i]
   p2=(1 - gauss_pyr_mask[i])*lapl_pyr_black[i]
   blended_pyr.append(p1 + p2)
  return blended_pyr
'''Reconstruct the image based on its laplacian pyramid.'''
def collapse(lapl_pyr):
  output = None
  
  output = np.zeros((lapl_pyr[0].shape[0],lapl_pyr[0].shape[1]), dtype=np.float64)
  for i in range(len(lapl_pyr)-1,0,-1):
    lap = iexpand(lapl_pyr[i])
    lapb = lapl_pyr[i-1]
    if lap.shape[0] > lapb.shape[0]:
      lap = np.delete(lap,(-1),axis=0)
    if lap.shape[1] > lapb.shape[1]:
      lap = np.delete(lap,(-1),axis=1)
    tmp = lap + lapb
    
    output = tmp
  return output

img= cv2.imread('2.png')

img1= simplest_cb(img,50)
  
lab1= cv2.cvtColor(img1, cv2.COLOR_BGR2LAB)
lab3= lab1.copy()
#-----Splitting the LAB image to different channels-------------------------
l, a, b = cv2.split(lab3)


#-----Applying CLAHE to L-channel-------------------------------------------
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
cl = clahe.apply(l)


#-----Merge the CLAHE enhanced L-channel with the a and b channel-----------
lab2 = cv2.merge((cl,a,b))

#-----Converting image from LAB Color model to RGB model--------------------
img2 = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

R1 = np.double(lab1[:,:,0])/255
WL1= cv2.Laplacian(R1,cv2.CV_64F)
h= np.array([1,4,6,4,1])/16
filt= (h.T).dot(h)
WC1= cv2.filter2D(R1,cv2.CV_64F,filt)
for i in np.where(WC1>(math.pi/2.75)):
    WC1[i]= math.pi/2.75
WC1= (R1-WC1)*(R1-WC1)
WS1= saliencyDetection(img1)
sigma= 0.25
aver= 0.5
WE1= np.exp(-(R1-aver)**2/(2*np.square(sigma)))

R2 = np.double(lab2[:,:,0])/255
WL2= cv2.Laplacian(R2,cv2.CV_64F)
h= np.array([1,4,6,4,1])/16
filt= (h.T).dot(h)
WC2= cv2.filter2D(R1,cv2.CV_64F,filt)
for i in np.where(WC2>(math.pi/2.75)):
    WC2[i]= math.pi/2.75
WC2= (R2-WC2)*(R2-WC2)
WS2= saliencyDetection(img1)
sigma= 0.25
aver= 0.5
WE2= np.exp(-(R2-aver)**2/(2*np.square(sigma)))
W1 = (WL1 + WC1 + WS1 + WE1)/(WL1 + WC1 + WS1 + WE1 + WL2 + WC2 + WS2 + WE2)
W2 = (WL2 + WC2 + WS2 + WE2)/(WL1 + WC1 + WS1 + WE1 + WL2 + WC2 + WS2 + WE2)
cv2.imshow("weight",np.hstack([W1,W2]))
levels=5
Weight1= gauss_pyramid(W1,5)
Weight2= gauss_pyramid(W2,5)



R1= None
G1= None
B1= None
R2= None
G2= None
B2= None
R_b= []
(R1,G1,B1)= split_rgb(img1)
(R2,G2,B2)= split_rgb(img2)

depth=5 
gauss_pyr_image1r = gauss_pyramid(R1, depth)
gauss_pyr_image1g = gauss_pyramid(G1, depth)
gauss_pyr_image1b = gauss_pyramid(B1, depth)
 
gauss_pyr_image2r = gauss_pyramid(R2, depth)
gauss_pyr_image2g = gauss_pyramid(G2, depth)
gauss_pyr_image2b = gauss_pyramid(B2, depth)
 
r1  = lapl_pyramid(gauss_pyr_image1r)
g1  = lapl_pyramid(gauss_pyr_image1g)
b1  = lapl_pyramid(gauss_pyr_image1b)
 
r2 = lapl_pyramid(gauss_pyr_image2r)
g2 = lapl_pyramid(gauss_pyr_image2g)
b2 = lapl_pyramid(gauss_pyr_image2b)
R_r = np.array(Weight1)* r1 + np.array(Weight2) * r2
R_g = np.array(Weight1)* g1 + np.array(Weight2) * g2
R_b = np.array(Weight1)* b1 + np.array(Weight2) * b2
R= collapse(R_r)
G= collapse(R_g)
B= collapse(R_b)
R[R < 0] = 0
R[R > 255] = 255
R = R.astype(np.uint8)
 
G[G < 0] = 0
G[G > 255] = 255
G = G.astype(np.uint8)
 
B[B < 0] = 0
B[B > 255] = 255
B = B.astype(np.uint8)
result = np.zeros(img.shape,dtype=img.dtype)
tmp = []
tmp.append(R)
tmp.append(G)
tmp.append(B)
result = cv2.merge(tmp,result)

cv2.imshow("",np.hstack([img,result]))
cv2.waitKey(0)
cv2.destroyAllWindows()
