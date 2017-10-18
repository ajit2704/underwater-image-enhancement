#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 13:38:25 2017

@author: ajit
"""

import sys
import os
import numpy as np
import cv2 
import pywt
import copy
def fuseCoeff(cooef1, cooef2, method):

    if (method == 'mean'):
        cooef = (cooef1 + cooef2) / 2
    elif (method == 'min'):
        cooef = np.minimum(cooef1,cooef2)
    elif (method == 'max'):
        cooef = np.maximum(cooef1,cooef2)
    else:
        cooef = []

    return cooef
def main():
    img1= cv2.imread("/home/ajit/Desktop/images.jpg")
    img2= copy.copy(img1)
    b1, g1, r1 =cv2.split(img2)
    
    hsv = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
    h, s, v =cv2.split(hsv)
    a=h.max()
    b=h.min()
    h-= h.min()
    h=h/(a-b)
   
    h*= 179
    
    
    s-= s.min()
    s= s/(s.max()-s.min())
    s*=255
    v-= v.min()
    v= v/(v.max()-v.min())
    v*=255
    hsv=cv2.merge([h,s,v])
    hsv1=hsv.astype(np.float32)
    rgb=cv2.cvtColor(hsv1, cv2.COLOR_HSV2RGB)
    b, g, r =cv2.split(rgb)
    r-= r.min()
    r*= 255/(r.max()-r.min())
    g-= g.min()
    g*= 255/(g.max()-g.min())
    b-= b.min()
    b*= 255/(b.max()-b.min())
    cl1=cv2.merge([b,g,r])
    clahe = cv2.createCLAHE()
    b2 = clahe.apply(b1)
    g2 = clahe.apply(g1)
    r2 = clahe.apply(r1)
    cl2 =cv2.merge([b2,g2,r2])
    
    cooef1 = pywt.wavedec2(cl1[:,:], 'db1')
    cooef2 = pywt.wavedec2(cl2[:,:], 'db1')
    FUSION_METHOD = 'max'
    fusedCooef = []
    for i in range(len(cooef1)-1):

   
      if(i == 0):

        fusedCooef.append(fuseCoeff(cooef1[0],cooef2[0],FUSION_METHOD))

      else:

        
        c1 = fuseCoeff(cooef1[i][0],cooef2[i][0],FUSION_METHOD)
        c2 = fuseCoeff(cooef1[i][1], cooef2[i][1], FUSION_METHOD)
        c3 = fuseCoeff(cooef1[i][2], cooef2[i][2], FUSION_METHOD)
        

        fusedCooef.append((c1,c2,c3))
    fusedImage = pywt.waverec2(fusedCooef, 'db1')
    fusedImage = np.multiply(np.divide(fusedImage - np.min(fusedImage),(np.max(fusedImage) - np.min(fusedImage))),255)
    fusedImage = fusedImage.astype(np.uint8)
    cv2.imshow("win",fusedImage)
if __name__ == "__main__":
    main()
    
    
