#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 10:14:11 2017

@author: ajit
"""

import sys
import os
import numpy as np
import cv2 


def main():
    img= cv2.imread("/home/ajit/Desktop/images.jpg")
    x,y,z=cv2.split(img)
    imdr = ((x.max()-x.min())/2) + x.min()
    imdg = ((y.max()-y.min())/2) + y.min()
    imdb = ((z.max()-z.min())/2) + z.min()
    x2=[]
    x1=[]
    y1=[]
    y2=[]
    z1=[]
    z2=[]
    for i in np.where(x>imdr):
        x2.append((255*(i-imdr)/((x.max()-imdr)))*np.exp((-np.square(255*(i-imdr)/(.001*(x.max()-imdr))))/2))
    for j in np.where(x<imdr): 
        x1.append((255*(x[j]-x.min())/((imdr-x.min())))*np.exp((-np.square(255*(x[j]-x.min())/((imdr-x.min()))))/2))
    for i in np.where(y>imdg):
        y2.append((255*(y[i]-imdg)/((y.max()-imdg)))*np.exp((-np.square(255*(y[i]-imdg)/(.001*(y.max()-imdg))))/2))
    for j in np.where(y<imdg): 
        y1.append((255*(y[j]-y.min())/((imdg-y.min())))*np.exp((-np.square(255*(y[j]-y.min())/(.001*(imdg-y.min()))))/2))
    for j in np.where(z<imdb): 
        z1[j]= (255*(z1[j]-z.min())/((imdb-z.min())))*np.exp((-np.square(255*(z1[j]-z.min())/(.001*(imdb-z.min()))))/2)
    for j in np.where(z2>imdb):
        z2.append((255*(z[i]-imdb)/((z.max()-imdb)))*np.exp((-np.square(255*(z[i]-imdb)/(.001*(z.max()-imdb))))/2))
    hc=cv2.merge((z1,y1,x1))
    lc=cv2.merge((z2,y2,x2))
    zstk=np.concatenate((hc,lc),axis=2)
    hsv = cv2.cvtColor(zstk, cv2.COLOR_BGR2HSV)
    equ = cv2.equalizeHist(hsv)
    cv2.imshow('image',equ)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
if __name__ == "__main__":
    main()


    

    
