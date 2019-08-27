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
from adjust_gamma import adjust_gamma
"""def split(arr, cond):
  return [arr[cond], arr[~cond]]
def ireduce(image):
  out = None
  h= np.array([1,4,6,4,1])/16
  filt= (h.T).dot(h)
  outimage = cv2.filter2D(np.array(image),cv2.CV_64F,filt)
  out = outimage[::2,::2]
  return out
def iexpand(image):
  out = None
  h= np.array([1,4,6,4,1])/16
  filt= (h.T).dot(h)
  outimage = np.zeros((image.shape[0]*2, image.shape[1]*2), dtype=np.float64)
  outimage[::2,::2]=image[:,:]
  out = cv2.filter2D(outimage,cv2.CV_64F,filt)
  return out
def gauss_pyramid(image, levels):
  output = []
  output.append(image)
  tmp = image
  for i in range(0,levels):
    tmp = ireduce(tmp)
    output.append(tmp)
  return output
def lapl_pyramid(gauss_pyr):
  output = []
  k = len(gauss_pyr)
  for i in range(0,k-1):
    gu = np.array(gauss_pyr[i])
    egu = np.array(iexpand(gauss_pyr[i+1]))
    if egu.shape[0] > gu.shape[0]:
       egu = np.delete(egu,(-1),axis=0)
    
    output.append(gu - egu[:,0])
  output.append(gauss_pyr.pop())
  return output
def collapse(lapl_pyr):
  output = None
  
  output = np.zeros((lapl_pyr[0].shape[0]), dtype=np.float64)
  for i in range(len(lapl_pyr)-1,0,-1):
    lap = iexpand(lapl_pyr[i])
    lapb = lapl_pyr[i-1]
    if lap.shape[0] > lapb.shape[0]:
      lap = np.delete(lap,(-1),axis=0)
    
      
    tmp = lap.dot(np.ones(np.transpose(lap).shape))+ lapb
    
    output = tmp
  return output"""
img= cv2.imread("/root/.config/spyder-py3/imenh/2.png")
img1=img.copy()
img2=img.copy()
b,g,r=cv2.split(img2)
b2,g2,r2= cv2.split(img)
b1,g1,r1=cv2.split(img1)
imdb = ((b2.max()-b2.min())/2) + b2.min()


alpha=0.9
b[b<imdb]=imdb
for index,value in np.ndenumerate( b ):
   new_value=((255* (value-imdb))/ ((b2.max()-b2.min())/(alpha**2)))
   b[index]= new_value
print(b.shape)
#print(b1.shape)
imdg = ((g2.max()-g2.min())/2) + g2.min()
g[g<imdg]=imdg
for index,value in np.ndenumerate(g):
    new_value=((255* (value-imdg))/((g2.max()-g2.min())*(alpha**2)))
    g[index]= new_value
print(g.shape)

imdr = ((r2.max()-r2.min())/2) + r2.min()
r[r<imdr]=imdr
for index,value in np.ndenumerate(r):
    new_value=((255* (value-imdr))/ ((r2.max()-r2.min())/(alpha**2)))
    r[index]= new_value
print(r.shape)
b1[b1>imdb]=imdb
for index,value in np.ndenumerate( b1 ):
    new_value=((255* (value-b2.min()))/ ((b2.max()-b2.min())*(alpha**2)))
    b1[index]= new_value
print(b1.shape)
#print(b1.shape)

g1[g1>imdg]=imdg
for index,value in np.ndenumerate(g1):
   new_value=((255* (value-g2.min()))/((g2.max()-g2.min())*(alpha**2)))
   g1[index]= new_value
print(g1.shape)

imdr = ((r2.max()-r2.min())/2) + r.min()
r1[r1>imdr]=imdr
for index,value in np.ndenumerate(r1):
    new_value=((255* (value-r2.min()))/ ((r2.max()-r2.min())*(alpha**2)))
    r1[index]=new_value
print(r.shape)
res=cv2.merge((b,g,r))
res1=cv2.merge((b1,g1,r1))
res2= cv2.addWeighted(res,.5,res1,.5,0)

cv2.imshow("",res)
cv2.imshow("1",res1)
cv2.imshow("2",img2)
cv2.imshow("avg",res2)
fin= adjust_gamma(res2,1.2)
cv2.imshow("final",fin)


cv2.waitKey(0)
cv2.destroyAllWindows()
"""
alpha=1.0
rb1= np.zeros(b1.shape,dtype=b1.dtype)
for index,value in np.ndenumerate( b1 ):
    new_value=((255* (value-imdb)/ (b.max()-b.min()))/(alpha**2))*np.exp(-np.square(255* (value-imdb)/ (b.max()-b.min()))/(2*(alpha**2)))
    rb1[index]= new_value
#rb1 = [((255* (b1[i]-imdb)/ (b.max()-b.min()))/(alpha**2))*np.exp(-np.square(255* (b1[i]-imdb)/ (b.max()-b.min()))/(2*(alpha**2)))for i in b1]
#rb2 = [((255* (b2[i]-b.min())/ (b.max()-b.min()))/(alpha**2))*np.exp(-np.square(255* (b2[i]-b.min())/ (b.max()-b.min()))/(2*(alpha**2)))for i in b2]
imdg = ((g.max()-g.min())/2) + g.min()


g1,g2= split(g,g>imdg)
rg1 = [((255* (g1[i]-imdg)/ (g.max()-g.min()))/(alpha**2))*np.exp(-np.square(255* (g1[i]-imdg)/ (g.max()-g.min()))/(2*(alpha**2)))for i in g1]
#rg2 = [((255* (g2[i]-g.min())/ (g.max()-g.min()))/(alpha**2))*np.exp(-np.square(255* (g2[i]-g.min())/ (g.max()-g.min()))/(2*(alpha**2)))for i in g2]
imdr = ((r.max()-r.min())/2) + r.min()
print(imdb)


r1,r2= split(r,r>imdr)

print(r1.shape)
print(r2.shape)
alpha=1.0


rr1 = [((255* (r1[i]-imdr)/ (r.max()-r.min()))/(alpha**2))*np.exp(-np.square(255* (r1[i]-imdr)/ (r.max()-r.min()))/(2*(alpha**2)))for i in r1]
#rr2 = [((255* (r2[i]-r.min())/ (r.max()-r.min()))/(alpha**2))*np.exp(-np.square(255* (r2[i]-r.min())/ (r.max()-r.min()))/(2*(alpha**2)))for i in r2]

print(imdr)
gauss_pyr_image1r = gauss_pyramid(rr1, depth)
gauss_pyr_image1g = gauss_pyramid(rg1, depth)
gauss_pyr_image1b = gauss_pyramid(rb1, depth)
""" 
"""
gauss_pyr_image2r = gauss_pyramid(R2, depth)
gauss_pyr_image2g = gauss_pyramid(G2, depth)
gauss_pyr_image2b = gauss_pyramid(B2, depth)
""" 
"""
r1  = lapl_pyramid(gauss_pyr_image1r)
g1  = lapl_pyramid(gauss_pyr_image1g)
b1  = lapl_pyramid(gauss_pyr_image1b)
""" 
"""
r2 = lapl_pyramid(gauss_pyr_image2r)
g2 = lapl_pyramid(gauss_pyr_image2g)
b2 = lapl_pyramid(gauss_pyr_image2b)
#* exp(-np.squa(255* (x1[i]-x.min())/ (x.max()-x.min()))/2) for i in x1]
"""
"""
R= collapse(r1)
G= collapse(g1)
B= collapse(b1)
"""
"""
rr1[rr1 < 0] = 0
rr1[rr1 > 255] = 255
rr1 = rr1.astype(np.uint8)
 
rg1[rg1 < 0] = 0
rg1[rg1 > 255] = 255
rg1 = rg1.astype(np.uint8)
 
rb1[rb1 < 0] = 0
rb1[rb1 > 255] = 255
rb1 = rb1.astype(np.uint8)
"""
"""
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
tmp.append((np.array(rb1,dtype=b.dtype)))
tmp.append((np.array(rg1,dtype=g.dtype)))
tmp.append((np.array(rr1,dtype=r.dtype)))
result = cv2.merge((tmp),result)
cv2.imshow("",result)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""


