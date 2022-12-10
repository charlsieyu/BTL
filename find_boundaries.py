# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 22:52:36 2022

@author: Charlsie
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import copy

path = "./data/image/0-081.jpg"

# BGR, read blue channel
img = cv2.imread(path)[:, :, 0]
#img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
img_copy = copy.deepcopy(img)

blurred = cv2.bilateralFilter(img, 8, 300, 150)

# create a black canvas for boundaries
bdr = np.zeros((512,512), np.uint8)

ys = []
for i in range(blurred.shape[1]):
    y = np.argmax(blurred[30:-30, i]) + 30
    ys.append(y)
    img_copy[y-2:y+2, i] = 255 # the width of boundary is four rows 
    bdr[y-2:y+2, i] = 255

img_crop = img[ys[0]-50:ys[0]+100, :]
bdr_crop = bdr[ys[0]-50:ys[0]+100, :]

plt.imshow(img)
plt.show()
plt.imshow(blurred)
plt.show()
plt.imshow(img_copy)
plt.show()
plt.imshow(img_crop)
plt.show()
plt.imshow(bdr_crop)
plt.show()