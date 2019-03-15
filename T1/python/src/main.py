#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import cv2

import utils.color as colorlib
import utils.image as imagelib


#img = cv2.imread('arroz.bmp')
img = cv2.imread('documento-3mp.bmp')
nimg = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
nimg = colorlib.threshold(nimg, 0.75)
nimg = np.where(nimg == 1, 0, 1)
cv2.imwrite('binarizado.bmp', cv2.normalize(nimg, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX))

cmps = imagelib.labeling(nimg, 4, 4, 9)
for cmp in cmps:
    cv2.rectangle(img, (cmp.yi, cmp.xi), (cmp.yf, cmp.xf), (0, 0, 255), 1)

cv2.imwrite('output.bmp', img)
print("componentes: " + str(len(cmps)))
