#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import cv2


def create_mask(img: np.ndarray, threshold: float) -> np.ndarray:

    w, h, c = img.shape
    if c == 3:
        gs_img = np.repeat((img * (0.114, 0.587, 0.299)).sum(axis=(2,)), 3, axis=1).reshape((w, h, c,))
        mask = np.where(gs_img < threshold, 0, img)
    else:
        mask = np.where(img < threshold, 0, img)

    return mask


def bloom1(img: np.ndarray,
           threshold: float,
           sigmas=[5, 10, 15, 20],
           mask_weight: float = 0.5,
           img_weight: float = 0.8):
    mask = create_mask(img, threshold)
    buffer = np.zeros(img.shape)

    for sigma in sigmas:
        shape = int(3 * sigma / 2) * 2 + 1
        buffer += cv2.GaussianBlur(mask, (shape, shape), sigma)

    bloom = (img * img_weight + buffer * mask_weight)
    bloom = np.where(bloom > 1, 1, bloom)
    return bloom


def bloom2(img: np.ndarray,
           threshold: float,
           window=[10, 15, 20, 25],
           mask_weight: float = 0.5,
           img_weight: float = 0.8):
    mask = create_mask(img, threshold)
    buffer = np.zeros(img.shape)

    for w in window:
        blur_layer = mask.copy()
        for i in range(0, 5):
            blur_layer = cv2.blur(blur_layer, (w, w))
        buffer += blur_layer

    bloom = (img * img_weight + buffer * mask_weight)
    bloom = np.where(bloom > 1, 1, bloom)
    return bloom
