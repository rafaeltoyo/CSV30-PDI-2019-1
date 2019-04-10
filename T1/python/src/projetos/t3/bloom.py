#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import cv2


def bloom1(img: np.ndarray, threshold: float):
    mask = np.where(img < threshold, 0, img)
    buffer = np.zeros(mask.shape)

    sigma = 5
    for i in range(0, 5):
        buffer += cv2.GaussianBlur(mask, (5 * i, 5 * i), sigma)
        sigma *= 2
    bloom = (img + buffer)

    bloom = np.where(bloom > 1, 1, bloom)

    return bloom


def bloom2(img: np.ndarray, threshold: float):
    mask = np.where(img < threshold, 0, img)

    buffer = np.zeros(mask.shape)

    for sigma in range(0, 25, 5):
        buffer += cv2.GaussianBlur(mask, (5, 5), sigmaX=sigma, sigmaY=sigma)
    bloom = (img + buffer)

    bloom = np.where(bloom > 1, 1, bloom)

    return bloom
