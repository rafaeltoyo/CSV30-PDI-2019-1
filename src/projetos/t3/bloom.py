#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import cv2


def create_mask(img: np.ndarray, threshold: float) -> np.ndarray:
    """
    Criar mascara para o bloom
    :param img:
    :param threshold:
    :return:
    """

    w, h, c = img.shape
    if c == 3:
        gs_img = np.repeat((img * (0.114, 0.587, 0.299)).sum(axis=(2,)), 3, axis=1).reshape((w, h, c,))
        mask = np.where(gs_img < threshold, 0, img)
    else:
        mask = np.where(img < threshold, 0, img)

    return mask


def bloom1(img: np.ndarray,
           threshold: float,
           sigmas=None,
           mask_weight: float = 0.5,
           img_weight: float = 0.8) -> np.ndarray:
    """
    Bloom com gaussian blur
    :param img:
    :param threshold:
    :param sigmas:
    :param mask_weight:
    :param img_weight:
    :return:
    """

    if sigmas is None:
        return img

    mask = create_mask(img, threshold)
    buffer = np.zeros(img.shape)

    for sigma in sigmas:
        shape = (3 * sigma) | 1
        buffer += cv2.GaussianBlur(mask, (shape, shape), sigma)

    bloom = (img * img_weight + buffer * mask_weight)
    bloom = np.where(bloom > 1, 1, bloom)
    return bloom


def bloom2(img: np.ndarray,
           threshold: float,
           window=None,
           mask_weight: float = 0.5,
           img_weight: float = 0.8,
           repeat_boxblur: int = 5) -> np.ndarray:
    """
    Bloom com box blur
    :param repeat_boxblur:
    :param img:
    :param threshold:
    :param window:
    :param mask_weight:
    :param img_weight:
    :return:
    """

    if window is None:
        return img

    mask = create_mask(img, threshold)
    buffer = np.zeros(img.shape)

    for w in window:
        blur_layer = mask.copy()
        for i in range(0, repeat_boxblur):
            w = int(w | 1)
            blur_layer = cv2.blur(blur_layer, (w, w))
        buffer += blur_layer

    bloom = (img * img_weight + buffer * mask_weight)
    bloom = np.where(bloom > 1, 1, bloom)
    return bloom
