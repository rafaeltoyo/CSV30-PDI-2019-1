#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import cv2


def box_blur_1(img: np.ndarray, window_h: int, window_w: int):
    """

    :param img:
    :param window_h:
    :param window_w:
    :return:
    """
    w, h, clr = img.shape

    border_h = int(window_h / 2)
    border_w = int(window_w / 2)

    img_blur = np.copy(img)

    for yi in range(border_h, h - border_h):
        for xi in range(border_w, w - border_w):
            for ci in range(0, clr):
                img_blur[xi][yi][ci] = sum([img[xj][yj][ci] for xj in range(xi - border_w, xi + border_w + 1) for yj in range(yi - border_h, yi + border_h + 1)])
                img_blur[xi][yi][ci] /= (window_h * window_w)
    return img_blur


def box_blur_2(img: np.ndarray, window_h: int, window_w: int):
    """"""

    w, h, num_c = img.shape

    border_h = int(window_h / 2)
    border_w = int(window_w / 2)

    img_blur = np.zeros(img.shape)

    for yi in range(border_h, h - border_h):

        for idx_color in range(0, num_c):
            img_blur[border_w + 1][yi][idx_color] = sum(img[0:window_w][yi][1][idx_color])

        for xi in range(border_w + 2, w - border_w):
            for ci in range(0, num_c):
                img_blur[xi][yi][ci] = img_blur[xi - 1][yi][ci] - img[xi - 1 - border_w][yi][ci] + \
                                       img[xi + border_w][yi][ci]

    for yi in range(border_h, h - border_h):

        for idx_color in range(0, num_c):
            img_blur[border_w + 1][yi][idx_color] = sum(img[0:window_w][yi][1][idx_color])

        for xi in range(border_w + 2, w - border_w):
            for ci in range(0, num_c):
                img_blur[xi][yi][ci] = img_blur[xi - 1][yi][ci] - img[xi - 1 - border_w][yi][ci] + \
                                       img[xi + border_w][yi][ci]
