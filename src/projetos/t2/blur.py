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
                img_blur[xi][yi][ci] = sum([img[xj][yj][ci] for xj in range(xi - border_w, xi + border_w + 1) for yj in
                                            range(yi - border_h, yi + border_h + 1)])
                img_blur[xi][yi][ci] /= (window_h * window_w)
    return img_blur


def box_blur_2(img: np.ndarray, window_h: int, window_w: int):
    """"""

    w, h, num_c = img.shape

    border_w = int(window_w / 2)
    border_h = int(window_h / 2)
    num_pixel = window_w * window_h

    buffer = np.copy(img)
    out = np.copy(img)

    # Soma por linha
    for yi in range(0, h):

        for idx_color in range(0, num_c):
            buffer[border_w][yi][idx_color] = sum([img[xj][yi][idx_color] for xj in range(0, window_w)])

        for xi in range(border_w + 1, w - border_w):
            for ci in range(0, num_c):
                buffer[xi][yi][ci] = (buffer[xi - 1][yi][ci] -
                                      img[xi - border_w - 1][yi][ci] +
                                      img[xi + border_w][yi][ci])

    # Soma por coluna
    for xi in range(border_w, w - border_w):

        for idx_color in range(0, num_c):
            out[xi][border_h][idx_color] = sum([buffer[xi][yj][idx_color] for yj in range(0, window_h)])

        for yi in range(border_h + 1, h - border_h):
            for ci in range(0, num_c):
                out[xi][yi][ci] = (out[xi][yi - 1][ci] -
                                   buffer[xi][yi - border_h - 1][ci] +
                                   buffer[xi][yi + border_h][ci])

    # Dividir as somas (média)
    for yi in range(border_h, h - border_h):
        for xi in range(border_w, w - border_w):
            for ci in range(0, num_c):
                out[xi][yi][ci] /= num_pixel

    return out


def box_blur_3(img: np.ndarray, window_h: int, window_w: int):
    """"""

    w, h, num_c = img.shape

    border_w = int(window_w / 2)
    border_h = int(window_h / 2)

    # Imagem integral
    buffer = np.zeros(img.shape).astype(np.float64)
    out = np.zeros(img.shape)

    for yi in range(0, h):
        for ci in range(0, num_c):
            buffer[0][yi][ci] = img[0][yi][ci]
            for xi in range(1, w):
                buffer[xi][yi][ci] = img[xi][yi][ci] + buffer[xi - 1][yi][ci]

    for yi in range(1, h):
        for xi in range(0, w):
            for ci in range(0, num_c):
                buffer[xi][yi][ci] = buffer[xi][yi][ci] + buffer[xi][yi - 1][ci]

    # Ajuste (blur)
    for y in range(0, h):
        for x in range(0, w):

            xi = max(0, x - border_w)
            yi = max(0, y - border_h)
            xf = min(w - 1, x + border_w)
            yf = min(h - 1, y + border_h)

            window_wi = xf - xi + 1
            window_hi = yf - yi + 1
            num_pixel = window_wi * window_hi

            for ci in range(0, num_c):
                out[x][y][ci] = buffer[xf][yf][ci]
                if xi > 0:
                    out[x][y][ci] -= buffer[xi - 1][yf][ci]
                if yi > 0:
                    out[x][y][ci] -= buffer[xf][yi - 1][ci]
                if xi > 0 and yi > 0:
                    out[x][y][ci] += buffer[xi - 1][yi - 1][ci]
                out[x][y][ci] /= num_pixel

    return out
