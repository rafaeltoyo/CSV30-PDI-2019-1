#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


def img_rgb_to_red(img: np.ndarray) -> np.ndarray:
    """
    Extrair o canal Red (vermelho) de uma imagem RGB
    :param img: Imagem normalizada aberta pelo OpenCV e com três canais
    :return: Imagem normalizada com apenas um canal
    """
    if img.shape[2] != 3:
        raise Exception('Imagem passada não possui três canais de cores (RGB).')
    return img[:, :, 2]


def img_rgb_to_green(img: np.ndarray) -> np.ndarray:
    """
    Extrair o canal Green (verde) de uma imagem RGB
    :param img: Imagem normalizada aberta pelo OpenCV e com três canais
    :return: Imagem normalizada com apenas um canal
    """
    if img.shape[2] != 3:
        raise Exception('Imagem passada não possui três canais de cores (RGB).')
    return img[:, :, 1]


def img_rgb_to_blue(img: np.ndarray) -> np.ndarray:
    """
    Extrair o canal Blue (azul) de uma imagem RGB
    :param img: Imagem normalizada aberta pelo OpenCV e com três canais
    :return: Imagem normalizada com apenas um canal
    """
    if img.shape[2] != 3:
        raise Exception('Imagem passada não possui três canais de cores (RGB).')
    return img[:, :, 0]


def img_rgb_to_gs(img: np.ndarray) -> np.ndarray:
    """
    Converte uma imagem RGB para GrayScale
    :param img: Imagem normalizada aberta pelo OpenCV e com três canais
    :return: Imagem normalizada com apenas um canal
    """
    w, h, c = img.shape
    if c != 3:
        raise Exception('Imagem passada não possui três canais de cores (RGB).')
    out = np.copy(img)
    out[:, :] *= (0.114, 0.587, 0.299)
    out = out.sum(axis=(2,))
    # out = np.array([img[x][y][2] * 0.299 + img[x][y][1] * 0.587 + img[x][y][0] * 0.114 for x in range(w) for y in range(h)]).reshape((w, h))
    return out


def threshold(img: np.ndarray, th: float) -> np.ndarray:
    """
    Binarização da imagem normalizada de canal único baseado em um threshold
    :param img: Imagem normalizada aberta pelo OpenCV e com três canais
    :param th: Threshold
    :return: Imagem binarizada com apenas um canal
    """
    w, h, c = img.shape
    if c == 3:
        img = img_rgb_to_gs(img)
    return np.where(img > th, 1, 0)
