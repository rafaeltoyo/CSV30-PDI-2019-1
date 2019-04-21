#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

from utils.pathbuilder import PathBuilder


def projeto4():
    # ================================================================================================================ #
    #   Constantes
    # ---------------------------------------------------------------------------------------------------------------- #

    GAMMA = 7
    RESIZE = 2

    VER_HLS = False

    # ================================================================================================================ #
    #   Imagens
    # ---------------------------------------------------------------------------------------------------------------- #

    images_name = ['60.bmp', '82.bmp', '114.bmp', '150.bmp', '205.bmp']
    prj_path = PathBuilder()

    # ================================================================================================================ #
    #   Processar para cada imagem o algoritmo
    # ---------------------------------------------------------------------------------------------------------------- #

    for input_img in images_name:
        num_gohan = int(os.path.splitext(input_img)[0])
        imgbasename = 'p04-' + str(num_gohan) + '-'

        # ============================================================================================================ #
        # Abrir imagem
        img = cv.imread(prj_path.inputdir(input_img))
        w, h, c = img.shape
        img = cv.resize(img, (RESIZE * h, RESIZE * w))

        # ============================================================================================================ #
        # Correção Gama
        img = np.uint8(((np.float32(img) / 255) ** GAMMA) * 255)

        # ============================================================================================================ #
        # Nitidez
        sharpness = cv.medianBlur(img, RESIZE * 70 + 1)
        cv.addWeighted(sharpness, -1.3, img, 0.7, 3, img)
        img = cv.normalize(img, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
        #cv.imwrite(imgbasename + 'tratada.bmp', img)

        # ============================================================================================================ #
        # Binarizar
        gsimg = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        bimg = cv.adaptiveThreshold(gsimg, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 251, -10)
        #cv.imwrite(imgbasename + 'binarizada.bmp', bimg)
        bimg = np.uint8(np.where(bimg == 0, 255, 0))

        # ============================================================================================================ #
        # Limpar ruidos
        bimg = cv.morphologyEx(bimg, cv.MORPH_OPEN, np.ones((7, 7), np.uint8), iterations=1)
        #cv.imwrite(imgbasename + 'limpada.bmp', bimg)

        # ============================================================================================================ #
        # Detectar bordas
        threshold_edges = 50
        edges = np.where(bimg > 0, gsimg, threshold_edges)
        edges = np.where(edges > threshold_edges, edges, threshold_edges)
        cv.resize(edges, (h * 5, w * 5), edges)
        edges = cv.normalize(edges, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
        cv.imwrite(imgbasename + 'teste.bmp', edges)
        edges = cv.Canny(edges, 70, 180)
        #edges = cv.Laplacian(edges, cv.CV_64F)
        #edges = np.where(edges > 0, 255, 0)
        #edges = cv.morphologyEx(edges, cv.MORPH_DILATE, cv.getStructuringElement(cv.MORPH_CROSS, (3, 3)))
        #edges = cv.morphologyEx(edges, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_RECT, (3, 3)), iterations=1)
        #cv.resize(edges, (int(edges.shape[0]/5), int(edges.shape[1]/5)), edges)
        #cv.imwrite(imgbasename + 'edges.bmp', edges)

        # ============================================================================================================ #
        # Tirar as bordas da imagem binarizada
        #bimg = np.where(edges > 0, 0, bimg)
        #bimg = cv.morphologyEx(bimg, cv.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1)
        #bimg = cv.morphologyEx(bimg, cv.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
        #bimg = cv.morphologyEx(bimg, cv.MORPH_ERODE, cv.getStructuringElement(cv.MORPH_CROSS, (3, 3)), iterations=3)
        cv.imwrite(imgbasename + 'final.bmp', bimg)

        # ============================================================================================================ #
        # Saída
        ret, labels = cv.connectedComponents(bimg)
        print("Arroz encontrados: " + str(ret))
        print("Arroz esperados: " + str(num_gohan))

        continue

        # ============================================================================================================ #
        # RGB -> HLS e tirar saturação
        hls_img = cv.cvtColor(img, cv.COLOR_BGR2HLS)
        hls_img = np.int32(hls_img)
        hls_img[:][:] *= (1, 1, 0)
        hls_img = np.uint8(hls_img)
        img = cv.cvtColor(hls_img, cv.COLOR_HLS2BGR)
        img = cv.normalize(img, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
        cv.imwrite(imgbasename + 'tratada.bmp', img)

        # ============================================================================================================ #
        # Binarizar
        gs_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        binarized_img = cv.adaptiveThreshold(gs_img,
                                             255,
                                             cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             cv.THRESH_BINARY_INV,
                                             251,
                                             -10)

        # ============================================================================================================ #
        # Tentar pegar bordas
        border_img = cv.adaptiveThreshold(cv.resize(gs_img, (h * 5, w * 5)),
                                          255,
                                          cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv.THRESH_BINARY_INV,
                                          45,
                                          -1)
        border_img = np.uint8(np.where(border_img == 0, 255, 0))
        border_img = cv.morphologyEx(border_img,
                                     cv.MORPH_OPEN,
                                     cv.getStructuringElement(cv.MORPH_ELLIPSE, (11, 11)),
                                     iterations=1)
        border_img = cv.erode(border_img,
                              cv.getStructuringElement(cv.MORPH_ELLIPSE, (11, 11)),
                              iterations=1)
        cv.imwrite(imgbasename + 'border.bmp', border_img)

        # ============================================================================================================ #
        # Limpar ruidos
        binarized_img = cv.resize(binarized_img, (h * 5, w * 5))
        binarized_img = np.uint8(np.where(binarized_img == 0, 255, 0))
        noise_img = cv.morphologyEx(binarized_img,
                                    cv.MORPH_OPEN,
                                    np.ones((7, 7), np.uint8),
                                    iterations=1)
        cv.imwrite(imgbasename + 'binarizada.bmp', noise_img)

        # ============================================================================================================ #
        # Erosão
        erosion_img = noise_img
        erosion_img = cv.morphologyEx(erosion_img,
                                      cv.MORPH_OPEN,
                                      np.array([
                                          [0, 0, 1, 0, 0],
                                          [0, 0, 1, 0, 0],
                                          [1, 1, 1, 1, 1],
                                          [0, 0, 1, 0, 0],
                                          [0, 0, 1, 0, 0]
                                      ], np.uint8),
                                      iterations=7)
        erosion_img = cv.erode(erosion_img,
                               np.array([
                                   [0, 0, 1, 0, 0],
                                   [0, 0, 1, 0, 0],
                                   [1, 1, 1, 1, 1],
                                   [0, 0, 1, 0, 0],
                                   [0, 0, 1, 0, 0]
                               ], np.uint8),
                               iterations=3)
        cv.imwrite(imgbasename + 'final.bmp', erosion_img)

        # ============================================================================================================ #
        quero = np.where(border_img > 0, 0, erosion_img)

        cv.imwrite(imgbasename + 'zzzzzzzzz.bmp', quero)

        # erosion_img = cv.resize(erosion_img, (h, w))
        # teste = np.where(erosion_img != 0, gs_img, 0)
        # cv.imwrite(imgbasename + 'teste.bmp', teste)

        ret, labels = cv.connectedComponents(quero)

        # ============================================================================================================ #
        # Saída
        print("Arroz encontrados: " + str(ret))
        print("Arroz esperados: " + str(num_gohan))

    # ================================================================================================================ #
