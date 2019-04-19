#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import cv2 as cv

from utils.pathbuilder import PathBuilder


def projeto4():
    # ================================================================================================================ #
    #   Constantes
    # ---------------------------------------------------------------------------------------------------------------- #

    THRESHOLD = 0.5
    GAMMA = 5
    BOX_SIZE_FACTOR = 500

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

        # Abrir imagem
        img = cv.imread(prj_path.inputdir(input_img))
        w, h, c = img.shape

        # Correção Gamma
        gamma_img = np.uint8(((np.float32(img) / 255) ** GAMMA) * 255)
        cv.imwrite(imgbasename + '1gamma.bmp', gamma_img)

        # Tirar iluminação
        #median_img = cv.medianBlur(gamma_img, 71)
        #median_img = cv.blur(gamma_img, (81, 81))
        #cv.imwrite(imgbasename + '2median.bmp', median_img)

        # Binarização adaptativa
        gs_img = cv.cvtColor(gamma_img, cv.COLOR_BGR2GRAY)
        box_size = max(int((min(w, h) / BOX_SIZE_FACTOR) / 2) * 2 + 1, 3)
        binarized_img = cv.adaptiveThreshold(gs_img,
                                             255,
                                             cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             cv.THRESH_BINARY_INV,
                                             155,
                                             -1)
        cv.imwrite(imgbasename + '3binarizado.bmp', binarized_img)

        # Erosao
        binarized_img = np.uint8(np.where(binarized_img == 0, 255, 0))
        noise_img = cv.morphologyEx(binarized_img,
                                    cv.MORPH_OPEN,
                                    np.ones((9, 9), np.uint8),
                                    iterations=1)
        cv.imwrite(imgbasename + '4limpar.bmp', noise_img)

        erosion_img = noise_img

        erosion_img = cv.morphologyEx(erosion_img,
                                      cv.MORPH_OPEN,
                                      np.array([
                                          [0, 0, 1, 0, 0],
                                          [0, 1, 1, 1, 0],
                                          [1, 1, 1, 1, 1],
                                          [0, 1, 1, 1, 0],
                                          [0, 0, 1, 0, 0]
                                      ], np.uint8),
                                      iterations=1)
        #erosion_img = cv.erode(erosion_img,
        #                       np.ones((7, 7), np.uint8),
        #                       iterations=1)
        cv.imwrite(imgbasename + '5final.bmp', erosion_img)

        ret, labels = cv.connectedComponents(erosion_img)

        # Saída
        print("Arroz encontrados: " + str(ret))
        print("Arroz esperados: " + str(num_gohan))

    # ================================================================================================================ #
