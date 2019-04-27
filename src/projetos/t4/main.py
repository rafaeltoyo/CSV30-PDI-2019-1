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
    GAMMA = 10
    RESIZE = 2

    # ================================================================================================================ #
    #   Imagens
    # ---------------------------------------------------------------------------------------------------------------- #
    images_name = ['60.bmp', '82.bmp', '114.bmp', '150.bmp', '205.bmp']
    prj_path = PathBuilder()

    # ================================================================================================================ #
    #   Processar para cada imagem o algoritmo
    # ---------------------------------------------------------------------------------------------------------------- #
    for input_img in images_name:
        # ============================================================================================================ #
        # Abrir imagem
        # ------------------------------------------------------------------------------------------------------------ #

        img = cv.imread(prj_path.inputdir(input_img))
        w, h, c = img.shape
        img = cv.resize(img, (RESIZE * h, RESIZE * w))

        num_gohan = int(os.path.splitext(input_img)[0])
        PREFIX = 'p04-' + str(num_gohan) + '-'

        # ============================================================================================================ #
        # Tratar imagem
        # ------------------------------------------------------------------------------------------------------------ #

        # Correção Gama
        simg = np.uint8(((np.float32(img.copy()) / 255) ** GAMMA) * 255)

        # Nitidez
        sharpness = cv.medianBlur(simg, RESIZE * 70 + 1)
        cv.addWeighted(sharpness, -1.3, simg, 0.7, 3, simg)
        simg = cv.normalize(simg, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)

        cv.imwrite(PREFIX + 'tratada.bmp', simg)

        # ============================================================================================================ #
        # Binarizar
        # ------------------------------------------------------------------------------------------------------------ #

        gsimg = cv.cvtColor(simg, cv.COLOR_BGR2GRAY)
        bimg = cv.adaptiveThreshold(gsimg, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 251, -10)
        # cv.imwrite(PREFIX + 'binarizada.bmp', bimg)
        bimg = np.uint8(np.where(bimg == 0, 255, 0))

        # Limpar ruidos
        bimg = cv.morphologyEx(bimg, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5)), iterations=1)
        # cv.imwrite(PREFIX + 'limpada.bmp', bimg)

        # ============================================================================================================ #
        # Detectar bordas
        # ------------------------------------------------------------------------------------------------------------ #

        # edges = cv.cvtColor(simg, cv.COLOR_BGR2GRAY)
        # edges = cv.adaptiveThreshold(edges, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 5, -1)
        # cv.imwrite(PREFIX + 'edges.bmp', edges)

        # threshold_edges = 0
        # edges = np.where(bimg > 0, gsimg, threshold_edges)
        # edges = np.where(edges > threshold_edges, edges, threshold_edges)
        # cv.resize(edges, (h * 5, w * 5), edges)
        # edges = cv.normalize(edges, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
        # cv.imwrite(PREFIX + 'teste.bmp', edges)

        # edges = cv.Canny(simg, 0, 255)
        # edges = cv.Laplacian(gsimg, cv.CV_64F)
        # cv.imwrite(PREFIX + 'edges.bmp', edges)

        # ============================================================================================================ #
        # Tirar as bordas da imagem binarizada
        # ------------------------------------------------------------------------------------------------------------ #

        # bimg = np.where(edges > 0, 0, bimg)
        # bimg = cv.morphologyEx(bimg, cv.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1)
        # bimg = cv.morphologyEx(bimg, cv.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
        # bimg = cv.morphologyEx(bimg, cv.MORPH_ERODE, cv.getStructuringElement(cv.MORPH_CROSS, (3, 3)), iterations=3)
        cv.imwrite(PREFIX + 'final.bmp', gsimg)

        edges = cv.Canny(bimg, 0, 255)
        circles = cv.HoughCircles(edges, cv.HOUGH_GRADIENT, 1.2, 5,
                                  param1=100, param2=30,
                                  minRadius=10, maxRadius=30)
        print(circles)
        if circles is not None:
            circles = circles.tolist()
            for cir in circles:
                for x, y, r in cir:
                    x, y, r = int(x), int(y), int(r)
                    cv.circle(img, (x, y), r, (0, 255, 0), 4)

            # show the output image
            cv.imwrite("output.bmp", cv.resize(img, (500, 500)))

        # ============================================================================================================ #
        # Contar os arroz
        # ------------------------------------------------------------------------------------------------------------ #

        ret, labels = cv.connectedComponents(bimg)
        tam_arroz = (np.sum(np.where(bimg > 0, 1, 0)) / ret) * 0.9

        num_gohan_sepa = 0
        for i in range(1, ret + 1):
            label_size = np.sum(np.where(labels == i, 1, 0))
            num_gohan_sepa += int(label_size / tam_arroz)
            if (label_size % tam_arroz) / tam_arroz > 0.5:
                num_gohan_sepa += 1

        print("Arroz encontrados: " + str(ret))
        print("Arroz estimados: " + str(num_gohan_sepa))
        print("Arroz esperados: " + str(num_gohan))

    # ================================================================================================================ #
