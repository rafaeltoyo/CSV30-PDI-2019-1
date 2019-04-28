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
        # Contar os arroz
        # ------------------------------------------------------------------------------------------------------------ #

        ret, labels = cv.connectedComponents(bimg)
        ngohan_enct = np.max(labels)

        comp_size = np.zeros(ngohan_enct)

        for i in range(1, ngohan_enct):
            comp_size[i] = (int((labels == i).sum() / 10)) * 10

        bins = np.unique(comp_size)
        hstack_labelsize = np.hstack(comp_size)
        n, bins, patches = plt.hist(hstack_labelsize, bins=bins)

        tam_arroz = np.unique(comp_size)[n.argmax()]

        aux_x = bins[n.argmax()]
        plt.plot([aux_x, aux_x], [0, n.max()])
        plt.show()

        # Mostrar resultado
        a = np.hstack(labels)
        plt.hist(a, bins=range(1, ngohan_enct))
        plt.plot([0, ngohan_enct], [tam_arroz, tam_arroz])
        plt.show()

        # Verificar cada componente e seu tamanho
        ngohan_sepa = 0
        for i in range(1, ngohan_enct):

            # Tamanho da componente
            label_size = np.sum(np.where(labels == i, 1, 0))

            # Quantos arroz entrariam nessa componente
            contar = int(label_size / tam_arroz)
            if contar > 0:
                ngohan_sepa += contar
            else:
                ngohan_sepa += 1
                continue

            # Faltou pouco para mais um arroz?
            if (label_size % tam_arroz) / tam_arroz > 0.6:
                ngohan_sepa += 1

        print("Arroz encontrados: " + str(ngohan_enct))
        print("Arroz estimados: " + str(ngohan_sepa))
        print("Arroz esperados: " + str(num_gohan))

    # ================================================================================================================ #
