#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

from utils.pathbuilder import PathBuilder


def projeto5():
    # ================================================================================================================ #
    #   Constantes
    # ---------------------------------------------------------------------------------------------------------------- #
    GAMMA = 10
    RESIZE = 2

    # ================================================================================================================ #
    #   Constantes
    # ---------------------------------------------------------------------------------------------------------------- #
    count = 1

    # ================================================================================================================ #
    #   Imagens
    # ---------------------------------------------------------------------------------------------------------------- #
    images_name = ["chromakey/{0}.bmp".format(i) for i in range(0, 9)]
    prj_path = PathBuilder()

    for imgname in images_name:
        # Abrir imagem
        img = cv2.imread(prj_path.inputdir(imgname))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        teste = img

        # Tratar

        # img[:, :, 0] = int(127)
        # img[:, :, 1] = int(127)
        # img[:, :, 2] = int(255)

        # hist = cv2.calcHist([img], [0], None, [256], [0, 256])

        plt.hist(img[:, :, 0].ravel(), 256, [0, 256])

        gsimg = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_HSV2BGR), cv2.COLOR_BGR2GRAY)
        ret, th = cv2.threshold(img[:, :, 2], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cv2.imwrite(prj_path.outputdir("chromakey{0}.bmp".format(count)), th)

        count += 1

    plt.show()

    # ================================================================================================================ #
