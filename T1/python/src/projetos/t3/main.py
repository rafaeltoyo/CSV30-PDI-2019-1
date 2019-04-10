#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import cv2

from utils.mytimer import Timer
from utils.pathbuilder import PathBuilder
from projetos.t3.bloom import bloom1


def projeto3():
    # ================================================================================================================ #
    #   Constantes
    # ---------------------------------------------------------------------------------------------------------------- #

    INPUT_IMG = 6
    THRESHOLD = 0.85

    # ================================================================================================================ #
    #   Timers
    # ---------------------------------------------------------------------------------------------------------------- #

    timer_bloom1 = Timer(txt="Bloom Gaussiano")
    timer_bloom2 = Timer(txt="Bloom Box Blur")

    # ================================================================================================================ #
    #   Imagens
    # ---------------------------------------------------------------------------------------------------------------- #

    images_name = ['arroz.bmp',
                   'documento-3mp.bmp',
                   'zumbi.bmp',
                   'a01 - Original.bmp',
                   'b01 - Original.bmp',
                   'chessboard.bmp',
                   'GT2.bmp']
    prj_path = PathBuilder()

    # ================================================================================================================ #
    #   Abrir a imagem e normalizar para trabalhar sobre ela
    # ---------------------------------------------------------------------------------------------------------------- #

    # Abrir imagem
    img = cv2.imread(prj_path.inputdir(images_name[INPUT_IMG]))
    # Normalizar com float 32 bits
    nimg = np.float32(img) / 255

    # ================================================================================================================ #
    #   Função de bloom 1 (Gaussiano)
    # ---------------------------------------------------------------------------------------------------------------- #

    # Iniciar o timer para avaliar a performace do blur
    timer_bloom1.start()
    # Função de blur
    nimg = bloom1(nimg, THRESHOLD)
    # Parar o timer do blur
    timer_bloom1.stop()
    # Gerar imagem borrada para visualização
    cv2.imwrite(prj_path.outputdir('bloom1.bmp'), np.uint8(nimg * 255))
    print(prj_path.outputdir('bloom1.bmp'))

    # ================================================================================================================ #
    #   Apresentar o número de componentes encontradas
    # ---------------------------------------------------------------------------------------------------------------- #

    print("Tempos de execução")
    timer_bloom1.result()

    # ================================================================================================================ #

