#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import cv2

from utils.mytimer import Timer
from utils.pathbuilder import PathBuilder

from projetos.t2.blur import box_blur_1, box_blur_2, box_blur_3


def projeto2():
    # ================================================================================================================ #
    #   Constantes
    # ---------------------------------------------------------------------------------------------------------------- #

    INPUT_IMG = 2
    WINDOW_WIDTH = 11
    WINDOW_HEIGHT = 11
    COMP_MIN_WIDTH = 4
    COMP_MIN_HEIGHT = 4
    COMP_MIN_PIXEL = 9

    # ================================================================================================================ #
    #   Timers
    # ---------------------------------------------------------------------------------------------------------------- #

    timer_general = Timer(txt="Total")
    timer_blur = Timer(txt="Blur")

    # ================================================================================================================ #
    #   Imagens
    # ---------------------------------------------------------------------------------------------------------------- #

    images_name = ['arroz.bmp', 'documento-3mp.bmp', 'zumbi.bmp', 'a01 - Original.bmp', 'b01 - Original.bmp']
    prj_path = PathBuilder()

    # ================================================================================================================ #
    #   Abrir a imagem e normalizar para trabalhar sobre ela
    # ---------------------------------------------------------------------------------------------------------------- #

    # Iniciar o timer do programa
    timer_general.start()
    # Abrir imagem
    img = cv2.imread(prj_path.inputdir(images_name[INPUT_IMG]))
    # Normalizar com float 32 bits
    nimg = np.float32(img) / 255

    # ================================================================================================================ #
    #   Função de blur
    # ---------------------------------------------------------------------------------------------------------------- #

    # Iniciar o timer para avaliar a performace do blur
    timer_blur.start()
    # Função de blur
    nimg = box_blur_3(nimg, window_w=WINDOW_WIDTH, window_h=WINDOW_HEIGHT)
    # Parar o timer do blur
    timer_blur.stop()
    # Gerar imagem borrada para visualização
    cv2.imwrite(prj_path.outputdir('T2-blur3.bmp'), np.uint8(nimg * 255))

    # ================================================================================================================ #
    #   Binalizar a imagem
    # ---------------------------------------------------------------------------------------------------------------- #

    # Inversão na imagem do documento
    if INPUT_IMG == 1:
        nimg = np.where(nimg == 1, 0, 1)

    # ================================================================================================================ #
    #   Gerar imagem das componentes encontradas
    # ---------------------------------------------------------------------------------------------------------------- #

    # cv2.imwrite(prj_path.outputdir('output.bmp'), img)
    timer_general.stop()

    # ================================================================================================================ #
    #   Apresentar o número de componentes encontradas
    # ---------------------------------------------------------------------------------------------------------------- #

    print("Tempos de execução")
    timer_general.result()
    timer_blur.result()

    # ================================================================================================================ #
