#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import cv2

from utils.mytimer import Timer
from utils.pathbuilder import PathBuilder
from projetos.t3.bloom import bloom1, bloom2


def projeto3():
    # ================================================================================================================ #
    #   Constantes
    # ---------------------------------------------------------------------------------------------------------------- #

    INPUT_IMG = 6
    THRESHOLD = 0.5
    N_BOXBLUR = 5

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
    #   Definir os sigmas
    # ---------------------------------------------------------------------------------------------------------------- #

    sigmas = range(0, 30, 5)
    window = [int(((sgm ** 2) * 12 / N_BOXBLUR + 1) ** 0.5) for sgm in sigmas]

    # ================================================================================================================ #
    #   Função de bloom 1 (Gaussiano)
    # ---------------------------------------------------------------------------------------------------------------- #

    # Iniciar o timer para avaliar a performace do blur
    timer_bloom1.start()
    # Função de blur
    nimg1 = bloom1(nimg, THRESHOLD,
                   sigmas=sigmas,
                   mask_weight=0.5,
                   img_weight=0.5)
    # Parar o timer do blur
    timer_bloom1.stop()
    # Gerar imagem borrada para visualização
    cv2.imwrite(prj_path.outputdir('bloom1.bmp'), np.uint8(nimg1 * 255))

    # ================================================================================================================ #
    #   Função de bloom 2 (Box Blur)
    # ---------------------------------------------------------------------------------------------------------------- #

    # Iniciar o timer para avaliar a performace do blur
    timer_bloom2.start()
    # Função de blur
    nimg2 = bloom2(nimg, THRESHOLD,
                   window=window,
                   mask_weight=0.5,
                   img_weight=0.5,
                   repeat_boxblur=N_BOXBLUR)
    # Parar o timer do blur
    timer_bloom2.stop()
    # Gerar imagem borrada para visualização
    cv2.imwrite(prj_path.outputdir('bloom2.bmp'), np.uint8(nimg2 * 255))

    # ================================================================================================================ #
    #   Apresentar o número de componentes encontradas
    # ---------------------------------------------------------------------------------------------------------------- #

    print("Tempos de execução")
    timer_bloom1.result()
    timer_bloom2.result()

    # ================================================================================================================ #
