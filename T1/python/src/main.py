#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import cv2
from utils.mytimer import Timer

# ==================================================================================================================== #
#   Constantes
# -------------------------------------------------------------------------------------------------------------------- #
INPUT_IMG = 1
THRESHOLD_VALUE = 0.75
COMP_MIN_WIDTH = 4
COMP_MIN_HEIGHT = 4
COMP_MIN_PIXEL = 9

# ==================================================================================================================== #
#   Timers
# -------------------------------------------------------------------------------------------------------------------- #
timer_general = Timer(txt="Total")
timer_threshold = Timer(txt="Binarização")
timer_labeling = Timer(txt="Rotulação")

# ==================================================================================================================== #
#   Abrir a imagem
# -------------------------------------------------------------------------------------------------------------------- #
timer_general.start()
img = cv2.imread(['arroz.bmp', 'documento-3mp.bmp'][INPUT_IMG])

# ==================================================================================================================== #
#   Normalizar a imagem
# -------------------------------------------------------------------------------------------------------------------- #
nimg = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

# ==================================================================================================================== #
#   Função de binarização
# -------------------------------------------------------------------------------------------------------------------- #
from utils.color import threshold
timer_threshold.start()
nimg = threshold(nimg, THRESHOLD_VALUE)
timer_threshold.stop()

# ==================================================================================================================== #
#   Gerar imagem binarizada
# -------------------------------------------------------------------------------------------------------------------- #
cv2.imwrite('binarizado.bmp', cv2.normalize(nimg, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX))

# Inversão na imagem do documento
if INPUT_IMG == 1:
    nimg = np.where(nimg == 1, 0, 1)

# ==================================================================================================================== #
#   Função de rotular componentes
# -------------------------------------------------------------------------------------------------------------------- #
from utils.image import labeling_recursive as labeling
timer_labeling.start()
cmps = labeling(nimg, min_width=COMP_MIN_WIDTH, min_height=COMP_MIN_HEIGHT, min_pixel=COMP_MIN_PIXEL)
timer_labeling.stop()

# Desenhar os retângulos das componentes encontradas na imagem original
for cmp in cmps:
    cv2.rectangle(img, (cmp.yi, cmp.xi), (cmp.yf, cmp.xf), (0, 0, 255), 1)

# ==================================================================================================================== #
#   Gerar imagem das componentes encontradas
# -------------------------------------------------------------------------------------------------------------------- #
cv2.imwrite('output.bmp', img)
timer_general.stop()

# ==================================================================================================================== #
#   Apresentar o número de componentes encontradas
# -------------------------------------------------------------------------------------------------------------------- #
print("Componentes encontradas: " + str(len(cmps)))
print("")
print("Tempos de execução")
timer_general.result()
timer_threshold.result()
timer_labeling.result()

# ==================================================================================================================== #
