#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


class Component:

    def __init__(self, label=None):
        self.xi = -1
        self.yi = -1

        self.xf = -1
        self.yf = -1

        self.num_pixel = 0
        self.label = label

    def increment(self, x, y):
        self.xi = min(x, self.xi) if self.xi >= 0 else x
        self.yi = min(y, self.yi) if self.yi >= 0 else y

        self.xf = max(x, self.xf) if self.xf >= 0 else x
        self.yf = max(y, self.yf) if self.yf >= 0 else y

        self.num_pixel += 1

    @property
    def height(self):
        return self.yf - self.yi

    @property
    def width(self):
        return self.xf - self.xi


def labeling(img: np.ndarray, min_width: int, min_height: int, min_pixel: int):
    w, h = img.shape

    UNEXPLORED_LABEL = -1
    label = 1

    wrk = np.copy(img)
    wrk = np.where(wrk == 1, UNEXPLORED_LABEL, 0)

    stack = []
    components = []

    for x in range(w):
        for y in range(h):

            # Possivel inicio de componente encontrado
            if wrk[x][y] == UNEXPLORED_LABEL:

                # Criar a componente
                cmp = Component(label=label)
                label += 1
                # Root na stack
                wrk[x][y] = cmp.label
                stack.append([x, y])

                # Enquanto a stack nao for vazia ...
                while len(stack) > 0:
                    # Tirar o primeiro elemento
                    xi, yi = stack.pop()
                    # Adicionar a componente
                    cmp.increment(xi, yi)

                    # Explorar vizinhos
                    if xi > 0 and wrk[xi-1][yi] == UNEXPLORED_LABEL:
                        wrk[xi-1][yi] = cmp.label
                        stack.append([xi-1, yi])
                    if xi < w-1 and wrk[xi+1][yi] == UNEXPLORED_LABEL:
                        wrk[xi+1][yi] = cmp.label
                        stack.append([xi+1, yi])
                    if yi > 0 and wrk[xi][yi-1] == UNEXPLORED_LABEL:
                        wrk[xi][yi-1] = cmp.label
                        stack.append([xi, yi-1])
                    if yi < h-1 and wrk[xi][yi+1] == UNEXPLORED_LABEL:
                        wrk[xi][yi+1] = cmp.label
                        stack.append([xi, yi+1])

                # Verificar validade da componente
                if cmp.height >= min_height and cmp.width >= min_width and cmp.num_pixel >= min_pixel:
                    components.append(cmp)

    return components
