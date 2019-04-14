#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import enum


class Neighborhood(enum.Enum):
    NB_4 = 4
    NB_8 = 8


class Component:

    def __init__(self, label=None):
        """
        Classe que representa uma componente detectada em uma imagem
        :param label: Rótulo da componente
        """
        self.xi = None
        self.yi = None

        self.xf = None
        self.yf = None

        self.num_pixel = 0
        self.label = label

    def increment(self, x: float, y: float):
        """
        Reajustar a componente com uma nova coordenada
        :param x:
        :param y:
        :return: Nada
        """
        self.xi = min(x, self.xi) if self.xi is not None else x
        self.yi = min(y, self.yi) if self.yi is not None else y

        self.xf = max(x, self.xf) if self.xf is not None else x
        self.yf = max(y, self.yf) if self.yf is not None else y

        self.num_pixel += 1

    @property
    def height(self):
        """
        Calcula e retorna a altura da componente
        :return: Altura da componente
        """
        if self.yf is None or self.yi is None:
            return 0
        return self.yf - self.yi

    @property
    def width(self):
        """
        Calcula e retorna a largura da componente
        :return: largura da componente
        """
        if self.xf is None or self.xi is None:
            return 0
        return self.xf - self.xi


def get_neighborhood(img: np.ndarray, xi: int, yi: int, target, type: Neighborhood = Neighborhood.NB_4):
    w, h = img.shape
    if type == Neighborhood.NB_4:
        if xi > 0 and img[xi - 1][yi] == target:
            yield [xi - 1, yi]
        if xi < w - 1 and img[xi + 1][yi] == target:
            yield [xi + 1, yi]
        if yi > 0 and img[xi][yi - 1] == target:
            yield [xi, yi - 1]
        if yi < h - 1 and img[xi][yi + 1] == target:
            yield [xi, yi + 1]
    elif type == Neighborhood.NB_8:
        if xi > 0 and yi > 0 and img[xi - 1][yi - 1] == target:
            yield [xi - 1, yi - 1]
        if yi > 0 and img[xi][yi - 1] == target:
            yield [xi, yi - 1]
        if xi < w - 1 and yi > 0 and img[xi + 1][yi - 1] == target:
            yield [xi + 1, yi - 1]

        if xi > 0 and img[xi - 1][yi] == target:
            yield [xi - 1, yi]
        if xi < w - 1 and img[xi + 1][yi] == target:
            yield [xi + 1, yi]

        if xi > 0 and yi < h - 1 and img[xi - 1][yi + 1] == target:
            yield [xi - 1, yi + 1]
        if yi < h - 1 and img[xi][yi + 1] == target:
            yield [xi, yi + 1]
        if xi < w - 1 and yi < h - 1 and img[xi + 1][yi + 1] == target:
            yield [xi + 1, yi + 1]


def labeling_recursive(img: np.ndarray,
                       neig: Neighborhood = Neighborhood.NB_4,
                       min_width: int = 1,
                       min_height: int = 1,
                       min_pixel: int = 1):
    """
    Rotularização com algoritmo recursivo
    :param img: Imagem
    :param neig: Número de vizinhos a ser considerados (4 ou 8)
    :param min_width: Largura mínima de uma componente
    :param min_height: Altura mínima de uma componente
    :param min_pixel: Número mínimo que uma componente deverá conter
    :return: Lista com as componentes encontradas
    """
    w, h = img.shape

    UNEXPLORED_LABEL = -1
    label = 1

    wrk = np.copy(img)
    wrk = np.where(wrk == 1, UNEXPLORED_LABEL, 0)

    components = []

    def floodfill(cmp: Component, xi: int, yi: int):
        wrk[xi][yi] = cmp.label
        cmp.increment(xi, yi)

        # Explorar vizinhos
        for xj, yj in get_neighborhood(wrk, xi, yi, UNEXPLORED_LABEL, type=neig):
            floodfill(cmp, xj, yj)

    for y in range(h):
        for x in range(w):

            # Possivel inicio de componente encontrado
            if wrk[x][y] == UNEXPLORED_LABEL:

                # Criar a componente
                cmp = Component(label=label)
                label += 1

                # Iniciar recursão na semente
                floodfill(cmp, x, y)

                # Verificar validade da componente
                if cmp.height >= min_height and cmp.width >= min_width and cmp.num_pixel >= min_pixel:
                    components.append(cmp)
    return components


def labeling_stack(img: np.ndarray,
                       neig: Neighborhood = Neighborhood.NB_4,
                       min_width: int = 1,
                       min_height: int = 1,
                       min_pixel: int = 1):
    """
    Rotularização com algoritmo recursivo
    :param img: Imagem
    :param neig: Número de vizinhos a ser considerados (4 ou 8)
    :param min_width: Largura mínima de uma componente
    :param min_height: Altura mínima de uma componente
    :param min_pixel: Número mínimo que uma componente deverá conter
    :return: Lista com as componentes encontradas
    """
    w, h = img.shape

    UNEXPLORED_LABEL = -1
    label = 1

    wrk = np.copy(img)
    wrk = np.where(wrk == 1, UNEXPLORED_LABEL, 0)

    stack = []
    components = []

    for y in range(h):
        for x in range(w):

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

                    #  Explorar vizinhos
                    for xj, yj in get_neighborhood(wrk, xi, yi, UNEXPLORED_LABEL, type=neig):
                        wrk[xj][yj] = cmp.label
                        stack.append([xj, yj])

                # Verificar validade da componente
                if cmp.height >= min_height and cmp.width >= min_width and cmp.num_pixel >= min_pixel:
                    components.append(cmp)
    return components
