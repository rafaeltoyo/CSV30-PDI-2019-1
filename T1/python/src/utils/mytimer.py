#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time


class Timer:

    def __init__(self, txt="Timer"):
        self.__txt = txt
        self.__start = 0
        self.__end = 0

    def start(self):
        self.__start = self.__end = time.time()

    def stop(self):
        self.__end = time.time()

    def result(self):
        print(self.__txt + ": {0}s".format(self.__end - self.__start))
