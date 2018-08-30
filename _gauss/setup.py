#!/usr/bin/env python

from distutils.core import setup, Extension
import numpy as np

ext_modules = [ Extension('_gauss', sources = ['_gauss.c']) ]

setup(
        name = 'Gauss test',
        version = '1.0',
        include_dirs = [np.get_include()], #Add Include path of numpy
        ext_modules = ext_modules
      )
