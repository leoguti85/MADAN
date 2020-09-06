#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@ Alexandre Bovet
alexandre.bovet@uclouvain.be
"""

from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("_cython_fast_funcs.pyx",
    compiler_directives = {'language_level': 3})
)
