#!-*- coding:utf-8 -*-

from distutils.core import setup, Extension
import os

libfacwt = Extension(
    'pyfacwt.libfacwt',
    sources = [os.path.join('pyfacwt', 'libfacwt.cpp')],
    include_dirs = [os.path.join('pyfacwt','include')],
    libraries = ['fftw3f', 'boost_python3', 'boost_numpy3'],
    library_dirs = [],
    extra_compile_args = ['-std=c++11','-w','-mavx2','-Ofast'],
    language = 'c++'
)

# TODO
with open('README.rst') as fp:
    long_description = fp.read()

setup(name = 'pyfacwt',
      version = '1.0.0',
      description = 'Python wrapper library of fast approximate continuous wavelet transform implementation written by c++',
      long_description = long_description,
      license='MIT',
      author='Tomohiko Nakamura',
      classifiers = [
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3',
          'Development Status :: 3 -Alpha',
          'Topic :: Multimedia :: Sound/Audio :: Analysis'
      ],
      keywords = 'continuous wavelet transform, constant-Q transform',
      packages = ['pyfacwt'],
      ext_modules = [libfacwt])
