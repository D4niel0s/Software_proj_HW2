from setuptools import Extension, setup

module = Extension("kmeans_module", sources=['kmeansmodule.c'])
setup(name='kmeans_module',
     version='1.0',
     description='An implementation of the K-Means clustering algorithm in C!',
     ext_modules=[module])