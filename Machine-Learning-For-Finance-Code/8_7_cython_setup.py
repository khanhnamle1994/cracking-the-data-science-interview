from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize("cython_fib_8_5.pyx"),
)