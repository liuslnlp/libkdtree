import sys
import os
from setuptools import setup, find_packages

if sys.platform == 'win32':
    LIB_NAME = "kdtree.dll"
elif sys.platform == 'darwin':
    LIB_NAME = "libkdtree.dylib"
else:
    LIB_NAME = "libkdtree.so"

LIB_PATH = os.path.join('libkdtree', LIB_NAME)

if not os.path.exists(LIB_PATH):
    raise Exception("Can not find C-dll file!")


print('Install libkdtree from: ' + LIB_PATH)

setup(name='kdtree',
      version='0.01',
      description="LibKD-Tree Python Package",
      install_requires=[
          'numpy',
      ],
      author = "liu",
      author_email = "wisedoge@outlook.com",
      packages=find_packages(),
      zip_safe=False,
      include_package_data=True,
      data_files=[('libkdtree', [LIB_PATH])],
      license = "MIT Licence",
      url='https://github.com/WiseDoge/libkdtree')