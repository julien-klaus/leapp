from setuptools import setup
from setuptools import find_packages

setup(name='leapp',
      version='1.0',
      description='Learning Probabilistic Programs',
      url='https://github.com/julien-klaus/leapp',
      author='Julien Klaus',
      author_email='julien.klaus@uni-jena.de',
      license='lgpl-3.0',
      packages=find_packages(),
      install_requires=[
          'numpy',
          'pandas',
          'graphviz',
          'rpy2',
          'networkx'
      ],
      zip_safe=False)

# use e.g. python setup.py install
# or pip install . (from local directory)