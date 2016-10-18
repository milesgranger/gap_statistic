import os


try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


def open_file(fname):
    return open(os.path.join(os.path.dirname(__file__), fname))


setup(name='gap-statistic',
      version='0.1',
      author='Miles Granger',
      author_email='miles.granger@outlook.com',
      description='Python implementation of the gap statistic',
      packages=['optimalK'],
      zip_safe=True
      )
