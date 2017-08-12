
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


setup(name='gap-statistic',
      version='1.0',
      author='Miles Granger',
      author_email='miles.granger@outlook.com',
      description='Python implementation of the gap statistic',
      packages=['gap_statistic'],
      zip_safe=True,
      install_requires=['numpy', 'scikit-learn', 'pandas', 'scipy', 'joblib'],
      setup_requires=['pytest-runner'],
      tests_require=['pytest']
      )
