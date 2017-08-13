
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


setup(name='gap-stat',
      version='1.0.1',
      author='Miles Granger',
      maintainer='Miles Granger',
      author_email='miles.granger@outlook.com',
      maintainer_email='miles.granger@outlook.com',
      keywords='kmeans unsupervised learning machine-learning clustering',
      description='Python implementation of the gap statistic.',
      long_description='Uses the gap statistic method by Tibshirani, Walther, Hastie to suggest n_clusters.',
      packages=['gap_statistic'],
      license='BSD',
      url='https://github.com/milesgranger/gap_statistic',
      zip_safe=True,
      install_requires=['numpy', 'pandas', 'scipy'],
      setup_requires=['pytest-runner'],
      tests_require=['pytest', 'scikit-learn'],
      classifiers=[
            'Programming Language :: Python :: 3',
            'Development Status :: 4 - Beta'
      ]
      )
