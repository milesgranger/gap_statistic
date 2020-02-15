from setuptools import setup
from gap_statistic import __version__

install_requires = ["numpy", "pandas", "scipy"]

setup_requires = [
    "setuptools-rust>=0.9.2",
    "pytest-runner",
    "pytest",
    "scikit-learn",
    "joblib",
    "scipy",
    "pandas",
]
tests_require = ["scikit-learn", "pytest", "joblib", "black", "click"]

setup(
    name="gap-stat",
    version=__version__,
    author="Miles Granger",
    maintainer="Miles Granger",
    author_email="miles59923@gmail.com",
    maintainer_email="miles59923@gmail.com",
    keywords="kmeans unsupervised learning machine-learning clustering",
    description="Python implementation of the gap statistic with Rust optimizations.",
    long_description="Uses the gap statistic method by Tibshirani, Walther, Hastie to suggest n_clusters.",
    packages=["gap_statistic"],
    license="MIT",
    url="https://github.com/milesgranger/gap_statistic",
    zip_safe=False,
    setup_requires=setup_requires,
    install_requires=install_requires,
    tests_require=tests_require,
    test_suite="tests",
    include_package_data=True,
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Information Technology",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Rust",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
    ],
)
