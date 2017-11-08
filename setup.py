from setuptools import setup, find_packages
from sklearn_features import __version__

setup(

    name='sklearn-features',
    version=__version__,
    description='Helpful tools for building feature extraction pipelines with scikit-learn',
    url='https://github.com/sixtwoeight-tech/sklearn-features',
    license='MIT',

    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering',
    ],

    install_requires=[
        'scikit-learn',
        'pandas',
        'scipy',
    ],

    python_requires='>=3.3',

    packages=find_packages(),
)
