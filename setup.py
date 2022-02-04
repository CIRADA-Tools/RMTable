import io
import os
import sys
from shutil import rmtree

from setuptools import find_packages, setup, Command

NAME = 'RMTable'
DESCRIPTION = 'Reading, writing, and manipulating RMTables (radio astronomy Faraday rotation catalogs)'
URL = 'https://github.com/CIRADA-Tools/RMTable'
REQUIRES_PYTHON = '>=3.5.0'
VERSION = '1.0.0'

REQUIRED = [
    'numpy', 'astropy',
]

here = os.path.abspath(os.path.dirname(__file__))


try:
    with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    python_requires=REQUIRES_PYTHON,
    url=URL,
    py_modules=['rmtable'],
    install_requires=REQUIRED,
    include_package_data=True,
    license='MIT',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Astronomy',
    ],
    maintainer='Cameron Van Eck',
    maintainer_email='cameron.van.eck@utoronto.ca',
)

