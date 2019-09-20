#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages


readme = """
A Python implementation of the Binary Matrix Decomposition algorithm for clustering binary data.

* Documentation: https://bmdcluster.readthedocs.io.
* GitHub: https://github.com/csprock/bmdcluster
"""

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['numpy>=1.14']

setup_requirements = ['pytest-runner', ]

test_requirements = ['pytest', ]

setup(
    author="Carson Sprock",
    author_email='csprock@gmail.com',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    description="Binary Matrix Decomposition algorithm for clustering binary data",
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='bmdcluster',
    name='bmdcluster',
    #packages=find_packages(include=['bmdcluster']),
    packages=['bmdcluster', 'bmdcluster.optimizers', 'bmdcluster.initializers'],
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/csprock/bmdcluster',
    version='0.2.3',
    zip_safe=False,
)
