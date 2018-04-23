from setuptools import setup, find_packages

def readme():
    with open('README.md') as f:
        return f.read()


def setup(
    name = 'bmdcluster',
    version = '0.0.1.dev1',
    description = 'BMD algorithm for clustering binary data',
    url = 'https://github.com/csprock/bmdcluster',
    author = 'Carson Sprock',
    author_email = 'csprock@gmail.com',
    license = 'MIT',
    keywords = 'clustering binary discrete',
    description = 'A Package for Clustering Binary Data',
    long_description = readme(),
    long_description_content_type = 'text/markdown',
    install_requires = ['numpy>=1.14'],
    packages = find_packages(exclude = ['tests']),
    classifiers = [
    'Development Status :: 2 - Pre-Alpha',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
    'Topic :: Scientific/Engineering :: Information Analysis'
    ]
    python_requires = '>=3'
)
