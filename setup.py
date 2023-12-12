"""Setup script for yastn."""
from setuptools import setup, find_packages

requirements = open('requirements.txt').readlines()

description = ('Scripts for transverse-field Ising chain.')

# README file as long_description.
long_description = open('README.md', encoding='utf-8').read()

__version__ = '0.0.1'

setup(
    name='IsingFF',
    version=__version__,
    author='Marek M. Rams',
    author_email='marek.rams@uj.edu.pl',
    license='Apache License 2.0',
    platform=['any'],
    python_requires=('>=3.7'),
    install_requires=requirements,
    description=description,
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages(exclude=['tests', 'examples'])
)
