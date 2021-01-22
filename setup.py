#!/usr/bin/env python

from setuptools import setup, find_packages

setup(name='fastaiextensions',
      version='0.0.1',
      description='Package of fast.ai extensions',
      author='Andrew Scribner',
      author_email='ca.scribner+1@gmail.com',
      packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
#       license='LICENSE.txt',
    )