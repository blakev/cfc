#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# >>
#   cfc, 2021
#   Blake VandeMerwe <blake@vidangel.com>
# <<

import setuptools

with open('README.md', 'r') as fp:
    long_description = fp.read()

with open('requirements.txt', 'r') as fp:
    requirements = [o.strip() for o in fp.read().split('\n')]

setuptools.setup(
    name='cfc',
    version='0.1.0',
    python_requires=">=3.8",
    author="Blake VandeMerwe",
    author_email='blakev@null.net',
    license='MIT',
    description='Small utility to keep cache folders under control',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=setuptools.find_packages(where="cfc"),
    entry_points={
        'console_scripts': ['cfc=cfc.__main__'],
    },
    zip_safe=False,
    include_package_data=True,
    install_requires=requirements,
    classifiers=[
        'Development Status :: 3 - Alpha',
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
)
