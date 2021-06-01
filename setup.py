#!/usr/bin/env python
"""
ChatterBot setup file.
"""
import os
import sys
import platform
import configparser
from setuptools import setup


if sys.version_info[0] < 3:
    raise Exception(
        'You are tying to install ChatterBot on Python version {}.\n'
        'Please install ChatterBot in Python 3 instead.'.format(
            platform.python_version()
        )
    )

config = configparser.ConfigParser()

current_directory = os.path.dirname(os.path.abspath(__file__))
config_file_path = os.path.join(current_directory, 'setup.cfg')

config.read(config_file_path)

VERSION = config['chatterbot']['version']
AUTHOR = config['chatterbot']['author']
AUTHOR_EMAIL = config['chatterbot']['email']
URL = config['chatterbot']['url']

with open('README.md') as f:
    LONG_DESCRIPTION = f.read()

REQUIREMENTS = []
DEPENDENCIES = []

with open('requirements.txt') as requirements:
    for requirement in requirements.readlines():
        if requirement.startswith('git+git://'):
            DEPENDENCIES.append(requirement)
        else:
            REQUIREMENTS.append(requirement)


setup(
    name='antyrec',
    version=VERSION,
    url=URL,
    download_url='{}/tarball/{}'.format(URL, VERSION),
    #project_urls={
    #    'Documentation': 'https://chatterbot.readthedocs.io',
    #},
    description='antyrec is a useful tool for recognition attacks',
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    packages=[
        'alpm',
        'alpm.changeit',
        'alpm.lp_util',
        'alpm.recit',
    ],
    #package_dir={'chatterbot': 'chatterbot'},
    include_package_data=True,
    install_requires=REQUIREMENTS,
    dependency_links=DEPENDENCIES,
    python_requires='>=3.7, <3.9',
    license='not yet',
    zip_safe=True,
    platforms=['any'],
    keywords=['antyrec', 'alpm', 'license plate',],
    classifiers=[
        'Development Status :: 0 - Alpha',
        'Intended Audience :: Developer',
        'License :: Not yet approved :: NO License',
        'Environment :: Console',
        'Environment :: Web Environment',
        'Operating System :: OS Independent',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: OCR :: antyrec',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3 :: Only',
    ],
    #test_suite='tests'
)
