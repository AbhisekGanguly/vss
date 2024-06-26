from setuptools import setup, find_packages

setup(
    name='vss',
    version='0.1.0',
    author='Abhisek Ganguly',
    author_email='abhisekganguly@icloud.com',
    description='A library for vector similarity search and clustering',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/AbhisekGanguly/vss',
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    install_requires=[
        'numpy',
    ],
)
