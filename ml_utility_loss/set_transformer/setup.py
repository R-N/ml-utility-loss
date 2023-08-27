from setuptools import setup

setup(
    name='set_transformer',
    version='0.1.0',    
    description='',
    url='https://github.com/R-N/set_transformer',
    author='',
    author_email='',
    license='',
    packages=setuptools.find_packages(),
    install_requires=[
        "torch>=1.0",
        "matplotlib",
        "scipy",
        "tqdm",
    ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
    ],
)