from setuptools import setup

setup(
    name='ml_utility_loss',
    version='0.1.1',    
    description='',
    url='https://github.com/R-N/ml-utility-loss',
    author='Muhammad Rizqi Nur',
    author_email='rizqinur2010@gmail.com',
    license='MIT',
    packages=setuptools.find_packages(),
    install_requires=[
        #"tab_ddpm @ git+https://github.com/R-N/tab-ddpm@main",
        #"ctgan @ git+https://github.com/R-N/CTGAN@master",
        #"lct_gan @ git+https://github.com/R-N/LCT-GAN@master",
        #"ctab_gan_plus @ git+https://github.com/R-N/CTAB-GAN-Plus@main",
        #"realtabformer @ git+https://github.com/R-N/REaLTabFormer@main",
        "alpharelu", # @ git+https://github.com/MaxatTezekbayev/alpha-relu@main",
        "entmax", # @ git+https://github.com/deep-spin/entmax@master",
        "accelerate>=0.20.3",
        "catboost>=1.0.3",
        "category-encoders>=2.3.0",
        "datasets>=2.6.1",
        "delu",
        "dython~=0.6.4.post1",
        "icecream>=2.1.2",
        "imbalanced-learn>=0.7.0",
        "jupyter>=1.0.0",
        "libzero>=0.0.8",
        "matplotlib>=3.5.2",
        "numpy>=1.25.0",
        "optuna>=2.10.1",
        "pandas==1.5.3",
        "pyarrow>=6.0.0",
        "rdt>=1.3.0",
        "rtdl>=0.0.9",
        "shapely>=1.8.5.post1",
        "scikit-learn==1.2.2",
        "scipy>=1.8.0",
        "seaborn>=0.11.2",
        "skorch",
        "tomli-w>=0.4.0",
        "tomli>=1.2.2",
        "torch>=1.13.0",
        "torchinfo",
        "torchvision>=0.12.0",
        "tqdm>=4.64.1",
        "transformers==4.28.0",
    ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
    ],
)