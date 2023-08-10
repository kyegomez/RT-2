from setuptools import setup, find_packages

setup(
    name='rt2',
    packages=find_packages(exclude=[]),
    version='0.0.3',
    license='MIT',
    description='rt-2 - PyTorch',
    author='Kye Gomez',
    author_email='kye@apac.ai',
    long_description_content_type='text/markdown',
    url='https://github.com/kyegomez/rt-2',
    keywords=[
        'artificial intelligence',
        'deep learning',
        'optimizers',
        'Prompt Engineering'
    ],
    install_requires=[
        'transformers',
        'torch',
        'einops',
        'beartype',
        'palme',
        'transformers',
        'palm-rlhf-pytorch',
        'tokenizers',
        'wandb',
        'classifier-free-guidance-pytorch'
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
    ],
)