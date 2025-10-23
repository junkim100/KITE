from setuptools import setup, find_packages

setup(
    name='kite-eval',
    version='1.0.0',
    description='KITE (Korean Instruction-following Task Evaluation) - A comprehensive benchmark for evaluating Korean instruction-following capabilities of LLMs',
    author='Dongjun Kim',
    author_email='junkim100@gmail.com',
    url='https://github.com/junkim100/KITE',
    packages=find_packages(exclude=['tests*']),
    install_requires=[
        'absl-py>=2.0.0',
        'fire>=0.5.0',
        'openai>=1.0.0',
        'transformers>=4.30.0',
        'torch>=2.0.0',
        'accelerate>=0.20.0',
        'datasets>=2.0.0',
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.9',
)
