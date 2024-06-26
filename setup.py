import setuptools

setuptools.setup(
    name='local_gemma_2',
    version='0.0.0',
    author='Hugging Face',
    packages=setuptools.find_packages(),
    entry_points={
        'console_scripts': ['local-gemma-2=local_gemma_2.cli:main']
        },
)
