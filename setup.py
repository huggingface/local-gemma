import setuptools


deps = [
    "torch",
    "accelerate",
    "transformers",
]


setuptools.setup(
    name='local_gemma_2',
    version='0.0.1',
    author="The Hugging Face team",
    packages=setuptools.find_packages(),
    entry_points={
        'console_scripts': ['local-gemma-2=local_gemma_2.cli:main']
        },
    install_requires=deps,
)
