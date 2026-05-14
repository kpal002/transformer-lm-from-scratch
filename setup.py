from setuptools import setup, find_packages

setup(
    name="transformer-lm-from-scratch",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0",
        "regex>=2023.0",
        "tqdm>=4.0",
        "datasets>=2.0",
    ],
    python_requires=">=3.10",
)
