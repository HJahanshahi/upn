from setuptools import setup, find_packages

setup(
    name="upn",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.7.0",
        "torchdiffeq>=0.2.0",
        "numpy>=1.19.0",
        "matplotlib>=3.3.0",
        "scipy>=1.5.0",
    ],
    authors="Jahanshahi, Hadi and Zhu, Zheng H.",
    author_email="hadij@yorku.ca",
    description="Uncertainty Propagation Networks for continuous-time uncertainty modeling",
    long_description="A library implementing Uncertainty Propagation Networks (UPNs) for modeling uncertainty in continuous-time dynamical systems.",
    url="https://github.com/yourusername/upn",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)