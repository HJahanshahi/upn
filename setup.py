from setuptools import setup, find_packages

setup(
    name="upn",
    version="1.0.0",
    description="Uncertainty Propagation Networks for Neural Ordinary Differential Equations",
    author="Hadi Jahanshahi, Zheng H. Zhu",
    author_email="gzhu@yorku.ca",
    url="https://github.com/HJahanshahi/upn",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=1.9",
        "torchdiffeq",
        "numpy",
        "scipy",
        "matplotlib",
        "tqdm",
        "pandas",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
