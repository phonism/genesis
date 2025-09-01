"""
Genesis Deep Learning Framework Setup
"""
from setuptools import setup, find_packages
import os

def read_file(filename):
    """Read file content."""
    with open(os.path.join(os.path.dirname(__file__), filename), encoding='utf-8') as f:
        return f.read()

def read_version():
    """Read version from genesis/__init__.py."""
    version_file = os.path.join('genesis', '__init__.py')
    with open(version_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('__version__'):
                return line.split('=')[1].strip().strip("'\"")
    raise RuntimeError("Unable to find version string.")

def read_requirements(filename):
    """Read requirements from file."""
    requirements = []
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    requirements.append(line)
    except FileNotFoundError:
        pass
    return requirements

setup(
    name="genesis-dl",
    version=read_version(),
    author="Genesis Team",
    author_email="genesis-dev@example.com",
    description="Genesis is a lightweight deep learning framework written from scratch in Python",
    long_description=read_file("README.md"),
    long_description_content_type="text/markdown",
    url="https://github.com/phonism/genesis",
    project_urls={
        "Documentation": "https://genesis-docs.example.com",
        "Repository": "https://github.com/phonism/genesis.git",
        "Issues": "https://github.com/phonism/genesis/issues",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research", 
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS :: MacOS X",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10", 
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords="deep learning, machine learning, neural networks, gpu, triton, cuda",
    python_requires=">=3.8",
    install_requires=read_requirements("requirements.txt"),
    extras_require={
        "llm": [
            "transformers>=4.20.0",
            "safetensors>=0.3.0",
            "dill>=0.3.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "isort>=5.0.0",
            "mypy>=1.0.0",
            "pre-commit>=2.0.0",
        ],
        "docs": [
            "mkdocs>=1.4.0", 
            "mkdocs-material>=8.0.0",
            "mkdocs-autorefs>=0.4.0",
        ],
        "benchmark": [
            "matplotlib>=3.5.0",
            "seaborn>=0.11.0",
            "pandas>=1.3.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "genesis-benchmark=benchmark.bench_ops:main",
        ],
    },
    package_data={
        "genesis": ["py.typed"],
    },
    include_package_data=True,
    zip_safe=False,
)