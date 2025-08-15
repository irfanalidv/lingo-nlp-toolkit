#!/usr/bin/env python3
"""
Minimal setup script for Lingo NLP Toolkit.
"""

from setuptools import setup, find_packages


# Read the README file
def read_readme():
    """Read README.md file."""
    try:
        with open("README.md", "r", encoding="utf-8") as fh:
            return fh.read()
    except FileNotFoundError:
        return "Advanced NLP Toolkit - Lightweight, Fast, and Transformer-Ready"


setup(
    name="lingo",
    version="0.1.0",
    author="Md Irfan Ali",
    author_email="irfanali29@hotmail.com",
    description="Advanced NLP Toolkit - Lightweight, Fast, and Transformer-Ready",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/irfanalidv/Lingo",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=1.12.0",
        "transformers>=4.20.0",
        "tokenizers>=0.13.0",
        "spacy>=3.5.0",
        "nltk>=3.8",
        "scikit-learn>=1.1.0",
        "numpy>=1.21.0",
        "pandas>=1.5.0",
        "scipy>=1.9.0",
        "tqdm>=4.64.0",
        "pyyaml>=6.0",
        "requests>=2.28.0",
        "beautifulsoup4>=4.11.0",
        "lxml>=4.9.0",
        "fastapi>=0.95.0",
        "uvicorn>=0.20.0",
        "pydantic>=1.10.0",
        "sentence-transformers>=2.2.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "pytest-mock>=3.10.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "flake8>=5.0.0",
            "mypy>=1.0.0",
            "pre-commit>=2.20.0",
        ],
        "full": [
            "pdfplumber>=0.7.0",
            "python-docx>=0.8.11",
            "openpyxl>=3.0.10",
            "matplotlib>=3.5.0",
            "seaborn>=0.11.0",
            "plotly>=5.10.0",
            "umap-learn>=0.5.3",
            "hdbscan>=0.8.29",
            "gensim>=4.3.0",
            "wordcloud>=1.9.0",
            "textstat>=0.7.3",
            "language-tool-python>=2.7.1",
        ],
        "gpu": [
            "torch>=1.12.0",
            "torchvision>=0.13.0",
            "torchaudio>=0.12.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "lingo=lingo.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Framework :: Jupyter",
        "Typing :: Typed",
    ],
    include_package_data=True,
    package_data={
        "lingo": [
            "configs/*.yaml",
            "configs/*.yml",
            "models/*.json",
            "data/*.txt",
            "*.pyi",
            "py.typed",
        ],
    },
)
