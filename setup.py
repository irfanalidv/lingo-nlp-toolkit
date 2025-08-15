#!/usr/bin/env python3
"""
Setup script for Lingo NLP Toolkit.
"""

import os
import re
from setuptools import setup, find_packages


# Read the README file
def read_readme():
    """Read README.md file."""
    try:
        with open("README.md", "r", encoding="utf-8") as fh:
            return fh.read()
    except FileNotFoundError:
        return "Advanced NLP Toolkit - Lightweight, Fast, and Transformer-Ready"


# Get long description
long_description = read_readme()

# Version
version = "2.3"

# Core requirements
install_requires = [
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
]

setup(
    name="lingo-nlp-toolkit",
    version=version,
    author="Md Irfan Ali",
    author_email="irfanali29@hotmail.com",
    maintainer="Md Irfan Ali",
    maintainer_email="irfanali29@hotmail.com",
    description="Advanced NLP Toolkit - Lightweight, Fast, and Transformer-Ready",
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords=[
        "nlp",
        "natural-language-processing",
        "transformers",
        "bert",
        "machine-learning",
        "ai",
        "text-processing",
        "sentiment-analysis",
        "named-entity-recognition",
        "text-classification",
        "embeddings",
    ],
    url="https://github.com/irfanalidv/lingo-nlp-toolkit",
    project_urls={
        "Homepage": "https://github.com/irfanalidv/lingo-nlp-toolkit",
        "Documentation": "https://github.com/irfanalidv/lingo-nlp-toolkit",
        "Repository": "https://github.com/irfanalidv/lingo-nlp-toolkit.git",
        "Bug Tracker": "https://github.com/irfanalidv/lingo-nlp-toolkit/issues",
        "Source Code": "https://github.com/irfanalidv/lingo-nlp-toolkit",
        "Changelog": "https://github.com/irfanalidv/lingo-nlp-toolkit/blob/main/CHANGELOG.md",
    },
    packages=find_packages(include=["lingo", "lingo.*"]),
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
    include_package_data=True,
    zip_safe=False,
    python_requires=">=3.9",
    install_requires=install_requires,
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
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "myst-parser>=0.18.0",
            "sphinx-autodoc-typehints>=1.19.0",
        ],
        "test": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "pytest-mock>=3.10.0",
            "pytest-xdist>=3.0.0",
            "coverage>=6.0.0",
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
        # "License :: OSI Approved :: MIT License",  # Removed - superseded by license expression
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
    platforms=["any"],
    provides=["lingo"],
)
