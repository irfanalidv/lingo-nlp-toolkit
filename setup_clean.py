#!/usr/bin/env python3
"""
Clean setup script for Lingo NLP Toolkit.
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
        "torch>=1.9.0",
        "transformers>=4.20.0",
        "spacy>=3.4.0",
        "nltk>=3.7",
        "scikit-learn>=1.0.0",
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "tqdm>=4.62.0",
        "pyyaml>=6.0",
        "sentence-transformers>=2.2.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=3.0",
            "black>=22.0",
            "flake8>=4.0",
            "mypy>=0.950",
            "isort>=5.0",
        ],
        "full": [
            "tensorflow>=2.8.0",
            "accelerate>=0.12.0",
            "optuna>=3.0.0",
            "wandb>=0.12.0",
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
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Text Processing :: Linguistic",
    ],
    keywords="nlp, natural language processing, transformers, bert, gpt, machine learning, ai",
    project_urls={
        "Bug Reports": "https://github.com/irfanalidv/Lingo/issues",
        "Source": "https://github.com/irfanalidv/Lingo",
        "Documentation": "https://github.com/irfanalidv/Lingo#readme",
    },
    include_package_data=True,
    zip_safe=False,
)
