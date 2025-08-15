#!/usr/bin/env python3
"""
Minimal setup script for Lingo NLP Toolkit - PyPI Ready.
"""

from setuptools import setup, find_packages

setup(
    name="lingo-nlp-toolkit",
    version="0.2.2",
    author="Md Irfan Ali",
    author_email="irfanali29@hotmail.com",
    description="Advanced NLP Toolkit - Lightweight, Fast, and Transformer-Ready",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/irfanalidv/lingo-nlp-toolkit",
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
        "Bug Reports": "https://github.com/irfanalidv/lingo-nlp-toolkit/issues",
        "Source": "https://github.com/irfanalidv/lingo-nlp-toolkit",
        "Documentation": "https://github.com/irfanalidv/lingo-nlp-toolkit#readme",
    },
    zip_safe=False,
)
