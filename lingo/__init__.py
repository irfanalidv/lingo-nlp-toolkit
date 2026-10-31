"""
Lingo: Advanced NLP Toolkit
Lightweight, Fast, and Transformer-Ready
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

try:
    __version__ = _pkg_version("lingo-nlp-toolkit")
except PackageNotFoundError:  # pragma: no cover - running from a source tree
    __version__ = "0.0.0+unknown"
__author__ = "Md Irfan Ali"
__email__ = "irfanali29@hotmail.com"

# Core imports
from .core import Pipeline
from .models import (
    EmbeddingModel,
    NERModel,
    QuestionAnsweringModel,
    SentimentAnalyzer,
    SummarizationModel,
    TextClassifier,
)
from .preprocessing import TextPreprocessor
from .utils import (
    download_model,
    get_available_models,
    load_model,
    save_model,
)

# Version info
__all__ = [
    "__version__",
    "__author__",
    "__email__",
    "Pipeline",
    "TextPreprocessor",
    "TextClassifier",
    "NERModel",
    "SentimentAnalyzer",
    "EmbeddingModel",
    "QuestionAnsweringModel",
    "SummarizationModel",
    "load_model",
    "save_model",
    "get_available_models",
    "download_model",
]
