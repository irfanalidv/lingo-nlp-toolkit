"""Lingo: one Pipeline API for common NLP tasks on Hugging Face models, plus preprocessing."""

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
from .exceptions import InvalidInputError, LingoError, UnsupportedTaskError
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
    "EmbeddingModel",
    "InvalidInputError",
    "LingoError",
    "NERModel",
    "Pipeline",
    "QuestionAnsweringModel",
    "SentimentAnalyzer",
    "SummarizationModel",
    "TextClassifier",
    "TextPreprocessor",
    "UnsupportedTaskError",
    "__author__",
    "__email__",
    "__version__",
    "download_model",
    "get_available_models",
    "load_model",
    "save_model",
]
