"""
Lingo: Advanced NLP Toolkit
Lightweight, Fast, and Transformer-Ready
"""

__version__ = "0.2.1"
__author__ = "Md Irfan Ali"
__email__ = "irfanali29@hotmail.com"

# Core imports
from .core import Pipeline
from .preprocessing import TextPreprocessor
from .models import (
    TextClassifier,
    NERModel,
    SentimentAnalyzer,
    EmbeddingModel,
    QuestionAnsweringModel,
    SummarizationModel,
)
from .utils import (
    load_model,
    save_model,
    get_available_models,
    download_model,
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
