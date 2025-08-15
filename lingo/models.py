"""
Model classes for Lingo NLP toolkit.
Provides implementations for various NLP tasks.
"""

import logging
from typing import Any, Dict, List, Optional, Union, Tuple
from pathlib import Path
import json
import torch
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoModelForQuestionAnswering,
    AutoModelForSeq2SeqLM,
    pipeline,
    Pipeline as HFPipeline,
)

logger = logging.getLogger(__name__)


class BaseModel:
    """Base class for all NLP models."""

    def __init__(
        self,
        model_name: str,
        config: Optional[Dict[str, Any]] = None,
        device: str = "cpu",
        cache_dir: Optional[str] = None,
        **kwargs,
    ):
        self.model_name = model_name
        self.config = config or {}
        self.device = device
        self.cache_dir = cache_dir
        self.kwargs = kwargs

        self.tokenizer = None
        self.model = None
        self.pipeline = None

        self._load_model()

    def _load_model(self):
        """Load the model and tokenizer."""
        raise NotImplementedError

    def __call__(self, inputs: Any, **kwargs) -> Any:
        """Process inputs through the model."""
        raise NotImplementedError

    def save(self, path: Union[str, Path]):
        """Save the model to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        if self.tokenizer:
            self.tokenizer.save_pretrained(path)

        if self.model:
            self.model.save_pretrained(path)

        # Save configuration
        config = {
            "model_name": self.model_name,
            "config": self.config,
            "device": self.device,
            "kwargs": self.kwargs,
        }

        with open(path / "model_config.json", "w") as f:
            json.dump(config, f, indent=2)

        logger.info(f"Model saved to {path}")

    def load(self, path: Union[str, Path]):
        """Load the model from disk."""
        path = Path(path)

        # Load configuration
        with open(path / "model_config.json", "r") as f:
            config = json.load(f)

        self.model_name = config["model_name"]
        self.config = config["config"]
        self.device = config["device"]
        self.kwargs = config["kwargs"]

        # Reload model
        self._load_model()

        logger.info(f"Model loaded from {path}")


class TextClassifier(BaseModel):
    """Text classification model."""

    def _load_model(self):
        """Load the classification model and tokenizer."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, cache_dir=self.cache_dir
            )

            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name, cache_dir=self.cache_dir
            )

            self.model.to(self.device)

            # Create HuggingFace pipeline
            self.pipeline = pipeline(
                "text-classification",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1,
            )

            logger.info(f"Loaded text classification model: {self.model_name}")

        except Exception as e:
            logger.error(f"Error loading model {self.model_name}: {e}")
            raise

    def __call__(self, inputs: Union[str, List[str]], **kwargs) -> List[Dict[str, Any]]:
        """
        Classify input text(s).

        Args:
            inputs: Text or list of texts to classify
            **kwargs: Additional arguments for the pipeline

        Returns:
            List of classification results
        """
        if isinstance(inputs, str):
            inputs = [inputs]

        results = self.pipeline(inputs, **kwargs)

        # Ensure consistent output format
        if not isinstance(results, list):
            results = [results]

        return results


class NERModel(BaseModel):
    """Named Entity Recognition model."""

    def _load_model(self):
        """Load the NER model and tokenizer."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, cache_dir=self.cache_dir
            )

            self.model = AutoModelForTokenClassification.from_pretrained(
                self.model_name, cache_dir=self.cache_dir
            )

            self.model.to(self.device)

            # Create HuggingFace pipeline
            self.pipeline = pipeline(
                "ner",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1,
            )

            logger.info(f"Loaded NER model: {self.model_name}")

        except Exception as e:
            logger.error(f"Error loading model {self.model_name}: {e}")
            raise

    def __call__(
        self, inputs: Union[str, List[str]], **kwargs
    ) -> List[List[Dict[str, Any]]]:
        """
        Extract named entities from input text(s).

        Args:
            inputs: Text or list of texts to process
            **kwargs: Additional arguments for the pipeline

        Returns:
            List of entity lists for each input
        """
        if isinstance(inputs, str):
            inputs = [inputs]

        results = self.pipeline(inputs, **kwargs)

        # Ensure consistent output format
        if not isinstance(results, list):
            results = [results]

        return results


class SentimentAnalyzer(BaseModel):
    """Sentiment analysis model."""

    def _load_model(self):
        """Load the sentiment analysis model and tokenizer."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, cache_dir=self.cache_dir
            )

            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name, cache_dir=self.cache_dir
            )

            self.model.to(self.device)

            # Create HuggingFace pipeline
            self.pipeline = pipeline(
                "sentiment-analysis",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1,
            )

            logger.info(f"Loaded sentiment analysis model: {self.model_name}")

        except Exception as e:
            logger.error(f"Error loading model {self.model_name}: {e}")
            raise

    def __call__(self, inputs: Union[str, List[str]], **kwargs) -> List[Dict[str, Any]]:
        """
        Analyze sentiment of input text(s).

        Args:
            inputs: Text or list of texts to analyze
            **kwargs: Additional arguments for the pipeline

        Returns:
            List of sentiment analysis results
        """
        if isinstance(inputs, str):
            inputs = [inputs]

        results = self.pipeline(inputs, **kwargs)

        # Ensure consistent output format
        if not isinstance(results, list):
            results = [results]

        return results


class EmbeddingModel(BaseModel):
    """Text embedding model."""

    def _load_model(self):
        """Load the embedding model and tokenizer."""
        try:
            # Use sentence-transformers for better embedding quality
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(
                self.model_name, cache_folder=self.cache_dir
            )
            self.model.to(self.device)

            logger.info(f"Loaded embedding model: {self.model_name}")

        except Exception as e:
            logger.error(f"Error loading model {self.model_name}: {e}")
            raise

    def __call__(
        self, inputs: Union[str, List[str]], **kwargs
    ) -> Union[List[float], List[List[float]]]:
        """
        Generate embeddings for input text(s).

        Args:
            inputs: Text or list of texts to embed
            **kwargs: Additional arguments for tokenization

        Returns:
            Embeddings for input text(s)
        """
        if isinstance(inputs, str):
            inputs = [inputs]

        # Generate embeddings using sentence-transformers
        embeddings = self.model.encode(inputs, **kwargs)

        # Convert to list format
        if len(inputs) == 1:
            return embeddings.tolist()
        else:
            return [emb.tolist() for emb in embeddings]

    def similarity(self, text1: str, text2: str, **kwargs) -> float:
        """
        Calculate cosine similarity between two texts.

        Args:
            text1: First text
            text2: Second text
            **kwargs: Additional arguments for embedding generation

        Returns:
            Cosine similarity score
        """
        import numpy as np

        emb1 = self(text1, **kwargs)
        emb2 = self(text2, **kwargs)

        # Convert to numpy arrays and ensure 1D
        emb1 = np.array(emb1).flatten()
        emb2 = np.array(emb2).flatten()

        # Calculate cosine similarity
        cos_sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

        return float(cos_sim)


class QuestionAnsweringModel(BaseModel):
    """Question answering model."""

    def _load_model(self):
        """Load the QA model and tokenizer."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, cache_dir=self.cache_dir
            )

            self.model = AutoModelForQuestionAnswering.from_pretrained(
                self.model_name, cache_dir=self.cache_dir
            )

            self.model.to(self.device)

            # Create HuggingFace pipeline
            self.pipeline = pipeline(
                "question-answering",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1,
            )

            logger.info(f"Loaded QA model: {self.model_name}")

        except Exception as e:
            logger.error(f"Error loading model {self.model_name}: {e}")
            raise

    def __call__(self, question: str, context: str, **kwargs) -> Dict[str, Any]:
        """
        Answer a question based on given context.

        Args:
            question: The question to answer
            context: The context to search for answers
            **kwargs: Additional arguments for the pipeline

        Returns:
            Answer with confidence score
        """
        result = self.pipeline(question=question, context=context, **kwargs)

        return result


class SummarizationModel(BaseModel):
    """Text summarization model."""

    def _load_model(self):
        """Load the summarization model and tokenizer."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, cache_dir=self.cache_dir
            )

            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name, cache_dir=self.cache_dir
            )

            self.model.to(self.device)

            # Create HuggingFace pipeline
            self.pipeline = pipeline(
                "summarization",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1,
            )

            logger.info(f"Loaded summarization model: {self.model_name}")

        except Exception as e:
            logger.error(f"Error loading model {self.model_name}: {e}")
            raise

    def __call__(self, inputs: Union[str, List[str]], **kwargs) -> List[Dict[str, Any]]:
        """
        Summarize input text(s).

        Args:
            inputs: Text or list of texts to summarize
            **kwargs: Additional arguments for the pipeline

        Returns:
            List of summarization results
        """
        if isinstance(inputs, str):
            inputs = [inputs]

        results = self.pipeline(inputs, **kwargs)

        # Ensure consistent output format
        if not isinstance(results, list):
            results = [results]

        return results
