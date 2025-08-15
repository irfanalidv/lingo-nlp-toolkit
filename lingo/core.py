"""
Core Pipeline class for Lingo NLP toolkit.
Provides a unified interface for all NLP tasks.
"""

import logging
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import yaml
import json

from .models import (
    TextClassifier,
    NERModel,
    SentimentAnalyzer,
    EmbeddingModel,
    QuestionAnsweringModel,
    SummarizationModel,
)
from .preprocessing import TextPreprocessor
from .utils import load_model, save_model, get_available_models

logger = logging.getLogger(__name__)


class Pipeline:
    """
    Main Pipeline class for Lingo NLP toolkit.

    This class provides a unified interface for all NLP tasks including:
    - Text classification
    - Named Entity Recognition (NER)
    - Sentiment analysis
    - Embedding generation
    - Question answering
    - Text summarization

    Args:
        task (str): The NLP task to perform
        model (str): Model name or path to use
        config (dict, optional): Configuration dictionary
        device (str, optional): Device to run on ('cpu', 'cuda', 'mps')
        cache_dir (str, optional): Directory to cache models
    """

    SUPPORTED_TASKS = {
        "text-classification": TextClassifier,
        "ner": NERModel,
        "sentiment-analysis": SentimentAnalyzer,
        "embedding": EmbeddingModel,
        "question-answering": QuestionAnsweringModel,
        "summarization": SummarizationModel,
    }

    def __init__(
        self,
        task: str,
        model: str,
        config: Optional[Dict[str, Any]] = None,
        device: str = "auto",
        cache_dir: Optional[str] = None,
        **kwargs,
    ):
        self.task = task
        self.model_name = model
        self.config = config or {}
        self.device = self._determine_device(device)
        self.cache_dir = cache_dir
        self.kwargs = kwargs

        # Initialize components
        self.preprocessor = None
        self.model = None
        self._initialize_pipeline()

    def _determine_device(self, device: str) -> str:
        """Determine the best available device."""
        if device == "auto":
            import torch

            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device

    def _initialize_pipeline(self):
        """Initialize the pipeline components."""
        if self.task not in self.SUPPORTED_TASKS:
            raise ValueError(
                f"Task '{self.task}' not supported. "
                f"Supported tasks: {list(self.SUPPORTED_TASKS.keys())}"
            )

        # Initialize preprocessor
        self.preprocessor = TextPreprocessor(
            config=self.config.get("preprocessing", {}), device=self.device
        )

        # Initialize model
        model_class = self.SUPPORTED_TASKS[self.task]
        self.model = model_class(
            model_name=self.model_name,
            config=self.config.get("model", {}),
            device=self.device,
            cache_dir=self.cache_dir,
            **self.kwargs,
        )

        logger.info(f"Initialized {self.task} pipeline with {self.model_name}")

    def __call__(self, inputs: Union[str, List[str]] = None, **kwargs) -> Any:
        """
        Process inputs through the pipeline.

        Args:
            inputs: Text input(s) to process (optional for some tasks like QA)
            **kwargs: Additional arguments for the model

        Returns:
            Processed results
        """
        # Handle special cases like question-answering
        if self.task == "question-answering":
            if "question" not in kwargs or "context" not in kwargs:
                raise ValueError(
                    "Question-answering requires 'question' and 'context' arguments"
                )
            return self.model(**kwargs)

        # Handle regular single-input tasks
        if inputs is None:
            raise ValueError("Inputs are required for this task")

        if isinstance(inputs, str):
            inputs = [inputs]

        # Preprocess inputs
        processed_inputs = self.preprocessor(inputs)

        # Run inference
        results = self.model(processed_inputs, **kwargs)

        return results[0] if len(inputs) == 1 else results

    def predict(self, inputs: Union[str, List[str]], **kwargs) -> Any:
        """Alias for __call__ method."""
        return self(inputs, **kwargs)

    def batch_predict(
        self, inputs: List[str], batch_size: int = 32, **kwargs
    ) -> List[Any]:
        """
        Process inputs in batches for better performance.

        Args:
            inputs: List of text inputs
            batch_size: Size of batches to process
            **kwargs: Additional arguments for the model

        Returns:
            List of processed results
        """
        results = []

        for i in range(0, len(inputs), batch_size):
            batch = inputs[i : i + batch_size]
            batch_results = self(batch, **kwargs)
            results.extend(batch_results)

        return results

    def save(self, path: Union[str, Path]):
        """Save the pipeline to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save configuration
        config = {
            "task": self.task,
            "model_name": self.model_name,
            "config": self.config,
            "device": self.device,
            "cache_dir": self.cache_dir,
            "kwargs": self.kwargs,
        }

        with open(path / "pipeline_config.json", "w") as f:
            json.dump(config, f, indent=2)

        # Save model
        if self.model:
            self.model.save(path / "model")

        # Save preprocessor
        if self.preprocessor:
            self.preprocessor.save(path / "preprocessor")

        logger.info(f"Pipeline saved to {path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "Pipeline":
        """Load a pipeline from disk."""
        path = Path(path)

        # Load configuration
        with open(path / "pipeline_config.json", "r") as f:
            config = json.load(f)

        # Create pipeline instance
        pipeline = cls(
            task=config["task"],
            model=config["model_name"],
            config=config["config"],
            device=config["device"],
            cache_dir=config["cache_dir"],
            **config["kwargs"],
        )

        # Load model and preprocessor
        if (path / "model").exists():
            pipeline.model.load(path / "model")

        if (path / "preprocessor").exists():
            pipeline.preprocessor.load(path / "preprocessor")

        logger.info(f"Pipeline loaded from {path}")
        return pipeline

    def get_info(self) -> Dict[str, Any]:
        """Get information about the pipeline."""
        return {
            "task": self.task,
            "model_name": self.model_name,
            "device": self.device,
            "config": self.config,
            "supported_tasks": list(self.SUPPORTED_TASKS.keys()),
        }

    def __repr__(self) -> str:
        return f"Pipeline(task='{self.task}', model='{self.model_name}', device='{self.device}')"
