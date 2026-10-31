"""
Core Pipeline class for Lingo NLP toolkit.
Provides a unified interface for all NLP tasks.
"""

import json
import logging
from pathlib import Path
from typing import Any, ClassVar

from .exceptions import InvalidInputError, UnsupportedTaskError
from .models import (
    BaseModel,
    EmbeddingModel,
    NERModel,
    QuestionAnsweringModel,
    SentimentAnalyzer,
    SummarizationModel,
    TextClassifier,
)
from .preprocessing import TextPreprocessor

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

    # Preprocessing applied before model inference unless overridden via
    # ``config["preprocessing"]``. Transformer tokenizers handle casing and
    # punctuation themselves, so by default we only normalize Unicode and
    # whitespace (lowercasing would hurt cased models such as most NER models).
    DEFAULT_PREPROCESSING: ClassVar[dict[str, Any]] = {
        "normalize_unicode": True,
        "lowercase": False,
        "remove_extra_whitespace": True,
        "use_spacy": False,
    }

    SUPPORTED_TASKS: ClassVar[dict[str, type[BaseModel]]] = {
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
        config: dict[str, Any] | None = None,
        device: str = "auto",
        cache_dir: str | None = None,
        **kwargs: Any,
    ) -> None:
        self.task = task
        self.model_name = model
        self.config = config or {}
        self.device = self._determine_device(device)
        self.cache_dir = cache_dir
        self.kwargs = kwargs

        # Initialize components
        self.preprocessor: TextPreprocessor
        self.model: BaseModel
        self._initialize_pipeline()

    def _determine_device(self, device: str) -> str:
        """Determine the best available device."""
        if device == "auto":
            import torch

            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        return device

    def _initialize_pipeline(self) -> None:
        """Initialize the pipeline components."""
        if self.task not in self.SUPPORTED_TASKS:
            raise UnsupportedTaskError(
                f"Task '{self.task}' not supported. "
                f"Supported tasks: {list(self.SUPPORTED_TASKS.keys())}"
            )

        # Initialize preprocessor
        preprocessing_config = {
            **self.DEFAULT_PREPROCESSING,
            **self.config.get("preprocessing", {}),
        }
        self.preprocessor = TextPreprocessor(config=preprocessing_config, device=self.device)

        # Initialize model
        model_class = self.SUPPORTED_TASKS[self.task]
        self.model = model_class(
            model_name=self.model_name,
            config=self.config.get("model", {}),
            device=self.device,
            cache_dir=self.cache_dir,
            **self.kwargs,
        )

        logger.info("Initialized %s pipeline with %s", self.task, self.model_name)

    def __call__(self, inputs: str | list[str] | None = None, **kwargs: Any) -> Any:
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
                raise InvalidInputError(
                    "Question-answering requires 'question' and 'context' arguments"
                )
            return self.model(**kwargs)

        # Handle regular single-input tasks
        if inputs is None:
            raise InvalidInputError("Inputs are required for this task")

        single = isinstance(inputs, str)
        results = self._run(inputs, **kwargs)
        return results[0] if single else results

    def _run(self, inputs: str | list[str], **kwargs: Any) -> list[Any]:
        """Preprocess and run the model, always returning one result per input."""
        if isinstance(inputs, str):
            inputs = [inputs]
        processed_inputs = self.preprocessor(list(inputs))
        results = self.model(processed_inputs, **kwargs)
        if not isinstance(results, list):
            results = [results]
        return results

    def predict(self, inputs: str | list[str], **kwargs: Any) -> Any:
        """Alias for __call__ method."""
        return self(inputs, **kwargs)

    def batch_predict(self, inputs: list[str], batch_size: int = 32, **kwargs: Any) -> list[Any]:
        """
        Process inputs in batches for better performance.

        Args:
            inputs: List of text inputs
            batch_size: Size of batches to process
            **kwargs: Additional arguments for the model

        Returns:
            List of processed results
        """
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.task == "question-answering":
            raise InvalidInputError("batch_predict is not supported for question-answering")

        results: list[Any] = []

        for i in range(0, len(inputs), batch_size):
            batch = inputs[i : i + batch_size]
            results.extend(self._run(batch, **kwargs))

        return results

    def save(self, path: str | Path) -> None:
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

        with (path / "pipeline_config.json").open("w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

        # Save model
        if self.model:
            self.model.save(path / "model")

        # Save preprocessor
        if self.preprocessor:
            self.preprocessor.save(path / "preprocessor")

        logger.info("Pipeline saved to %s", path)

    @classmethod
    def load(cls, path: str | Path, device: str | None = None) -> "Pipeline":
        """Load a pipeline saved with :meth:`save`.

        Args:
            path: Directory written by :meth:`save`.
            device: Override the saved device (for example ``"cpu"`` or ``"auto"`` when a
                pipeline saved on a GPU machine is loaded on a CPU-only one). Defaults to
                the saved device.
        """
        path = Path(path)

        # Load configuration
        with (path / "pipeline_config.json").open(encoding="utf-8") as f:
            config = json.load(f)

        # Prefer the locally saved model weights over re-downloading from the Hub
        model_dir = path / "model"
        model_source = str(model_dir) if model_dir.exists() else config["model_name"]

        preprocessor_dir = path / "preprocessor"
        pipeline_config = dict(config.get("config") or {})
        if (preprocessor_dir / "preprocessor_config.json").exists():
            with (preprocessor_dir / "preprocessor_config.json").open(encoding="utf-8") as f:
                saved = json.load(f)
            pipeline_config["preprocessing"] = saved.get("config", {})

        pipeline = cls(
            task=config["task"],
            model=model_source,
            config=pipeline_config,
            device=device or config.get("device", "auto"),
            cache_dir=config.get("cache_dir"),
            **config.get("kwargs", {}),
        )
        # Keep the original model identifier for get_info()/repr
        pipeline.model_name = config["model_name"]
        pipeline.config = config.get("config") or {}

        logger.info("Pipeline loaded from %s", path)
        return pipeline

    def get_info(self) -> dict[str, Any]:
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
