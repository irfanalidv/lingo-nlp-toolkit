"""
Model classes for Lingo NLP toolkit.
Provides implementations for various NLP tasks.
"""

import json
import logging
from pathlib import Path
from typing import Any

from transformers import (
    AutoModelForQuestionAnswering,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
    pipeline,
)

logger = logging.getLogger(__name__)


class BaseModel:
    """Base class for all NLP models."""

    def __init__(
        self,
        model_name: str,
        config: dict[str, Any] | None = None,
        device: str = "cpu",
        cache_dir: str | None = None,
        **kwargs: Any,
    ) -> None:
        self.model_name = model_name
        self.config = config or {}
        self.device = device
        self.cache_dir = cache_dir
        self.kwargs = kwargs

        self.tokenizer: Any = None
        self.model: Any = None
        self.pipeline: Any = None

        self._load_model()

    def _load_model(self) -> None:
        """Load the model and tokenizer."""
        raise NotImplementedError

    def _load_hf_pipeline(self, hf_task: str, model_cls: Any, source: str | None = None) -> None:
        """Load a tokenizer + model with ``model_cls`` and wrap them in a HF pipeline."""
        source = source or self.model_name
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(source, cache_dir=self.cache_dir)
            self.model = model_cls.from_pretrained(source, cache_dir=self.cache_dir)
            self.model.to(self.device)
            self.pipeline = pipeline(
                hf_task,
                model=self.model,
                tokenizer=self.tokenizer,
                device=self.device,
                # Extra keyword arguments given to Pipeline(...) / the model class are
                # forwarded to the Hugging Face pipeline (e.g. aggregation_strategy="simple").
                **self.kwargs,
            )
            logger.info(f"Loaded {hf_task} model: {source}")
        except Exception as e:
            logger.error(f"Error loading model {source}: {e}")
            raise

    def _run_pipeline(self, inputs: str | list[str], **kwargs: Any) -> list[Any]:
        """Run the HF pipeline on one or more texts and always return a list."""
        if isinstance(inputs, str):
            inputs = [inputs]
        results = self.pipeline(inputs, **kwargs)
        if not isinstance(results, list):
            results = [results]
        return results

    def __call__(self, inputs: Any, **kwargs: Any) -> Any:
        """Process inputs through the model."""
        raise NotImplementedError

    def save(self, path: str | Path) -> None:
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

    def load(self, path: str | Path) -> None:
        """Load model weights and configuration previously written by :meth:`save`."""
        path = Path(path)

        # Load configuration
        with open(path / "model_config.json") as f:
            config = json.load(f)

        self.config = config["config"]
        self.device = config["device"]
        self.kwargs = config["kwargs"]

        # Reload weights from the saved directory, then restore the original identifier
        self.model_name = str(path)
        self._load_model()
        self.model_name = config["model_name"]

        logger.info(f"Model loaded from {path}")


class TextClassifier(BaseModel):
    """Text classification model."""

    def _load_model(self) -> None:
        """Load the classification model and tokenizer."""
        self._load_hf_pipeline("text-classification", AutoModelForSequenceClassification)

    def __call__(self, inputs: str | list[str], **kwargs: Any) -> list[dict[str, Any]]:
        """
        Classify input text(s).

        Args:
            inputs: Text or list of texts to classify
            **kwargs: Additional arguments for the pipeline

        Returns:
            List of classification results
        """
        return self._run_pipeline(inputs, **kwargs)


class NERModel(BaseModel):
    """Named Entity Recognition model."""

    def _load_model(self) -> None:
        """Load the NER model and tokenizer."""
        self._load_hf_pipeline("ner", AutoModelForTokenClassification)

    def __call__(self, inputs: str | list[str], **kwargs: Any) -> list[list[dict[str, Any]]]:
        """
        Extract named entities from input text(s).

        Args:
            inputs: Text or list of texts to process
            **kwargs: Additional arguments for the pipeline

        Returns:
            List of entity lists for each input
        """
        return self._run_pipeline(inputs, **kwargs)


class SentimentAnalyzer(BaseModel):
    """Sentiment analysis model."""

    def _load_model(self) -> None:
        """Load the sentiment analysis model and tokenizer."""
        self._load_hf_pipeline("sentiment-analysis", AutoModelForSequenceClassification)

    def __call__(self, inputs: str | list[str], **kwargs: Any) -> list[dict[str, Any]]:
        """
        Analyze sentiment of input text(s).

        Args:
            inputs: Text or list of texts to analyze
            **kwargs: Additional arguments for the pipeline

        Returns:
            List of sentiment analysis results
        """
        return self._run_pipeline(inputs, **kwargs)


class EmbeddingModel(BaseModel):
    """Text embedding model."""

    def _load_model(self) -> None:
        """Load the embedding model and tokenizer."""
        try:
            # Use sentence-transformers for better embedding quality
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(self.model_name, cache_folder=self.cache_dir)
            self.model.to(self.device)

            logger.info(f"Loaded embedding model: {self.model_name}")

        except Exception as e:
            logger.error(f"Error loading model {self.model_name}: {e}")
            raise

    def __call__(self, inputs: str | list[str], **kwargs: Any) -> list[float] | list[list[float]]:
        """
        Generate embeddings for input text(s).

        Args:
            inputs: Text or list of texts to embed
            **kwargs: Additional arguments for tokenization

        Returns:
            A single embedding vector when ``inputs`` is a string, otherwise a
            list with one embedding vector per input text.
        """
        single = isinstance(inputs, str)
        texts = [inputs] if single else list(inputs)

        # Generate embeddings using sentence-transformers
        embeddings = self.model.encode(texts, **kwargs)
        vectors = [list(map(float, emb)) for emb in embeddings]
        return vectors[0] if single else vectors

    def save(self, path: str | Path) -> None:
        """Save the sentence-transformers model and configuration to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.model.save(str(path))
        config = {
            "model_name": self.model_name,
            "config": self.config,
            "device": self.device,
            "kwargs": self.kwargs,
        }
        with open(path / "model_config.json", "w") as f:
            json.dump(config, f, indent=2)
        logger.info(f"Model saved to {path}")

    def similarity(self, text1: str, text2: str, **kwargs: Any) -> float:
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
    """Extractive question answering model.

    Implemented directly on top of ``AutoModelForQuestionAnswering`` (rather than the
    Hugging Face ``question-answering`` pipeline, which was removed in transformers 5),
    so it works with both transformers 4.x and 5.x. Long contexts are split into
    overlapping windows and the best-scoring span across windows is returned.
    """

    def _load_model(self) -> None:
        """Load the QA model and tokenizer."""
        source = self.model_name
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(source, cache_dir=self.cache_dir)
            self.model = AutoModelForQuestionAnswering.from_pretrained(
                source, cache_dir=self.cache_dir
            )
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"Loaded question-answering model: {source}")
        except Exception as e:
            logger.error(f"Error loading model {source}: {e}")
            raise

    def __call__(
        self,
        question: str,
        context: str,
        max_answer_len: int = 30,
        max_seq_len: int = 384,
        doc_stride: int = 128,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Answer a question based on given context.

        Args:
            question: The question to answer
            context: The context to search for answers
            max_answer_len: Maximum answer length in tokens
            max_seq_len: Maximum tokens per (question, context window) pair
            doc_stride: Token overlap between consecutive context windows

        Returns:
            Dict with ``answer``, ``score`` (0-1), and character offsets ``start``/``end``
            of the answer within ``context``.
        """
        import torch

        if not question or not context:
            raise ValueError("Both 'question' and 'context' must be non-empty strings")

        encoding = self.tokenizer(
            question,
            context,
            truncation="only_second",
            max_length=max_seq_len,
            stride=doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding=True,
            return_tensors="pt",
        )
        offsets = encoding.pop("offset_mapping")
        encoding.pop("overflow_to_sample_mapping", None)
        model_inputs = {k: v.to(self.model.device) for k, v in encoding.items()}

        with torch.no_grad():
            outputs = self.model(**model_inputs)

        best: tuple[float, int, int, int] | None = None
        for i in range(outputs.start_logits.shape[0]):
            context_mask = torch.tensor(
                [sid == 1 for sid in encoding.sequence_ids(i)], dtype=torch.bool
            )
            if not context_mask.any():
                continue
            start_logits = outputs.start_logits[i].float().cpu()
            end_logits = outputs.end_logits[i].float().cpu()
            start_probs = torch.softmax(start_logits.masked_fill(~context_mask, -1e4), dim=-1)
            end_probs = torch.softmax(end_logits.masked_fill(~context_mask, -1e4), dim=-1)

            # Joint span scores restricted to end >= start and length <= max_answer_len
            scores = torch.outer(start_probs, end_probs).triu().tril(max_answer_len - 1)
            scores = scores * context_mask[:, None] * context_mask[None, :]
            flat_idx = int(torch.argmax(scores))
            start_tok, end_tok = divmod(flat_idx, scores.shape[1])
            score = float(scores[start_tok, end_tok])
            if best is None or score > best[0]:
                best = (score, i, start_tok, end_tok)

        if best is None:
            return {"answer": "", "score": 0.0, "start": 0, "end": 0}

        score, window, start_tok, end_tok = best
        start_char = int(offsets[window][start_tok][0])
        end_char = int(offsets[window][end_tok][1])
        return {
            "score": score,
            "start": start_char,
            "end": end_char,
            "answer": context[start_char:end_char].strip(),
        }


class SummarizationModel(BaseModel):
    """Abstractive summarization model (BART, T5, Pegasus, ...).

    Implemented with ``AutoModelForSeq2SeqLM.generate`` (the Hugging Face
    ``summarization`` pipeline was removed in transformers 5), so it works with both
    transformers 4.x and 5.x. Model-specific defaults such as T5's ``"summarize: "``
    prefix are taken from ``config.task_specific_params["summarization"]``.
    """

    def _load_model(self) -> None:
        """Load the summarization model and tokenizer."""
        source = self.model_name
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(source, cache_dir=self.cache_dir)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(source, cache_dir=self.cache_dir)
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"Loaded summarization model: {source}")
        except Exception as e:
            logger.error(f"Error loading model {source}: {e}")
            raise

    def __call__(self, inputs: str | list[str], **kwargs: Any) -> list[dict[str, Any]]:
        """
        Summarize input text(s).

        Args:
            inputs: Text or list of texts to summarize
            **kwargs: Generation arguments forwarded to ``model.generate``
                (e.g. ``max_length``, ``min_length``, ``num_beams``)

        Returns:
            List of ``{"summary_text": ...}`` dicts, one per input
        """
        import torch

        texts = [inputs] if isinstance(inputs, str) else list(inputs)
        task_params = dict(
            (getattr(self.model.config, "task_specific_params", None) or {}).get(
                "summarization", {}
            )
        )
        prefix = task_params.pop("prefix", "") or ""
        generation_kwargs = {**task_params, **kwargs}

        max_input_len = getattr(self.tokenizer, "model_max_length", 1024)
        if not isinstance(max_input_len, int) or max_input_len > 100_000:
            max_input_len = 1024

        encoded = self.tokenizer(
            [prefix + t for t in texts],
            truncation=True,
            max_length=max_input_len,
            padding=True,
            return_tensors="pt",
        )
        encoded = {k: v.to(self.model.device) for k, v in encoded.items()}
        with torch.no_grad():
            output_ids = self.model.generate(**encoded, **generation_kwargs)

        summaries = self.tokenizer.batch_decode(
            output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        return [{"summary_text": summary.strip()} for summary in summaries]
