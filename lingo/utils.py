"""
Utility functions for Lingo NLP toolkit.
Provides helper functions for model management, evaluation, and common operations.
"""

import logging
import re
from pathlib import Path
from typing import Any

import requests
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from tqdm import tqdm

logger = logging.getLogger(__name__)


def load_model(path: str | Path) -> Any:
    """
    Load a saved model from disk.

    Args:
        path: Path to the saved model

    Returns:
        Loaded model
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Model path does not exist: {path}")

    # Check if it's a pipeline
    if (path / "pipeline_config.json").exists():
        from .core import Pipeline

        return Pipeline.load(path)

    # Check if it's a standalone model
    if (path / "model_config.json").exists():
        # This would need to be implemented based on the model type
        raise NotImplementedError("Standalone model loading not yet implemented")

    raise ValueError(f"Invalid model path: {path}")


def save_model(model: Any, path: str | Path):
    """
    Save a model to disk.

    Args:
        model: Model to save
        path: Path to save the model
    """
    path = Path(path)

    if hasattr(model, "save"):
        model.save(path)
    else:
        raise ValueError("Model does not have a save method")


def get_available_models() -> dict[str, list[str]]:
    """
    Get list of available pre-trained models for different tasks.

    Returns:
        Dictionary mapping tasks to available models
    """
    return {
        "text-classification": [
            "bert-base-uncased",
            "roberta-base",
            "distilbert-base-uncased",
            "albert-base-v2",
            "xlnet-base-cased",
        ],
        "ner": [
            "dbmdz/bert-large-cased-finetuned-conll03-english",
            "dslim/bert-base-NER",
            "Jean-Baptiste/roberta-large-ner-english",
        ],
        "sentiment-analysis": [
            "cardiffnlp/twitter-roberta-base-sentiment-latest",
            "nlptown/bert-base-multilingual-uncased-sentiment",
            "finiteautomata/bertweet-base-sentiment-analysis",
        ],
        "embedding": [
            "sentence-transformers/all-MiniLM-L6-v2",
            "sentence-transformers/all-mpnet-base-v2",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        ],
        "question-answering": [
            "deepset/roberta-base-squad2",
            "distilbert-base-cased-distilled-squad",
            "deepset/minilm-uncased-squad2",
        ],
        "summarization": ["facebook/bart-large-cnn", "t5-base", "google/pegasus-xsum"],
    }


def download_model(model_name: str, cache_dir: str | None = None) -> str:
    """
    Download a model from Hugging Face Hub.

    Args:
        model_name: Name of the model to download
        cache_dir: Directory to cache the model

    Returns:
        Path to the downloaded model
    """
    from transformers import AutoModel, AutoTokenizer

    try:
        # Download tokenizer
        AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)

        # Download model (this will cache it)
        AutoModel.from_pretrained(model_name, cache_dir=cache_dir)

        logger.info("Successfully downloaded model: %s", model_name)

        # Return the cache path
        folder = "models--" + model_name.replace("/", "--")
        if cache_dir:
            return str(Path(cache_dir) / folder)
        return str(Path("~/.cache/huggingface/hub").expanduser() / folder)

    except Exception as e:
        logger.error("Error downloading model %s: %s", model_name, e)
        raise


def evaluate_classification(
    y_true: list[Any], y_pred: list[Any], labels: list[Any] | None = None
) -> dict[str, Any]:
    """
    Evaluate classification performance.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: List of label names

    Returns:
        Dictionary with evaluation metrics
    """
    metrics = {}

    # Basic metrics
    metrics["accuracy"] = accuracy_score(y_true, y_pred)

    # Precision, recall, F1
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    metrics["precision"] = precision
    metrics["recall"] = recall
    metrics["f1"] = f1
    metrics["support"] = support

    # Per-class metrics
    if labels:
        # Pass labels explicitly so per-class scores line up with the given label order
        per_class = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, average=None, zero_division=0
        )
        metrics["per_class"] = {
            label: {
                "precision": per_class[0][i],
                "recall": per_class[1][i],
                "f1": per_class[2][i],
                "support": per_class[3][i],
            }
            for i, label in enumerate(labels)
        }

    # Confusion matrix
    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=labels).tolist()

    # Classification report
    metrics["classification_report"] = classification_report(
        y_true, y_pred, labels=labels, output_dict=True, zero_division=0
    )

    return metrics


def evaluate_ner(
    y_true: list[list[dict[str, Any]]], y_pred: list[list[dict[str, Any]]]
) -> dict[str, Any]:
    """
    Evaluate NER performance.

    Args:
        y_true: True entities for each text
        y_pred: Predicted entities for each text

    Returns:
        Dictionary with NER evaluation metrics
    """
    # This is a simplified NER evaluation
    # In practice, you'd want more sophisticated entity-level evaluation

    total_true = sum(len(entities) for entities in y_true)
    total_pred = sum(len(entities) for entities in y_pred)

    # Simple overlap-based evaluation
    correct = 0
    for true_entities, pred_entities in zip(y_true, y_pred, strict=False):
        for true_entity in true_entities:
            for pred_entity in pred_entities:
                if (
                    true_entity.get("entity") == pred_entity.get("entity")
                    and true_entity.get("start") == pred_entity.get("start")
                    and true_entity.get("end") == pred_entity.get("end")
                ):
                    correct += 1
                    break

    precision = correct / total_pred if total_pred > 0 else 0
    recall = correct / total_true if total_true > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "total_true": total_true,
        "total_pred": total_pred,
        "correct": correct,
    }


def batch_texts(texts: list[str], batch_size: int = 32) -> list[list[str]]:
    """
    Split texts into batches.

    Args:
        texts: List of texts to batch
        batch_size: Size of each batch

    Returns:
        List of text batches
    """
    return [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]


def chunk_text(
    text: str, max_length: int = 512, overlap: int = 50, separator: str = "\n"
) -> list[str]:
    """
    Split long text into overlapping chunks.

    Args:
        text: Text to chunk
        max_length: Maximum length of each chunk
        overlap: Number of characters to overlap between chunks
        separator: Character to use for splitting

    Returns:
        List of text chunks
    """
    if max_length < 1:
        raise ValueError("max_length must be >= 1")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if len(text) <= max_length:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = start + max_length

        if end < len(text):
            # Try to find a good break point
            break_point = text.rfind(separator, start, end)
            if break_point > start:
                end = break_point + 1

        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        # Always move forward, even when a separator shortened the chunk below `overlap`
        start = max(end - overlap, start + 1)

    return chunks


def clean_text(text: str) -> str:
    """
    Basic text cleaning.

    Args:
        text: Text to clean

    Returns:
        Cleaned text
    """
    # Normalize curly quotes to their ASCII equivalents (before stripping symbols,
    # otherwise they would be removed instead of normalized)
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2018", "'").replace("\u2019", "'")

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text)

    # Remove special characters (keep alphanumeric, spaces, and basic punctuation)
    text = re.sub(r'[^\w\s.,!?;:()"\'-]', "", text)

    return text.strip()


def get_text_statistics(text: str) -> dict[str, Any]:
    """
    Get basic statistics about a text.

    Args:
        text: Text to analyze

    Returns:
        Dictionary with text statistics
    """
    words = text.split()
    sentences = text.split(".")

    # Filter out empty sentences
    sentences = [s.strip() for s in sentences if s.strip()]

    # Count characters (excluding spaces)
    char_count = len(text.replace(" ", ""))

    # Count syllables (approximate: vowel groups, at least one per word)
    syllable_count = sum(
        max(1, len(re.findall(r"[aeiouy]+", word.lower()))) for word in words if word
    )

    # Calculate readability scores
    avg_sentence_length = len(words) / len(sentences) if sentences else 0
    avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
    avg_syllables_per_word = syllable_count / len(words) if words else 0

    # Flesch Reading Ease (Flesch, 1948), clamped to 0-100
    flesch_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
    flesch_score = max(0.0, min(100.0, flesch_score)) if words else 0.0

    return {
        "characters": len(text),
        "characters_no_spaces": char_count,
        "words": len(words),
        "sentences": len(sentences),
        "syllables": syllable_count,
        "avg_sentence_length": round(avg_sentence_length, 2),
        "avg_word_length": round(avg_word_length, 2),
        "flesch_reading_ease": round(flesch_score, 2),
        "reading_level": _get_reading_level(flesch_score),
    }


def _get_reading_level(flesch_score: float) -> str:
    """Get reading level based on Flesch score."""
    if flesch_score >= 90:
        return "Very Easy"
    if flesch_score >= 80:
        return "Easy"
    if flesch_score >= 70:
        return "Fairly Easy"
    if flesch_score >= 60:
        return "Standard"
    if flesch_score >= 50:
        return "Fairly Difficult"
    if flesch_score >= 30:
        return "Difficult"
    return "Very Difficult"


def download_file(
    url: str, filepath: str | Path, show_progress: bool = True, timeout: float = 30.0
) -> str:
    """
    Download a file from URL with progress bar.

    Args:
        url: URL to download from
        filepath: Local path to save the file
        show_progress: Whether to show download progress
        timeout: Seconds to wait for the server to connect or send data

    Returns:
        Path to the downloaded file
    """
    filepath = Path(filepath)
    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        total_size = int(response.headers.get("content-length", 0))

        with filepath.open("wb") as f:
            if show_progress and total_size > 0:
                with tqdm(total=total_size, unit="B", unit_scale=True) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))
            else:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

    return str(filepath)
