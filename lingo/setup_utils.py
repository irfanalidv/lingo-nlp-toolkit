"""
Setup utilities for Lingo NLP toolkit.
Automatically downloads and installs required NLP data.
"""

import logging
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


# NLTK resources used by lingo, mapped to their location inside nltk_data
NLTK_RESOURCES = {
    "punkt": "tokenizers/punkt",
    "punkt_tab": "tokenizers/punkt_tab",
    "stopwords": "corpora/stopwords",
    "wordnet": "corpora/wordnet",
}


def download_spacy_model(model_name: str = "en_core_web_sm") -> bool:
    """Download spaCy model if not already present."""
    try:
        import spacy

        # Check if model is already installed
        try:
            spacy.load(model_name)
            logger.info("spaCy model '%s' is already available", model_name)
            return True
        except OSError:
            pass

        # Download the model
        logger.info("Downloading spaCy model: %s", model_name)
        subprocess.run(
            [sys.executable, "-m", "spacy", "download", model_name],
            check=False,
        )

        # Verify download
        try:
            spacy.load(model_name)
            logger.info("Successfully downloaded spaCy model: %s", model_name)
            return True
        except OSError as e:
            logger.error("Failed to download spaCy model: %s", e)
            return False

    except ImportError:
        logger.warning("spaCy not available, skipping model download")
        return False


def download_nltk_data() -> bool:
    """Download required NLTK data."""
    try:
        import nltk

        for package, resource_path in NLTK_RESOURCES.items():
            try:
                nltk.data.find(resource_path)
                logger.info("NLTK package '%s' is already available", package)
            except LookupError:
                logger.info("Downloading NLTK package: %s", package)
                nltk.download(package, quiet=True)
                logger.info("Successfully downloaded NLTK package: %s", package)

        return True

    except ImportError:
        logger.warning("NLTK not available, skipping data download")
        return False


def download_transformers_data() -> bool:
    """Download and cache common transformer models."""
    try:
        from transformers import AutoModel, AutoTokenizer

        # List of essential models to pre-download
        essential_models = [
            "bert-base-uncased",
            "distilbert-base-uncased",
            "cardiffnlp/twitter-roberta-base-sentiment-latest",
            "dslim/bert-base-NER",
        ]

        logger.info("Pre-downloading essential transformer models...")

        for model_name in essential_models:
            try:
                logger.info("Downloading model: %s", model_name)
                AutoTokenizer.from_pretrained(model_name)
                AutoModel.from_pretrained(model_name)
                logger.info("Successfully downloaded: %s", model_name)
            except Exception as e:
                logger.warning("Failed to download %s: %s", model_name, e)

        return True

    except ImportError:
        logger.warning("Transformers not available, skipping model download")
        return False


def setup_lingo_environment() -> bool:
    """Set up the complete Lingo environment."""
    logger.info("Setting up Lingo NLP environment...")

    # Create necessary directories
    cache_dir = Path.home() / ".lingo_cache"
    cache_dir.mkdir(exist_ok=True)

    # Download spaCy model
    spacy_success = download_spacy_model()

    # Download NLTK data
    nltk_success = download_nltk_data()

    # Download transformer models
    transformers_success = download_transformers_data()

    # Create a completion marker
    completion_file = cache_dir / "setup_complete.txt"
    with completion_file.open("w", encoding="utf-8") as f:
        f.write("Lingo setup finished.\n")
        f.write(f"spaCy: {'ok' if spacy_success else 'failed'}\n")
        f.write(f"NLTK: {'ok' if nltk_success else 'failed'}\n")
        f.write(f"Transformers: {'ok' if transformers_success else 'failed'}\n")

    logger.info("Lingo environment setup completed!")
    return spacy_success and nltk_success and transformers_success


def check_lingo_environment() -> bool:
    """Check if Lingo environment is properly set up."""
    cache_dir = Path.home() / ".lingo_cache"
    completion_file = cache_dir / "setup_complete.txt"

    if completion_file.exists():
        logger.info("Lingo environment appears to be set up")
        return True

    logger.info("Lingo environment not set up, running setup...")
    return setup_lingo_environment()


if __name__ == "__main__":
    # This can be run directly to set up the environment
    logging.basicConfig(level=logging.INFO)
    setup_lingo_environment()
