"""
Setup utilities for Lingo NLP toolkit.
Automatically downloads and installs required NLP data.
"""

import os
import sys
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def download_spacy_model(model_name: str = "en_core_web_sm"):
    """Download spaCy model if not already present."""
    try:
        import spacy

        # Check if model is already installed
        try:
            spacy.load(model_name)
            logger.info(f"spaCy model '{model_name}' is already available")
            return True
        except OSError:
            pass

        # Download the model
        logger.info(f"Downloading spaCy model: {model_name}")
        os.system(f"python -m spacy download {model_name}")

        # Verify download
        try:
            spacy.load(model_name)
            logger.info(f"Successfully downloaded spaCy model: {model_name}")
            return True
        except OSError as e:
            logger.error(f"Failed to download spaCy model: {e}")
            return False

    except ImportError:
        logger.warning("spaCy not available, skipping model download")
        return False


def download_nltk_data():
    """Download required NLTK data."""
    try:
        import nltk

        required_packages = ["punkt", "stopwords", "wordnet", "punkt_tab"]

        for package in required_packages:
            try:
                nltk.data.find(f"tokenizers/{package}")
                logger.info(f"NLTK package '{package}' is already available")
            except LookupError:
                logger.info(f"Downloading NLTK package: {package}")
                nltk.download(package, quiet=True)
                logger.info(f"Successfully downloaded NLTK package: {package}")

        return True

    except ImportError:
        logger.warning("NLTK not available, skipping data download")
        return False


def download_transformers_data():
    """Download and cache common transformer models."""
    try:
        from transformers import AutoTokenizer, AutoModel

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
                logger.info(f"Downloading model: {model_name}")
                # Download tokenizer
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                # Download model
                model = AutoModel.from_pretrained(model_name)
                logger.info(f"Successfully downloaded: {model_name}")
            except Exception as e:
                logger.warning(f"Failed to download {model_name}: {e}")

        return True

    except ImportError:
        logger.warning("Transformers not available, skipping model download")
        return False


def setup_lingo_environment():
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
    with open(completion_file, "w") as f:
        f.write("Lingo setup completed successfully!\n")
        f.write(f"spaCy: {'✓' if spacy_success else '✗'}\n")
        f.write(f"NLTK: {'✓' if nltk_success else '✗'}\n")
        f.write(f"Transformers: {'✓' if transformers_success else '✗'}\n")

    logger.info("Lingo environment setup completed!")
    return True


def check_lingo_environment():
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
