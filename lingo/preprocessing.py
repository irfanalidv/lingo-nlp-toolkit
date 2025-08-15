"""
Text preprocessing module for Lingo NLP toolkit.
Handles text normalization, cleaning, and tokenization.
"""

import re
import logging
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import json
import unicodedata

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize
import spacy

logger = logging.getLogger(__name__)


class TextPreprocessor:
    """
    Comprehensive text preprocessing pipeline.

    Features:
    - Unicode normalization
    - Text cleaning and normalization
    - Stopword removal
    - Lemmatization and stemming
    - Tokenization (word, sentence, subword)
    - Spell correction
    - Slang expansion
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        device: str = "cpu",
        language: str = "english",
    ):
        self.config = config or {}
        self.device = device
        self.language = language

        # Initialize NLTK components
        self._init_nltk()

        # Initialize spaCy for advanced NLP
        self.spacy_model = None
        if self.config.get("use_spacy", True):
            self._init_spacy()

        # Initialize tokenizers
        self._init_tokenizers()

        # Load stopwords
        self.stopwords = set()
        if self.config.get("remove_stopwords", False):
            self._load_stopwords()

        # Initialize lemmatizer and stemmer
        self.lemmatizer = None
        self.stemmer = None
        if self.config.get("lemmatize", False):
            self.lemmatizer = WordNetLemmatizer()
        if self.config.get("stem", False):
            self.stemmer = PorterStemmer()

    def _init_nltk(self):
        """Initialize NLTK components and download required data."""
        try:
            # Try to find required NLTK data
            required_packages = ["punkt", "stopwords", "wordnet", "punkt_tab"]

            for package in required_packages:
                try:
                    if package in ["punkt", "punkt_tab"]:
                        nltk.data.find(f"tokenizers/{package}")
                    else:
                        nltk.data.find(f"corpora/{package}")
                except LookupError:
                    logger.info(f"Downloading NLTK package: {package}")
                    nltk.download(package, quiet=True)

        except Exception as e:
            logger.warning(f"Failed to initialize NLTK: {e}")
            logger.warning(
                "Some NLTK features may not work. Run 'lingo setup' to fix this."
            )

    def _init_spacy(self):
        """Initialize spaCy model."""
        try:
            model_name = self.config.get("spacy_model", "en_core_web_sm")
            self.spacy_model = spacy.load(model_name)
            logger.info(f"Loaded spaCy model: {model_name}")
        except OSError:
            logger.warning(
                f"spaCy model {model_name} not found. Install with: python -m spacy download {model_name}"
            )
            self.spacy_model = None

    def _init_tokenizers(self):
        """Initialize tokenizers."""
        self.word_tokenizer = word_tokenize
        self.sent_tokenizer = sent_tokenize

        # Initialize subword tokenizers if specified
        self.subword_tokenizer = None
        if self.config.get("use_subword_tokenization", False):
            try:
                from tokenizers import Tokenizer

                # This would be configured based on the specific model
                pass
            except ImportError:
                logger.warning(
                    "tokenizers library not available for subword tokenization"
                )

    def _load_stopwords(self):
        """Load stopwords for the specified language."""
        try:
            self.stopwords = set(stopwords.words(self.language))
        except LookupError:
            logger.warning(f"Stopwords for language '{self.language}' not found")
            self.stopwords = set()

    def __call__(self, texts: Union[str, List[str]]) -> Union[str, List[str]]:
        """
        Apply preprocessing pipeline to input text(s).

        Args:
            texts: Single text or list of texts to preprocess

        Returns:
            Preprocessed text(s)
        """
        if isinstance(texts, str):
            return self.preprocess_text(texts)
        else:
            return [self.preprocess_text(text) for text in texts]

    def preprocess_text(self, text: str) -> str:
        """
        Apply complete preprocessing pipeline to a single text.

        Args:
            text: Input text to preprocess

        Returns:
            Preprocessed text
        """
        if not text or not isinstance(text, str):
            return text

        # Apply preprocessing steps based on configuration
        if self.config.get("normalize_unicode", True):
            text = self.normalize_unicode(text)

        if self.config.get("lowercase", True):
            text = text.lower()

        if self.config.get("remove_punctuation", False):
            text = self.remove_punctuation(text)

        if self.config.get("remove_numbers", False):
            text = self.remove_numbers(text)

        if self.config.get("remove_special_chars", False):
            text = self.remove_special_chars(text)

        if self.config.get("expand_contractions", False):
            text = self.expand_contractions(text)

        if self.config.get("correct_spelling", False):
            text = self.correct_spelling(text)

        if self.config.get("expand_slang", False):
            text = self.expand_slang(text)

        if self.config.get("remove_extra_whitespace", True):
            text = self.remove_extra_whitespace(text)

        return text.strip()

    def normalize_unicode(self, text: str) -> str:
        """Normalize Unicode characters."""
        return unicodedata.normalize("NFC", text)

    def remove_punctuation(self, text: str) -> str:
        """Remove punctuation marks."""
        return re.sub(r"[^\w\s]", "", text)

    def remove_numbers(self, text: str) -> str:
        """Remove numbers from text."""
        return re.sub(r"\d+", "", text)

    def remove_special_chars(self, text: str) -> str:
        """Remove special characters."""
        return re.sub(r"[^a-zA-Z\s]", "", text)

    def expand_contractions(self, text: str) -> str:
        """Expand common contractions."""
        contractions = {
            "n't": " not",
            "'ll": " will",
            "'re": " are",
            "'ve": " have",
            "'m": " am",
            "'d": " would",
            "'s": " is",  # Note: this is simplified
        }

        for contraction, expansion in contractions.items():
            text = text.replace(contraction, expansion)

        return text

    def correct_spelling(self, text: str) -> str:
        """Basic spell correction (placeholder for more advanced methods)."""
        # This is a placeholder - in practice, you'd use libraries like
        # language-tool-python, pyspellchecker, or custom models
        return text

    def expand_slang(self, text: str) -> str:
        """Expand common slang terms."""
        slang_dict = {
            "lol": "laugh out loud",
            "omg": "oh my god",
            "btw": "by the way",
            "imo": "in my opinion",
            "fyi": "for your information",
        }

        words = text.split()
        expanded_words = [slang_dict.get(word.lower(), word) for word in words]
        return " ".join(expanded_words)

    def remove_extra_whitespace(self, text: str) -> str:
        """Remove extra whitespace."""
        return re.sub(r"\s+", " ", text)

    def tokenize_words(self, text: str) -> List[str]:
        """Tokenize text into words."""
        if self.spacy_model:
            doc = self.spacy_model(text)
            return [token.text for token in doc if not token.is_space]
        else:
            return self.word_tokenizer(text)

    def tokenize_sentences(self, text: str) -> List[str]:
        """Tokenize text into sentences."""
        return self.sent_tokenizer(text)

    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """Remove stopwords from token list."""
        if not self.stopwords:
            return tokens
        return [token for token in tokens if token.lower() not in self.stopwords]

    def lemmatize_tokens(self, tokens: List[str]) -> List[str]:
        """Lemmatize tokens."""
        if not self.lemmatizer:
            return tokens
        return [self.lemmatizer.lemmatize(token) for token in tokens]

    def stem_tokens(self, tokens: List[str]) -> List[str]:
        """Stem tokens."""
        if not self.stemmer:
            return tokens
        return [self.stemmer.stem(token) for token in tokens]

    def get_preprocessing_pipeline(self, text: str) -> Dict[str, Any]:
        """
        Get complete preprocessing pipeline results.

        Args:
            text: Input text

        Returns:
            Dictionary with all preprocessing steps
        """
        original = text
        normalized = self.normalize_unicode(text)
        cleaned = self.preprocess_text(text)
        words = self.tokenize_words(cleaned)
        sentences = self.tokenize_sentences(original)

        result = {
            "original": original,
            "normalized": normalized,
            "cleaned": cleaned,
            "words": words,
            "sentences": sentences,
            "word_count": len(words),
            "sentence_count": len(sentences),
        }

        if self.stopwords:
            result["words_no_stopwords"] = self.remove_stopwords(words)

        if self.lemmatizer:
            result["lemmatized"] = self.lemmatize_tokens(words)

        if self.stemmer:
            result["stemmed"] = self.stem_tokens(words)

        return result

    def save(self, path: Union[str, Path]):
        """Save preprocessor configuration."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        config = {
            "config": self.config,
            "language": self.language,
            "device": self.device,
        }

        with open(path / "preprocessor_config.json", "w") as f:
            json.dump(config, f, indent=2)

        logger.info(f"Preprocessor saved to {path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "TextPreprocessor":
        """Load preprocessor from disk."""
        path = Path(path)

        with open(path / "preprocessor_config.json", "r") as f:
            config = json.load(f)

        preprocessor = cls(
            config=config["config"],
            device=config["device"],
            language=config["language"],
        )

        logger.info(f"Preprocessor loaded from {path}")
        return preprocessor
