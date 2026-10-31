"""
Text preprocessing module for Lingo NLP toolkit.
Handles text normalization, cleaning, and tokenization.
"""

import json
import logging
import re
import unicodedata
from pathlib import Path
from typing import Any, overload

import nltk
import spacy
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize

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
        config: dict[str, Any] | None = None,
        device: str = "cpu",
        language: str = "english",
    ) -> None:
        self.config = config or {}
        self.device = device
        self.language = language

        # Initialize NLTK components
        self._init_nltk()

        # Initialize spaCy for advanced NLP
        self.spacy_model: Any = None
        if self.config.get("use_spacy", True):
            self._init_spacy()

        # Initialize tokenizers
        self._init_tokenizers()

        # Load stopwords
        self.stopwords: set[str] = set()
        if self.config.get("remove_stopwords", False):
            self._load_stopwords()

        # Initialize lemmatizer and stemmer
        self.lemmatizer = None
        self.stemmer = None
        if self.config.get("lemmatize", False):
            self.lemmatizer = WordNetLemmatizer()
        if self.config.get("stem", False):
            self.stemmer = PorterStemmer()

    def _init_nltk(self) -> None:
        """Initialize NLTK components and download required data."""
        try:
            # Try to find required NLTK data
            from .setup_utils import NLTK_RESOURCES

            for package, resource_path in NLTK_RESOURCES.items():
                try:
                    nltk.data.find(resource_path)
                except LookupError:
                    logger.info("Downloading NLTK package: %s", package)
                    nltk.download(package, quiet=True)

        except Exception as e:
            logger.warning("Failed to initialize NLTK: %s", e)
            logger.warning("Some NLTK features may not work. Run 'lingo setup' to fix this.")

    def _init_spacy(self) -> None:
        """Initialize spaCy model."""
        model_name = self.config.get("spacy_model", "en_core_web_sm")
        try:
            self.spacy_model = spacy.load(model_name)
            logger.info("Loaded spaCy model: %s", model_name)
        except OSError:
            logger.warning(
                f"spaCy model {model_name} not found. Install with: python -m spacy download {model_name}"
            )
            self.spacy_model = None

    def _init_tokenizers(self) -> None:
        """Initialize tokenizers."""
        self.word_tokenizer = word_tokenize
        self.sent_tokenizer = sent_tokenize

        # Initialize subword tokenizers if specified
        self.subword_tokenizer = None
        if self.config.get("use_subword_tokenization", False):
            logger.warning(
                "use_subword_tokenization is not implemented yet; "
                "use a model tokenizer from transformers instead"
            )

    def _load_stopwords(self) -> None:
        """Load stopwords for the specified language."""
        try:
            self.stopwords = set(stopwords.words(self.language))
        except (LookupError, OSError):  # OSError: corpus present but language file missing
            logger.warning("Stopwords for language '%s' not found", self.language)
            self.stopwords = set()

    @overload
    def __call__(self, texts: str) -> str: ...

    @overload
    def __call__(self, texts: list[str]) -> list[str]: ...

    def __call__(self, texts: str | list[str]) -> str | list[str]:
        """
        Apply preprocessing pipeline to input text(s).

        Args:
            texts: Single text or list of texts to preprocess

        Returns:
            Preprocessed text(s)
        """
        if isinstance(texts, str):
            return self.preprocess_text(texts)
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
        # Irregular forms first, otherwise "won't" would become "wo not"
        irregular = {"won't": "will not", "can't": "cannot", "shan't": "shall not"}
        for contraction, expansion in irregular.items():
            text = re.sub(re.escape(contraction), expansion, text, flags=re.IGNORECASE)

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

    def tokenize_words(self, text: str) -> list[str]:
        """Tokenize text into words."""
        if self.spacy_model:
            doc = self.spacy_model(text)
            return [token.text for token in doc if not token.is_space]
        return self.word_tokenizer(text)

    def tokenize_sentences(self, text: str) -> list[str]:
        """Tokenize text into sentences."""
        return self.sent_tokenizer(text)

    def remove_stopwords(self, tokens: list[str]) -> list[str]:
        """Remove stopwords from token list."""
        if not self.stopwords:
            return tokens
        return [token for token in tokens if token.lower() not in self.stopwords]

    def lemmatize_tokens(self, tokens: list[str]) -> list[str]:
        """Lemmatize tokens."""
        if not self.lemmatizer:
            return tokens
        return [self.lemmatizer.lemmatize(token) for token in tokens]

    def stem_tokens(self, tokens: list[str]) -> list[str]:
        """Stem tokens."""
        if not self.stemmer:
            return tokens
        return [self.stemmer.stem(token) for token in tokens]

    def get_preprocessing_pipeline(self, text: str) -> dict[str, Any]:
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

    def save(self, path: str | Path) -> None:
        """Save preprocessor configuration."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        config = {
            "config": self.config,
            "language": self.language,
            "device": self.device,
        }

        with (path / "preprocessor_config.json").open("w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

        logger.info("Preprocessor saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "TextPreprocessor":
        """Load preprocessor from disk."""
        path = Path(path)

        with (path / "preprocessor_config.json").open(encoding="utf-8") as f:
            config = json.load(f)

        preprocessor = cls(
            config=config["config"],
            device=config["device"],
            language=config["language"],
        )

        logger.info("Preprocessor loaded from %s", path)
        return preprocessor
