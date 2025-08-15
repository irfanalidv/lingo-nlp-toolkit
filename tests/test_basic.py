"""
Basic tests for Lingo NLP toolkit.
Tests core functionality without requiring heavy models.
"""

import pytest
import tempfile
import os
from pathlib import Path

from lingo import TextPreprocessor
from lingo.utils import clean_text, get_text_statistics, batch_texts, chunk_text


class TestTextPreprocessor:
    """Test TextPreprocessor class."""
    
    def test_basic_preprocessing(self):
        """Test basic text preprocessing."""
        preprocessor = TextPreprocessor()
        
        text = "Hello, World! This is a test."
        result = preprocessor(text)
        
        assert isinstance(result, str)
        assert "hello" in result.lower()
    
    def test_preprocessing_config(self):
        """Test preprocessing with custom configuration."""
        config = {
            "lowercase": True,
            "remove_punctuation": True,
            "remove_extra_whitespace": True
        }
        
        preprocessor = TextPreprocessor(config=config)
        
        text = "Hello, World! This is a test."
        result = preprocessor(text)
        
        assert "," not in result
        assert "!" not in result
        assert result == "hello world this is a test"
    
    def test_preprocessing_pipeline(self):
        """Test complete preprocessing pipeline."""
        config = {
            "lowercase": True,
            "remove_stopwords": True,
            "lemmatize": True
        }
        
        preprocessor = TextPreprocessor(config=config)
        
        text = "The quick brown foxes are jumping over the lazy dogs"
        result = preprocessor.get_preprocessing_pipeline(text)
        
        assert "original" in result
        assert "cleaned" in result
        assert "words" in result
        assert "lemmatized" in result
        assert isinstance(result["word_count"], int)
        assert isinstance(result["sentence_count"], int)
    
    def test_save_load(self):
        """Test saving and loading preprocessor."""
        config = {"lowercase": True, "remove_punctuation": True}
        preprocessor = TextPreprocessor(config=config)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "preprocessor"
            preprocessor.save(save_path)
            
            # Check if files were created
            assert (save_path / "preprocessor_config.json").exists()
            
            # Load preprocessor
            loaded_preprocessor = TextPreprocessor.load(save_path)
            
            # Test that configuration is preserved
            assert loaded_preprocessor.config["lowercase"] == config["lowercase"]
            assert loaded_preprocessor.config["remove_punctuation"] == config["remove_punctuation"]


class TestUtils:
    """Test utility functions."""
    
    def test_clean_text(self):
        """Test text cleaning utility."""
        text = "  Hello,   World!  "
        cleaned = clean_text(text)
        
        assert cleaned == "Hello, World!"
        assert len(cleaned.split()) == 2
    
    def test_get_text_statistics(self):
        """Test text statistics utility."""
        text = "Hello world. This is a test. It has multiple sentences."
        stats = get_text_statistics(text)
        
        assert "characters" in stats
        assert "words" in stats
        assert "sentences" in stats
        assert stats["sentences"] == 3
        assert stats["words"] == 10
        assert "reading_level" in stats
    
    def test_batch_texts(self):
        """Test text batching utility."""
        texts = ["text1", "text2", "text3", "text4", "text5"]
        batches = batch_texts(texts, batch_size=2)
        
        assert len(batches) == 3
        assert len(batches[0]) == 2
        assert len(batches[1]) == 2
        assert len(batches[2]) == 1
    
    def test_chunk_text(self):
        """Test text chunking utility."""
        text = "This is a long text that needs to be chunked into smaller pieces."
        chunks = chunk_text(text, max_length=20, overlap=5)
        
        assert len(chunks) > 1
        assert all(len(chunk) <= 20 for chunk in chunks)
        
        # Test with short text
        short_text = "Short text"
        chunks = chunk_text(short_text, max_length=20)
        assert len(chunks) == 1
        assert chunks[0] == short_text


class TestIntegration:
    """Test integration between components."""
    
    def test_preprocessor_with_clean_text(self):
        """Test preprocessor with clean_text utility."""
        text = "  Hello,   World!  "
        cleaned = clean_text(text)
        
        preprocessor = TextPreprocessor()
        processed = preprocessor(cleaned)
        
        assert isinstance(processed, str)
        assert len(processed) > 0
    
    def test_text_statistics_with_processed_text(self):
        """Test text statistics with preprocessed text."""
        text = "Hello world. This is a test."
        
        preprocessor = TextPreprocessor()
        processed = preprocessor(text)
        
        stats = get_text_statistics(processed)
        
        assert stats["words"] > 0
        assert stats["sentences"] > 0


if __name__ == "__main__":
    pytest.main([__file__])
