# 🚀 Lingo Installation Guide

## **One-Command Installation**

Lingo is designed to be **install-and-use** - no manual setup required! When you install Lingo, it automatically downloads and configures all the necessary NLP data and models.

## **Quick Installation**

### **Option 1: Install from PyPI (Recommended)**

```bash
pip install lingo-nlp-toolkit
```

**📦 PyPI Package**: [lingo-nlp-toolkit on PyPI](https://pypi.org/project/lingo-nlp-toolkit/#files)

### **Option 2: Install from Source**

```bash
git clone https://github.com/irfanalidv/lingo-nlp-toolkit.git
cd lingo-nlp-toolkit
pip install -e .
```

## **What Happens During Installation**

When you install Lingo, it automatically:

1. ✅ **Downloads spaCy English model** (`en_core_web_sm`)
2. ✅ **Downloads NLTK data** (punkt, stopwords, wordnet, punkt_tab)
3. ✅ **Pre-downloads essential transformer models** (BERT, RoBERTa, etc.)
4. ✅ **Sets up caching directories** for optimal performance
5. ✅ **Verifies all components** are working correctly

## **First-Time Usage**

### **Automatic Setup (Recommended)**

Simply import Lingo and it will set up everything automatically:

```python
from lingo import Pipeline

# This will trigger auto-setup if needed
nlp = Pipeline(task="sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")
```

### **Manual Setup (Optional)**

If you prefer to set up manually or want to re-download data:

```bash
# Set up the environment
lingo setup

# Force re-download of all data
lingo setup --force
```

## **What Gets Downloaded**

### **spaCy Models**

- `en_core_web_sm` - English language model for text processing

### **NLTK Data**

- `punkt` - Sentence and word tokenization
- `stopwords` - Common stop words
- `wordnet` - Lexical database for lemmatization
- `punkt_tab` - Advanced tokenization rules

### **Transformer Models**

- `bert-base-uncased` - Base BERT model
- `distilbert-base-uncased` - Distilled BERT model
- `cardiffnlp/twitter-roberta-base-sentiment-latest` - Sentiment analysis
- `dslim/bert-base-NER` - Named Entity Recognition

## **Installation Verification**

After installation, verify everything is working:

```bash
# Check if Lingo is installed
python -c "import lingo; print('✅ Lingo installed successfully!')"

# Run the demo
python demo.py

# Test CLI
lingo list-models
```

## **Troubleshooting**

### **Common Issues**

#### **1. Out of Disk Space**

Lingo downloads several GB of models. Ensure you have at least 5GB free space.

#### **2. Network Issues**

If downloads fail, run the setup manually:

```bash
lingo setup
```

#### **3. Permission Issues**

On some systems, you might need:

```bash
pip install --user lingo
```

#### **4. Missing Dependencies**

If you get import errors:

```bash
pip install --upgrade pip
pip install lingo-nlp-toolkit[full]
```

### **Manual Data Downloads**

If auto-setup fails, you can manually download:

```bash
# Download spaCy model
python -m spacy download en_core_web_sm

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('punkt_tab')"
```

## **Environment Variables**

You can customize the installation:

```bash
# Custom cache directory
export LINGO_CACHE_DIR="/path/to/custom/cache"
pip install lingo-nlp-toolkit

# Skip auto-setup (not recommended)
export LINGO_SKIP_SETUP=1
pip install lingo-nlp-toolkit
```

## **Development Installation**

For development work:

```bash
git clone https://github.com/irfanalidv/lingo-nlp-toolkit.git
cd lingo-nlp-toolkit

# Install with development dependencies
pip install -e .[dev]

# Set up environment
lingo setup

# Run tests
make test
```

## **System Requirements**

### **Minimum Requirements**

- **Python**: 3.9 or higher
- **RAM**: 4GB minimum, 8GB recommended
- **Disk**: 5GB free space
- **OS**: Windows 10+, macOS 10.14+, or Linux

### **Recommended Requirements**

- **Python**: 3.11 or higher
- **RAM**: 16GB or more
- **Disk**: 10GB free space
- **GPU**: CUDA-compatible GPU (optional, for faster inference)

## **Performance Optimization**

### **GPU Support**

```bash
# Install with GPU support
pip install lingo-nlp-toolkit[gpu]

# Or install PyTorch with CUDA manually
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### **Memory Optimization**

```bash
# Use smaller models for limited memory
lingo run sentiment-analysis --model distilbert-base-uncased --text "Your text here"
```

## **Uninstallation**

To completely remove Lingo:

```bash
# Remove the package
pip uninstall lingo-nlp-toolkit

# Remove cached data (optional)
rm -rf ~/.lingo_cache
rm -rf ~/.cache/huggingface
```

## **Support**

If you encounter issues:

1. **Check the logs**: Look for error messages during installation
2. **Run setup manually**: `lingo setup --force`
3. **Check disk space**: Ensure you have enough free space
4. **Check network**: Ensure you can access the internet
5. **Open an issue**: [GitHub Issues](https://github.com/irfanalidv/lingo-nlp-toolkit/issues)

## **Next Steps**

After successful installation:

1. **Try the demo**: `python demo.py`
2. **Run examples**: `python examples/basic_usage.py`
3. **Use the CLI**: `lingo list-models`
4. **Build your first pipeline**: Check the [Quickstart Guide](QUICKSTART.md)

---

**🎉 You're all set! Lingo will handle everything automatically from here on out.**
