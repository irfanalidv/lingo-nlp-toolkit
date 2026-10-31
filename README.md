<div align="center">

# Lingo: Advanced NLP Toolkit

**One `Pipeline` API for sentiment, NER, classification, embeddings, question answering and summarization on Hugging Face models, plus spaCy/NLTK text preprocessing.**

[![PyPI version](https://img.shields.io/pypi/v/lingo-nlp-toolkit.svg)](https://pypi.org/project/lingo-nlp-toolkit/)
[![Python versions](https://img.shields.io/pypi/pyversions/lingo-nlp-toolkit.svg)](https://pypi.org/project/lingo-nlp-toolkit/)
[![Tests](https://github.com/irfanalidv/lingo-nlp-toolkit/actions/workflows/tests.yml/badge.svg)](https://github.com/irfanalidv/lingo-nlp-toolkit/actions/workflows/tests.yml)
[![PyPI Downloads](https://static.pepy.tech/badge/lingo-nlp-toolkit)](https://pepy.tech/projects/lingo-nlp-toolkit)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/irfanalidv/lingo-nlp-toolkit/blob/main/LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

</div>

## **Overview**

Lingo is a modern, high-performance **Natural Language Processing (NLP) toolkit** designed for researchers, data scientists, and developers building intelligent language-powered applications. It combines **ease of use**, **speed**, and **state-of-the-art transformer capabilities**, offering an end-to-end pipeline — from text preprocessing to advanced contextual understanding.

Lingo bridges the gap between traditional NLP techniques (spaCy, NLTK) and transformer models from the Hugging Face Hub (BERT, RoBERTa, BART, sentence-transformers, ...) behind a single, consistent API.

### **Why Lingo?**

- **One interface for six tasks**: `Pipeline(task=..., model=...)` works the same way for every task, accepts a string or a list, and returns one result per input.
- **Any Hub model**: pass any compatible model id or local path; Lingo handles tokenizer/model loading and device placement (`auto` picks CUDA, Apple MPS or CPU).
- **Works on transformers 4.x and 5.x**: question answering and summarization are implemented directly on the model classes, so they keep working after transformers 5 removed those pipelines.
- **Classic preprocessing included**: Unicode normalization, cleaning, stopwords, lemmatization and stemming via `TextPreprocessor`.
- **Save/load and CLI**: persist a pipeline (weights + config) to disk and run any task from the `lingo` command.

---

## **🚀 Quick Start**

### **Installation**

```bash
# One-command installation (recommended)
pip install lingo-nlp-toolkit

# Optional extras
pip install "lingo-nlp-toolkit[api]"       # FastAPI + uvicorn for serving a pipeline
pip install "lingo-nlp-toolkit[examples]"  # pandas, used by the scripts in examples/
pip install "lingo-nlp-toolkit[full]"      # api + examples + document/visualization helpers
pip install "lingo-nlp-toolkit[gpu]"       # torchvision / torchaudio
pip install "lingo-nlp-toolkit[dev]"       # pytest, ruff, build, twine

# Optional: spaCy English model (used by TextPreprocessor for word tokenization if present)
python -m spacy download en_core_web_sm
```

**Auto-setup**: required NLTK data is downloaded on first use, and Hugging Face models are downloaded and cached the first time you load them. Run `lingo setup` to fetch everything up front.

Requires Python 3.10+.

**📦 PyPI Package**: [lingo-nlp-toolkit on PyPI](https://pypi.org/project/lingo-nlp-toolkit/#files)

### **Examples & Use Cases**

```bash
# Basic usage
python examples/basic_usage.py

# Advanced real-world applications
python examples/advanced_use_cases.py

# Enterprise-grade NLP workflows
python examples/enterprise_nlp.py

# Capability showcase
python examples/showcase.py

# Interactive demo
python demo.py
```

### **First Steps**

```python
from lingo import Pipeline

# Create a sentiment analysis pipeline
nlp = Pipeline(task="sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")

# Run inference
text = "I absolutely love the new product update!"
result = nlp(text)
print(result)
# Output: {'label': 'positive', 'score': 0.99...}
```

### **Command Line Usage**

```bash
# Sentiment analysis
lingo run sentiment-analysis --model cardiffnlp/twitter-roberta-base-sentiment-latest --text "I love this product!"
# {"label": "positive", "score": 0.98...}

# Question answering (the input text is the context)
lingo run question-answering --model deepset/roberta-base-squad2 \
  --question "Who created Python?" --text "Python was created by Guido van Rossum in 1991."
# {"score": 0.97..., "start": 22, "end": 38, "answer": "Guido van Rossum"}

# List available models
lingo list-models

# Download a model
lingo download-model --model bert-base-uncased

# Show the installed version
lingo --version
```

---

## **✨ Key Features**

### **1. Text Preprocessing & Normalization**

- ✅ Unicode normalization (NFC/NFD)
- ✅ Lowercasing, punctuation removal, special character stripping
- ✅ Stopword removal (multi-language support)
- ✅ Lemmatization & stemming
- ✅ Word & sentence tokenization (spaCy when available, otherwise NLTK)
- ✅ Contraction and slang expansion

### **2. Core NLP Tasks**

- ✅ **Text Classification** - any sequence-classification model
- ✅ **Named Entity Recognition (NER)** - token classification, with optional entity grouping
- ✅ **Sentiment Analysis** - binary, ternary or fine-grained, depending on the model
- ✅ **Text Embeddings** - sentence-transformers models, plus cosine similarity
- ✅ **Question Answering** - extractive, with long contexts split into overlapping windows
- ✅ **Text Summarization** - abstractive (BART, T5, Pegasus, ...)

### **3. Hugging Face Integration**

- ✅ Load any model from the Hugging Face Hub or a local directory
- ✅ Extra keyword arguments are forwarded to the underlying Hugging Face pipeline
- ✅ Save a pipeline (weights + config) and load it back offline

### **4. Performance**

- ✅ Automatic device selection: CUDA, Apple Silicon (MPS) or CPU
- ✅ Batched inference with `batch_predict`
- ✅ Evaluation helpers for classification and NER

---

## **📚 Comprehensive Examples**

### **Text Classification**

```python
from lingo import Pipeline

# Create classifier
classifier = Pipeline(task="text-classification", model="bert-base-uncased")

# Classify texts
texts = [
    "This is a positive review about the product.",
    "I'm not satisfied with the service quality.",
    "The product meets my expectations.",
]

results = classifier(texts)
for text, result in zip(texts, results):
    print(f"{text[:30]}... → {result['label']}")

# Note: bert-base-uncased has no fine-tuned classification head, so its labels
# (LABEL_0/LABEL_1) are not meaningful. Use a fine-tuned model for real predictions,
# e.g. "distilbert-base-uncased-finetuned-sst-2-english".
```

### **Named Entity Recognition**

```python
# Create NER pipeline
ner = Pipeline(task="ner", model="dslim/bert-base-NER")

# Extract entities
text = "Apple Inc. is headquartered in Cupertino, California. Tim Cook is the CEO."
entities = ner(text)

for entity in entities:
    print(f"Entity: {entity['entity']}, Type: {entity['word']}, Score: {entity['score']:.3f}")

# Output:
# Entity: B-LOC, Type: cup, Score: 0.940
# Entity: B-LOC, Type: ##ert, Score: 0.671
# Entity: I-LOC, Type: ##ino, Score: 0.437
# Entity: B-LOC, Type: ca, Score: 0.506
```

### **Sentiment Analysis**

```python
# Create sentiment analyzer
sentiment = Pipeline(
    task="sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest"
)

# Analyze sentiment
texts = [
    "I love this amazing product!",
    "This is terrible, worst purchase ever.",
    "It's okay, nothing special.",
]

results = sentiment(texts)
for text, result in zip(texts, results):
    print(f"{text[:30]}... → {result['label']} ({result['score']:.3f})")

# Output:
# I love this amazing product!... → positive (0.986)
# This is terrible, worst purcha... → negative (0.941)
# It's okay, nothing special.... → neutral (0.599)
```

### **Text Embeddings & Similarity**

```python
# Create embedding pipeline
embeddings = Pipeline(task="embedding", model="sentence-transformers/all-MiniLM-L6-v2")

# Generate embeddings
texts = [
    "The cat is on the mat.",
    "A cat is sitting on the mat.",
    "The weather is beautiful today.",
]

embeds = embeddings(texts)

# Calculate similarity
from lingo.models import EmbeddingModel

embedding_model = embeddings.model

similarity = embedding_model.similarity(texts[0], texts[1])
print(f"Similarity: {similarity:.3f}")

# Output:
# Similarity: 0.907
```

### **Question Answering**

```python
# Create QA pipeline
qa = Pipeline(task="question-answering", model="deepset/roberta-base-squad2")

# Answer questions
context = """
Python is a high-level programming language created by Guido van Rossum in 1991.
It's known for its simplicity and readability, making it popular for beginners.
Python is widely used in data science, machine learning, and web development.
"""

question = "Who created Python?"
answer = qa(question=question, context=context)

print(f"Q: {question}")
print(f"A: {answer['answer']} (confidence: {answer['score']:.3f})")

# Output:
# Q: Who created Python?
# A: Guido van Rossum (confidence: 0.98...)
```

### **Text Summarization**

```python
# Create summarization pipeline
summarizer = Pipeline(task="summarization", model="facebook/bart-large-cnn")

# Summarize long text
long_text = """
Artificial Intelligence (AI) has emerged as one of the most transformative technologies of the 21st century.
It encompasses a wide range of capabilities including machine learning, natural language processing, computer vision,
and robotics. AI systems can now perform tasks that were once thought to be exclusively human, such as recognizing
speech, translating languages, making decisions, and solving complex problems.
"""

summary = summarizer(long_text, max_length=60, min_length=20)
print(f"Summary: {summary['summary_text']}")

# Output:
# Summary: Artificial Intelligence has emerged as one of the most transformative technologies of the 21st century. It encompasses a wide range of capabilities including machine learning, natural language processing, computer vision, and robotics.
```

---

## **🔧 Advanced Usage**

### **Custom Preprocessing**

```python
from lingo import TextPreprocessor

# Configure preprocessing
preprocessor = TextPreprocessor(
    config={
        "lowercase": True,
        "remove_punctuation": True,
        "remove_stopwords": True,
        "lemmatize": True,
        "use_spacy": True,
        "spacy_model": "en_core_web_sm",
    }
)

# Process text
text = "The quick brown foxes are jumping over the lazy dogs! 🦊🐕"
cleaned = preprocessor(text)
print(f"Cleaned: {cleaned}")

# Get detailed preprocessing results
pipeline_result = preprocessor.get_preprocessing_pipeline(text)
print(f"Words: {pipeline_result['words']}")
print(f"Lemmatized: {pipeline_result['lemmatized']}")

# Output:
# Cleaned: the quick brown foxes are jumping over the lazy dogs
# Words: ['the', 'quick', 'brown', 'foxes', 'are', 'jumping', 'over', 'the', 'lazy', 'dogs']
# Lemmatized: ['the', 'quick', 'brown', 'fox', 'are', 'jumping', 'over', 'the', 'lazy', 'dog']
# (pipeline_result["words_no_stopwords"] holds the words with stopwords removed)
```

### **Batch Processing**

```python
# Process large datasets efficiently
texts = ["Text 1", "Text 2", "Text 3", ...]  # Large list

# Batch processing
results = pipeline.batch_predict(texts, batch_size=32)

# Or use utility function
from lingo.utils import batch_texts

batches = batch_texts(texts, batch_size=32)
```

### **Model Evaluation**

```python
from lingo.utils import evaluate_classification

# Evaluate model performance
y_true = ["positive", "negative", "positive", "neutral"]
y_pred = ["positive", "negative", "positive", "positive"]

metrics = evaluate_classification(y_true, y_pred)
print(f"Accuracy: {metrics['accuracy']:.3f}")
print(f"F1 Score: {metrics['f1']:.3f}")

# Output:
# Accuracy: 0.750
# F1 Score: 0.650   (weighted average)
```

### **Pipeline Configuration**

```python
# Load configuration from file
import yaml

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Create pipeline with custom config. The "preprocessing" section controls the
# TextPreprocessor applied before inference. By default a Pipeline only normalizes
# Unicode and whitespace (it does NOT lowercase, which would hurt cased models).
pipeline = Pipeline(
    task="sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
    config=config,
)
```

---

## **📁 Project Structure**

```
lingo/
├── lingo/                    # Core package
│   ├── __init__.py          # Main imports
│   ├── core.py              # Pipeline class
│   ├── preprocessing.py      # Text preprocessing
│   ├── models.py            # NLP model classes
│   ├── utils.py             # Utility functions
│   ├── setup_utils.py       # `lingo setup` data/model downloads
│   ├── cli.py               # Command-line interface
│   └── configs/default.yaml # Default config
├── examples/                # Usage examples
├── tests/                   # Test suite (pytest)
├── pyproject.toml           # Packaging metadata (single source of truth)
└── README.md                # This file
```

---

## **⚡ Performance & Optimization**

### **Device Selection**

```python
# Automatic device detection
pipeline = Pipeline(task="sentiment-analysis", model="...", device="auto")

# Manual device selection
pipeline = Pipeline(task="sentiment-analysis", model="...", device="cuda")
pipeline = Pipeline(task="sentiment-analysis", model="...", device="mps")  # Apple Silicon
```

### **Batch Processing**

```python
# Optimize for large datasets
results = pipeline.batch_predict(texts, batch_size=64)
```

### **Memory Management**

```python
# Use mixed precision for faster inference
pipeline = Pipeline(task="sentiment-analysis", model="...", config={"use_mixed_precision": True})
```

---

## **🔌 Integration & Extensibility**

### **With Existing Libraries**

```python
# spaCy integration
import spacy

nlp = spacy.load("en_core_web_sm")

# NLTK integration
import nltk
from nltk.tokenize import word_tokenize

# scikit-learn integration
from sklearn.metrics import classification_report
```

### **Custom Models**

```python
# Extend base model class
from lingo.models import BaseModel


class CustomModel(BaseModel):
    def _load_model(self):
        # Custom model loading logic
        pass

    def __call__(self, inputs, **kwargs):
        # Custom inference logic
        pass
```

---

## **🚀 Deployment & Production**

### **Save & Load Pipelines**

```python
# Save pipeline
pipeline.save("./saved_pipeline")

# Load pipeline
loaded_pipeline = Pipeline.load("./saved_pipeline")
```

### **REST API Template**

Requires the `api` extra: `pip install "lingo-nlp-toolkit[api]"`.

```python
from fastapi import FastAPI
from lingo import Pipeline

app = FastAPI()

# Load pipeline
pipeline = Pipeline.load("./saved_pipeline")


@app.post("/analyze")
async def analyze_text(text: str):
    result = pipeline(text)
    return {"result": result}
```

### **Docker Deployment**

```dockerfile
FROM python:3.12-slim

WORKDIR /app
RUN pip install "lingo-nlp-toolkit[api]"

COPY . .
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## **🧩 API Overview**

| Object | What it does |
| --- | --- |
| `Pipeline(task, model, config=None, device="auto", cache_dir=None, **kwargs)` | Unified entry point. Tasks: `text-classification`, `ner`, `sentiment-analysis`, `embedding`, `question-answering`, `summarization`. Extra `kwargs` go to the Hugging Face pipeline (e.g. `aggregation_strategy`). |
| `Pipeline.__call__(text_or_list, **kwargs)` | One result for a string, a list of results for a list. QA uses `nlp(question=..., context=...)`. |
| `Pipeline.batch_predict(texts, batch_size=32)` | Batched inference, always returns a list. |
| `Pipeline.save(path)` / `Pipeline.load(path)` | Persist and restore weights, preprocessing and config. |
| `TextPreprocessor(config)` | Cleaning, normalization, tokenization, stopwords, lemmatization, stemming. |
| `TextClassifier`, `NERModel`, `SentimentAnalyzer`, `EmbeddingModel`, `QuestionAnsweringModel`, `SummarizationModel` | Task model wrappers used by `Pipeline` (usable directly). |
| `lingo.utils` | `evaluate_classification`, `evaluate_ner`, `clean_text`, `chunk_text`, `batch_texts`, `get_text_statistics`, `get_available_models`, `download_model`. |

## **⚙️ Configuration & Environment**

- `device`: `"auto"` (default), `"cpu"`, `"cuda"` or `"mps"`.
- `cache_dir`: where Hugging Face models are cached (default: the standard HF cache; `HF_HOME` is respected).
- `HF_HUB_OFFLINE=1`: run fully offline once models are cached.
- `HF_TOKEN`: needed only for gated/private Hub models.
- `lingo setup` downloads the spaCy English model, NLTK data and a few common models; its marker file lives in `~/.lingo_cache/`.

---

## **🤝 Contributing**

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### **Development Setup**

```bash
# Clone repository
git clone https://github.com/irfanalidv/lingo-nlp-toolkit.git
cd lingo-nlp-toolkit

# Install in development mode
pip install -e .[dev]

# Run tests
pytest

# Lint and format
ruff check .
ruff format .
```

---

## **📄 License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## **🙏 Acknowledgments**

- **Hugging Face** for the amazing transformers library
- **spaCy** for excellent NLP tools
- **NLTK** for foundational NLP capabilities
- **PyTorch** for deep learning framework
- **scikit-learn** for machine learning utilities

---

## **📞 Support & Community**

- **Documentation**: [GitHub Repository](https://github.com/irfanalidv/lingo-nlp-toolkit)
- **Issues**: [GitHub Issues](https://github.com/irfanalidv/lingo-nlp-toolkit/issues)
- **Discussions**: [GitHub Discussions](https://github.com/irfanalidv/lingo-nlp-toolkit/discussions)

---

## **⭐ Star History**

[![Star History Chart](https://api.star-history.com/svg?repos=irfanalidv/lingo-nlp-toolkit&type=Date)](https://star-history.com/#irfanalidv/lingo-nlp-toolkit&Date)

---

**Made with ❤️ by Md Irfan Ali**
