# **Lingo: Advanced NLP Toolkit**

**Lightweight, Fast, and Transformer-Ready**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://badge.fury.io/py/lingo.svg)](https://badge.fury.io/py/lingo)

## **Overview**

Lingo is a modern, high-performance **Natural Language Processing (NLP) toolkit** designed for researchers, data scientists, and developers building intelligent language-powered applications. It combines **ease of use**, **speed**, and **state-of-the-art transformer capabilities**, offering an end-to-end pipeline — from text preprocessing to advanced contextual understanding.

Lingo bridges the gap between traditional NLP techniques and next-generation transformer-based architectures like **BERT**, **GPT**, and **LLaMA**, ensuring flexibility, scalability, and cutting-edge accuracy.

---

## **🚀 Quick Start**

### **Installation**

```bash
# One-command installation (recommended)
pip install lingo-nlp-toolkit

# Full installation with all dependencies
pip install lingo-nlp-toolkit[full]

# Development installation
pip install lingo-nlp-toolkit[dev]

# GPU support
pip install lingo-nlp-toolkit[gpu]
```

**✨ Auto-Setup**: Lingo automatically downloads all required NLP data and models on first use!

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
# Output: {'label': 'POSITIVE', 'score': 0.988}
```

### **Command Line Usage**

```bash
# Sentiment analysis
lingo run sentiment-analysis --model cardiffnlp/twitter-roberta-base-sentiment-latest --text "I love this product!"

# List available models
lingo list-models

# Download a model
lingo download-model --model bert-base-uncased
```

---

## **✨ Key Features**

### **1. Text Preprocessing & Normalization**

- ✅ Unicode normalization (NFC/NFD)
- ✅ Lowercasing, punctuation removal, special character stripping
- ✅ Stopword removal (multi-language support)
- ✅ Lemmatization & stemming
- ✅ Advanced tokenization (Word, Sentence, Subword)
- ✅ Spell correction & slang expansion

### **2. Core NLP Tasks**

- ✅ **Text Classification** - Multi-class & multi-label
- ✅ **Named Entity Recognition (NER)** - Domain-specific models
- ✅ **Sentiment Analysis** - Binary, ternary, fine-grained
- ✅ **Text Embeddings** - BERT, Sentence-BERT, LLaMA
- ✅ **Question Answering** - Extractive & generative
- ✅ **Text Summarization** - Abstractive & extractive

### **3. Hugging Face Integration**

- ✅ Load any model from Hugging Face Hub
- ✅ Fine-tune pre-trained transformers
- ✅ Export models to Hugging Face Hub
- ✅ Mixed precision training

### **4. Performance & Scalability**

- ✅ GPU & multi-core CPU support
- ✅ Asynchronous batch processing
- ✅ Memory-efficient tokenization
- ✅ Lightweight deployment mode

---

## **📚 Comprehensive Examples**

### **Text Classification**

```python
from lingo import Pipeline

# Create classifier
classifier = Pipeline(
    task="text-classification",
    model="bert-base-uncased"
)

# Classify texts
texts = [
    "This is a positive review about the product.",
    "I'm not satisfied with the service quality.",
    "The product meets my expectations."
]

results = classifier(texts)
for text, result in zip(texts, results):
    print(f"{text[:30]}... → {result['label']}")

# Output:
# This is a positive review abou... → LABEL_0
# I'm not satisfied with the ser... → LABEL_0
# The product meets my expectati... → LABEL_0
```

### **Named Entity Recognition**

```python
# Create NER pipeline
ner = Pipeline(
    task="ner",
    model="dslim/bert-base-NER"
)

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
    task="sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest"
)

# Analyze sentiment
texts = [
    "I love this amazing product!",
    "This is terrible, worst purchase ever.",
    "It's okay, nothing special."
]

results = sentiment(texts)
for text, result in zip(texts, results):
    print(f"{text[:30]}... → {result['label']} ({result['score']:.3f})")

# Output:
# I love this amazing product!... → positive (0.987)
# This is terrible, worst purcha... → negative (0.953)
# It's okay, nothing special.... → neutral (0.596)
```

### **Text Embeddings & Similarity**

```python
# Create embedding pipeline
embeddings = Pipeline(
    task="embedding",
    model="sentence-transformers/all-MiniLM-L6-v2"
)

# Generate embeddings
texts = [
    "The cat is on the mat.",
    "A cat is sitting on the mat.",
    "The weather is beautiful today."
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
qa = Pipeline(
    task="question-answering",
    model="deepset/roberta-base-squad2"
)

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
# A: Guido van Rossum (confidence: 0.990)
```

### **Text Summarization**

```python
# Create summarization pipeline
summarizer = Pipeline(
    task="summarization",
    model="facebook/bart-large-cnn"
)

# Summarize long text
long_text = """
Artificial Intelligence (AI) has emerged as one of the most transformative technologies of the 21st century.
It encompasses a wide range of capabilities including machine learning, natural language processing, computer vision,
and robotics. AI systems can now perform tasks that were once thought to be exclusively human, such as recognizing
speech, translating languages, making decisions, and solving complex problems.
"""

summary = summarizer(long_text)
print(f"Summary: {summary[0]['summary_text']}")

# Output:
# Summary: artificial intelligence (ai) has emerged as one of the most transformative technologies of the 21st century. ai systems can now perform tasks that were once thought to be exclusively human.
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
        "spacy_model": "en_core_web_sm"
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
# F1 Score: 0.800
```

### **Pipeline Configuration**

```python
# Load configuration from file
import yaml

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Create pipeline with custom config
pipeline = Pipeline(
    task="sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
    config=config
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
│   └── cli.py               # Command-line interface
├── examples/                 # Usage examples
│   └── basic_usage.py       # Basic examples
├── lingo/configs/           # Configuration files
│   └── default.yaml         # Default config
├── tests/                   # Test suite
├── setup.py                 # Package setup
├── requirements.txt          # Dependencies
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
pipeline = Pipeline(
    task="sentiment-analysis",
    model="...",
    config={"use_mixed_precision": True}
)
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
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## **📊 Benchmarks & Performance**

| Task                | Model         | Speed (CPU) | Speed (GPU) | Memory Usage |
| ------------------- | ------------- | ----------- | ----------- | ------------ |
| Sentiment Analysis  | RoBERTa-base  | 50 ms       | 15 ms       | 500 MB       |
| NER                 | BERT-base-NER | 80 ms       | 25 ms       | 400 MB       |
| Text Classification | DistilBERT    | 30 ms       | 10 ms       | 300 MB       |
| Embeddings          | MiniLM-L6     | 40 ms       | 12 ms       | 200 MB       |

_Benchmarks on Intel i7-10700K (CPU) and RTX 3080 (GPU)_

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

# Format code
black lingo/
isort lingo/
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

- **Documentation**: [https://lingo.readthedocs.io](https://lingo.readthedocs.io)
- **Issues**: [GitHub Issues](https://github.com/irfanalidv/lingo-nlp-toolkit/issues)
- **Discussions**: [GitHub Discussions](https://github.com/irfanalidv/lingo-nlp-toolkit/discussions)
- **Email**: irfanali29@hotmail.com

---

## **⭐ Star History**

[![Star History Chart](https://api.star-history.com/svg?repos=irfanalidv/lingo-nlp-toolkit&type=Date)](https://star-history.com/#irfanalidv/lingo-nlp-toolkit&Date)

---

**Made with ❤️ by Md Irfan Ali**
