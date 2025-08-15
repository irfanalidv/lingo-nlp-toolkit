# 🚀 Lingo Quickstart Guide

Get up and running with Lingo NLP toolkit in minutes!

## **Prerequisites**

- Python 3.9 or higher
- pip package manager
- Basic knowledge of Python

## **Installation**

### **Option 1: Install from PyPI (recommended for users)**

```bash
# One-command installation
pip install lingo-nlp-toolkit
```

**✨ Auto-Setup**: Lingo automatically downloads all required NLP data and models on first use!

### **Option 2: Install from source (for development)**

```bash
# Clone the repository
git clone https://github.com/irfanalidv/lingo-nlp-toolkit.git
cd lingo-nlp-toolkit

# Install in development mode
pip install -e .

# Set up environment (optional - auto-setup will handle this)
lingo setup
```

### **Option 2: Install from PyPI (when available)**

```bash
pip install lingo-nlp-toolkit
```

## **Quick Test**

### **Automatic Setup (Recommended)**

Simply import Lingo and it will set up everything automatically:

```python
from lingo import Pipeline

# This will trigger auto-setup if needed
nlp = Pipeline(task="sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")
```

### **Manual Setup (Optional)**

If you prefer to set up manually:

```bash
# Set up the environment
lingo setup

# Force re-download of all data
lingo setup --force
```

### **Run the Demo**

```bash
python demo.py
```

## **Your First NLP Task**

### **Sentiment Analysis**

```python
from lingo import Pipeline

# Create a sentiment analysis pipeline
nlp = Pipeline(
    task="sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest"
)

# Analyze sentiment
text = "I absolutely love this product! It's amazing!"
result = nlp(text)
print(result)
# Output: {'label': 'POSITIVE', 'score': 0.988}
```

### **Named Entity Recognition**

```python
# Create an NER pipeline
ner = Pipeline(
    task="ner",
    model="dslim/bert-base-NER"
)

# Extract entities
text = "Apple Inc. is headquartered in Cupertino, California."
entities = ner(text)
print(entities)
```

## **Command Line Usage**

### **List available models**

```bash
lingo list-models
```

### **Download a model**

```bash
lingo download-model --model bert-base-uncased
```

### **Run sentiment analysis**

```bash
lingo run sentiment-analysis \
  --model cardiffnlp/twitter-roberta-base-sentiment-latest \
  --text "I love this product!"
```

### **Process text from file**

```bash
echo "This is a test text." > input.txt
lingo run sentiment-analysis \
  --model cardiffnlp/twitter-roberta-base-sentiment-latest \
  --input-file input.txt \
  --output-file results.json
```

## **Common Tasks**

### **Text Classification**

```python
classifier = Pipeline(
    task="text-classification",
    model="bert-base-uncased"
)
results = classifier(["Text 1", "Text 2", "Text 3"])
```

### **Text Embeddings**

```python
embeddings = Pipeline(
    task="embedding",
    model="sentence-transformers/all-MiniLM-L6-v2"
)
embeds = embeddings(["Sentence 1", "Sentence 2"])
```

### **Question Answering**

```python
qa = Pipeline(
    task="question-answering",
    model="deepset/roberta-base-squad2"
)
answer = qa(question="Who created Python?", context="Python was created by Guido van Rossum...")
```

### **Text Summarization**

```python
summarizer = Pipeline(
    task="summarization",
    model="facebook/bart-large-cnn"
)
summary = summarizer("Long text to summarize...")
```

## **Text Preprocessing**

```python
from lingo import TextPreprocessor

# Configure preprocessing
preprocessor = TextPreprocessor(
    config={
        "lowercase": True,
        "remove_punctuation": True,
        "remove_stopwords": True,
        "lemmatize": True
    }
)

# Process text
cleaned = preprocessor("The quick brown foxes are jumping!")
print(cleaned)

# Get detailed results
result = preprocessor.get_preprocessing_pipeline("Your text here")
print(f"Words: {result['words']}")
print(f"Lemmatized: {result['lemmatized']}")
```

## **Configuration**

Create a custom configuration file:

```yaml
# config.yaml
preprocessing:
  lowercase: true
  remove_punctuation: true
  remove_stopwords: true

models:
  sentiment_analysis:
    max_length: 512
    return_all_scores: false
```

Use it in your pipeline:

```python
import yaml

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

pipeline = Pipeline(
    task="sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
    config=config
)
```

## **Performance Tips**

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
# Process large datasets efficiently
texts = ["Text 1", "Text 2", "Text 3", ...]  # Large list
results = pipeline.batch_predict(texts, batch_size=64)
```

## **Troubleshooting**

### **Common Issues**

1. **Out of Memory**: Reduce batch size or use smaller models
2. **Model Download Fails**: Check internet connection and try again
3. **Import Errors**: Ensure all dependencies are installed

### **Setup Issues**

If auto-setup fails:

```bash
# Run setup manually
lingo setup

# Check setup status
lingo setup --help

# Force re-download
lingo setup --force
```

### **Manual Data Downloads**

If you need to download data manually:

```bash
# Download spaCy model
python -m spacy download en_core_web_sm

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('punkt_tab')"
```

### **Get Help**

- Check the [full documentation](README.md)
- Run `python demo.py` to see working examples
- Use `lingo --help` for CLI options
- Check the [examples directory](examples/) for more code samples

## **Next Steps**

1. **Explore Examples**: Check out `examples/basic_usage.py`
2. **Custom Models**: Learn how to extend the toolkit
3. **Production**: Deploy your pipelines with FastAPI
4. **Contributing**: Help improve Lingo!

---

**Happy NLP-ing! 🎉**
