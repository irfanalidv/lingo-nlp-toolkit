"""Clean product reviews, measure readability and score a classifier, with no models.

Use case: before (or instead of) running a transformer, you want normalised text,
token-level views, readability statistics, chunks that fit a model's context window, and
metrics for predictions you already have.

Needs: ``pip install lingo-nlp-toolkit``. No Hugging Face model is downloaded; NLTK
tokenizer/stopword/WordNet data (a few MB) is fetched automatically the first time.

Run: python examples/text_toolkit_offline.py
"""

from __future__ import annotations

from lingo import TextPreprocessor
from lingo.utils import chunk_text, clean_text, evaluate_classification, get_text_statistics

reviews = [
    "I can't believe how GOOD this blender is!!! Worth every $$$ btw.",
    "Arrived late and the lid was cracked. Won't order again.",
    "It's okay. Does the job, nothing special.",
]

preprocessor = TextPreprocessor(
    config={
        "use_spacy": False,  # NLTK tokenisation; set True after `python -m spacy download en_core_web_sm`
        "lowercase": True,
        "expand_contractions": True,
        "expand_slang": True,
        "remove_punctuation": True,
        "remove_stopwords": True,
        "lemmatize": True,
    }
)

print("== Preprocessing")
for review in reviews:
    steps = preprocessor.get_preprocessing_pipeline(review)
    print(f"raw:        {review}")
    print(f"cleaned:    {steps['cleaned']}")
    print(f"content:    {steps['words_no_stopwords']}")
    print(f"lemmas:     {preprocessor.lemmatize_tokens(steps['words_no_stopwords'])}\n")

print("== Readability (Flesch Reading Ease)")
for text in [
    "The lid broke. I sent it back.",
    "Notwithstanding considerable manufacturing irregularities, the replacement "
    "procedure was administratively straightforward.",
]:
    stats = get_text_statistics(text)
    print(f"{stats['flesch_reading_ease']:6.1f}  {stats['reading_level']:<15} {text[:50]}")

print("\n== Chunking a long document for a 120-character context window")
manual = clean_text(" ".join(reviews) * 3)
chunks = chunk_text(manual, max_length=120, overlap=20, separator=" ")
print(f"{len(manual)} characters -> {len(chunks)} chunks, longest {max(map(len, chunks))}")

print("\n== Evaluating predictions you already have")
y_true = ["positive", "negative", "neutral", "negative", "positive", "neutral"]
y_pred = ["positive", "negative", "positive", "negative", "positive", "neutral"]
metrics = evaluate_classification(y_true, y_pred, labels=["positive", "neutral", "negative"])
print(f"accuracy={metrics['accuracy']:.3f}  weighted F1={metrics['f1']:.3f}")
for label, scores in metrics["per_class"].items():
    print(f"  {label:<8} precision={scores['precision']:.2f} recall={scores['recall']:.2f}")
print(f"confusion matrix (rows=true): {metrics['confusion_matrix']}")
