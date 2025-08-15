#!/usr/bin/env python3
"""
Basic usage examples for Lingo NLP toolkit.
This script demonstrates the core functionality.
"""

import json
from lingo import Pipeline, TextPreprocessor
from lingo.utils import get_available_models, evaluate_classification


def main():
    print("🚀 Lingo NLP Toolkit - Basic Usage Examples\n")

    # Example 1: Sentiment Analysis
    print("=" * 50)
    print("1. SENTIMENT ANALYSIS")
    print("=" * 50)

    sentiment_pipeline = Pipeline(
        task="sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
    )

    texts = [
        "I absolutely love this product! It's amazing!",
        "This is the worst experience I've ever had.",
        "The product is okay, nothing special.",
    ]

    print("Analyzing sentiment for:")
    for text in texts:
        print(f"  • {text}")

    results = sentiment_pipeline(texts)

    print("\nResults:")
    for text, result in zip(texts, results):
        print(
            f"  • {text[:50]}... → {result['label']} (confidence: {result['score']:.3f})"
        )

    # Expected output:
    # Results:
    #   • I absolutely love this product! It's amazing!... → positive (confidence: 0.987)
    #   • This is the worst experience I've ever had.... → negative (confidence: 0.953)
    #   • The product is okay, nothing special.... → neutral (confidence: 0.596)

    # Example 2: Named Entity Recognition
    print("\n" + "=" * 50)
    print("2. NAMED ENTITY RECOGNITION")
    print("=" * 50)

    ner_pipeline = Pipeline(task="ner", model="dslim/bert-base-NER")

    text = "Apple Inc. is headquartered in Cupertino, California. Tim Cook is the CEO."

    print(f"Extracting entities from: {text}")

    entities = ner_pipeline(text)

    print("\nExtracted entities:")
    for entity in entities:
        print(f"  • {entity['entity']} ({entity['word']}) - {entity['score']:.3f}")

    # Expected output:
    # Extracted entities:
    #   • B-LOC (cup) - 0.940
    #   • B-LOC (##ert) - 0.671
    #   • I-LOC (##ino) - 0.437
    #   • B-LOC (ca) - 0.506

    # Example 3: Text Classification
    print("\n" + "=" * 50)
    print("3. TEXT CLASSIFICATION")
    print("=" * 50)

    # Using a general-purpose classifier
    classifier = Pipeline(task="text-classification", model="bert-base-uncased")

    # Note: This is a basic example - you'd typically use a fine-tuned model
    # for specific classification tasks
    print("Note: Using base BERT model for demonstration")

    # Example 4: Text Embeddings
    print("\n" + "=" * 50)
    print("4. TEXT EMBEDDINGS")
    print("=" * 50)

    embedding_pipeline = Pipeline(
        task="embedding", model="sentence-transformers/all-MiniLM-L6-v2"
    )

    text1 = "The cat is on the mat."
    text2 = "A cat is sitting on the mat."
    text3 = "The weather is beautiful today."

    print("Generating embeddings for:")
    print(f"  • {text1}")
    print(f"  • {text2}")
    print(f"  • {text3}")

    embeddings = embedding_pipeline([text1, text2, text3])

    # Calculate similarities
    from lingo.models import EmbeddingModel

    embedding_model = embedding_pipeline.model

    sim_1_2 = embedding_model.similarity(text1, text2)
    sim_1_3 = embedding_model.similarity(text1, text3)

    print(f"\nSimilarity between text 1 and 2: {sim_1_2:.3f}")
    print(f"Similarity between text 1 and 3: {sim_1_3:.3f}")

    # Expected output:
    # Similarity between text 1 and 2: 0.907
    # Similarity between text 1 and 3: 0.234

    # Example 5: Question Answering
    print("\n" + "=" * 50)
    print("5. QUESTION ANSWERING")
    print("=" * 50)

    qa_pipeline = Pipeline(
        task="question-answering", model="deepset/roberta-base-squad2"
    )

    context = """
    The Python programming language was created by Guido van Rossum and was released in 1991. 
    Python is known for its simplicity and readability. It has become one of the most popular 
    programming languages for data science, machine learning, and web development.
    """

    question = "Who created Python?"

    print(f"Context: {context.strip()}")
    print(f"Question: {question}")

    answer = qa_pipeline(question=question, context=context)

    print(f"\nAnswer: {answer['answer']}")
    print(f"Confidence: {answer['score']:.3f}")

    # Expected output:
    # Answer: Guido van Rossum
    # Confidence: 0.990

    # Example 6: Text Summarization
    print("\n" + "=" * 50)
    print("6. TEXT SUMMARIZATION")
    print("=" * 50)

    summarization_pipeline = Pipeline(
        task="summarization", model="facebook/bart-large-cnn"
    )

    long_text = """
    Artificial Intelligence (AI) has emerged as one of the most transformative technologies 
    of the 21st century. From virtual assistants like Siri and Alexa to self-driving cars 
    and medical diagnosis systems, AI is revolutionizing how we live and work. Machine 
    learning, a subset of AI, enables computers to learn from data without being explicitly 
    programmed. Deep learning, which uses neural networks with multiple layers, has achieved 
    remarkable breakthroughs in image recognition, natural language processing, and game playing. 
    Companies across industries are investing heavily in AI to gain competitive advantages, 
    while researchers continue to push the boundaries of what's possible. However, the rapid 
    advancement of AI also raises important questions about ethics, privacy, and the future 
    of human work.
    """

    print(f"Original text ({len(long_text)} characters):")
    print(long_text.strip())

    summary = summarization_pipeline(long_text)

    # Handle different summary output formats
    if isinstance(summary, list) and len(summary) > 0:
        summary_text = summary[0]["summary_text"]
    elif isinstance(summary, dict) and "summary_text" in summary:
        summary_text = summary["summary_text"]
    else:
        summary_text = str(summary)

    print(f"\nSummary ({len(summary_text)} characters):")
    print(summary_text)

    # Expected output:
    # Summary (156 characters):
    # artificial intelligence (ai) has emerged as one of the most transformative technologies of the 21st century. ai systems can now perform tasks that were once thought to be exclusively human.

    # Example 7: Text Preprocessing
    print("\n" + "=" * 50)
    print("7. TEXT PREPROCESSING")
    print("=" * 50)

    preprocessor = TextPreprocessor(
        config={
            "lowercase": True,
            "remove_punctuation": True,
            "remove_stopwords": True,
            "lemmatize": True,
        }
    )

    sample_text = "The quick brown foxes are jumping over the lazy dogs! 🦊🐕"

    print(f"Original text: {sample_text}")

    pipeline_result = preprocessor.get_preprocessing_pipeline(sample_text)

    print(f"Cleaned text: {pipeline_result['cleaned']}")
    print(f"Words: {pipeline_result['words']}")
    print(f"Words without stopwords: {pipeline_result['words_no_stopwords']}")
    print(f"Lemmatized: {pipeline_result['lemmatized']}")

    # Expected output:
    # Cleaned text: the quick brown foxes are jumping over the lazy dogs
    # Words: ['the', 'quick', 'brown', 'foxes', 'are', 'jumping', 'over', 'the', 'lazy', 'dogs']
    # Words without stopwords: ['quick', 'brown', 'foxes', 'jumping', 'lazy', 'dogs']
    # Lemmatized: ['the', 'quick', 'brown', 'fox', 'are', 'jumping', 'over', 'the', 'lazy', 'dog']

    # Example 8: Available Models
    print("\n" + "=" * 50)
    print("8. AVAILABLE MODELS")
    print("=" * 50)

    models = get_available_models()

    for task, model_list in models.items():
        print(f"\n{task.upper()}:")
        for model in model_list[:3]:  # Show first 3 models
            print(f"  • {model}")
        if len(model_list) > 3:
            print(f"  ... and {len(model_list) - 3} more")

    print("\n" + "=" * 50)
    print("✅ All examples completed successfully!")
    print("=" * 50)


if __name__ == "__main__":
    main()
