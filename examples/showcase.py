#!/usr/bin/env python3
"""
Lingo NLP Toolkit - Capability Showcase
"""

from lingo import Pipeline, TextPreprocessor


def main():
    print("🚀 Lingo NLP Toolkit - Capability Showcase")
    print("=" * 60)

    # Text preprocessing showcase
    print("\n📝 Text Preprocessing")
    print("-" * 30)

    preprocessor = TextPreprocessor(
        {
            "normalize_unicode": True,
            "lowercase": True,
            "expand_contractions": True,
            "expand_slang": True,
            "remove_stopwords": True,
            "lemmatize": True,
        }
    )

    sample_text = "I'm gonna love this product! It's AWESOME!!! 😍 #amazing"
    results = preprocessor.get_preprocessing_pipeline(sample_text)

    print(f"Original: {sample_text}")
    print(f"Cleaned: {results['cleaned']}")
    print(f"Words: {len(results['words'])}")
    print(f"Lemmatized: {results['lemmatized'][:5]}")
    
    # Expected output:
    # Original: I'm gonna love this product! It's AWESOME!!! 😍 #amazing
    # Cleaned: i am gonna love this product! it is awesome!!! 😍 #amazing
    # Words: 17
    # Lemmatized: ['i', 'am', 'gon', 'na', 'love']

    # Sentiment analysis showcase
    print("\n🎭 Sentiment Analysis")
    print("-" * 30)

    sentiment_pipeline = Pipeline(
        task="sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
    )

    texts = [
        "I love this product! It's amazing!",
        "This is terrible quality.",
        "The service was okay.",
    ]

    for text in texts:
        result = sentiment_pipeline(text)
        print(f"{text}: {result['label']} ({result['score']:.3f})")
    
    # Expected output:
    # I love this product! It's amazing!: positive (0.987)
    # This is terrible quality.: negative (0.937)
    # The service was okay.: positive (0.805)

    # NER showcase
    print("\n🔍 Named Entity Recognition")
    print("-" * 30)

    ner_pipeline = Pipeline(task="ner", model="dslim/bert-base-NER")

    ner_text = "Apple Inc. CEO Tim Cook announced new products in San Francisco."
    entities = ner_pipeline(ner_text)

    print(f"Text: {ner_text}")
    for entity in entities:
        print(f"  {entity['entity_group']}: {entity['word']}")
    
    # Expected output:
    # Text: Apple Inc. CEO Tim Cook announced new products in San Francisco.
    #   B-LOC: cup
    #   B-LOC: ##ert
    #   I-LOC: ##ino

        # Embeddings showcase
    print("\n🔢 Text Embeddings")
    print("-" * 30)

    embedding_pipeline = Pipeline(
        task="embedding", model="sentence-transformers/all-MiniLM-L6-v2"
    )

    embedding_text = "Machine learning is transforming industries."
    embedding = embedding_pipeline(embedding_text)

    print(f"Text: {embedding_text}")
    if isinstance(embedding, list):
        print(f"Embedding dimensions: {len(embedding)}")
    else:
        print(f"Embedding result: {type(embedding)}")
        if hasattr(embedding, "shape"):
            print(f"Embedding shape: {embedding.shape}")
        elif hasattr(embedding, "__len__"):
            print(f"Embedding length: {len(embedding)}")
    
    # Expected output:
    # Text: Machine learning is transforming industries.
    # Embedding dimensions: 384

    print("\n✅ Showcase completed successfully!")


if __name__ == "__main__":
    main()
