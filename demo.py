#!/usr/bin/env python3
"""
Lingo NLP Toolkit - Advanced Demo
Showcasing comprehensive NLP capabilities and real-world use cases.
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Any


def print_header(title: str):
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(f"🚀 {title}")
    print("=" * 60)


def print_section(title: str):
    """Print a formatted section."""
    print(f"\n📋 {title}")
    print("-" * 40)


def demo_text_preprocessing():
    """Demonstrate advanced text preprocessing capabilities."""
    print_header("Advanced Text Preprocessing")

    try:
        from lingo import TextPreprocessor

        # Sample text with various preprocessing challenges
        sample_texts = [
            "I'm gonna love this product! It's AWESOME!!! 😍",
            "The company's revenue increased by 25% in Q3 2024.",
            "btw, imo this is the best solution ever!",
            "Apple Inc. is headquartered in Cupertino, California.",
            "This is a test sentence. Here's another one. And a third.",
        ]

        # Create preprocessor with custom configuration
        config = {
            "normalize_unicode": True,
            "lowercase": True,
            "remove_punctuation": False,
            "remove_numbers": False,
            "remove_special_chars": False,
            "expand_contractions": True,
            "correct_spelling": False,
            "expand_slang": True,
            "remove_extra_whitespace": True,
            "remove_stopwords": True,
            "lemmatize": True,
            "stem": False,
        }

        preprocessor = TextPreprocessor(config=config)

        print_section("Sample Texts")
        for i, text in enumerate(sample_texts, 1):
            print(f"{i}. {text}")

        print_section("Preprocessing Results")
        for i, text in enumerate(sample_texts, 1):
            print(f"\n📝 Original {i}: {text}")

            # Get comprehensive preprocessing results
            results = preprocessor.get_preprocessing_pipeline(text)

            print(f"   🧹 Cleaned: {results['cleaned']}")
            print(f"   🔤 Words: {len(results['words'])} tokens")
            print(f"   📊 Sentences: {results['sentence_count']}")

            if results.get("words_no_stopwords"):
                print(
                    f"   🚫 No stopwords: {len(results['words_no_stopwords'])} tokens"
                )

            if results.get("lemmatized"):
                print(f"   🔍 Lemmatized: {results['lemmatized'][:5]}...")

        # Demonstrate tokenization
        print_section("Advanced Tokenization")
        complex_text = "Dr. Smith's appointment is at 2:30 PM on Dec. 15th, 2024."
        print(f"Complex text: {complex_text}")

        # Word tokenization
        words = preprocessor.tokenize_words(complex_text)
        print(f"Word tokens: {words}")

        # Sentence tokenization
        sentences = preprocessor.tokenize_sentences(complex_text)
        print(f"Sentence tokens: {sentences}")

        return True

    except Exception as e:
        print(f"❌ Text preprocessing demo failed: {e}")
        return False


def demo_sentiment_analysis():
    """Demonstrate advanced sentiment analysis capabilities."""
    print_header("Advanced Sentiment Analysis")

    try:
        from lingo import Pipeline

        # Sample texts for sentiment analysis
        sentiment_texts = [
            "I absolutely love this product! It's amazing and works perfectly! 😍",
            "This is the worst purchase I've ever made. Terrible quality and broke immediately.",
            "The product is okay, nothing special but it gets the job done.",
            "Mixed feelings about this. Good features but expensive and slow.",
            "I'm neutral about this product. It exists and functions as expected.",
        ]

        # Create sentiment analysis pipeline
        sentiment_pipeline = Pipeline(
            task="sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        )

        print_section("Sentiment Analysis Results")

        for i, text in enumerate(sentiment_texts, 1):
            print(f"\n📝 Text {i}: {text[:80]}...")

            # Analyze sentiment
            start_time = time.time()
            result = sentiment_pipeline(text)
            processing_time = time.time() - start_time

            print(f"   🎭 Sentiment: {result['label']}")
            print(f"   📊 Confidence: {result['score']:.3f}")
            print(f"   ⚡ Processing time: {processing_time:.3f}s")

        # Batch processing demonstration
        print_section("Batch Processing")
        print("Processing all texts in batch...")

        start_time = time.time()
        batch_results = sentiment_pipeline.batch_predict(sentiment_texts)
        batch_time = time.time() - start_time

        print(f"Batch processing time: {batch_time:.3f}s")
        print(
            f"Individual processing time: {batch_time/len(sentiment_texts):.3f}s per text"
        )

        return True

    except Exception as e:
        print(f"❌ Sentiment analysis demo failed: {e}")
        return False


def demo_named_entity_recognition():
    """Demonstrate advanced NER capabilities."""
    print_header("Advanced Named Entity Recognition")

    try:
        from lingo import Pipeline

        # Sample texts with various entity types
        ner_texts = [
            "Apple Inc. CEO Tim Cook announced new products at WWDC 2024 in San Francisco, California.",
            "The Great Wall of China was built during the Ming Dynasty and spans over 13,000 miles.",
            "NASA's Perseverance rover landed on Mars on February 18, 2021, at Jezero Crater.",
            "Shakespeare's Hamlet was first performed in 1603 at the Globe Theatre in London, England.",
            "The COVID-19 pandemic began in Wuhan, China in December 2019 and affected millions worldwide.",
        ]

        # Create NER pipeline
        ner_pipeline = Pipeline(task="ner", model="dslim/bert-base-NER")

        print_section("NER Results")

        for i, text in enumerate(ner_texts, 1):
            print(f"\n📝 Text {i}: {text}")

            # Extract entities
            start_time = time.time()
            entities = ner_pipeline(text)
            processing_time = time.time() - start_time

            print(f"   🔍 Entities found: {len(entities)}")

            # Group entities by type
            entity_types = {}
            for entity in entities:
                # Handle different field names for entity type
                entity_type = (
                    entity.get("entity_group")
                    or entity.get("entity_type")
                    or entity.get("label", "UNKNOWN")
                )
                if entity_type not in entity_types:
                    entity_types[entity_type] = []
                entity_types[entity_type].append(
                    entity.get("word", entity.get("text", "UNKNOWN"))
                )

            for entity_type, words in entity_types.items():
                print(f"   📌 {entity_type}: {', '.join(set(words))}")

            print(f"   ⚡ Processing time: {processing_time:.3f}s")

        return True

    except Exception as e:
        print(f"❌ NER demo failed: {e}")
        return False


def demo_text_classification():
    """Demonstrate advanced text classification capabilities."""
    print_header("Advanced Text Classification")

    try:
        from lingo import Pipeline

        # Sample texts for classification
        classification_texts = [
            "The stock market reached new highs today with technology stocks leading gains.",
            "Scientists discovered a new species of deep-sea creatures in the Pacific Ocean.",
            "The new restaurant downtown offers authentic Italian cuisine with fresh ingredients.",
            "Breaking news: Major breakthrough in renewable energy technology announced.",
            "Local sports team wins championship after dramatic overtime victory.",
        ]

        # Create text classification pipeline
        classifier = Pipeline(task="text-classification", model="bert-base-uncased")

        print_section("Text Classification Results")

        for i, text in enumerate(classification_texts, 1):
            print(f"\n📝 Text {i}: {text}")

            # Classify text
            start_time = time.time()
            result = classifier(text)
            processing_time = time.time() - start_time

            print(f"   🏷️  Classification: {result['label']}")
            print(f"   📊 Confidence: {result['score']:.3f}")
            print(f"   ⚡ Processing time: {processing_time:.3f}s")

        return True

    except Exception as e:
        print(f"❌ Text classification demo failed: {e}")
        return False


def demo_embeddings_and_similarity():
    """Demonstrate advanced embedding and similarity capabilities."""
    print_header("Advanced Embeddings & Similarity")

    try:
        from lingo import Pipeline
        import numpy as np

        # Sample texts for similarity analysis
        similarity_texts = [
            "The cat sat on the mat.",
            "A feline is resting on the carpet.",
            "The weather is sunny today.",
            "It's a beautiful day with clear skies.",
            "Machine learning is a subset of artificial intelligence.",
            "AI includes various techniques like deep learning and neural networks.",
        ]

        # Create embedding pipeline
        embedding_pipeline = Pipeline(
            task="embedding", model="sentence-transformers/all-MiniLM-L6-v2"
        )

        print_section("Text Embeddings")

        # Generate embeddings
        embeddings = []
        for i, text in enumerate(similarity_texts):
            print(f"📝 Text {i+1}: {text}")

            start_time = time.time()
            embedding = embedding_pipeline(text)
            processing_time = time.time() - start_time

            embeddings.append(embedding)
            print(f"   🔢 Embedding shape: {len(embedding)}")
            print(f"   ⚡ Processing time: {processing_time:.3f}s")

        # Calculate similarity matrix
        print_section("Similarity Analysis")

        similarity_matrix = np.zeros((len(embeddings), len(embeddings)))
        for i in range(len(embeddings)):
            for j in range(len(embeddings)):
                # Cosine similarity
                dot_product = np.dot(embeddings[i], embeddings[j])
                norm_i = np.linalg.norm(embeddings[i])
                norm_j = np.linalg.norm(embeddings[j])
                similarity_matrix[i][j] = dot_product / (norm_i * norm_j)

        print("Similarity Matrix (higher values = more similar):")
        print("      ", end="")
        for i in range(len(similarity_texts)):
            print(f"T{i+1:2d} ", end="")
        print()

        for i in range(len(similarity_texts)):
            print(f"T{i+1:2d} ", end="")
            for j in range(len(similarity_texts)):
                print(f"{similarity_matrix[i][j]:.2f} ", end="")
            print()

        # Find most similar pairs
        print_section("Most Similar Text Pairs")
        most_similar = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                similarity = similarity_matrix[i][j]
                most_similar.append((i, j, similarity))

        most_similar.sort(key=lambda x: x[2], reverse=True)

        for i, j, similarity in most_similar[:3]:
            print(f"📊 Similarity {similarity:.3f}:")
            print(f"   Text {i+1}: {similarity_texts[i]}")
            print(f"   Text {j+1}: {similarity_texts[j]}")

        return True

    except Exception as e:
        print(f"❌ Embeddings demo failed: {e}")
        return False


def demo_question_answering():
    """Demonstrate advanced question answering capabilities."""
    print_header("Advanced Question Answering")

    try:
        from lingo import Pipeline

        # Context and questions
        context = """
        The Python programming language was created by Guido van Rossum and was released in 1991. 
        Python is known for its simplicity and readability, making it an excellent choice for beginners. 
        It supports multiple programming paradigms including procedural, object-oriented, and functional programming. 
        Python has a large standard library and is widely used in web development, data science, artificial intelligence, 
        and scientific computing. The language emphasizes code readability with its notable use of significant whitespace.
        """

        questions = [
            "Who created Python?",
            "When was Python released?",
            "What are Python's main features?",
            "What is Python used for?",
            "How does Python handle code readability?",
        ]

        # Create QA pipeline
        qa_pipeline = Pipeline(
            task="question-answering", model="deepset/roberta-base-squad2"
        )

        print_section("Context")
        print(context.strip())

        print_section("Question Answering Results")

        for i, question in enumerate(questions, 1):
            print(f"\n❓ Question {i}: {question}")

            # Get answer
            start_time = time.time()
            result = qa_pipeline(question=question, context=context)
            processing_time = time.time() - start_time

            print(f"   💡 Answer: {result['answer']}")
            print(f"   📊 Confidence: {result['score']:.3f}")
            print(f"   📍 Start: {result['start']}, End: {result['end']}")
            print(f"   ⚡ Processing time: {processing_time:.3f}s")

        return True

    except Exception as e:
        print(f"❌ Question answering demo failed: {e}")
        return False


def demo_text_summarization():
    """Demonstrate advanced text summarization capabilities."""
    print_header("Advanced Text Summarization")

    try:
        from lingo import Pipeline

        # Long text for summarization
        long_text = """
        Artificial Intelligence (AI) has emerged as one of the most transformative technologies of the 21st century. 
        It encompasses a wide range of capabilities including machine learning, natural language processing, computer vision, 
        and robotics. AI systems can now perform tasks that were once thought to be exclusively human, such as recognizing 
        speech, translating languages, making decisions, and solving complex problems. The technology has applications 
        across virtually every industry, from healthcare and finance to transportation and entertainment. Machine learning, 
        a subset of AI, enables computers to learn and improve from experience without being explicitly programmed. 
        Deep learning, which uses neural networks with multiple layers, has been particularly successful in areas like 
        image recognition and natural language understanding. However, the rapid advancement of AI also raises important 
        questions about ethics, privacy, job displacement, and the future of human work. As AI continues to evolve, 
        it will be crucial to develop frameworks for responsible AI development and deployment that maximize benefits 
        while minimizing potential risks and ensuring that the technology serves humanity's best interests.
        """

        # Create summarization pipeline
        summarizer = Pipeline(task="summarization", model="facebook/bart-large-cnn")

        print_section("Original Text")
        print(f"Length: {len(long_text.split())} words")
        print(long_text.strip())

        print_section("Summarization Results")

        # Generate summary
        start_time = time.time()
        summary = summarizer(long_text)
        processing_time = time.time() - start_time

        print(f"📝 Summary: {summary['summary_text']}")
        print(f"📊 Summary length: {len(summary['summary_text'].split())} words")
        print(
            f"📉 Compression ratio: {len(summary['summary_text'].split())/len(long_text.split())*100:.1f}%"
        )
        print(f"⚡ Processing time: {processing_time:.3f}s")

        return True

    except Exception as e:
        print(f"❌ Text summarization demo failed: {e}")
        return False


def demo_pipeline_management():
    """Demonstrate advanced pipeline management capabilities."""
    print_header("Advanced Pipeline Management")

    try:
        from lingo import Pipeline
        import tempfile
        import os

        # Create a custom pipeline configuration
        pipeline_config = {
            "preprocessing": {
                "normalize_unicode": True,
                "lowercase": True,
                "remove_punctuation": False,
                "expand_contractions": True,
                "expand_slang": True,
            },
            "models": {
                "sentiment": "cardiffnlp/twitter-roberta-base-sentiment-latest",
                "ner": "dslim/bert-base-NER",
                "embedding": "sentence-transformers/all-MiniLM-L6-v2",
            },
        }

        print_section("Pipeline Configuration")
        print(json.dumps(pipeline_config, indent=2))

        # Create and test pipeline
        print_section("Pipeline Creation & Testing")

        # Create sentiment pipeline
        sentiment_pipeline = Pipeline(
            task="sentiment-analysis", model=pipeline_config["models"]["sentiment"]
        )

        test_text = "This is an amazing product that exceeded all my expectations!"
        result = sentiment_pipeline(test_text)

        print(f"📝 Test text: {test_text}")
        print(f"🎭 Sentiment: {result['label']} (confidence: {result['score']:.3f})")

        # Save and load pipeline
        print_section("Pipeline Persistence")

        with tempfile.TemporaryDirectory() as temp_dir:
            pipeline_path = os.path.join(temp_dir, "my_pipeline")

            print(f"💾 Saving pipeline to: {pipeline_path}")
            sentiment_pipeline.save(pipeline_path)

            print(f"📂 Loading pipeline from: {pipeline_path}")
            loaded_pipeline = Pipeline.load(pipeline_path)

            # Test loaded pipeline
            loaded_result = loaded_pipeline(test_text)
            print(
                f"✅ Loaded pipeline result: {loaded_result['label']} (confidence: {loaded_result['score']:.3f})"
            )

            # Verify they're the same
            if (
                result["label"] == loaded_result["label"]
                and abs(result["score"] - loaded_result["score"]) < 0.001
            ):
                print("🎉 Pipeline save/load successful!")
            else:
                print("⚠️  Pipeline save/load may have issues")

        return True

    except Exception as e:
        print(f"❌ Pipeline management demo failed: {e}")
        return False


def demo_performance_benchmarks():
    """Demonstrate performance benchmarking capabilities."""
    print_header("Performance Benchmarks")

    try:
        from lingo import Pipeline
        import time
        import statistics

        # Test texts of varying lengths
        test_texts = [
            "Short text.",
            "This is a medium length text for testing purposes.",
            "This is a longer text that contains more words and should take more time to process. It includes multiple sentences and various types of content to simulate real-world usage scenarios.",
            "This is a very long text that contains many words and should take significantly more time to process. It includes multiple sentences, various types of content, and complex language structures to simulate real-world usage scenarios. The text continues with additional information about various topics including technology, science, and current events to provide a comprehensive test of the system's performance capabilities.",
        ]

        # Create pipeline
        pipeline = Pipeline(
            task="sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        )

        print_section("Performance Testing")

        results = {}
        for i, text in enumerate(test_texts):
            print(f"\n📝 Text {i+1} ({len(text.split())} words): {text[:50]}...")

            # Warm-up run
            pipeline(text)

            # Benchmark runs
            times = []
            for run in range(5):
                start_time = time.time()
                pipeline(text)
                end_time = time.time()
                times.append(end_time - start_time)

            avg_time = statistics.mean(times)
            std_time = statistics.stdev(times) if len(times) > 1 else 0

            results[i] = {
                "words": len(text.split()),
                "avg_time": avg_time,
                "std_time": std_time,
                "words_per_second": len(text.split()) / avg_time,
            }

            print(f"   ⚡ Average time: {avg_time:.4f}s ± {std_time:.4f}s")
            print(f"   📊 Words per second: {results[i]['words_per_second']:.1f}")

        # Performance summary
        print_section("Performance Summary")

        total_words = sum(r["words"] for r in results.values())
        total_time = sum(r["avg_time"] for r in results.values())
        overall_wps = total_words / total_time

        print(f"📊 Total words processed: {total_words}")
        print(f"⏱️  Total processing time: {total_time:.4f}s")
        print(f"🚀 Overall throughput: {overall_wps:.1f} words/second")

        # Efficiency analysis
        print_section("Efficiency Analysis")

        for i, result in results.items():
            efficiency = result["words_per_second"] / overall_wps
            print(f"Text {i+1}: {efficiency:.2f}x efficiency")

        return True

    except Exception as e:
        print(f"❌ Performance benchmarks demo failed: {e}")
        return False


def main():
    """Run all demos."""
    print_header("Lingo NLP Toolkit - Advanced Demo")
    print("Showcasing comprehensive NLP capabilities and real-world use cases")

    demos = [
        ("Text Preprocessing", demo_text_preprocessing),
        ("Sentiment Analysis", demo_sentiment_analysis),
        ("Named Entity Recognition", demo_named_entity_recognition),
        ("Text Classification", demo_text_classification),
        ("Embeddings & Similarity", demo_embeddings_and_similarity),
        ("Question Answering", demo_question_answering),
        ("Text Summarization", demo_text_summarization),
        ("Pipeline Management", demo_pipeline_management),
        ("Performance Benchmarks", demo_performance_benchmarks),
    ]

    successful_demos = 0
    total_demos = len(demos)

    for demo_name, demo_func in demos:
        try:
            success = demo_func()
            if success:
                successful_demos += 1
                print(f"✅ {demo_name} completed successfully")
            else:
                print(f"⚠️  {demo_name} completed with warnings")
        except Exception as e:
            print(f"❌ {demo_name} failed: {e}")

    print_header("Demo Summary")
    print(f"🎯 Total demos: {total_demos}")
    print(f"✅ Successful: {successful_demos}")
    print(f"⚠️  Warnings: {total_demos - successful_demos}")

    if successful_demos == total_demos:
        print("\n🎉 All demos completed successfully!")
        print("🚀 Lingo is ready for production use!")
    else:
        print(f"\n⚠️  {total_demos - successful_demos} demo(s) had issues.")
        print("Check the output above for details.")

    print("\n📚 For more examples, check:")
    print("   - examples/basic_usage.py")
    print("   - README.md")
    print("   - INSTALLATION.md")


if __name__ == "__main__":
    main()
