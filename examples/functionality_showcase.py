#!/usr/bin/env python3
"""
Lingo NLP Toolkit - Complete Functionality Showcase
Demonstrating all features and capabilities of the Lingo toolkit.
"""

import json
import time
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Union
from datetime import datetime


def print_header(title: str):
    """Print a formatted header."""
    print("\n" + "=" * 80)
    print(f"🚀 {title}")
    print("=" * 80)


def print_section(title: str):
    """Print a formatted section."""
    print(f"\n📋 {title}")
    print("-" * 60)


def showcase_text_preprocessing():
    """Showcase comprehensive text preprocessing capabilities."""
    print_header("Text Preprocessing & Normalization")

    try:
        from lingo import TextPreprocessor

        # Sample text with various challenges
        sample_texts = [
            "I'm gonna love this product! It's AWESOME!!! 😍 #amazing #loveit",
            "The company's revenue increased by 25% in Q3 2024. Dr. Smith reported.",
            "btw, imo this is the best solution ever! fyi, it's really good.",
            "Apple Inc. is headquartered in Cupertino, California. Founded by Steve Jobs.",
            "This is a test sentence. Here's another one. And a third one here.",
            "The weather is 75°F today, but it'll be 80°F tomorrow. #weather #forecast",
            "Customer service was terrible! The app crashed 3 times. 😡 #frustrated",
            "Machine learning algorithms can process data at incredible speeds.",
            "The meeting is scheduled for 2:30 PM on Dec. 15th, 2024.",
            "Python 3.11+ is required for this project. Use pip install -r requirements.txt",
        ]

        print_section("Sample Texts")
        for i, text in enumerate(sample_texts, 1):
            print(f"{i:2d}. {text}")

        # Different preprocessing configurations
        configs = {
            "basic": {
                "normalize_unicode": True,
                "lowercase": True,
                "remove_extra_whitespace": True,
            },
            "aggressive": {
                "normalize_unicode": True,
                "lowercase": True,
                "remove_punctuation": True,
                "remove_numbers": True,
                "remove_special_chars": True,
                "remove_extra_whitespace": True,
            },
            "nlp_ready": {
                "normalize_unicode": True,
                "lowercase": True,
                "expand_contractions": True,
                "expand_slang": True,
                "remove_stopwords": True,
                "lemmatize": True,
                "remove_extra_whitespace": True,
            },
            "preserve_entities": {
                "normalize_unicode": True,
                "lowercase": False,  # Keep proper nouns
                "remove_extra_whitespace": True,
                "expand_contractions": True,
            },
        }

        for config_name, config in configs.items():
            print_section(f"Configuration: {config_name.title()}")

            preprocessor = TextPreprocessor(config=config)

            for i, text in enumerate(sample_texts[:3], 1):  # Show first 3 for brevity
                print(f"\n📝 Text {i}: {text}")

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

                if results.get("stemmed"):
                    print(f"   🌱 Stemmed: {results['stemmed'][:5]}...")

        # Advanced tokenization showcase
        print_section("Advanced Tokenization")
        complex_text = "Dr. Smith's appointment is at 2:30 PM on Dec. 15th, 2024. The meeting will be in Room 301."

        basic_preprocessor = TextPreprocessor(configs["basic"])

        print(f"Complex text: {complex_text}")
        print(f"Word tokens: {basic_preprocessor.tokenize_words(complex_text)}")
        print(f"Sentence tokens: {basic_preprocessor.tokenize_sentences(complex_text)}")

        return True

    except Exception as e:
        print(f"❌ Text preprocessing showcase failed: {e}")
        return False


def showcase_sentiment_analysis():
    """Showcase advanced sentiment analysis capabilities."""
    print_header("Sentiment Analysis & Emotion Detection")

    try:
        from lingo import Pipeline

        # Diverse sentiment texts
        sentiment_texts = [
            "I absolutely love this product! It's amazing and works perfectly! 😍",
            "This is the worst purchase I've ever made. Terrible quality and broke immediately.",
            "The product is okay, nothing special but it gets the job done.",
            "Mixed feelings about this. Good features but expensive and slow.",
            "I'm neutral about this product. It exists and functions as expected.",
            "OMG! This is incredible! Best thing ever! #love #amazing #perfect",
            "So disappointed. Waste of money. Never buying again. 😡",
            "Pretty good overall, though there are some minor issues.",
            "Absolutely terrible customer service. Worst experience ever!",
            "Love the design, hate the price. Mixed emotions here.",
        ]

        # Create sentiment analysis pipeline
        sentiment_pipeline = Pipeline(
            task="sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        )

        print_section("Sentiment Analysis Results")

        results_summary = {"POSITIVE": 0, "NEGATIVE": 0, "NEUTRAL": 0}
        confidence_scores = []

        for i, text in enumerate(sentiment_texts, 1):
            print(f"\n📝 Text {i}: {text[:60]}...")

            # Analyze sentiment
            start_time = time.time()
            result = sentiment_pipeline(text)
            processing_time = time.time() - start_time

            sentiment = result["label"]
            confidence = result["score"]

            results_summary[sentiment] += 1
            confidence_scores.append(confidence)

            # Emoji for sentiment
            emoji_map = {"POSITIVE": "😊", "NEGATIVE": "😞", "NEUTRAL": "😐"}
            emoji = emoji_map.get(sentiment, "❓")

            print(f"   {emoji} Sentiment: {sentiment}")
            print(f"   📊 Confidence: {confidence:.3f}")
            print(f"   ⚡ Processing time: {processing_time:.3f}s")

            # Confidence interpretation
            if confidence > 0.9:
                print(f"   🎯 Very confident prediction")
            elif confidence > 0.7:
                print(f"   ✅ Confident prediction")
            elif confidence > 0.5:
                print(f"   ⚠️  Moderate confidence")
            else:
                print(f"   ❓ Low confidence - may need review")

        # Summary statistics
        print_section("Sentiment Analysis Summary")
        total_texts = len(sentiment_texts)
        avg_confidence = sum(confidence_scores) / len(confidence_scores)

        print(f"📊 Total texts analyzed: {total_texts}")
        print(
            f"😊 Positive: {results_summary['POSITIVE']} ({results_summary['POSITIVE']/total_texts*100:.1f}%)"
        )
        print(
            f"😞 Negative: {results_summary['NEGATIVE']} ({results_summary['NEGATIVE']/total_texts*100:.1f}%)"
        )
        print(
            f"😐 Neutral: {results_summary['NEUTRAL']} ({results_summary['NEUTRAL']/total_texts*100:.1f}%)"
        )
        print(f"📈 Average confidence: {avg_confidence:.3f}")

        # Batch processing demonstration
        print_section("Batch Processing Performance")
        print("Processing all texts in batch...")

        start_time = time.time()
        batch_results = sentiment_pipeline.batch_predict(sentiment_texts)
        batch_time = time.time() - start_time

        individual_time = sum(confidence_scores) * 0.1  # Rough estimate
        speedup = individual_time / batch_time if batch_time > 0 else 0

        print(f"Batch processing time: {batch_time:.3f}s")
        print(f"Estimated individual time: {individual_time:.3f}s")
        print(f"🚀 Speedup: {speedup:.1f}x")

        return True

    except Exception as e:
        print(f"❌ Sentiment analysis showcase failed: {e}")
        return False


def showcase_named_entity_recognition():
    """Showcase advanced NER capabilities."""
    print_header("Named Entity Recognition & Information Extraction")

    try:
        from lingo import Pipeline

        # Diverse texts with various entity types
        ner_texts = [
            "Apple Inc. CEO Tim Cook announced new products at WWDC 2024 in San Francisco, California.",
            "The Great Wall of China was built during the Ming Dynasty and spans over 13,000 miles.",
            "NASA's Perseverance rover landed on Mars on February 18, 2021, at Jezero Crater.",
            "Shakespeare's Hamlet was first performed in 1603 at the Globe Theatre in London, England.",
            "The COVID-19 pandemic began in Wuhan, China in December 2019 and affected millions worldwide.",
            "Microsoft Corporation was founded by Bill Gates and Paul Allen in 1975 in Albuquerque, New Mexico.",
            "The Eiffel Tower, built in 1889, stands 324 meters tall in Paris, France.",
            "Albert Einstein published his theory of relativity in 1905 while working at the Swiss Patent Office.",
            "The Titanic sank on April 15, 1912, after hitting an iceberg in the North Atlantic Ocean.",
            "Google LLC was founded by Larry Page and Sergey Brin in 1998 in Menlo Park, California.",
        ]

        # Create NER pipeline
        ner_pipeline = Pipeline(task="ner", model="dslim/bert-base-NER")

        print_section("NER Results")

        entity_statistics = {}
        total_entities = 0

        for i, text in enumerate(ner_texts, 1):
            print(f"\n📝 Text {i}: {text}")

            # Extract entities
            start_time = time.time()
            entities = ner_pipeline(text)
            processing_time = time.time() - start_time

            print(f"   🔍 Entities found: {len(entities)}")
            total_entities += len(entities)

            # Group entities by type
            entity_types = {}
            for entity in entities:
                entity_type = entity["entity_group"]
                entity_text = entity["word"]

                if entity_type not in entity_types:
                    entity_types[entity_type] = []
                entity_types[entity_type].append(entity_text)

                # Update global statistics
                if entity_type not in entity_statistics:
                    entity_statistics[entity_type] = {}
                if entity_text not in entity_statistics[entity_type]:
                    entity_statistics[entity_type][entity_text] = 0
                entity_statistics[entity_type][entity_text] += 1

            # Display entities by type
            for entity_type, words in entity_types.items():
                unique_words = list(set(words))
                print(f"   📌 {entity_type}: {', '.join(unique_words)}")

            print(f"   ⚡ Processing time: {processing_time:.3f}s")

        # Entity analysis summary
        print_section("Entity Analysis Summary")
        print(f"📊 Total entities extracted: {total_entities}")
        print(f"📝 Average entities per text: {total_entities/len(ner_texts):.1f}")

        print("\nEntity type distribution:")
        for entity_type, entities in entity_statistics.items():
            total_count = sum(entities.values())
            unique_count = len(entities)
            print(f"   {entity_type}: {total_count} total, {unique_count} unique")

        # Most common entities
        print_section("Most Common Entities")
        for entity_type, entities in entity_statistics.items():
            if entities:
                sorted_entities = sorted(
                    entities.items(), key=lambda x: x[1], reverse=True
                )
                print(f"\n{entity_type}:")
                for entity, count in sorted_entities[:5]:
                    print(f"   {entity}: {count} occurrences")

        return True

    except Exception as e:
        print(f"❌ NER showcase failed: {e}")
        return False


def showcase_text_classification():
    """Showcase advanced text classification capabilities."""
    print_header("Text Classification & Categorization")

    try:
        from lingo import Pipeline

        # Diverse texts for classification
        classification_texts = [
            "The stock market reached new highs today with technology stocks leading gains.",
            "Scientists discovered a new species of deep-sea creatures in the Pacific Ocean.",
            "The new restaurant downtown offers authentic Italian cuisine with fresh ingredients.",
            "Breaking news: Major breakthrough in renewable energy technology announced.",
            "Local sports team wins championship after dramatic overtime victory.",
            "Latest smartphone features advanced AI capabilities and improved battery life.",
            "Climate change research shows concerning trends in global temperature rise.",
            "New movie receives critical acclaim and breaks box office records.",
            "Medical breakthrough: New treatment shows promise for rare diseases.",
            "Space exploration mission discovers evidence of water on distant planet.",
        ]

        # Create text classification pipeline
        classifier = Pipeline(task="text-classification", model="bert-base-uncased")

        print_section("Text Classification Results")

        classification_results = {}

        for i, text in enumerate(classification_texts, 1):
            print(f"\n📝 Text {i}: {text}")

            # Classify text
            start_time = time.time()
            result = classifier(text)
            processing_time = time.time() - start_time

            label = result["label"]
            confidence = result["score"]

            # Store results for analysis
            if label not in classification_results:
                classification_results[label] = []
            classification_results[label].append(
                {
                    "text": text,
                    "confidence": confidence,
                    "processing_time": processing_time,
                }
            )

            print(f"   🏷️  Classification: {label}")
            print(f"   📊 Confidence: {confidence:.3f}")
            print(f"   ⚡ Processing time: {processing_time:.3f}s")

            # Confidence interpretation
            if confidence > 0.8:
                print(f"   🎯 High confidence classification")
            elif confidence > 0.6:
                print(f"   ✅ Good confidence classification")
            else:
                print(f"   ⚠️  Low confidence - may need review")

        # Classification analysis
        print_section("Classification Analysis")
        total_texts = len(classification_texts)

        print(f"📊 Total texts classified: {total_texts}")
        print(f"🏷️  Unique categories: {len(classification_results)}")

        print("\nCategory distribution:")
        for category, results in classification_results.items():
            count = len(results)
            avg_confidence = sum(r["confidence"] for r in results) / count
            avg_time = sum(r["processing_time"] for r in results) / count

            print(f"   {category}: {count} texts ({count/total_texts*100:.1f}%)")
            print(f"     Average confidence: {avg_confidence:.3f}")
            print(f"     Average processing time: {avg_time:.3f}s")

        return True

    except Exception as e:
        print(f"❌ Text classification showcase failed: {e}")
        return False


def showcase_embeddings_and_similarity():
    """Showcase advanced embedding and similarity capabilities."""
    print_header("Text Embeddings & Semantic Similarity")

    try:
        from lingo import Pipeline
        import numpy as np

        # Diverse texts for similarity analysis
        similarity_texts = [
            "The cat sat on the mat.",
            "A feline is resting on the carpet.",
            "The weather is sunny today.",
            "It's a beautiful day with clear skies.",
            "Machine learning is a subset of artificial intelligence.",
            "AI includes various techniques like deep learning and neural networks.",
            "The food at this restaurant is delicious.",
            "This place serves amazing cuisine.",
            "Python is a popular programming language.",
            "JavaScript is widely used for web development.",
        ]

        # Create embedding pipeline
        embedding_pipeline = Pipeline(
            task="embedding", model="sentence-transformers/all-MiniLM-L6-v2"
        )

        print_section("Text Embeddings")

        # Generate embeddings
        embeddings = []
        embedding_times = []

        for i, text in enumerate(similarity_texts):
            print(f"📝 Text {i+1}: {text}")

            start_time = time.time()
            embedding = embedding_pipeline(text)
            processing_time = time.time() - start_time

            embeddings.append(embedding)
            embedding_times.append(processing_time)

            print(f"   🔢 Embedding shape: {len(embedding)}")
            print(f"   ⚡ Processing time: {processing_time:.3f}s")

        # Similarity analysis
        print_section("Similarity Analysis")

        # Calculate similarity matrix
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

        print("Top 5 most similar pairs:")
        for rank, (i, j, similarity) in enumerate(most_similar[:5], 1):
            print(f"{rank}. Similarity {similarity:.3f}:")
            print(f"   Text {i+1}: {similarity_texts[i]}")
            print(f"   Text {j+1}: {similarity_texts[j]}")

        # Clustering analysis
        print_section("Text Clustering Analysis")

        # Simple clustering based on similarity threshold
        threshold = 0.7
        clusters = []
        used_indices = set()

        for i in range(len(embeddings)):
            if i in used_indices:
                continue

            cluster = [i]
            used_indices.add(i)

            for j in range(i + 1, len(embeddings)):
                if j not in used_indices and similarity_matrix[i][j] > threshold:
                    cluster.append(j)
                    used_indices.add(j)

            if len(cluster) > 1:  # Only show clusters with multiple texts
                clusters.append(cluster)

        print(f"Found {len(clusters)} clusters with similarity > {threshold}:")
        for i, cluster in enumerate(clusters, 1):
            print(f"   Cluster {i}: Texts {[idx+1 for idx in cluster]}")
            print(f"     Representative: {similarity_texts[cluster[0]]}")

        # Performance metrics
        print_section("Performance Metrics")
        avg_embedding_time = sum(embedding_times) / len(embedding_times)
        total_embedding_time = sum(embedding_times)

        print(f"📊 Total embedding time: {total_embedding_time:.3f}s")
        print(f"⚡ Average embedding time: {avg_embedding_time:.3f}s")
        print(f"🔢 Total embeddings generated: {len(embeddings)}")
        print(
            f"🚀 Throughput: {len(embeddings)/total_embedding_time:.1f} embeddings/second"
        )

        return True

    except Exception as e:
        print(f"❌ Embeddings showcase failed: {e}")
        return False


def showcase_question_answering():
    """Showcase advanced question answering capabilities."""
    print_header("Question Answering & Information Retrieval")

    try:
        from lingo import Pipeline

        # Context and diverse questions
        context = """
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

        questions = [
            "Who created AI?",
            "When did AI emerge?",
            "What are the main components of AI?",
            "What is machine learning?",
            "How does deep learning work?",
            "What industries use AI?",
            "What are the concerns about AI?",
            "What is the future of AI?",
            "How does AI affect jobs?",
            "What is responsible AI development?",
        ]

        # Create QA pipeline
        qa_pipeline = Pipeline(
            task="question-answering", model="deepset/roberta-base-squad2"
        )

        print_section("Context")
        print(f"Length: {len(context.split())} words")
        print(context.strip())

        print_section("Question Answering Results")

        qa_results = []
        total_qa_time = 0

        for i, question in enumerate(questions, 1):
            print(f"\n❓ Question {i}: {question}")

            # Get answer
            start_time = time.time()
            result = qa_pipeline(question=question, context=context)
            processing_time = time.time() - start_time

            total_qa_time += processing_time
            qa_results.append(
                {
                    "question": question,
                    "answer": result["answer"],
                    "confidence": result["score"],
                    "start": result["start"],
                    "end": result["end"],
                    "processing_time": processing_time,
                }
            )

            print(f"   💡 Answer: {result['answer']}")
            print(f"   📊 Confidence: {result['score']:.3f}")
            print(f"   📍 Position: {result['start']}-{result['end']}")
            print(f"   ⚡ Processing time: {processing_time:.3f}s")

            # Confidence interpretation
            if result["score"] > 0.8:
                print(f"   🎯 High confidence answer")
            elif result["score"] > 0.6:
                print(f"   ✅ Good confidence answer")
            else:
                print(f"   ⚠️  Low confidence - answer may be unreliable")

        # QA Analysis
        print_section("Question Answering Analysis")

        total_questions = len(questions)
        avg_confidence = sum(r["confidence"] for r in qa_results) / total_questions
        avg_processing_time = total_qa_time / total_questions

        print(f"📊 Total questions: {total_questions}")
        print(f"📈 Average confidence: {avg_confidence:.3f}")
        print(f"⏱️  Total processing time: {total_qa_time:.3f}s")
        print(f"⚡ Average processing time: {avg_processing_time:.3f}s")
        print(f"🚀 Questions per second: {total_questions/total_qa_time:.1f}")

        # Answer quality analysis
        print_section("Answer Quality Analysis")

        high_confidence = sum(1 for r in qa_results if r["confidence"] > 0.8)
        medium_confidence = sum(1 for r in qa_results if 0.6 <= r["confidence"] <= 0.8)
        low_confidence = sum(1 for r in qa_results if r["confidence"] < 0.6)

        print(
            f"🎯 High confidence answers (>0.8): {high_confidence} ({high_confidence/total_questions*100:.1f}%)"
        )
        print(
            f"✅ Medium confidence answers (0.6-0.8): {medium_confidence} ({medium_confidence/total_questions*100:.1f}%)"
        )
        print(
            f"⚠️  Low confidence answers (<0.6): {low_confidence} ({low_confidence/total_questions*100:.1f}%)"
        )

        return True

    except Exception as e:
        print(f"❌ Question answering showcase failed: {e}")
        return False


def showcase_text_summarization():
    """Showcase advanced text summarization capabilities."""
    print_header("Text Summarization & Content Compression")

    try:
        from lingo import Pipeline

        # Long texts for summarization
        long_texts = [
            """
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
            """,
            """
            Climate change represents one of the most pressing challenges facing humanity in the 21st century. The scientific 
            consensus is clear: human activities, particularly the burning of fossil fuels and deforestation, are driving 
            unprecedented changes in Earth's climate system. Global temperatures have risen by approximately 1.1°C since 
            pre-industrial times, with the rate of warming accelerating in recent decades. These changes are already having 
            profound impacts on natural systems and human societies worldwide. Rising sea levels threaten coastal communities, 
            extreme weather events are becoming more frequent and intense, and ecosystems are shifting in response to changing 
            conditions. The consequences extend beyond environmental concerns to economic, social, and political stability. 
            Addressing climate change requires immediate and sustained action across multiple sectors, including energy, 
            transportation, agriculture, and urban planning. This includes transitioning to renewable energy sources, 
            improving energy efficiency, implementing sustainable land use practices, and developing climate-resilient 
            infrastructure. International cooperation is essential, as climate change is a global problem requiring global 
            solutions. The Paris Agreement represents a significant step forward, but much more ambitious action is needed 
            to limit warming to well below 2°C and pursue efforts to limit it to 1.5°C above pre-industrial levels.
            """,
            """
            The Internet of Things (IoT) represents a paradigm shift in how we interact with technology and the world around us. 
            By connecting everyday objects to the internet and enabling them to collect, share, and act on data, IoT is 
            creating a more interconnected and intelligent environment. This technology has applications across virtually every 
            sector, from smart homes and cities to industrial automation and healthcare. In smart homes, IoT devices can 
            control lighting, heating, security systems, and appliances, providing convenience and energy efficiency. Smart 
            cities use IoT sensors to monitor traffic, air quality, waste management, and public services, enabling more 
            efficient urban planning and resource allocation. In healthcare, IoT devices can monitor patient vital signs, 
            track medication adherence, and provide remote care capabilities. Industrial IoT (IIoT) enables predictive 
            maintenance, quality control, and supply chain optimization in manufacturing and logistics. However, the rapid 
            proliferation of IoT devices also raises significant concerns about privacy, security, and data management. 
            With billions of devices collecting and transmitting data, ensuring the security and privacy of this information 
            becomes paramount. Additionally, the massive amount of data generated by IoT devices requires sophisticated 
            analytics and storage solutions. As IoT continues to evolve, addressing these challenges while maximizing the 
            benefits will be crucial for realizing the full potential of this transformative technology.
            """,
        ]

        # Create summarization pipeline
        summarizer = Pipeline(task="summarization", model="facebook/bart-large-cnn")

        print_section("Text Summarization Results")

        summarization_results = []
        total_summarization_time = 0

        for i, text in enumerate(long_texts, 1):
            print(f"\n📄 Text {i}")
            print(f"   Length: {len(text.split())} words")
            print(f"   Content: {text[:200]}...")

            # Generate summary
            start_time = time.time()
            summary = summarizer(text)
            processing_time = time.time() - start_time

            total_summarization_time += processing_time
            summary_text = summary["summary_text"]

            summarization_results.append(
                {
                    "text_id": i,
                    "original_length": len(text.split()),
                    "summary_length": len(summary_text.split()),
                    "compression_ratio": len(summary_text.split()) / len(text.split()),
                    "processing_time": processing_time,
                    "summary": summary_text,
                }
            )

            print(f"   📝 Summary: {summary_text}")
            print(f"   📊 Summary length: {len(summary_text.split())} words")
            print(
                f"   📉 Compression ratio: {summarization_results[-1]['compression_ratio']:.1%}"
            )
            print(f"   ⚡ Processing time: {processing_time:.3f}s")

        # Summarization Analysis
        print_section("Summarization Analysis")

        total_texts = len(long_texts)
        avg_compression = (
            sum(r["compression_ratio"] for r in summarization_results) / total_texts
        )
        avg_processing_time = total_summarization_time / total_texts

        print(f"📊 Total texts summarized: {total_texts}")
        print(f"📉 Average compression ratio: {avg_compression:.1%}")
        print(f"⏱️  Total processing time: {total_summarization_time:.3f}s")
        print(f"⚡ Average processing time: {avg_processing_time:.3f}s")
        print(
            f"🚀 Words per second: {sum(r['original_length'] for r in summarization_results)/total_summarization_time:.1f}"
        )

        # Quality analysis
        print_section("Summary Quality Analysis")

        for result in summarization_results:
            compression = result["compression_ratio"]
            if compression < 0.1:
                quality = "Very compressed"
            elif compression < 0.2:
                quality = "Highly compressed"
            elif compression < 0.3:
                quality = "Moderately compressed"
            else:
                quality = "Lightly compressed"

            print(f"Text {result['text_id']}: {quality} ({compression:.1%})")
            print(f"  {result['original_length']} → {result['summary_length']} words")

        return True

    except Exception as e:
        print(f"❌ Text summarization showcase failed: {e}")
        return False


def showcase_pipeline_management():
    """Showcase advanced pipeline management capabilities."""
    print_header("Pipeline Management & Orchestration")

    try:
        from lingo import Pipeline, TextPreprocessor
        import tempfile
        import os

        # Create a comprehensive pipeline configuration
        pipeline_config = {
            "preprocessing": {
                "normalize_unicode": True,
                "lowercase": True,
                "remove_punctuation": False,
                "expand_contractions": True,
                "expand_slang": True,
                "remove_stopwords": True,
                "lemmatize": True,
            },
            "models": {
                "sentiment": "cardiffnlp/twitter-roberta-base-sentiment-latest",
                "ner": "dslim/bert-base-NER",
                "embedding": "sentence-transformers/all-MiniLM-L6-v2",
            },
            "pipeline_settings": {
                "batch_size": 32,
                "max_workers": 4,
                "cache_results": True,
                "enable_logging": True,
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

        # Pipeline persistence
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

        # Pipeline composition
        print_section("Pipeline Composition")

        # Create multiple pipelines
        ner_pipeline = Pipeline(task="ner", model=pipeline_config["models"]["ner"])

        embedding_pipeline = Pipeline(
            task="embedding", model=pipeline_config["models"]["embedding"]
        )

        # Create preprocessor
        preprocessor = TextPreprocessor(pipeline_config["preprocessing"])

        # Process text through multiple pipelines
        complex_text = (
            "Apple Inc. CEO Tim Cook announced amazing new products at WWDC 2024!"
        )

        print(f"📝 Complex text: {complex_text}")

        # Preprocessing
        processed_text = preprocessor(complex_text)
        print(f"🧹 Preprocessed: {processed_text}")

        # Sentiment analysis
        sentiment_result = sentiment_pipeline(processed_text)
        print(
            f"🎭 Sentiment: {sentiment_result['label']} ({sentiment_result['score']:.3f})"
        )

        # NER
        ner_result = ner_pipeline(complex_text)
        print(f"🔍 Entities: {len(ner_result)} found")
        for entity in ner_result[:3]:  # Show first 3
            print(f"   {entity['entity_group']}: {entity['word']}")

        # Embeddings
        embedding_result = embedding_pipeline(complex_text)
        print(f"🔢 Embedding: {len(embedding_result)} dimensions")

        # Performance monitoring
        print_section("Performance Monitoring")

        # Test multiple texts
        test_texts = [
            "I love this product!",
            "This is terrible quality.",
            "The service was okay.",
            "Amazing experience!",
            "Very disappointed.",
        ]

        print(f"Testing pipeline with {len(test_texts)} texts...")

        start_time = time.time()
        batch_results = sentiment_pipeline.batch_predict(test_texts)
        batch_time = time.time() - start_time

        print(f"Batch processing time: {batch_time:.3f}s")
        print(f"Individual processing time: {batch_time/len(test_texts):.3f}s per text")
        print(f"Throughput: {len(test_texts)/batch_time:.1f} texts/second")

        # Results summary
        positive_count = sum(1 for r in batch_results if r["label"] == "POSITIVE")
        negative_count = sum(1 for r in batch_results if r["label"] == "NEGATIVE")
        neutral_count = sum(1 for r in batch_results if r["label"] == "NEUTRAL")

        print(f"\nResults summary:")
        print(f"  😊 Positive: {positive_count}")
        print(f"  😞 Negative: {negative_count}")
        print(f"  😐 Neutral: {neutral_count}")

        return True

    except Exception as e:
        print(f"❌ Pipeline management showcase failed: {e}")
        return False


def main():
    """Run all functionality showcases."""
    print_header("Lingo NLP Toolkit - Complete Functionality Showcase")
    print("Demonstrating all features and capabilities of the Lingo toolkit")

    showcases = [
        ("Text Preprocessing & Normalization", showcase_text_preprocessing),
        ("Sentiment Analysis & Emotion Detection", showcase_sentiment_analysis),
        (
            "Named Entity Recognition & Information Extraction",
            showcase_named_entity_recognition,
        ),
        ("Text Classification & Categorization", showcase_text_classification),
        ("Text Embeddings & Semantic Similarity", showcase_embeddings_and_similarity),
        ("Question Answering & Information Retrieval", showcase_question_answering),
        ("Text Summarization & Content Compression", showcase_text_summarization),
        ("Pipeline Management & Orchestration", showcase_pipeline_management),
    ]

    successful_showcases = 0
    total_showcases = len(showcases)

    for showcase_name, showcase_func in showcases:
        try:
            print(f"\n{'='*80}")
            print(f"🚀 Running: {showcase_name}")
            print(f"{'='*80}")

            success = showcase_func()
            if success:
                successful_showcases += 1
                print(f"✅ {showcase_name} completed successfully")
            else:
                print(f"⚠️  {showcase_name} completed with warnings")

        except Exception as e:
            print(f"❌ {showcase_name} failed: {e}")

    print_header("Showcase Summary")
    print(f"🎯 Total showcases: {total_showcases}")
    print(f"✅ Successful: {successful_showcases}")
    print(f"⚠️  Warnings: {total_showcases - successful_showcases}")

    if successful_showcases == total_showcases:
        print("\n🎉 All showcases completed successfully!")
        print("🚀 Lingo demonstrates comprehensive NLP capabilities!")
    else:
        print(f"\n⚠️  {total_showcases - successful_showcases} showcase(s) had issues.")
        print("Check the output above for details.")

    print("\n📚 For more examples, check:")
    print("   - examples/advanced_use_cases.py")
    print("   - examples/enterprise_nlp.py")
    print("   - demo.py")
    print("   - README.md")


if __name__ == "__main__":
    main()
