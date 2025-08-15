#!/usr/bin/env python3
"""
Lingo NLP Toolkit - Advanced Use Cases
Real-world applications and advanced NLP workflows.
"""

import json
import time
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

@dataclass
class Document:
    """Document structure for processing."""
    id: str
    title: str
    content: str
    category: str
    timestamp: str

class SocialMediaAnalyzer:
    """Advanced social media content analysis."""
    
    def __init__(self):
        from lingo import Pipeline
        
        self.sentiment_pipeline = Pipeline(
            task="sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest"
        )
        
        self.ner_pipeline = Pipeline(
            task="ner",
            model="dslim/bert-base-NER"
        )
        
        self.classifier = Pipeline(
            task="text-classification",
            model="bert-base-uncased"
        )
    
    def analyze_social_media_posts(self, posts: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Analyze social media posts comprehensively."""
        results = []
        
        for post in posts:
            analysis = {
                'post_id': post['id'],
                'text': post['text'],
                'timestamp': post['timestamp']
            }
            
            # Sentiment analysis
            sentiment = self.sentiment_pipeline(post['text'])
            analysis['sentiment'] = sentiment['label']
            analysis['sentiment_score'] = sentiment['score']
            
            # Named entity recognition
            entities = self.ner_pipeline(post['text'])
            analysis['entities'] = entities
            
            # Content classification
            classification = self.classifier(post['text'])
            analysis['category'] = classification['label']
            analysis['category_score'] = classification['score']
            
            # Engagement prediction (simplified)
            analysis['engagement_score'] = self._predict_engagement(post['text'])
            
            results.append(analysis)
        
        return results
    
    def _predict_engagement(self, text: str) -> float:
        """Predict engagement based on text characteristics."""
        # Simple heuristic-based engagement prediction
        engagement_factors = {
            'hashtags': len([word for word in text.split() if word.startswith('#')]),
            'mentions': len([word for word in text.split() if word.startswith('@')]),
            'exclamation_marks': text.count('!'),
            'question_marks': text.count('?'),
            'length': len(text.split()),
            'has_emoji': any(char in '😀😃😄😁😆😅😂🤣😊😇🙂🙃😉😌😍🥰😘😗😙😚😋😛😝😜🤪🤨🧐🤓😎🤩🥳😏😒😞😔😟😕🙁☹️😣😖😫😩🥺😢😭😤😠😡🤬🤯😳🥵🥶😱😨😰😥😓🤗🤔🤭🤫🤥😶😐😑😯😦😧😮😲🥱😴🤤😪😵🤐🥴🤢🤮🤧😷🤒🤕' for char in text)
        }
        
        # Calculate engagement score
        score = 0
        score += engagement_factors['hashtags'] * 0.1
        score += engagement_factors['mentions'] * 0.1
        score += engagement_factors['exclamation_marks'] * 0.05
        score += engagement_factors['question_marks'] * 0.05
        score += min(engagement_factors['length'] * 0.01, 0.2)
        score += 0.1 if engagement_factors['has_emoji'] else 0
        
        return min(score, 1.0)

class CustomerFeedbackAnalyzer:
    """Advanced customer feedback analysis system."""
    
    def __init__(self):
        from lingo import Pipeline, TextPreprocessor
        
        self.sentiment_pipeline = Pipeline(
            task="sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest"
        )
        
        self.qa_pipeline = Pipeline(
            task="question-answering",
            model="deepset/roberta-base-squad2"
        )
        
        self.preprocessor = TextPreprocessor({
            "normalize_unicode": True,
            "lowercase": True,
            "expand_contractions": True,
            "remove_stopwords": True,
            "lemmatize": True
        })
    
    def analyze_feedback_batch(self, feedback_list: List[Dict[str, str]]) -> Dict[str, Any]:
        """Analyze a batch of customer feedback."""
        results = {
            'total_feedback': len(feedback_list),
            'sentiment_distribution': {},
            'common_issues': {},
            'improvement_suggestions': [],
            'response_priorities': []
        }
        
        # Process each feedback
        for feedback in feedback_list:
            # Sentiment analysis
            sentiment = self.sentiment_pipeline(feedback['text'])
            sentiment_label = sentiment['label']
            
            results['sentiment_distribution'][sentiment_label] = \
                results['sentiment_distribution'].get(sentiment_label, 0) + 1
            
            # Extract key information
            processed_text = self.preprocessor(feedback['text'])
            results['common_issues'].update(self._extract_issues(processed_text))
            
            # Generate improvement suggestions
            if sentiment_label == 'NEGATIVE':
                suggestion = self._generate_improvement_suggestion(feedback['text'])
                results['improvement_suggestions'].append(suggestion)
                
                # Prioritize responses
                priority = self._calculate_response_priority(feedback['text'], sentiment['score'])
                results['response_priorities'].append({
                    'feedback_id': feedback['id'],
                    'priority': priority,
                    'text': feedback['text'][:100] + '...'
                })
        
        # Sort response priorities
        results['response_priorities'].sort(key=lambda x: x['priority'], reverse=True)
        
        return results
    
    def _extract_issues(self, text: str) -> Dict[str, int]:
        """Extract common issues from feedback text."""
        # Simple keyword-based issue extraction
        issue_keywords = {
            'slow': 'performance',
            'bug': 'technical_issue',
            'error': 'technical_issue',
            'crash': 'technical_issue',
            'expensive': 'pricing',
            'cost': 'pricing',
            'price': 'pricing',
            'difficult': 'usability',
            'hard': 'usability',
            'confusing': 'usability',
            'broken': 'technical_issue',
            'doesn\'t work': 'technical_issue',
            'not working': 'technical_issue'
        }
        
        issues = {}
        text_lower = text.lower()
        
        for keyword, issue_type in issue_keywords.items():
            if keyword in text_lower:
                issues[issue_type] = issues.get(issue_type, 0) + 1
        
        return issues
    
    def _generate_improvement_suggestion(self, text: str) -> str:
        """Generate improvement suggestions based on feedback."""
        # Simple rule-based suggestion generation
        suggestions = {
            'performance': 'Consider optimizing system performance and response times.',
            'technical_issue': 'Investigate and fix the reported technical issues.',
            'pricing': 'Review pricing strategy and consider offering more competitive rates.',
            'usability': 'Improve user interface and user experience design.',
            'customer_service': 'Enhance customer support and response times.'
        }
        
        # Determine the main issue
        main_issue = 'general'
        for issue_type in suggestions.keys():
            if issue_type in text.lower():
                main_issue = issue_type
                break
        
        return suggestions.get(main_issue, 'Review and address the customer concerns.')
    
    def _calculate_response_priority(self, text: str, sentiment_score: float) -> float:
        """Calculate response priority for negative feedback."""
        # Higher priority for more negative sentiment
        base_priority = 1.0 - sentiment_score
        
        # Additional factors
        urgency_indicators = ['urgent', 'immediate', 'asap', 'critical', 'emergency']
        has_urgency = any(indicator in text.lower() for indicator in urgency_indicators)
        
        if has_urgency:
            base_priority += 0.3
        
        return min(base_priority, 1.0)

class DocumentProcessor:
    """Advanced document processing and analysis."""
    
    def __init__(self):
        from lingo import Pipeline, TextPreprocessor
        
        self.ner_pipeline = Pipeline(
            task="ner",
            model="dslim/bert-base-NER"
        )
        
        self.summarizer = Pipeline(
            task="summarization",
            model="facebook/bart-large-cnn"
        )
        
        self.preprocessor = TextPreprocessor({
            "normalize_unicode": True,
            "lowercase": False,  # Keep case for proper nouns
            "remove_extra_whitespace": True
        })
    
    def process_document_collection(self, documents: List[Document]) -> Dict[str, Any]:
        """Process a collection of documents."""
        results = {
            'total_documents': len(documents),
            'document_summaries': [],
            'entity_analysis': {},
            'category_insights': {},
            'processing_metadata': {}
        }
        
        start_time = time.time()
        
        for doc in documents:
            # Preprocess document
            processed_content = self.preprocessor(doc.content)
            
            # Generate summary
            summary = self.summarizer(doc.content)
            
            # Extract entities
            entities = self.ner_pipeline(doc.content)
            
            # Store results
            doc_result = {
                'id': doc.id,
                'title': doc.title,
                'category': doc.category,
                'summary': summary['summary_text'],
                'entities': entities,
                'word_count': len(doc.content.split()),
                'summary_ratio': len(summary['summary_text'].split()) / len(doc.content.split())
            }
            
            results['document_summaries'].append(doc_result)
            
            # Aggregate entity information
            for entity in entities:
                entity_type = entity['entity_group']
                entity_text = entity['word']
                
                if entity_type not in results['entity_analysis']:
                    results['entity_analysis'][entity_type] = {}
                
                if entity_text not in results['entity_analysis'][entity_type]:
                    results['entity_analysis'][entity_type][entity_text] = 0
                
                results['entity_analysis'][entity_type][entity_text] += 1
            
            # Category insights
            if doc.category not in results['category_insights']:
                results['category_insights'][doc.category] = {
                    'count': 0,
                    'total_words': 0,
                    'avg_summary_ratio': 0
                }
            
            cat_insights = results['category_insights'][doc.category]
            cat_insights['count'] += 1
            cat_insights['total_words'] += doc_result['word_count']
            cat_insights['avg_summary_ratio'] = \
                (cat_insights['avg_summary_ratio'] * (cat_insights['count'] - 1) + doc_result['summary_ratio']) / cat_insights['count']
        
        # Calculate processing metadata
        processing_time = time.time() - start_time
        results['processing_metadata'] = {
            'total_processing_time': processing_time,
            'avg_time_per_document': processing_time / len(documents),
            'total_words_processed': sum(doc['word_count'] for doc in results['document_summaries']),
            'words_per_second': sum(doc['word_count'] for doc in results['document_summaries']) / processing_time
        }
        
        return results

class ContentRecommendationEngine:
    """Content recommendation system using embeddings."""
    
    def __init__(self):
        from lingo import Pipeline
        
        self.embedding_pipeline = Pipeline(
            task="embedding",
            model="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        self.sentiment_pipeline = Pipeline(
            task="sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest"
        )
    
    def build_content_index(self, content_items: List[Dict[str, str]]) -> Dict[str, Any]:
        """Build a searchable content index."""
        index = {
            'content_items': {},
            'embeddings': {},
            'metadata': {}
        }
        
        for item in content_items:
            item_id = item['id']
            
            # Store content
            index['content_items'][item_id] = {
                'title': item['title'],
                'content': item['content'],
                'category': item.get('category', 'general'),
                'tags': item.get('tags', [])
            }
            
            # Generate embeddings
            combined_text = f"{item['title']} {item['content']}"
            embedding = self.embedding_pipeline(combined_text)
            index['embeddings'][item_id] = embedding
            
            # Analyze sentiment
            sentiment = self.sentiment_pipeline(item['content'])
            index['metadata'][item_id] = {
                'sentiment': sentiment['label'],
                'sentiment_score': sentiment['score'],
                'word_count': len(item['content'].split()),
                'created_at': item.get('created_at', 'unknown')
            }
        
        return index
    
    def find_similar_content(self, query: str, content_index: Dict[str, Any], top_k: int = 5) -> List[Dict[str, Any]]:
        """Find content similar to the query."""
        import numpy as np
        
        # Generate query embedding
        query_embedding = self.embedding_pipeline(query)
        
        # Calculate similarities
        similarities = []
        for item_id, embedding in content_index['embeddings'].items():
            # Cosine similarity
            dot_product = np.dot(query_embedding, embedding)
            norm_query = np.linalg.norm(query_embedding)
            norm_item = np.linalg.norm(embedding)
            similarity = dot_product / (norm_query * norm_item)
            
            similarities.append({
                'item_id': item_id,
                'similarity': similarity,
                'content': content_index['content_items'][item_id],
                'metadata': content_index['metadata'][item_id]
            })
        
        # Sort by similarity and return top-k
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        return similarities[:top_k]
    
    def get_personalized_recommendations(self, user_profile: Dict[str, Any], content_index: Dict[str, Any], top_k: int = 10) -> List[Dict[str, Any]]:
        """Get personalized content recommendations."""
        import numpy as np
        
        # Create user preference vector
        user_preferences = {
            'positive_sentiment': 0.3,
            'negative_sentiment': -0.2,
            'neutral_sentiment': 0.1,
            'preferred_categories': user_profile.get('preferred_categories', []),
            'content_length_preference': user_profile.get('preferred_content_length', 'medium')
        }
        
        # Score each content item
        scored_items = []
        for item_id, metadata in content_index['metadata'].items():
            score = 0
            
            # Sentiment preference
            if metadata['sentiment'] == 'POSITIVE':
                score += user_preferences['positive_sentiment']
            elif metadata['sentiment'] == 'NEGATIVE':
                score += user_preferences['negative_sentiment']
            else:
                score += user_preferences['neutral_sentiment']
            
            # Category preference
            content_item = content_index['content_items'][item_id]
            if content_item['category'] in user_preferences['preferred_categories']:
                score += 0.5
            
            # Content length preference
            word_count = metadata['word_count']
            if user_preferences['content_length_preference'] == 'short' and word_count < 500:
                score += 0.3
            elif user_preferences['content_length_preference'] == 'medium' and 500 <= word_count <= 2000:
                score += 0.3
            elif user_preferences['content_length_preference'] == 'long' and word_count > 2000:
                score += 0.3
            
            scored_items.append({
                'item_id': item_id,
                'score': score,
                'content': content_item,
                'metadata': metadata
            })
        
        # Sort by score and return top-k
        scored_items.sort(key=lambda x: x['score'], reverse=True)
        return scored_items[:top_k]

def main():
    """Run advanced use case examples."""
    print("🚀 Lingo NLP Toolkit - Advanced Use Cases")
    print("=" * 60)
    
    # Example 1: Social Media Analysis
    print("\n📱 Example 1: Social Media Content Analysis")
    print("-" * 40)
    
    social_posts = [
        {'id': '1', 'text': 'Just tried the new AI tool and it\'s amazing! #AI #Innovation', 'timestamp': '2024-01-15'},
        {'id': '2', 'text': 'Frustrated with the slow performance of this app. Needs improvement.', 'timestamp': '2024-01-15'},
        {'id': '3', 'text': 'Great customer service from @TechSupport! They solved my issue quickly.', 'timestamp': '2024-01-15'}
    ]
    
    analyzer = SocialMediaAnalyzer()
    results = analyzer.analyze_social_media_posts(social_posts)
    
    for result in results:
        print(f"\nPost {result['post_id']}:")
        print(f"  Sentiment: {result['sentiment']} ({result['sentiment_score']:.3f})")
        print(f"  Category: {result['category']} ({result['category_score']:.3f})")
        print(f"  Engagement Score: {result['engagement_score']:.3f}")
    
    # Example 2: Customer Feedback Analysis
    print("\n📝 Example 2: Customer Feedback Analysis")
    print("-" * 40)
    
    feedback_data = [
        {'id': '1', 'text': 'The app crashes every time I try to upload a file. This is urgent!'},
        {'id': '2', 'text': 'Love the new interface! Much easier to use now.'},
        {'id': '3', 'text': 'The pricing is too expensive for what you get.'}
    ]
    
    feedback_analyzer = CustomerFeedbackAnalyzer()
    feedback_results = feedback_analyzer.analyze_feedback_batch(feedback_data)
    
    print(f"Total feedback: {feedback_results['total_feedback']}")
    print(f"Sentiment distribution: {feedback_results['sentiment_distribution']}")
    print(f"Common issues: {feedback_results['common_issues']}")
    
    # Example 3: Document Processing
    print("\n📄 Example 3: Document Processing & Analysis")
    print("-" * 40)
    
    documents = [
        Document(
            id='1',
            title='AI in Healthcare',
            content='Artificial Intelligence is revolutionizing healthcare by enabling early disease detection, personalized treatment plans, and improved patient outcomes. Machine learning algorithms can analyze medical images, predict patient risks, and assist doctors in making better decisions.',
            category='Technology',
            timestamp='2024-01-15'
        ),
        Document(
            id='2',
            title='Climate Change Impact',
            content='Climate change is affecting global weather patterns, leading to more frequent extreme weather events. Rising temperatures are causing sea level rise and threatening biodiversity worldwide.',
            category='Environment',
            timestamp='2024-01-15'
        )
    ]
    
    doc_processor = DocumentProcessor()
    doc_results = doc_processor.process_document_collection(documents)
    
    print(f"Processed {doc_results['total_documents']} documents")
    print(f"Total processing time: {doc_results['processing_metadata']['total_processing_time']:.2f}s")
    print(f"Words per second: {doc_results['processing_metadata']['words_per_second']:.1f}")
    
    # Example 4: Content Recommendation
    print("\n🎯 Example 4: Content Recommendation Engine")
    print("-" * 40)
    
    content_items = [
        {'id': '1', 'title': 'Introduction to Machine Learning', 'content': 'Machine learning is a subset of artificial intelligence that enables computers to learn from data without being explicitly programmed.', 'category': 'AI', 'tags': ['machine-learning', 'ai', 'tutorial']},
        {'id': '2', 'title': 'Python Programming Basics', 'content': 'Python is a versatile programming language known for its simplicity and readability. It\'s perfect for beginners and experts alike.', 'category': 'Programming', 'tags': ['python', 'programming', 'tutorial']}
    ]
    
    rec_engine = ContentRecommendationEngine()
    content_index = rec_engine.build_content_index(content_items)
    
    # Find similar content
    query = "artificial intelligence and machine learning"
    similar_content = rec_engine.find_similar_content(query, content_index, top_k=2)
    
    print(f"Query: {query}")
    for item in similar_content:
        print(f"  Similar item: {item['content']['title']} (similarity: {item['similarity']:.3f})")
    
    print("\n✅ Advanced use cases completed successfully!")
    print("🚀 Lingo is ready for production NLP applications!")

if __name__ == "__main__":
    main()
