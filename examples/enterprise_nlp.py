#!/usr/bin/env python3
"""
Lingo NLP Toolkit - Enterprise NLP Examples
Production-ready NLP workflows for enterprise applications.
"""

import json
import time
import logging
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import queue
import threading

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class NLPResult:
    """Standardized NLP result structure."""
    text_id: str
    task_type: str
    result: Dict[str, Any]
    confidence: float
    processing_time: float
    timestamp: str
    metadata: Dict[str, Any]

@dataclass
class BatchJob:
    """Batch processing job definition."""
    job_id: str
    task_type: str
    texts: List[str]
    config: Dict[str, Any]
    priority: int = 1
    created_at: str = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()

class EnterpriseNLPPipeline:
    """Enterprise-grade NLP pipeline with advanced features."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the enterprise NLP pipeline.
        
        Args:
            config: Configuration dictionary with pipeline settings
        """
        self.config = config
        self.models = {}
        self.preprocessors = {}
        self.executor = ThreadPoolExecutor(max_workers=config.get('max_workers', 4))
        self.job_queue = queue.PriorityQueue()
        self.results_cache = {}
        self.performance_metrics = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'avg_processing_time': 0.0,
            'total_processing_time': 0.0
        }
        
        self._initialize_models()
        self._start_background_processor()
    
    def _initialize_models(self):
        """Initialize all required NLP models."""
        try:
            from lingo import Pipeline, TextPreprocessor
            
            # Initialize preprocessors
            self.preprocessors['default'] = TextPreprocessor({
                "normalize_unicode": True,
                "lowercase": True,
                "remove_extra_whitespace": True,
                "expand_contractions": True
            })
            
            self.preprocessors['aggressive'] = TextPreprocessor({
                "normalize_unicode": True,
                "lowercase": True,
                "remove_punctuation": True,
                "remove_numbers": True,
                "remove_stopwords": True,
                "lemmatize": True
            })
            
            # Initialize models based on configuration
            model_configs = self.config.get('models', {})
            
            for task, model_config in model_configs.items():
                try:
                    self.models[task] = Pipeline(
                        task=task,
                        model=model_config['model_name'],
                        device=model_config.get('device', 'auto')
                    )
                    logger.info(f"Initialized {task} model: {model_config['model_name']}")
                except Exception as e:
                    logger.error(f"Failed to initialize {task} model: {e}")
            
        except Exception as e:
            logger.error(f"Failed to initialize models: {e}")
            raise
    
    def _start_background_processor(self):
        """Start background job processor."""
        def process_jobs():
            while True:
                try:
                    # Get job from queue (non-blocking)
                    try:
                        priority, job = self.job_queue.get_nowait()
                    except queue.Empty:
                        time.sleep(0.1)
                        continue
                    
                    # Process the job
                    self._process_job(job)
                    self.job_queue.task_done()
                    
                except Exception as e:
                    logger.error(f"Error in background processor: {e}")
                    time.sleep(1)
        
        thread = threading.Thread(target=process_jobs, daemon=True)
        thread.start()
        logger.info("Background job processor started")
    
    def _process_job(self, job: BatchJob):
        """Process a batch job."""
        try:
            logger.info(f"Processing job {job.job_id} with {len(job.texts)} texts")
            
            start_time = time.time()
            results = []
            
            # Process texts based on task type
            if job.task_type == 'sentiment-analysis':
                results = self._batch_sentiment_analysis(job.texts, job.config)
            elif job.task_type == 'ner':
                results = self._batch_ner(job.texts, job.config)
            elif job.task_type == 'text-classification':
                results = self._batch_classification(job.texts, job.config)
            elif job.task_type == 'summarization':
                results = self._batch_summarization(job.texts, job.config)
            else:
                logger.warning(f"Unknown task type: {job.task_type}")
                return
            
            processing_time = time.time() - start_time
            
            # Store results
            for i, (text, result) in enumerate(zip(job.texts, results)):
                nlp_result = NLPResult(
                    text_id=f"{job.job_id}_{i}",
                    task_type=job.task_type,
                    result=result,
                    confidence=result.get('score', 0.0),
                    processing_time=processing_time / len(job.texts),
                    timestamp=datetime.now().isoformat(),
                    metadata={'job_id': job.job_id, 'config': job.config}
                )
                
                self.results_cache[nlp_result.text_id] = nlp_result
            
            # Update metrics
            self.performance_metrics['total_requests'] += len(job.texts)
            self.performance_metrics['successful_requests'] += len(job.texts)
            self.performance_metrics['total_processing_time'] += processing_time
            self.performance_metrics['avg_processing_time'] = \
                self.performance_metrics['total_processing_time'] / self.performance_metrics['total_requests']
            
            logger.info(f"Job {job.job_id} completed successfully in {processing_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Failed to process job {job.job_id}: {e}")
            self.performance_metrics['failed_requests'] += len(job.texts)
    
    def _batch_sentiment_analysis(self, texts: List[str], config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Perform batch sentiment analysis."""
        if 'sentiment-analysis' not in self.models:
            raise ValueError("Sentiment analysis model not available")
        
        model = self.models['sentiment-analysis']
        preprocessor = self.preprocessors.get(config.get('preprocessor', 'default'))
        
        results = []
        for text in texts:
            # Preprocess if specified
            if preprocessor and config.get('preprocess', True):
                processed_text = preprocessor(text)
            else:
                processed_text = text
            
            # Analyze sentiment
            result = model(processed_text)
            results.append(result)
        
        return results
    
    def _batch_ner(self, texts: List[str], config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Perform batch named entity recognition."""
        if 'ner' not in self.models:
            raise ValueError("NER model not available")
        
        model = self.models['ner']
        results = []
        
        for text in texts:
            result = model(text)
            results.append(result)
        
        return results
    
    def _batch_classification(self, texts: List[str], config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Perform batch text classification."""
        if 'text-classification' not in self.models:
            raise ValueError("Text classification model not available")
        
        model = self.models['text-classification']
        results = []
        
        for text in texts:
            result = model(text)
            results.append(result)
        
        return results
    
    def _batch_summarization(self, texts: List[str], config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Perform batch text summarization."""
        if 'summarization' not in self.models:
            raise ValueError("Summarization model not available")
        
        model = self.models['summarization']
        results = []
        
        for text in texts:
            result = model(text)
            results.append(result)
        
        return results
    
    def submit_batch_job(self, job: BatchJob) -> str:
        """Submit a batch job for processing."""
        # Add job to priority queue
        self.job_queue.put((job.priority, job))
        logger.info(f"Submitted batch job {job.job_id} with {len(job.texts)} texts")
        return job.job_id
    
    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get the status of a batch job."""
        # Count completed results for this job
        completed_count = sum(1 for result in self.results_cache.values() 
                            if result.metadata.get('job_id') == job_id)
        
        # Find the job in the queue
        queue_size = self.job_queue.qsize()
        
        return {
            'job_id': job_id,
            'status': 'completed' if completed_count > 0 else 'queued',
            'completed_texts': completed_count,
            'queue_position': queue_size,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_results(self, job_id: str) -> List[NLPResult]:
        """Get results for a completed job."""
        results = [result for result in self.results_cache.values() 
                  if result.metadata.get('job_id') == job_id]
        return results
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        return self.performance_metrics.copy()
    
    def clear_cache(self, older_than_hours: int = 24):
        """Clear old results from cache."""
        cutoff_time = datetime.now() - timedelta(hours=older_than_hours)
        old_results = [text_id for text_id, result in self.results_cache.items()
                      if datetime.fromisoformat(result.timestamp) < cutoff_time]
        
        for text_id in old_results:
            del self.results_cache[text_id]
        
        logger.info(f"Cleared {len(old_results)} old results from cache")

class DocumentIntelligenceSystem:
    """Advanced document intelligence system for enterprise use."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.pipeline = EnterpriseNLPPipeline(config)
        self.document_store = {}
        self.analytics_engine = DocumentAnalytics()
    
    def process_document(self, doc_id: str, content: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process a single document comprehensively."""
        try:
            start_time = time.time()
            
            # Store document
            self.document_store[doc_id] = {
                'content': content,
                'metadata': metadata or {},
                'processed_at': datetime.now().isoformat(),
                'word_count': len(content.split()),
                'char_count': len(content)
            }
            
            # Create comprehensive analysis job
            analysis_job = BatchJob(
                job_id=f"doc_analysis_{doc_id}",
                task_type='comprehensive',
                texts=[content],
                config={'preprocessor': 'default', 'preprocess': True},
                priority=1
            )
            
            # Submit for processing
            self.pipeline.submit_batch_job(analysis_job)
            
            # Wait for completion (in production, this would be async)
            while True:
                status = self.pipeline.get_job_status(analysis_job.job_id)
                if status['status'] == 'completed':
                    break
                time.sleep(0.1)
            
            # Get results
            results = self.pipeline.get_results(analysis_job.job_id)
            
            # Generate analytics
            analytics = self.analytics_engine.analyze_document(content, results)
            
            processing_time = time.time() - start_time
            
            return {
                'doc_id': doc_id,
                'status': 'completed',
                'processing_time': processing_time,
                'results': results,
                'analytics': analytics,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to process document {doc_id}: {e}")
            return {
                'doc_id': doc_id,
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def batch_process_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process multiple documents in batch."""
        results = []
        
        for doc in documents:
            result = self.process_document(
                doc_id=doc['id'],
                content=doc['content'],
                metadata=doc.get('metadata', {})
            )
            results.append(result)
        
        return results
    
    def get_document_insights(self, doc_id: str) -> Dict[str, Any]:
        """Get comprehensive insights for a document."""
        if doc_id not in self.document_store:
            raise ValueError(f"Document {doc_id} not found")
        
        doc = self.document_store[doc_id]
        
        # Get processing results
        results = self.pipeline.get_results(f"doc_analysis_{doc_id}")
        
        # Generate insights
        insights = self.analytics_engine.generate_insights(doc, results)
        
        return insights
    
    def export_analytics(self, format: str = 'json') -> str:
        """Export analytics data in specified format."""
        if format == 'json':
            return json.dumps(self.analytics_engine.get_summary_stats(), indent=2)
        elif format == 'csv':
            return self.analytics_engine.export_to_csv()
        else:
            raise ValueError(f"Unsupported format: {format}")

class DocumentAnalytics:
    """Document analytics and insights generation."""
    
    def __init__(self):
        self.analytics_data = {
            'total_documents': 0,
            'total_words': 0,
            'avg_document_length': 0,
            'sentiment_distribution': {},
            'entity_frequency': {},
            'category_distribution': {},
            'processing_times': []
        }
    
    def analyze_document(self, content: str, results: List[NLPResult]) -> Dict[str, Any]:
        """Analyze a single document and update analytics."""
        # Update basic metrics
        self.analytics_data['total_documents'] += 1
        word_count = len(content.split())
        self.analytics_data['total_words'] += word_count
        
        # Update average document length
        total_docs = self.analytics_data['total_documents']
        total_words = self.analytics_data['total_words']
        self.analytics_data['avg_document_length'] = total_words / total_docs
        
        # Analyze results
        analysis = {
            'word_count': word_count,
            'char_count': len(content),
            'sentences': len(content.split('.')),
            'paragraphs': len(content.split('\n\n')),
            'results_summary': {}
        }
        
        for result in results:
            task_type = result.task_type
            if task_type not in analysis['results_summary']:
                analysis['results_summary'][task_type] = []
            
            analysis['results_summary'][task_type].append({
                'confidence': result.confidence,
                'processing_time': result.processing_time,
                'result': result.result
            })
            
            # Update sentiment distribution
            if task_type == 'sentiment-analysis':
                sentiment = result.result.get('label', 'unknown')
                self.analytics_data['sentiment_distribution'][sentiment] = \
                    self.analytics_data['sentiment_distribution'].get(sentiment, 0) + 1
            
            # Update entity frequency
            elif task_type == 'ner':
                for entity in result.result:
                    entity_type = entity.get('entity_group', 'unknown')
                    entity_text = entity.get('word', 'unknown')
                    
                    if entity_type not in self.analytics_data['entity_frequency']:
                        self.analytics_data['entity_frequency'][entity_type] = {}
                    
                    if entity_text not in self.analytics_data['entity_frequency'][entity_type]:
                        self.analytics_data['entity_frequency'][entity_type][entity_text] = 0
                    
                    self.analytics_data['entity_frequency'][entity_type][entity_text] += 1
        
        return analysis
    
    def generate_insights(self, document: Dict[str, Any], results: List[NLPResult]) -> Dict[str, Any]:
        """Generate insights for a specific document."""
        insights = {
            'document_metrics': {
                'id': document.get('id', 'unknown'),
                'word_count': document.get('word_count', 0),
                'char_count': document.get('char_count', 0),
                'processed_at': document.get('processed_at', 'unknown')
            },
            'nlp_insights': {},
            'recommendations': []
        }
        
        # Process NLP results
        for result in results:
            task_type = result.task_type
            
            if task_type == 'sentiment-analysis':
                sentiment = result.result.get('label', 'unknown')
                confidence = result.confidence
                
                insights['nlp_insights']['sentiment'] = {
                    'label': sentiment,
                    'confidence': confidence,
                    'interpretation': self._interpret_sentiment(sentiment, confidence)
                }
                
                # Generate recommendations based on sentiment
                if sentiment == 'NEGATIVE' and confidence > 0.7:
                    insights['recommendations'].append(
                        "High-confidence negative sentiment detected. Consider immediate review."
                    )
            
            elif task_type == 'ner':
                entities = result.result
                insights['nlp_insights']['entities'] = {
                    'count': len(entities),
                    'types': list(set(entity.get('entity_group', 'unknown') for entity in entities)),
                    'key_entities': [entity.get('word', 'unknown') for entity in entities[:5]]
                }
                
                # Generate recommendations based on entities
                if len(entities) > 10:
                    insights['recommendations'].append(
                        "High entity density detected. Consider content categorization."
                    )
        
        return insights
    
    def _interpret_sentiment(self, sentiment: str, confidence: float) -> str:
        """Interpret sentiment with confidence level."""
        if confidence < 0.5:
            return f"Low confidence {sentiment.lower()} sentiment - results may be unreliable"
        elif confidence < 0.8:
            return f"Moderate confidence {sentiment.lower()} sentiment"
        else:
            return f"High confidence {sentiment.lower()} sentiment"
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics."""
        return self.analytics_data.copy()
    
    def export_to_csv(self) -> str:
        """Export analytics data to CSV format."""
        # This would generate CSV data in a real implementation
        return "CSV export functionality would be implemented here"

def main():
    """Run enterprise NLP examples."""
    print("🏢 Lingo NLP Toolkit - Enterprise Examples")
    print("=" * 60)
    
    # Configuration for enterprise pipeline
    config = {
        'max_workers': 4,
        'models': {
            'sentiment-analysis': {
                'model_name': 'cardiffnlp/twitter-roberta-base-sentiment-latest',
                'device': 'auto'
            },
            'ner': {
                'model_name': 'dslim/bert-base-NER',
                'device': 'auto'
            },
            'text-classification': {
                'model_name': 'bert-base-uncased',
                'device': 'auto'
            },
            'summarization': {
                'model_name': 'facebook/bart-large-cnn',
                'device': 'auto'
            }
        }
    }
    
    try:
        # Initialize enterprise pipeline
        print("\n🚀 Initializing Enterprise NLP Pipeline...")
        pipeline = EnterpriseNLPPipeline(config)
        
        # Example 1: Batch Processing
        print("\n📊 Example 1: Batch Processing")
        print("-" * 40)
        
        # Create batch job
        batch_job = BatchJob(
            job_id="batch_001",
            task_type="sentiment-analysis",
            texts=[
                "This product exceeded all my expectations!",
                "I'm very disappointed with the quality.",
                "The service was okay, nothing special.",
                "Absolutely love this company and their products!"
            ],
            config={'preprocessor': 'default', 'preprocess': True},
            priority=1
        )
        
        # Submit job
        job_id = pipeline.submit_batch_job(batch_job)
        print(f"Submitted batch job: {job_id}")
        
        # Monitor progress
        while True:
            status = pipeline.get_job_status(job_id)
            print(f"Job status: {status['status']}, Completed: {status['completed_texts']}")
            
            if status['status'] == 'completed':
                break
            
            time.sleep(0.5)
        
        # Get results
        results = pipeline.get_results(job_id)
        print(f"\nBatch processing completed! Processed {len(results)} texts")
        
        for result in results:
            print(f"  Text {result.text_id}: {result.result.get('label', 'unknown')} "
                  f"(confidence: {result.confidence:.3f})")
        
        # Example 2: Document Intelligence
        print("\n📄 Example 2: Document Intelligence System")
        print("-" * 40)
        
        doc_system = DocumentIntelligenceSystem(config)
        
        # Process sample documents
        sample_docs = [
            {
                'id': 'doc_001',
                'content': 'Artificial Intelligence is transforming industries worldwide. Companies are adopting AI to improve efficiency and create new opportunities.',
                'metadata': {'category': 'Technology', 'source': 'Industry Report'}
            },
            {
                'id': 'doc_002',
                'content': 'Customer satisfaction has declined significantly this quarter. We need to address service quality issues immediately.',
                'metadata': {'category': 'Business', 'source': 'Customer Survey'}
            }
        ]
        
        print("Processing documents...")
        doc_results = doc_system.batch_process_documents(sample_docs)
        
        for result in doc_results:
            print(f"\nDocument {result['doc_id']}: {result['status']}")
            if result['status'] == 'completed':
                print(f"  Processing time: {result['processing_time']:.2f}s")
                print(f"  Word count: {result['analytics']['word_count']}")
        
        # Example 3: Performance Monitoring
        print("\n📈 Example 3: Performance Monitoring")
        print("-" * 40)
        
        metrics = pipeline.get_performance_metrics()
        print("Performance Metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")
        
        # Example 4: Analytics Export
        print("\n📊 Example 4: Analytics Export")
        print("-" * 40)
        
        analytics_export = doc_system.export_analytics('json')
        print("Analytics Summary:")
        print(analytics_export[:500] + "..." if len(analytics_export) > 500 else analytics_export)
        
        print("\n✅ Enterprise NLP examples completed successfully!")
        print("🚀 Lingo is ready for enterprise production use!")
        
    except Exception as e:
        logger.error(f"Enterprise examples failed: {e}")
        print(f"❌ Examples failed: {e}")

if __name__ == "__main__":
    main()
