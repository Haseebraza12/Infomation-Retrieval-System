"""
Reranking Module for Cortex IR System
Neural reranking with cross-encoder and optimizations
"""

from typing import List, Dict, Tuple
import numpy as np

from sentence_transformers import CrossEncoder

import config
from utils import setup_logging, timer, batch_iterator, truncate_text

logger = setup_logging(__name__)


class NeuralReranker:
    """
    Neural reranking using cross-encoder with optimizations
    """
    
    def __init__(self, model_name: str = None, use_onnx: bool = False):
        """
        Initialize neural reranker
        
        Args:
            model_name: Cross-encoder model name (default from config)
            use_onnx: Whether to use ONNX optimization
        """
        if model_name is None:
            model_name = config.CROSS_ENCODER_MODEL
        
        logger.info(f"Loading cross-encoder model: {model_name}")
        
        try:
            # Load cross-encoder
            self.model = CrossEncoder(model_name, max_length=512)
            self.available = True
            
            logger.info("Neural reranker initialized successfully")
            
        except Exception as e:
            logger.error(f"Error loading cross-encoder: {e}")
            logger.warning("Reranking will be disabled")
            self.model = None
            self.available = False
    
    def _prepare_pairs(
        self, 
        query: str, 
        documents: List[Dict]
    ) -> List[Tuple[str, str]]:
        """
        Prepare (query, document) pairs for cross-encoder
        
        Args:
            query: Query string
            documents: List of document dictionaries
            
        Returns:
            List of (query, doc_text) tuples
        """
        pairs = []
        
        for doc in documents:
            # Combine title and truncated content
            title = doc.get('title', '')
            content = doc.get('content', '')
            
            # Truncate content to max length
            content_truncated = truncate_text(
                content, 
                max_length=config.MAX_CONTENT_LENGTH * 5,  # Roughly 5 chars per token
                suffix=''
            )
            
            doc_text = f"{title}. {content_truncated}"
            pairs.append((query, doc_text))
        
        return pairs
    
    @timer
    def rerank(
        self, 
        query: str, 
        documents: List[Dict], 
        top_k: int = None
    ) -> List[Dict]:
        """
        Rerank documents using cross-encoder
        
        Args:
            query: Query string
            documents: List of document dictionaries
            top_k: Number of top documents to return (default from config)
            
        Returns:
            Reranked list of documents with scores
        """
        if not self.available:
            logger.warning("Reranker not available, returning original order")
            return documents[:top_k] if top_k else documents
        
        if not documents:
            return []
        
        if top_k is None:
            top_k = config.RERANK_TOP_K
        
        logger.info(f"Reranking {len(documents)} documents with cross-encoder")
        
        # Prepare pairs
        pairs = self._prepare_pairs(query, documents)
        
        # Batch processing for efficiency
        batch_size = config.RERANK_BATCH_SIZE
        all_scores = []
        
        for batch_pairs in batch_iterator(pairs, batch_size):
            batch_scores = self.model.predict(batch_pairs)
            all_scores.extend(batch_scores)
        
        # Add reranking scores to documents
        for doc, score in zip(documents, all_scores):
            doc['rerank_score'] = float(score)
        
        # Sort by rerank score
        reranked = sorted(
            documents, 
            key=lambda x: x['rerank_score'], 
            reverse=True
        )
        
        logger.info(f"Reranking complete. Top score: {reranked[0]['rerank_score']:.4f}")
        
        return reranked[:top_k]
    
    @timer
    def rerank_with_early_stopping(
        self,
        query: str,
        documents: List[Dict],
        top_k: int = None,
        stability_threshold: int = 10
    ) -> List[Dict]:
        """
        Rerank with early stopping when top results stabilize
        
        Args:
            query: Query string
            documents: List of document dictionaries
            top_k: Number of top documents to return
            stability_threshold: Number of documents to check for stability
            
        Returns:
            Reranked list of documents
        """
        if not self.available or len(documents) <= stability_threshold:
            return self.rerank(query, documents, top_k)
        
        # First pass: score top documents
        top_docs = documents[:stability_threshold * 2]
        pairs = self._prepare_pairs(query, top_docs)
        scores = self.model.predict(pairs)
        
        # Add scores
        for doc, score in zip(top_docs, scores):
            doc['rerank_score'] = float(score)
        
        # Sort
        top_docs_sorted = sorted(
            top_docs,
            key=lambda x: x['rerank_score'],
            reverse=True
        )[:stability_threshold]
        
        # Score remaining documents in batches
        remaining_docs = documents[stability_threshold * 2:]
        
        if remaining_docs:
            pairs_remaining = self._prepare_pairs(query, remaining_docs)
            scores_remaining = self.model.predict(pairs_remaining)
            
            for doc, score in zip(remaining_docs, scores_remaining):
                doc['rerank_score'] = float(score)
            
            # Combine and sort
            all_scored = top_docs_sorted + remaining_docs
            all_sorted = sorted(
                all_scored,
                key=lambda x: x['rerank_score'],
                reverse=True
            )
        else:
            all_sorted = top_docs_sorted
        
        if top_k:
            return all_sorted[:top_k]
        return all_sorted


class EnsembleReranker:
    """
    Ensemble reranking combining multiple signals
    """
    
    def __init__(self):
        """Initialize ensemble reranker"""
        self.neural_reranker = NeuralReranker()
        logger.info("EnsembleReranker initialized")
    
    @timer
    def rerank(
        self,
        query: str,
        documents: List[Dict],
        top_k: int = None,
        weights: Dict[str, float] = None
    ) -> List[Dict]:
        """
        Rerank using ensemble of signals
        
        Args:
            query: Query string
            documents: List of document dictionaries
            top_k: Number of top documents to return
            weights: Weights for different signals
            
        Returns:
            Reranked documents
        """
        if weights is None:
            weights = {
                'neural': 0.7,      # Neural reranker
                'retrieval': 0.2,   # Original retrieval score
                'title_match': 0.1  # Title keyword match
            }
        
        # Neural reranking
        if self.neural_reranker.available:
            documents = self.neural_reranker.rerank(
                query, 
                documents, 
                top_k=len(documents)
            )
        else:
            # Fallback: use retrieval score
            for doc in documents:
                doc['rerank_score'] = doc.get('retrieval_score', 0.0)
        
        # Calculate title match score
        query_terms = set(query.lower().split())
        for doc in documents:
            title_terms = set(doc.get('title', '').lower().split())
            title_match = len(query_terms & title_terms) / max(len(query_terms), 1)
            doc['title_match_score'] = title_match
        
        # Normalize scores
        if documents:
            # Normalize neural scores
            neural_scores = [doc.get('rerank_score', 0.0) for doc in documents]
            max_neural = max(neural_scores) if neural_scores else 1.0
            min_neural = min(neural_scores) if neural_scores else 0.0
            range_neural = max_neural - min_neural if max_neural > min_neural else 1.0
            
            for doc in documents:
                doc['rerank_score_norm'] = (
                    (doc.get('rerank_score', 0.0) - min_neural) / range_neural
                )
            
            # Normalize retrieval scores
            retrieval_scores = [doc.get('retrieval_score', 0.0) for doc in documents]
            max_ret = max(retrieval_scores) if retrieval_scores else 1.0
            min_ret = min(retrieval_scores) if retrieval_scores else 0.0
            range_ret = max_ret - min_ret if max_ret > min_ret else 1.0
            
            for doc in documents:
                doc['retrieval_score_norm'] = (
                    (doc.get('retrieval_score', 0.0) - min_ret) / range_ret
                )
        
        # Calculate ensemble score
        for doc in documents:
            ensemble_score = (
                weights['neural'] * doc.get('rerank_score_norm', 0.0) +
                weights['retrieval'] * doc.get('retrieval_score_norm', 0.0) +
                weights['title_match'] * doc.get('title_match_score', 0.0)
            )
            doc['ensemble_score'] = ensemble_score
        
        # Sort by ensemble score
        reranked = sorted(
            documents,
            key=lambda x: x['ensemble_score'],
            reverse=True
        )
        
        logger.info(f"Ensemble reranking complete. Top ensemble score: {reranked[0]['ensemble_score']:.4f}")
        
        if top_k:
            return reranked[:top_k]
        return reranked


def main():
    """Test reranking"""
    import time
    from retrieval import HybridRetriever
    from query_processor import QueryProcessor
    
    # Initialize
    retriever = HybridRetriever()
    processor = QueryProcessor()
    reranker = NeuralReranker()
    ensemble = EnsembleReranker()
    
    # Test query
    query = "impact of inflation on economy"
    
    print("\n=== Reranking Test ===\n")
    print(f"Query: {query}\n")
    
    # Process query
    query_info = processor.process(query)
    
    # Retrieve
    print("1. Retrieving documents...")
    results, documents = retriever.retrieve_with_documents(
        query=query_info['corrected'],
        query_tokens=query_info['tokens'],
        k=100
    )
    print(f"   Retrieved {len(documents)} documents")
    
    # Neural reranking
    print("\n2. Neural reranking...")
    start = time.time()
    reranked = reranker.rerank(query, documents, top_k=20)
    elapsed = (time.time() - start) * 1000
    print(f"   Reranked to {len(reranked)} documents in {elapsed:.2f}ms")
    
    print("\n3. Top 5 results after reranking:")
    for i, doc in enumerate(reranked[:5], 1):
        print(f"{i}. {doc['title']}")
        print(f"   Rerank score: {doc['rerank_score']:.4f}")
        print(f"   Retrieval score: {doc['retrieval_score']:.4f}")
        print()
    
    # Ensemble reranking
    print("\n4. Ensemble reranking...")
    start = time.time()
    ensemble_reranked = ensemble.rerank(query, documents, top_k=20)
    elapsed = (time.time() - start) * 1000
    print(f"   Ensemble reranked in {elapsed:.2f}ms")
    
    print("\n5. Top 5 results after ensemble reranking:")
    for i, doc in enumerate(ensemble_reranked[:5], 1):
        print(f"{i}. {doc['title']}")
        print(f"   Ensemble score: {doc['ensemble_score']:.4f}")
        print()


if __name__ == "__main__":
    main()
