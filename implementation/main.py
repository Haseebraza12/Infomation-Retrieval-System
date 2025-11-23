"""
Main Pipeline for Cortex IR System
Orchestrates the complete retrieval pipeline
"""

import time
from typing import Dict, List, Tuple
from pathlib import Path

import config
from utils import setup_logging, timer, index_exists, generate_snippet
from query_processor import QueryProcessor
from retrieval import HybridRetriever
from reranker import EnsembleReranker
from post_processor import PostProcessor

logger = setup_logging(__name__)


class CortexIRPipeline:
    """
    Complete IR pipeline orchestrator
    """
    
    def __init__(self):
        """Initialize the complete IR pipeline"""
        logger.info("Initializing Cortex IR Pipeline...")
        
        # Check if indices exist
        if not self._check_indices():
            logger.error("Indices not found. Please run preprocessing and indexing first.")
            raise FileNotFoundError(
                "Indices not found. Run: python preprocessing.py && python indexing.py"
            )
        
        # Initialize components
        self.query_processor = QueryProcessor()
        self.retriever = HybridRetriever()
        self.reranker = EnsembleReranker()
        self.post_processor = PostProcessor()
        
        logger.info("Cortex IR Pipeline initialized successfully")
    
    def _check_indices(self) -> bool:
        """Check if all required indices exist"""
        checks = {
            'BM25': index_exists('bm25'),
            'Processed Data': index_exists('processed'),
            'Metadata': index_exists('metadata')
        }
        
        all_exist = all(checks.values())
        
        if not all_exist:
            logger.warning("Missing indices:")
            for name, exists in checks.items():
                logger.warning(f"  {name}: {'✓' if exists else '✗'}")
        
        return checks['BM25'] and checks['Processed Data']  # Minimum required
    
    @timer
    def search(
        self,
        query: str,
        top_k: int = None,
        enable_reranking: bool = True,
        enable_post_processing: bool = True
    ) -> Dict:
        """
        Execute complete search pipeline
        
        Args:
            query: User query
            top_k: Number of results to return
            enable_reranking: Whether to use neural reranking
            enable_post_processing: Whether to apply post-processing
            
        Returns:
            Dictionary with results and metadata
        """
        if top_k is None:
            top_k = config.FINAL_TOP_K
        
        start_time = time.time()
        
        logger.info(f"Processing query: '{query}'")
        
        # Stage 1: Query Processing
        stage1_start = time.time()
        query_info = self.query_processor.process(query)
        stage1_time = (time.time() - stage1_start) * 1000
        
        # Stage 2: Hybrid Retrieval
        stage2_start = time.time()
        results, documents = self.retriever.retrieve_with_documents(
            query=query_info['corrected'],
            query_tokens=query_info['tokens'],
            k=100  # Retrieve more for reranking
        )
        stage2_time = (time.time() - stage2_start) * 1000
        
        # Stage 3: Neural Reranking (optional)
        if enable_reranking and self.reranker.neural_reranker.available:
            stage3_start = time.time()
            documents = self.reranker.rerank(
                query_info['corrected'],
                documents,
                top_k=50
            )
            stage3_time = (time.time() - stage3_start) * 1000
        else:
            stage3_time = 0
            documents = documents[:50]
        
        # Stage 4: Post-Processing (optional)
        clusters = {}
        if enable_post_processing:
            stage4_start = time.time()
            documents, clusters = self.post_processor.process(
                query=query_info['corrected'],
                query_type=query_info['type'],
                documents=documents,
                top_k=top_k
            )
            stage4_time = (time.time() - stage4_start) * 1000
        else:
            stage4_time = 0
            documents = documents[:top_k]
        
        total_time = (time.time() - start_time) * 1000
        
        # Generate snippets
        for doc in documents:
            doc['snippet'] = generate_snippet(
                doc.get('content', ''),
                query_info['corrected'],
                max_length=200
            )
        
        # Prepare result
        result = {
            'query': {
                'original': query,
                'corrected': query_info['corrected'],
                'type': query_info['type'],
                'tokens': query_info['tokens'],
                'entities': query_info['entities']
            },
            'results': documents,
            'clusters': clusters,
            'metadata': {
                'total_results': len(documents),
                'total_time_ms': total_time,
                'stage_times': {
                    'query_processing': stage1_time,
                    'retrieval': stage2_time,
                    'reranking': stage3_time,
                    'post_processing': stage4_time
                },
                'reranking_enabled': enable_reranking,
                'post_processing_enabled': enable_post_processing
            }
        }
        
        logger.info(f"Search completed in {total_time:.2f}ms")
        logger.info(f"  Retrieval: {stage2_time:.2f}ms")
        if enable_reranking:
            logger.info(f"  Reranking: {stage3_time:.2f}ms")
        if enable_post_processing:
            logger.info(f"  Post-processing: {stage4_time:.2f}ms")
        
        return result
    
    def batch_search(self, queries: List[str], **kwargs) -> List[Dict]:
        """
        Execute batch search for multiple queries
        
        Args:
            queries: List of queries
            **kwargs: Arguments for search()
            
        Returns:
            List of result dictionaries
        """
        logger.info(f"Batch searching {len(queries)} queries")
        
        results = []
        for query in queries:
            result = self.search(query, **kwargs)
            results.append(result)
        
        return results


def main():
    """Test the pipeline"""
    print("\n" + "="*70)
    print(" "*20 + "CORTEX IR SYSTEM")
    print("="*70 + "\n")
    
    # Initialize pipeline
    print("Initializing pipeline...")
    pipeline = CortexIRPipeline()
    print("✓ Pipeline ready\n")
    
    # Test queries
    test_queries = [
        "latest COVID updates",
        "who won the super bowl?",
        "impact of inflation on economy",
        "sports team performance"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*70}")
        print(f"Query {i}/{len(test_queries)}: {query}")
        print('='*70)
        
        # Execute search
        result = pipeline.search(query, top_k=5)
        
        # Display results
        print(f"\nQuery Type: {result['query']['type']}")
        print(f"Total Time: {result['metadata']['total_time_ms']:.2f}ms")
        print(f"\nTop {len(result['results'])} Results:\n")
        
        for j, doc in enumerate(result['results'], 1):
            print(f"{j}. {doc['title']}")
            print(f"   Category: {doc['category']}")
            
            # Show score
            score = doc.get('ensemble_score', 
                           doc.get('rerank_score', 
                                  doc.get('retrieval_score', 0.0)))
            print(f"   Score: {score:.4f}")
            
            # Show snippet
            print(f"   {doc['snippet'][:100]}...")
            print()
        
        # Show clusters if available
        if result['clusters']:
            print(f"Topic Clusters:")
            for cluster_name, cluster_docs in result['clusters'].items():
                print(f"  • {cluster_name}: {len(cluster_docs)} documents")
        
        print()
    
    print("\n" + "="*70)
    print("Pipeline test complete!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
