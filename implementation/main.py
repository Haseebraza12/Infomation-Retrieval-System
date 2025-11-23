"""
Main Pipeline - Orchestrates the complete IR system
"""

from typing import Dict, List, Optional
import time

from config import Config
from utils import logger, Timer
from query_processor import QueryProcessor
from retrieval import ParallelRetriever
from reranker import EnsembleReranker
from post_processor import PostProcessor
from indexing import IndexBuilder


class CortexIRPipeline:
    """
    Main IR pipeline orchestrator
    """
    
    def __init__(self, config: Config = None):
        """Initialize pipeline with all components"""
        if config is None:
            config = Config()
        
        self.config = config
        
        logger.info("Initializing Cortex IR Pipeline...")
        
        # Initialize components
        self.query_processor = QueryProcessor(config)
        self.retriever = ParallelRetriever(config)
        self.reranker = EnsembleReranker(config)
        self.post_processor = PostProcessor(config)
        
        # Load indices
        self._load_indices()
        
        logger.info("Pipeline initialized successfully!")
    
    def _load_indices(self):
        """Load all pre-built indices"""
        logger.info("Loading indices...")
        
        builder = IndexBuilder(self.config)
        
        # Load BM25 index
        logger.info("Loading BM25 index...")
        bm25_index = builder.load_bm25_index()
        
        # Load articles
        logger.info("Loading article metadata...")
        articles = builder.load_metadata()
        
        # Load dense embeddings (optional)
        try:
            logger.info("Loading dense embeddings...")
            dense_embeddings = builder.load_dense_embeddings()
        except FileNotFoundError:
            logger.warning("Dense embeddings not found, using BM25 only")
            dense_embeddings = None
        
        # Initialize retriever with indices
        self.retriever.load_indices(bm25_index, articles, dense_embeddings)
        
        logger.info(f"Indices loaded: {len(articles)} articles indexed")
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        enable_reranking: bool = True,
        enable_post_processing: bool = True,
        return_metadata: bool = True
    ) -> Dict:
        """
        Execute end-to-end search pipeline
        
        Args:
            query: Search query string
            top_k: Number of final results to return
            enable_reranking: Whether to use neural reranking
            enable_post_processing: Whether to apply post-processing
            return_metadata: Whether to include performance metadata
            
        Returns:
            Dictionary with results and metadata
        """
        start_time = time.time()
        stage_times = {}
        
        # Stage 1: Query Processing
        with Timer("Query Processing") as t:
            processed_query = self.query_processor.process(query)
        stage_times['query_processing'] = t.elapsed * 1000
        
        # Stage 2: Retrieval
        with Timer("Retrieval") as t:
            documents = self.retriever.retrieve_with_documents(
                query_tokens=processed_query['tokens'],
                query_text=processed_query['corrected'],
                top_k=self.config.TOP_K_RETRIEVAL,
                use_hybrid=self.config.USE_DENSE_RETRIEVAL
            )
        stage_times['retrieval'] = t.elapsed * 1000
        
        # Stage 3: Reranking (optional)
        if enable_reranking and documents:
            with Timer("Reranking") as t:
                documents = self.reranker.rerank(
                    query=processed_query['corrected'],
                    documents=documents,
                    top_k=self.config.RERANK_TOP_K
                )
            stage_times['reranking'] = t.elapsed * 1000
        else:
            stage_times['reranking'] = 0
        
        # Stage 4: Post-Processing (optional)
        clusters = None
        if enable_post_processing and documents:
            with Timer("Post-Processing") as t:
                documents, clusters = self.post_processor.process(
                    documents=documents,
                    query=processed_query,
                    top_k=top_k
                )
            stage_times['post_processing'] = t.elapsed * 1000
        else:
            stage_times['post_processing'] = 0
            documents = documents[:top_k]
        
        # Calculate total time
        total_time = time.time() - start_time
        
        # Prepare result
        result = {
            'results': documents,
            'query': processed_query,
            'clusters': clusters
        }
        
        # Add metadata if requested
        if return_metadata:
            result['metadata'] = {
                'total_time_ms': total_time * 1000,
                'stage_times': stage_times,
                'num_results': len(documents),
                'reranking_enabled': enable_reranking,
                'post_processing_enabled': enable_post_processing
            }
        
        return result
    
    def batch_search(
        self,
        queries: List[str],
        top_k: int = 10,
        enable_reranking: bool = True,
        enable_post_processing: bool = True
    ) -> List[Dict]:
        """
        Execute batch search for multiple queries
        
        Args:
            queries: List of query strings
            top_k: Number of results per query
            enable_reranking: Whether to use reranking
            enable_post_processing: Whether to use post-processing
            
        Returns:
            List of result dictionaries
        """
        results = []
        
        for query in queries:
            result = self.search(
                query=query,
                top_k=top_k,
                enable_reranking=enable_reranking,
                enable_post_processing=enable_post_processing
            )
            results.append(result)
        
        return results


def main():
    """Example usage of the pipeline"""
    # Initialize pipeline
    pipeline = CortexIRPipeline()
    
    # Test queries
    test_queries = [
        "latest sports news",
        "economic impact of inflation",
        "who won the championship?",
        "business merger announcements"
    ]
    
    print("\n" + "="*70)
    print("CORTEX IR SYSTEM - DEMO")
    print("="*70)
    
    for query in test_queries:
        print(f"\n{'='*70}")
        print(f"Query: {query}")
        print('='*70)
        
        # Execute search
        result = pipeline.search(
            query=query,
            top_k=5,
            enable_reranking=True,
            enable_post_processing=True
        )
        
        # Display results
        print(f"\nQuery Type: {result['query']['type']}")
        print(f"Total Time: {result['metadata']['total_time_ms']:.0f}ms")
        print(f"\nTop {len(result['results'])} Results:")
        print("-"*70)
        
        for i, doc in enumerate(result['results'], 1):
            score = doc.get('ensemble_score', doc.get('rerank_score', doc.get('retrieval_score', 0)))
            print(f"\n{i}. {doc['title']}")
            print(f"   Category: {doc['category']} | Score: {score:.4f}")
            print(f"   {doc['content'][:150]}...")
        
        print("\n" + "="*70)


if __name__ == "__main__":
    main()
