"""
Retrieval Module for Cortex IR System
Handles hybrid retrieval with BM25 and ColBERT, plus RRF fusion
"""

from typing import List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor
import time

import bm25s
import numpy as np

import config
from utils import setup_logging, timer, load_pickle
from indexing import BM25Indexer, ColBERTIndexer

logger = setup_logging(__name__)


class BM25Retriever:
    """
    BM25 sparse retrieval
    """
    
    def __init__(self, index_path: str = None):
        """
        Initialize BM25 retriever
        
        Args:
            index_path: Path to BM25 index (default from config)
        """
        self.indexer = BM25Indexer()
        
        if index_path is None:
            index_path = config.BM25_INDEX_PATH
        
        # Load index
        self.indexer.load_index(index_path)
        
        logger.info("BM25Retriever initialized and index loaded")
    
    @timer
    def retrieve(self, query: str, k: int = None) -> List[Tuple[int, float]]:
        """
        Retrieve documents using BM25
        
        Args:
            query: Query string (can be preprocessed tokens)
            k: Number of documents to retrieve (default from config)
            
        Returns:
            List of (doc_id, score) tuples
        """
        if k is None:
            k = config.TOP_K_SPARSE
        
        # Tokenize query
        query_tokens = query.split() if isinstance(query, str) else query
        
        # Retrieve
        results, scores = self.indexer.retriever.retrieve(
            bm25s.tokenize([query_tokens], stopwords=None),
            k=k
        )
        
        # Convert to list of tuples
        results_list = [
            (int(doc_id), float(score)) 
            for doc_id, score in zip(results[0], scores[0])
        ]
        
        logger.debug(f"BM25 retrieved {len(results_list)} documents")
        
        return results_list
    
    def get_documents(self, doc_ids: List[int]) -> List[Dict]:
        """
        Get document content for given IDs
        
        Args:
            doc_ids: List of document IDs
            
        Returns:
            List of document dictionaries
        """
        documents = []
        for doc_id in doc_ids:
            if 0 <= doc_id < len(self.indexer.corpus):
                documents.append(self.indexer.corpus[doc_id])
        
        return documents


class ColBERTRetriever:
    """
    ColBERT dense retrieval
    """
    
    def __init__(self, index_name: str = None):
        """
        Initialize ColBERT retriever
        
        Args:
            index_name: Name of ColBERT index (default from config)
        """
        self.indexer = ColBERTIndexer()
        
        if index_name is None:
            index_name = self.indexer.index_name
        
        # Load index
        try:
            self.indexer.load_index(index_name)
            self.available = True
        except:
            logger.warning("ColBERT index not available")
            self.available = False
        
        if self.available:
            logger.info("ColBERTRetriever initialized and index loaded")
    
    @timer
    def retrieve(self, query: str, k: int = None) -> List[Tuple[int, float]]:
        """
        Retrieve documents using ColBERT
        
        Args:
            query: Query string
            k: Number of documents to retrieve (default from config)
            
        Returns:
            List of (doc_id, score) tuples
        """
        if not self.available:
            logger.warning("ColBERT not available, returning empty results")
            return []
        
        if k is None:
            k = config.TOP_K_DENSE
        
        try:
            # Search with ColBERT
            results = self.indexer.model.search(query, k=k)
            
            # Convert to list of tuples
            results_list = [
                (int(result['document_id']), float(result['score']))
                for result in results
            ]
            
            logger.debug(f"ColBERT retrieved {len(results_list)} documents")
            
            return results_list
            
        except Exception as e:
            logger.error(f"Error in ColBERT retrieval: {e}")
            return []


class ReciprocalRankFusion:
    """
    Reciprocal Rank Fusion for combining rankings
    """
    
    def __init__(self, k: int = None):
        """
        Initialize RRF
        
        Args:
            k: RRF k parameter (default from config)
        """
        self.k = k if k is not None else config.RRF_K
        logger.info(f"ReciprocalRankFusion initialized with k={self.k}")
    
    def fuse(
        self, 
        rankings: List[List[Tuple[int, float]]]
    ) -> List[Tuple[int, float]]:
        """
        Fuse multiple rankings using RRF
        
        Args:
            rankings: List of rankings, each is list of (doc_id, score) tuples
            
        Returns:
            Fused ranking as list of (doc_id, score) tuples
        """
        # Calculate RRF scores
        rrf_scores = {}
        
        for ranking in rankings:
            for rank, (doc_id, _) in enumerate(ranking, start=1):
                if doc_id not in rrf_scores:
                    rrf_scores[doc_id] = 0.0
                rrf_scores[doc_id] += 1.0 / (self.k + rank)
        
        # Sort by RRF score
        fused = sorted(
            rrf_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        logger.debug(f"RRF fused {len(fused)} unique documents")
        
        return fused


class HybridRetriever:
    """
    Hybrid retrieval combining BM25 and ColBERT with RRF fusion
    """
    
    def __init__(self):
        """Initialize hybrid retriever"""
        self.bm25_retriever = BM25Retriever()
        self.colbert_retriever = ColBERTRetriever()
        self.rrf = ReciprocalRankFusion()
        
        logger.info("HybridRetriever initialized")
    
    @timer
    def retrieve(
        self, 
        query: str, 
        query_tokens: List[str] = None,
        k: int = None,
        use_parallel: bool = True
    ) -> List[Tuple[int, float]]:
        """
        Retrieve documents using hybrid approach
        
        Args:
            query: Query string
            query_tokens: Preprocessed query tokens (optional)
            k: Number of final documents to return
            use_parallel: Whether to use parallel retrieval
            
        Returns:
            List of (doc_id, score) tuples
        """
        if k is None:
            k = config.TOP_K_SPARSE  # Will be fused to fewer
        
        # Prepare queries
        bm25_query = " ".join(query_tokens) if query_tokens else query
        colbert_query = query
        
        if use_parallel and self.colbert_retriever.available:
            # Parallel retrieval
            with ThreadPoolExecutor(max_workers=2) as executor:
                # Submit both retrieval tasks
                future_bm25 = executor.submit(
                    self.bm25_retriever.retrieve, 
                    bm25_query, 
                    k
                )
                future_colbert = executor.submit(
                    self.colbert_retriever.retrieve, 
                    colbert_query, 
                    k
                )
                
                # Get results
                bm25_results = future_bm25.result()
                colbert_results = future_colbert.result()
        else:
            # Sequential retrieval
            bm25_results = self.bm25_retriever.retrieve(bm25_query, k)
            
            if self.colbert_retriever.available:
                colbert_results = self.colbert_retriever.retrieve(colbert_query, k)
            else:
                colbert_results = []
        
        # Fuse results
        if colbert_results:
            rankings = [bm25_results, colbert_results]
            fused_results = self.rrf.fuse(rankings)
        else:
            # Only BM25 available
            fused_results = bm25_results
        
        logger.info(f"Hybrid retrieval returned {len(fused_results)} documents")
        
        return fused_results[:k]
    
    def get_documents(self, results: List[Tuple[int, float]]) -> List[Dict]:
        """
        Get document content for retrieval results
        
        Args:
            results: List of (doc_id, score) tuples
            
        Returns:
            List of document dictionaries with scores
        """
        doc_ids = [doc_id for doc_id, _ in results]
        documents = self.bm25_retriever.get_documents(doc_ids)
        
        # Add scores to documents
        score_map = dict(results)
        for doc in documents:
            doc['retrieval_score'] = score_map.get(doc['id'], 0.0)
        
        return documents
    
    @timer
    def retrieve_with_documents(
        self,
        query: str,
        query_tokens: List[str] = None,
        k: int = None
    ) -> Tuple[List[Tuple[int, float]], List[Dict]]:
        """
        Retrieve documents and get their content in one call
        
        Args:
            query: Query string
            query_tokens: Preprocessed query tokens
            k: Number of documents to return
            
        Returns:
            Tuple of (results, documents)
        """
        results = self.retrieve(query, query_tokens, k)
        documents = self.get_documents(results)
        
        return results, documents


def main():
    """Test retrieval"""
    from query_processor import QueryProcessor
    
    # Initialize
    retriever = HybridRetriever()
    processor = QueryProcessor()
    
    # Test queries
    test_queries = [
        "latest COVID updates",
        "who won the super bowl?",
        "impact of inflation on economy"
    ]
    
    print("\n=== Retrieval Test ===\n")
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        
        # Process query
        query_info = processor.process(query)
        
        # Retrieve
        start_time = time.time()
        results, documents = retriever.retrieve_with_documents(
            query=query_info['corrected'],
            query_tokens=query_info['tokens'],
            k=10
        )
        elapsed = (time.time() - start_time) * 1000
        
        print(f"Retrieved {len(documents)} documents in {elapsed:.2f}ms")
        print("\nTop 3 results:")
        
        for i, doc in enumerate(documents[:3], 1):
            print(f"{i}. {doc['title']}")
            print(f"   Score: {doc['retrieval_score']:.4f}")
            print(f"   Category: {doc['category']}")
            print()


if __name__ == "__main__":
    main()
