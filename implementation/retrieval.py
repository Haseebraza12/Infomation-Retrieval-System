"""
Retrieval Module - Handles BM25 and dense retrieval
"""
from typing import List, Dict, Tuple
import numpy as np
from tqdm import tqdm
import bm25s
from sentence_transformers import SentenceTransformer

from config import Config
from utils import logger, timer

class HybridRetriever:
    """Hybrid retrieval combining BM25 and dense search"""
    
    def __init__(self, config: Config):
        self.config = config
        self.bm25_index = None
        self.articles = None
        self.dense_embeddings = None
        self.dense_model = None
        
    def load_indices(self, bm25_index, articles: List[Dict], dense_embeddings=None):
        """Load pre-built indices"""
        self.bm25_index = bm25_index
        self.articles = articles
        self.dense_embeddings = dense_embeddings
        
        # Load dense model if embeddings are available
        if dense_embeddings is not None and self.config.USE_DENSE_RETRIEVAL:
            logger.info(f"Loading dense retrieval model: {self.config.EMBEDDING_MODEL}")
            self.dense_model = SentenceTransformer(self.config.EMBEDDING_MODEL)
        
        logger.info(f"Retriever loaded with {len(articles)} articles")
    
    @timer
    def retrieve(
        self, 
        query_tokens: List[str],
        query_text: str = "",
        top_k: int = 100,
        use_hybrid: bool = True
    ) -> List[Tuple[int, float]]:
        """
        Retrieve relevant documents
        
        Args:
            query_tokens: Tokenized query (for compatibility)
            query_text: Original query text (used for retrieval)
            top_k: Number of results
            use_hybrid: Use hybrid retrieval if available
            
        Returns:
            List of (article_id, score) tuples
        """
        if use_hybrid and self.dense_embeddings is not None and self.dense_model is not None:
            return self._hybrid_retrieve(query_tokens, query_text, top_k)
        else:
            return self._bm25_retrieve(query_tokens, query_text, top_k)
    
    @timer
    def _bm25_retrieve(self, query_tokens: List[str], query_text: str, top_k: int) -> List[Tuple[int, float]]:
        """BM25 sparse retrieval"""
        if self.bm25_index is None:
            raise ValueError("BM25 index not loaded")
        
        # Use query_text if available, otherwise join tokens
        if query_text:
            query_str = query_text
        else:
            query_str = ' '.join(query_tokens)
        
        # Tokenize query for bm25s (it expects raw text strings)
        query_tokens_array = bm25s.tokenize([query_str], stopwords='en')
        
        # Get results
        results, scores = self.bm25_index.retrieve(
            query_tokens_array, 
            k=top_k,
            return_as="tuple"
        )
        
        # Format results
        ranked_results = []
        for idx, score in zip(results[0], scores[0]):
            if idx < len(self.articles):
                ranked_results.append((idx, float(score)))
        
        return ranked_results
    
    @timer
    def _hybrid_retrieve(
        self, 
        query_tokens: List[str],
        query_text: str,
        top_k: int
    ) -> List[Tuple[int, float]]:
        """Hybrid retrieval using RRF"""
        # BM25 results
        bm25_results = self._bm25_retrieve(query_tokens, query_text, top_k)
        
        # Dense results
        dense_results = self._dense_retrieve(query_text, top_k)
        
        # Reciprocal Rank Fusion
        return self._reciprocal_rank_fusion(
            bm25_results, 
            dense_results, 
            k=self.config.RRF_K
        )
    
    @timer
    def _dense_retrieve(self, query_text: str, top_k: int) -> List[Tuple[int, float]]:
        """Dense retrieval using sentence-transformers"""
        if self.dense_model is None or self.dense_embeddings is None:
            return []
        
        # Encode query
        query_embedding = self.dense_model.encode(
            [query_text], 
            normalize_embeddings=True
        )[0]
        
        # Compute similarities (cosine similarity with normalized vectors = dot product)
        similarities = np.dot(self.dense_embeddings, query_embedding)
        
        # Get top-k
        top_indices = np.argsort(similarities)[::-1][:top_k]
        dense_results = [(int(idx), float(similarities[idx])) for idx in top_indices]
        
        return dense_results
    
    def _reciprocal_rank_fusion(
        self,
        results1: List[Tuple[int, float]],
        results2: List[Tuple[int, float]],
        k: int = 60
    ) -> List[Tuple[int, float]]:
        """Combine rankings using RRF"""
        scores = {}
        
        # Add scores from first ranker
        for rank, (doc_id, _) in enumerate(results1, 1):
            scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank)
        
        # Add scores from second ranker
        for rank, (doc_id, _) in enumerate(results2, 1):
            scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank)
        
        # Sort by combined score
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        return ranked

    def get_article_by_id(self, article_id: int) -> Dict:
        """Retrieve article by ID"""
        if 0 <= article_id < len(self.articles):
            return self.articles[article_id]
        return None


class ParallelRetriever:
    """Optimized parallel retrieval for production use"""
    
    def __init__(self, config: Config):
        self.config = config
        self.base_retriever = HybridRetriever(config)
        
    def load_indices(self, bm25_index, articles, dense_embeddings=None):
        """Load indices into base retriever"""
        self.base_retriever.load_indices(bm25_index, articles, dense_embeddings)
    
    @timer    
    def retrieve_with_documents(
        self,
        query_tokens: List[str],
        query_text: str,
        top_k: int = 100,
        use_hybrid: bool = True
    ) -> List[Dict]:
        """
        Retrieve documents with full details
        
        Args:
            query_tokens: Tokenized query (for compatibility)
            query_text: Original query text
            top_k: Number of results
            use_hybrid: Use hybrid retrieval
            
        Returns:
            List of article dictionaries with scores
        """
        # Retrieve
        results = self.base_retriever.retrieve(
            query_tokens,
            query_text,
            top_k=top_k,
            use_hybrid=use_hybrid
        )
        
        # Get full articles
        articles = []
        for doc_id, score in results:
            article = self.base_retriever.get_article_by_id(doc_id)
            if article:
                article_copy = article.copy()
                article_copy['retrieval_score'] = score
                article_copy['rank'] = len(articles) + 1
                articles.append(article_copy)
        
        return articles


def test_retrieval():
    """Test retrieval system"""
    from indexing import IndexBuilder
    
    config = Config()
    
    # Load indices
    builder = IndexBuilder(config)
    bm25_index = builder.load_bm25_index()
    articles = builder.load_metadata()
    
    try:
        dense_embeddings = builder.load_dense_embeddings()
    except FileNotFoundError:
        logger.warning("Dense embeddings not found, using BM25 only")
        dense_embeddings = None
    
    # Create retriever
    retriever = ParallelRetriever(config)
    retriever.load_indices(bm25_index, articles, dense_embeddings)
    
    # Test queries
    test_queries = [
        "artificial intelligence in healthcare",
        "climate change effects",
        "stock market trends"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        results = retriever.retrieve_with_documents(
            query_tokens=query.lower().split(),
            query_text=query,
            top_k=5
        )
        
        for i, article in enumerate(results, 1):
            print(f"{i}. {article['title']} (Score: {article['retrieval_score']:.4f})")


if __name__ == "__main__":
    test_retrieval()
