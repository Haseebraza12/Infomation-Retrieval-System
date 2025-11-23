"""
Reranking Module - Neural reranking with cross-encoders
"""

from typing import List, Dict, Tuple
import numpy as np
from sentence_transformers import CrossEncoder
import torch

from config import Config
from utils import logger, timer, batch_data

config = Config()


class NeuralReranker:
    """Cross-encoder based neural reranker"""
    
    def __init__(self, config: Config = None):
        """Initialize neural reranker"""
        if config is None:
            config = Config()
        
        self.config = config
        
        model_name = config.CROSS_ENCODER_MODEL
        
        logger.info(f"Loading cross-encoder model: {model_name}")
        
        # Determine device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {self.device}")
        
        # Load model
        self.model = CrossEncoder(model_name, device=self.device)
        
        logger.info("Neural reranker initialized")
    
    def _prepare_pairs(
        self,
        query: str,
        documents: List[Dict],
        max_length: int = None
    ) -> List[Tuple[str, str]]:
        """
        Prepare query-document pairs for reranking
        
        Args:
            query: Query string
            documents: List of documents
            max_length: Maximum content length
            
        Returns:
            List of (query, document) pairs
        """
        if max_length is None:
            max_length = self.config.MAX_CONTENT_LENGTH * 5  # Roughly 5 chars per token
        
        pairs = []
        for doc in documents:
            # Combine title and content
            title = doc.get('title', '')
            content = doc.get('content', '')
            
            # Truncate content if too long
            if len(content) > max_length:
                content = content[:max_length] + "..."
            
            doc_text = f"{title}. {content}"
            pairs.append((query, doc_text))
        
        return pairs
    
    @timer
    def rerank(
        self,
        query: str,
        documents: List[Dict],
        top_k: int = 50
    ) -> List[Dict]:
        """
        Rerank documents using cross-encoder
        
        Args:
            query: Query string
            documents: List of documents with retrieval scores
            top_k: Number of documents to return
            
        Returns:
            Reranked list of documents
        """
        if not documents:
            return documents
        
        logger.info(f"Reranking {len(documents)} documents with cross-encoder")
        
        # Prepare pairs
        pairs = self._prepare_pairs(query, documents)
        
        # Get reranking scores in batches
        batch_size = self.config.RERANK_BATCH_SIZE
        all_scores = []
        
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i + batch_size]
            scores = self.model.predict(batch, show_progress_bar=False)
            all_scores.extend(scores)
        
        # Convert to numpy array
        scores = np.array(all_scores)
        
        # Add rerank scores to documents
        for doc, score in zip(documents, scores):
            doc['rerank_score'] = float(score)
        
        # Sort by rerank score
        reranked = sorted(documents, key=lambda x: x['rerank_score'], reverse=True)
        
        logger.info(f"Reranking complete. Top score: {reranked[0]['rerank_score']:.4f}")
        
        return reranked[:top_k]


class EnsembleReranker:
    """
    Ensemble reranker combining multiple signals
    """
    
    def __init__(self, config: Config = None):
        """Initialize ensemble reranker"""
        if config is None:
            config = Config()
        
        self.config = config
        self.neural_reranker = NeuralReranker(config)
        
        # Weights for ensemble
        self.weights = {
            'retrieval': 0.3,
            'neural': 0.5,
            'entity': 0.2
        }
        
        logger.info("Ensemble reranker initialized")
    
    def _calculate_entity_score(
        self,
        query_entities: List[Dict],
        doc_entities: List[Dict]
    ) -> float:
        """
        Calculate entity overlap score
        
        Args:
            query_entities: List of query entities
            doc_entities: List of document entities
            
        Returns:
            Entity overlap score (0-1)
        """
        if not query_entities or not doc_entities:
            return 0.0
        
        # Extract entity texts
        query_entity_texts = {e['text'].lower() for e in query_entities}
        doc_entity_texts = {e['text'].lower() for e in doc_entities}
        
        # Calculate Jaccard similarity
        intersection = len(query_entity_texts & doc_entity_texts)
        union = len(query_entity_texts | doc_entity_texts)
        
        return intersection / union if union > 0 else 0.0
    
    @timer
    def rerank(
        self,
        query: str,
        documents: List[Dict],
        query_entities: List[Dict] = None,
        top_k: int = 50
    ) -> List[Dict]:
        """
        Rerank using ensemble of methods
        
        Args:
            query: Query string
            documents: List of documents
            query_entities: Query entities (optional)
            top_k: Number of results
            
        Returns:
            Reranked documents
        """
        if not documents:
            return documents
        
        # Get neural reranking scores
        documents = self.neural_reranker.rerank(query, documents, top_k=len(documents))
        
        # Normalize scores
        retrieval_scores = [doc.get('retrieval_score', 0) for doc in documents]
        neural_scores = [doc.get('rerank_score', 0) for doc in documents]
        
        retrieval_scores_norm = self._normalize(retrieval_scores)
        neural_scores_norm = self._normalize(neural_scores)
        
        # Calculate ensemble scores
        for i, doc in enumerate(documents):
            # Base scores
            retrieval_norm = retrieval_scores_norm[i]
            neural_norm = neural_scores_norm[i]
            
            # Entity score
            entity_score = 0.0
            if query_entities:
                doc_entities = doc.get('entities', [])
                entity_score = self._calculate_entity_score(query_entities, doc_entities)
            
            # Ensemble score
            ensemble_score = (
                self.weights['retrieval'] * retrieval_norm +
                self.weights['neural'] * neural_norm +
                self.weights['entity'] * entity_score
            )
            
            doc['ensemble_score'] = ensemble_score
        
        # Sort by ensemble score
        documents = sorted(documents, key=lambda x: x['ensemble_score'], reverse=True)
        
        logger.info(f"Ensemble reranking complete. Top ensemble score: {documents[0]['ensemble_score']:.4f}")
        
        return documents[:top_k]
    
    def _normalize(self, scores: List[float]) -> List[float]:
        """Min-max normalization"""
        if not scores:
            return scores
        
        min_score = min(scores)
        max_score = max(scores)
        
        if min_score == max_score:
            return [1.0] * len(scores)
        
        return [(s - min_score) / (max_score - min_score) for s in scores]


def test_reranker():
    """Test reranking system"""
    from indexing import IndexBuilder
    from retrieval import ParallelRetriever
    from query_processor import QueryProcessor
    
    cfg = Config()
    
    # Load indices
    builder = IndexBuilder(cfg)
    bm25_index = builder.load_bm25_index()
    articles = builder.load_metadata()
    
    try:
        dense_embeddings = builder.load_dense_embeddings()
    except FileNotFoundError:
        dense_embeddings = None
    
    # Create components
    query_processor = QueryProcessor(cfg)
    retriever = ParallelRetriever(cfg)
    retriever.load_indices(bm25_index, articles, dense_embeddings)
    reranker = EnsembleReranker(cfg)
    
    # Test query
    query = "artificial intelligence applications"
    
    # Process query
    processed_query = query_processor.process(query)
    
    # Retrieve
    documents = retriever.retrieve_with_documents(
        query_tokens=processed_query['tokens'],
        query_text=processed_query['corrected'],
        top_k=20
    )
    
    print(f"\nBefore reranking:")
    for i, doc in enumerate(documents[:5], 1):
        print(f"{i}. {doc['title']} (Score: {doc['retrieval_score']:.4f})")
    
    # Rerank
    reranked = reranker.rerank(
        query=processed_query['corrected'],
        documents=documents,
        query_entities=processed_query['entities'],
        top_k=10
    )
    
    print(f"\nAfter reranking:")
    for i, doc in enumerate(reranked, 1):
        print(f"{i}. {doc['title']} (Ensemble: {doc['ensemble_score']:.4f})")


if __name__ == "__main__":
    test_reranker()
