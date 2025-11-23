"""
Post-Processing Module - Diversity ranking, deduplication, temporal boosting
"""

from typing import List, Dict, Tuple, Optional
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
import re

from config import Config
from utils import logger, timer, cosine_similarity, calculate_entity_overlap

config = Config()


class DiversityRanker:
    """Maximal Marginal Relevance (MMR) for diversity"""
    
    def __init__(self, config: Config = None):
        if config is None:
            config = Config()
        self.config = config
    
    @timer
    def rerank(
        self,
        documents: List[Dict],
        lambda_param: float = 0.7,
        top_k: int = 10
    ) -> List[Dict]:
        """
        Apply MMR for diversity-aware ranking
        
        Args:
            documents: List of documents with scores
            lambda_param: Balance between relevance and diversity (0-1)
            top_k: Number of results
            
        Returns:
            Diversified list of documents
        """
        if not documents or len(documents) <= 1:
            return documents[:top_k]
        
        # Get document vectors (use content tokens as simple representation)
        doc_vectors = []
        for doc in documents:
            # Create simple bag-of-words vector
            tokens = set(doc.get('content_tokens', []))
            doc_vectors.append(tokens)
        
        # Initialize
        selected = []
        selected_indices = set()
        remaining = list(range(len(documents)))
        
        # Select first document (highest relevance)
        first_idx = 0
        selected.append(documents[first_idx])
        selected_indices.add(first_idx)
        remaining.remove(first_idx)
        
        # Select remaining documents
        while len(selected) < top_k and remaining:
            best_score = -float('inf')
            best_idx = None
            
            for idx in remaining:
                # Relevance score
                relevance = documents[idx].get('ensemble_score',
                           documents[idx].get('rerank_score',
                           documents[idx].get('retrieval_score', 0)))
                
                # Calculate diversity (minimum similarity to selected docs)
                max_sim = 0
                for sel_idx in selected_indices:
                    # Jaccard similarity
                    intersection = len(doc_vectors[idx] & doc_vectors[sel_idx])
                    union = len(doc_vectors[idx] | doc_vectors[sel_idx])
                    similarity = intersection / union if union > 0 else 0
                    max_sim = max(max_sim, similarity)
                
                # MMR score
                mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx
            
            if best_idx is not None:
                selected.append(documents[best_idx])
                selected_indices.add(best_idx)
                remaining.remove(best_idx)
            else:
                break
        
        return selected


class TemporalBooster:
    """Apply temporal boosting based on recency"""
    
    def __init__(self, config: Config = None):
        if config is None:
            config = Config()
        self.config = config
    
    def boost(
        self,
        documents: List[Dict],
        query_type: str = 'general',
        boost_factor: float = None
    ) -> List[Dict]:
        """
        Apply temporal boosting
        
        Args:
            documents: List of documents
            query_type: Type of query (breaking, factual, etc.)
            boost_factor: Boost factor for recent articles
            
        Returns:
            Documents with adjusted scores
        """
        if boost_factor is None:
            # Query-type specific boosting
            if query_type == 'breaking':
                boost_factor = self.config.TEMPORAL_BOOST_BREAKING
            elif query_type == 'factual':
                boost_factor = self.config.TEMPORAL_BOOST_FACTUAL
            else:
                boost_factor = self.config.TEMPORAL_BOOST_DEFAULT
        
        if boost_factor == 1.0:
            return documents  # No boosting needed
        
        now = datetime.now()
        decay_days = self.config.TEMPORAL_DECAY_DAYS
        
        for doc in documents:
            parsed_date = doc.get('parsed_date')
            
            if parsed_date:
                try:
                    # Convert to datetime if string
                    if isinstance(parsed_date, str):
                        date_obj = datetime.fromisoformat(parsed_date)
                    else:
                        date_obj = parsed_date
                    
                    # Calculate age in days
                    age_days = (now - date_obj).days
                    
                    # Calculate boost (exponential decay)
                    if age_days < decay_days:
                        temporal_weight = boost_factor * np.exp(-age_days / decay_days)
                    else:
                        temporal_weight = 1.0
                    
                    # Apply boost to score
                    current_score = doc.get('ensemble_score',
                                   doc.get('rerank_score',
                                   doc.get('retrieval_score', 0)))
                    
                    doc['temporal_boosted_score'] = current_score * temporal_weight
                    doc['temporal_weight'] = temporal_weight
                except:
                    # If date parsing fails, no boost
                    doc['temporal_boosted_score'] = doc.get('ensemble_score',
                                                     doc.get('rerank_score',
                                                     doc.get('retrieval_score', 0)))
                    doc['temporal_weight'] = 1.0
            else:
                # No date, no boost
                doc['temporal_boosted_score'] = doc.get('ensemble_score',
                                                 doc.get('rerank_score',
                                                 doc.get('retrieval_score', 0)))
                doc['temporal_weight'] = 1.0
        
        # Re-sort by boosted scores
        documents = sorted(documents, key=lambda x: x['temporal_boosted_score'], reverse=True)
        
        return documents


class Deduplicator:
    """Remove duplicate or highly similar documents"""
    
    def __init__(self, config: Config = None):
        if config is None:
            config = Config()
        self.config = config
    
    def deduplicate(
        self,
        documents: List[Dict],
        threshold: float = None
    ) -> List[Dict]:
        """
        Remove duplicate documents
        
        Args:
            documents: List of documents
            threshold: Similarity threshold for deduplication
            
        Returns:
            Deduplicated list
        """
        if threshold is None:
            threshold = self.config.ENTITY_SIMILARITY_THRESHOLD
        
        if not documents:
            return documents
        
        unique_docs = []
        seen_titles = set()
        
        for doc in documents:
            title = doc.get('title', '').lower().strip()
            
            # Check for exact title match
            if title in seen_titles:
                continue
            
            # Check for entity similarity with existing docs
            is_duplicate = False
            doc_entities = doc.get('entities', [])
            
            for existing in unique_docs:
                existing_entities = existing.get('entities', [])
                
                # Calculate entity overlap
                if doc_entities and existing_entities:
                    overlap = calculate_entity_overlap(doc_entities, existing_entities)
                    
                    if overlap >= threshold:
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                unique_docs.append(doc)
                seen_titles.add(title)
        
        logger.info(f"Deduplication: {len(documents)} -> {len(unique_docs)} documents")
        
        return unique_docs


class TopicClusterer:
    """Group documents by topics"""
    
    def __init__(self, config: Config = None):
        if config is None:
            config = Config()
        self.config = config
    
    def cluster(
        self,
        documents: List[Dict]
    ) -> Dict[str, List[Dict]]:
        """
        Cluster documents by category and entities
        
        Args:
            documents: List of documents
            
        Returns:
            Dictionary of topic -> documents
        """
        clusters = defaultdict(list)
        
        for doc in documents:
            # Primary clustering by category
            category = doc.get('category', 'Unknown')
            clusters[category].append(doc)
        
        return dict(clusters)


class PostProcessor:
    """Main post-processing pipeline"""
    
    def __init__(self, config: Config = None):
        if config is None:
            config = Config()
        
        self.config = config
        self.diversity_ranker = DiversityRanker(config)
        self.temporal_booster = TemporalBooster(config)
        self.deduplicator = Deduplicator(config)
        self.topic_clusterer = TopicClusterer(config)
        
        logger.info("Post-processor initialized")
    
    @timer
    def process(
        self,
        documents: List[Dict],
        query: Dict,
        top_k: int = 10
    ) -> Tuple[List[Dict], Dict]:
        """
        Apply complete post-processing pipeline
        
        Args:
            documents: List of documents from reranking
            query: Processed query dictionary
            top_k: Final number of results
            
        Returns:
            Tuple of (processed documents, topic clusters)
        """
        if not documents:
            return documents, {}
        
        logger.info(f"Starting post-processing pipeline for {len(documents)} documents")
        
        # 1. Temporal boosting (query-type aware)
        query_type = query.get('type', 'general')
        documents = self.temporal_booster.boost(documents, query_type=query_type)
        
        # 2. Deduplication
        documents = self.deduplicator.deduplicate(documents)
        
        # 3. Diversity ranking with query-type specific lambda
        if query_type == 'breaking':
            lambda_param = self.config.DIVERSITY_LAMBDA_BREAKING
        elif query_type == 'factual':
            lambda_param = self.config.DIVERSITY_LAMBDA_FACTUAL
        elif query_type == 'analytical':
            lambda_param = self.config.DIVERSITY_LAMBDA_ANALYTICAL
        elif query_type == 'historical':
            lambda_param = self.config.DIVERSITY_LAMBDA_HISTORICAL
        elif query_type == 'exploratory':
            lambda_param = self.config.DIVERSITY_LAMBDA_EXPLORATORY
        else:
            lambda_param = self.config.DIVERSITY_LAMBDA_DEFAULT
        
        documents = self.diversity_ranker.rerank(
            documents,
            lambda_param=lambda_param,
            top_k=top_k
        )
        
        # 4. Topic clustering (for display)
        clusters = self.topic_clusterer.cluster(documents)
        
        logger.info(f"Post-processing complete: {len(documents)} final results")
        
        return documents, clusters


def test_post_processor():
    """Test post-processing"""
    from indexing import IndexBuilder
    from retrieval import ParallelRetriever
    from query_processor import QueryProcessor
    from reranker import EnsembleReranker
    
    cfg = Config()
    
    # Load indices
    builder = IndexBuilder(cfg)
    bm25_index = builder.load_bm25_index()
    articles = builder.load_metadata()
    
    try:
        dense_embeddings = builder.load_dense_embeddings()
    except:
        dense_embeddings = None
    
    # Components
    query_processor = QueryProcessor(cfg)
    retriever = ParallelRetriever(cfg)
    retriever.load_indices(bm25_index, articles, dense_embeddings)
    reranker = EnsembleReranker(cfg)
    post_processor = PostProcessor(cfg)
    
    # Test
    query = "latest sports news"
    processed_query = query_processor.process(query)
    
    documents = retriever.retrieve_with_documents(
        query_tokens=processed_query['tokens'],
        query_text=processed_query['corrected'],
        top_k=50
    )
    
    documents = reranker.rerank(
        query=processed_query['corrected'],
        documents=documents,
        query_entities=processed_query['entities'],
        top_k=20
    )
    
    print(f"\nBefore post-processing: {len(documents)} docs")
    
    final_docs, clusters = post_processor.process(
        documents=documents,
        query=processed_query,
        top_k=10
    )
    
    print(f"After post-processing: {len(final_docs)} docs")
    print(f"\nClusters: {list(clusters.keys())}")
    
    for i, doc in enumerate(final_docs, 1):
        print(f"\n{i}. {doc['title']}")
        print(f"   Category: {doc['category']}")
        print(f"   Temporal weight: {doc.get('temporal_weight', 1.0):.3f}")


if __name__ == "__main__":
    test_post_processor()
