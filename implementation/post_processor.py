"""
Post-processing Module for Cortex IR System
Handles diversity, temporal intelligence, deduplication, and clustering
"""

from typing import List, Dict, Tuple, Set
from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np

from sentence_transformers import SentenceTransformer

import config
from utils import setup_logging, timer

logger = setup_logging(__name__)


class DiversityReranker:
    """
    Diversity-aware reranking using Maximal Marginal Relevance (MMR)
    """
    
    def __init__(self, embedding_model: str = None):
        """
        Initialize diversity reranker
        
        Args:
            embedding_model: Sentence transformer model for embeddings
        """
        if embedding_model is None:
            embedding_model = config.EMBEDDING_MODEL
        
        logger.info(f"Loading embedding model: {embedding_model}")
        
        try:
            self.model = SentenceTransformer(embedding_model)
            self.available = True
            logger.info("DiversityReranker initialized")
        except Exception as e:
            logger.error(f"Error loading embedding model: {e}")
            self.model = None
            self.available = False
    
    def _compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity between embeddings"""
        return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))
    
    @timer
    def rerank_mmr(
        self,
        query: str,
        documents: List[Dict],
        lambda_param: float = 0.7,
        top_k: int = None
    ) -> List[Dict]:
        """
        Rerank using Maximal Marginal Relevance for diversity
        
        Args:
            query: Query string
            documents: List of documents with scores
            lambda_param: Balance between relevance (1.0) and diversity (0.0)
            top_k: Number of documents to return
            
        Returns:
            Diversified list of documents
        """
        if not self.available or not documents:
            return documents[:top_k] if top_k else documents
        
        if top_k is None:
            top_k = config.FINAL_TOP_K
        
        logger.info(f"Applying MMR with lambda={lambda_param}")
        
        # Get embeddings for query and documents
        query_embedding = self.model.encode(query)
        
        doc_texts = [
            f"{doc.get('title', '')} {doc.get('content', '')[:500]}"
            for doc in documents
        ]
        doc_embeddings = self.model.encode(doc_texts)
        
        # MMR algorithm
        selected = []
        selected_embeddings = []
        remaining = list(range(len(documents)))
        
        # Get initial relevance scores
        relevance_scores = [
            doc.get('ensemble_score', doc.get('rerank_score', doc.get('retrieval_score', 0.0)))
            for doc in documents
        ]
        
        # Normalize relevance scores
        max_rel = max(relevance_scores) if relevance_scores else 1.0
        min_rel = min(relevance_scores) if relevance_scores else 0.0
        range_rel = max_rel - min_rel if max_rel > min_rel else 1.0
        
        relevance_scores_norm = [
            (score - min_rel) / range_rel
            for score in relevance_scores
        ]
        
        while len(selected) < top_k and remaining:
            mmr_scores = []
            
            for idx in remaining:
                # Relevance component
                relevance = relevance_scores_norm[idx]
                
                # Diversity component (maximum similarity to selected)
                if selected_embeddings:
                    max_sim = max(
                        self._compute_similarity(doc_embeddings[idx], sel_emb)
                        for sel_emb in selected_embeddings
                    )
                else:
                    max_sim = 0.0
                
                # MMR score
                mmr = lambda_param * relevance - (1 - lambda_param) * max_sim
                mmr_scores.append((idx, mmr))
            
            # Select document with highest MMR score
            best_idx, best_mmr = max(mmr_scores, key=lambda x: x[1])
            
            selected.append(best_idx)
            selected_embeddings.append(doc_embeddings[best_idx])
            remaining.remove(best_idx)
        
        # Return diversified documents
        diversified = [documents[idx] for idx in selected]
        
        logger.info(f"MMR reranking complete. Selected {len(diversified)} diverse documents")
        
        return diversified


class TemporalProcessor:
    """
    Temporal intelligence for boosting recent/relevant articles
    """
    
    def __init__(self):
        """Initialize temporal processor"""
        logger.info("TemporalProcessor initialized")
    
    @timer
    def apply_temporal_boost(
        self,
        query_type: str,
        documents: List[Dict],
        reference_date: datetime = None
    ) -> List[Dict]:
        """
        Apply temporal boosting based on query type
        
        Args:
            query_type: Type of query (breaking, historical, analytical, factual)
            documents: List of documents
            reference_date: Reference date for comparisons (default: now)
            
        Returns:
            Documents with temporal boost applied
        """
        if reference_date is None:
            reference_date = datetime.now()
        
        temporal_settings = config.TEMPORAL_SETTINGS.get(
            query_type,
            config.TEMPORAL_SETTINGS['analytical']
        )
        
        logger.info(f"Applying temporal boost for query type: {query_type}")
        
        for doc in documents:
            parsed_date = doc.get('parsed_date')
            
            if parsed_date is None:
                # No date info, neutral boost
                doc['temporal_boost'] = 1.0
                continue
            
            # Convert to datetime if needed
            if isinstance(parsed_date, str):
                try:
                    parsed_date = datetime.fromisoformat(parsed_date)
                except:
                    doc['temporal_boost'] = 1.0
                    continue
            
            # Calculate age in days
            age_days = (reference_date - parsed_date).days
            
            # Apply boost based on query type
            if query_type == 'breaking':
                # Boost recent articles
                recency_days = temporal_settings['recency_days']
                boost_factor = temporal_settings['boost_factor']
                
                if age_days <= recency_days:
                    boost = 1.0 + boost_factor * (1.0 - age_days / recency_days)
                else:
                    boost = 1.0 / (1.0 + np.log1p(age_days - recency_days))
                
                doc['temporal_boost'] = boost
                
            elif query_type == 'historical':
                # Date matching (neutral boost for now)
                doc['temporal_boost'] = 1.0
                
            elif query_type == 'factual':
                # Slight recency preference
                recency_days = temporal_settings['recency_days']
                boost_factor = temporal_settings['boost_factor']
                
                if age_days <= recency_days:
                    boost = 1.0 + (boost_factor - 1.0) * (1.0 - age_days / recency_days)
                else:
                    boost = 1.0
                
                doc['temporal_boost'] = boost
                
            else:  # analytical
                # No temporal bias
                doc['temporal_boost'] = 1.0
        
        # Apply boost to scores
        for doc in documents:
            current_score = doc.get(
                'ensemble_score',
                doc.get('rerank_score', doc.get('retrieval_score', 0.0))
            )
            doc['temporal_score'] = current_score * doc['temporal_boost']
        
        # Re-sort by temporal score
        documents.sort(key=lambda x: x['temporal_score'], reverse=True)
        
        logger.info("Temporal boost applied")
        
        return documents


class EntityDeduplicator:
    """
    Entity-based deduplication
    """
    
    def __init__(self):
        """Initialize entity deduplicator"""
        logger.info("EntityDeduplicator initialized")
    
    def _get_entity_set(self, doc: Dict) -> Set[str]:
        """Get set of high-confidence entities from document"""
        entities = doc.get('entities', [])
        
        entity_texts = set()
        for entity in entities:
            if entity.get('confidence', 0.0) >= config.ENTITY_CONFIDENCE_THRESHOLD:
                entity_texts.add(entity['text'].lower())
        
        return entity_texts
    
    def _jaccard_similarity(self, set1: Set, set2: Set) -> float:
        """Calculate Jaccard similarity between two sets"""
        if not set1 or not set2:
            return 0.0
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0
    
    @timer
    def deduplicate(
        self,
        documents: List[Dict],
        similarity_threshold: float = None
    ) -> List[Dict]:
        """
        Remove duplicate documents based on entity similarity
        
        Args:
            documents: List of documents
            similarity_threshold: Jaccard similarity threshold for duplicates
            
        Returns:
            Deduplicated list of documents
        """
        if similarity_threshold is None:
            similarity_threshold = config.ENTITY_SIMILARITY_THRESHOLD
        
        logger.info(f"Deduplicating with threshold={similarity_threshold}")
        
        deduplicated = []
        seen_entity_sets = []
        duplicates_removed = 0
        
        for doc in documents:
            entity_set = self._get_entity_set(doc)
            
            # Check similarity with already selected documents
            is_duplicate = False
            
            for seen_set in seen_entity_sets:
                similarity = self._jaccard_similarity(entity_set, seen_set)
                
                if similarity >= similarity_threshold:
                    is_duplicate = True
                    duplicates_removed += 1
                    break
            
            if not is_duplicate:
                deduplicated.append(doc)
                seen_entity_sets.append(entity_set)
        
        logger.info(f"Deduplication complete. Removed {duplicates_removed} duplicates")
        
        return deduplicated


class CategoryBalancer:
    """
    Balance results across categories
    """
    
    def __init__(self):
        """Initialize category balancer"""
        logger.info("CategoryBalancer initialized")
    
    @timer
    def balance(
        self,
        documents: List[Dict],
        preserve_top_k: int = 3
    ) -> List[Dict]:
        """
        Balance documents across categories
        
        Args:
            documents: List of documents
            preserve_top_k: Number of top documents to preserve regardless
            
        Returns:
            Balanced list of documents
        """
        if len(documents) <= preserve_top_k:
            return documents
        
        logger.info("Applying category balancing")
        
        # Preserve top documents
        preserved = documents[:preserve_top_k]
        remaining = documents[preserve_top_k:]
        
        # Group by category
        by_category = defaultdict(list)
        for doc in remaining:
            category = doc.get('category', 'Unknown')
            by_category[category].append(doc)
        
        # Interleave categories
        balanced = preserved.copy()
        categories = list(by_category.keys())
        
        if not categories:
            return documents
        
        # Round-robin through categories
        max_per_category = max(len(docs) for docs in by_category.values())
        
        for i in range(max_per_category):
            for category in categories:
                if i < len(by_category[category]):
                    balanced.append(by_category[category][i])
        
        logger.info(f"Category balancing complete. Categories: {categories}")
        
        return balanced


class ResultClusterer:
    """
    Cluster results by topic for presentation
    """
    
    def __init__(self):
        """Initialize result clusterer"""
        logger.info("ResultClusterer initialized")
    
    @timer
    def cluster(
        self,
        documents: List[Dict],
        n_clusters: int = 5
    ) -> Dict[str, List[Dict]]:
        """
        Cluster documents by topic
        
        Args:
            documents: List of documents with topic_id
            n_clusters: Maximum number of clusters
            
        Returns:
            Dictionary mapping cluster names to documents
        """
        logger.info(f"Clustering {len(documents)} documents into {n_clusters} clusters")
        
        # Group by topic_id
        by_topic = defaultdict(list)
        for doc in documents:
            topic_id = doc.get('topic_id', -1)
            by_topic[topic_id].append(doc)
        
        # Sort topics by size
        sorted_topics = sorted(
            by_topic.items(),
            key=lambda x: len(x[1]),
            reverse=True
        )
        
        # Create clusters
        clusters = {}
        for idx, (topic_id, docs) in enumerate(sorted_topics[:n_clusters]):
            cluster_name = f"Topic {idx + 1}"
            clusters[cluster_name] = docs
        
        # Add remaining to "Other" cluster
        if len(sorted_topics) > n_clusters:
            other_docs = []
            for topic_id, docs in sorted_topics[n_clusters:]:
                other_docs.extend(docs)
            if other_docs:
                clusters["Other Topics"] = other_docs
        
        logger.info(f"Created {len(clusters)} clusters")
        
        return clusters


class PostProcessor:
    """
    Complete post-processing pipeline
    """
    
    def __init__(self):
        """Initialize post-processor"""
        self.diversity_reranker = DiversityReranker()
        self.temporal_processor = TemporalProcessor()
        self.deduplicator = EntityDeduplicator()
        self.category_balancer = CategoryBalancer()
        self.clusterer = ResultClusterer()
        
        logger.info("PostProcessor initialized")
    
    @timer
    def process(
        self,
        query: str,
        query_type: str,
        documents: List[Dict],
        apply_diversity: bool = True,
        apply_temporal: bool = True,
        apply_deduplication: bool = True,
        apply_balancing: bool = False,
        top_k: int = None
    ) -> Tuple[List[Dict], Dict[str, List[Dict]]]:
        """
        Apply complete post-processing pipeline
        
        Args:
            query: Query string
            query_type: Type of query
            documents: List of documents
            apply_diversity: Whether to apply diversity reranking
            apply_temporal: Whether to apply temporal boost
            apply_deduplication: Whether to deduplicate
            apply_balancing: Whether to balance categories
            top_k: Final number of documents to return
            
        Returns:
            Tuple of (processed_documents, clusters)
        """
        if top_k is None:
            top_k = config.FINAL_TOP_K
        
        logger.info(f"Starting post-processing pipeline for {len(documents)} documents")
        
        processed = documents.copy()
        
        # 1. Diversity reranking
        if apply_diversity and self.diversity_reranker.available:
            lambda_param = (
                config.DIVERSITY_LAMBDA_FACTUAL 
                if query_type == 'factual' 
                else config.DIVERSITY_LAMBDA_EXPLORATORY
            )
            processed = self.diversity_reranker.rerank_mmr(
                query,
                processed,
                lambda_param=lambda_param,
                top_k=top_k * 2  # Get more for further processing
            )
        
        # 2. Temporal intelligence
        if apply_temporal:
            processed = self.temporal_processor.apply_temporal_boost(
                query_type,
                processed
            )
        
        # 3. Entity deduplication
        if apply_deduplication:
            processed = self.deduplicator.deduplicate(processed)
        
        # 4. Category balancing (optional)
        if apply_balancing:
            processed = self.category_balancer.balance(processed)
        
        # Limit to top_k
        processed = processed[:top_k]
        
        # 5. Topic clustering for presentation
        clusters = self.clusterer.cluster(processed)
        
        logger.info(f"Post-processing complete. Final: {len(processed)} documents")
        
        return processed, clusters


def main():
    """Test post-processing"""
    from retrieval import HybridRetriever
    from query_processor import QueryProcessor
    from reranker import EnsembleReranker
    
    # Initialize
    retriever = HybridRetriever()
    processor = QueryProcessor()
    reranker = EnsembleReranker()
    post_processor = PostProcessor()
    
    # Test query
    query = "impact of inflation on economy"
    
    print("\n=== Post-Processing Test ===\n")
    print(f"Query: {query}\n")
    
    # Process query
    query_info = processor.process(query)
    print(f"Query type: {query_info['type']}")
    
    # Retrieve
    print("\n1. Retrieving documents...")
    results, documents = retriever.retrieve_with_documents(
        query=query_info['corrected'],
        query_tokens=query_info['tokens'],
        k=100
    )
    
    # Rerank
    print(f"\n2. Reranking {len(documents)} documents...")
    reranked = reranker.rerank(query, documents, top_k=50)
    
    # Post-process
    print(f"\n3. Post-processing {len(reranked)} documents...")
    final_docs, clusters = post_processor.process(
        query=query,
        query_type=query_info['type'],
        documents=reranked,
        top_k=20
    )
    
    print(f"\n4. Final results: {len(final_docs)} documents")
    print(f"   Clustered into {len(clusters)} topic groups")
    
    print("\n5. Top 5 results:")
    for i, doc in enumerate(final_docs[:5], 1):
        print(f"{i}. {doc['title']}")
        print(f"   Category: {doc['category']}")
        print(f"   Temporal boost: {doc.get('temporal_boost', 1.0):.2f}")
        print()
    
    print("\n6. Clusters:")
    for cluster_name, cluster_docs in clusters.items():
        print(f"   {cluster_name}: {len(cluster_docs)} documents")


if __name__ == "__main__":
    main()
