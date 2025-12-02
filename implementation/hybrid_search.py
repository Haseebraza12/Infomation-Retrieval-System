"""
Hybrid Search Engine
Combines Boolean retrieval, Metadata filtering, and Ranked retrieval
"""

from typing import List, Dict, Optional, Set, Tuple, Union
import numpy as np

from config import Config
from utils import logger, timer
from boolean_retrieval import BooleanRetriever
from metadata_manager import MetadataManager
from retrieval import ParallelRetriever

class HybridSearchEngine:
    """
    Orchestrates search across multiple retrieval methods
    """
    
    def __init__(self, config: Config = None):
        if config is None:
            config = Config()
        self.config = config
        
        # Initialize components
        self.boolean_retriever = BooleanRetriever(config)
        self.metadata_manager = MetadataManager(config)
        self.retriever = ParallelRetriever(config)
        
        # Load indices for retriever
        self._load_retriever_indices()
        
    def _load_retriever_indices(self):
        """Load indices for the underlying retriever"""
        try:
            from indexing import IndexBuilder
            builder = IndexBuilder(self.config)
            
            bm25_index = builder.load_bm25_index()
            articles = builder.load_metadata()
            
            try:
                dense_embeddings = builder.load_dense_embeddings()
            except FileNotFoundError:
                logger.warning("Dense embeddings not found, using BM25 only")
                dense_embeddings = None
                
            self.retriever.load_indices(bm25_index, articles, dense_embeddings)
            
        except Exception as e:
            logger.error(f"Error loading retriever indices: {e}")
            
    def _is_boolean_query(self, query: str) -> bool:
        """Check if query contains boolean operators, wildcards, or phrases"""
        operators = {'AND', 'OR', 'NOT'}
        parts = query.split()
        # Check for operators, parentheses, wildcards, or quotes
        return (any(part in operators for part in parts) or 
                '(' in query or 
                '*' in query or 
                '"' in query)
        
    @timer
    def search_with_documents(
        self,
        query: str,
        category: str = None,
        start_date: str = None,
        end_date: str = None,
        entity: str = None,
        top_k: int = 10,
        use_hybrid: bool = True
    ) -> Dict:
        """
        Execute hybrid search with filters
        
        Args:
            query: Search query
            category: Category filter
            start_date: Start date filter
            end_date: End date filter
            entity: Entity filter
            top_k: Number of results
            use_hybrid: Use hybrid retrieval
            
        Returns:
            Dictionary with results and metadata
        """
        # 1. Candidate Selection (Boolean + Metadata)
        candidate_ids = self._get_candidates(query, category, start_date, end_date, entity)
        
        # 2. Ranked Retrieval
        # If we have candidates, we need to restrict retrieval to them
        # Since bm25s doesn't support easy restriction, we'll retrieve more and filter
        # OR if candidates are few, we might just score them (but we need a scorer)
        
        # For now, we'll use a "Retrieve then Filter" approach for BM25
        # But for Boolean queries, we MUST respect the boolean logic
        
        is_boolean = self._is_boolean_query(query)
        
        # If boolean query, use the boolean candidates as the base
        # If not boolean, use standard retrieval and filter
        
        if is_boolean:
            if not candidate_ids:
                return {'results': [], 'total': 0, 'query_type': 'boolean'}
                
            # We have specific documents from Boolean search
            results = self._rank_candidates(list(candidate_ids), query, top_k)
            query_type = 'boolean'
            
        elif candidate_ids is not None:
            # Filters are active (Category, Date, Entity)
            # We must restrict search to these candidates
            if not candidate_ids:
                return {'results': [], 'total': 0, 'query_type': 'filtered'}
                
            # Use the same ranking logic as Boolean search
            results = self._rank_candidates(list(candidate_ids), query, top_k)
            query_type = 'filtered'
            
        else:
            # Standard search with no filters
            # Use the parallel retriever
            results = self.retriever.retrieve_with_documents(
                query_tokens=query.lower().split(), # Simple fallback tokenization
                query_text=query,
                top_k=top_k,
                use_hybrid=use_hybrid
            )
            query_type = 'standard'
            
        return {
            'results': results,
            'total': len(results),
            'query_type': query_type
        }
        
    def _get_candidates(
        self, 
        query: str, 
        category: str, 
        start_date: str, 
        end_date: str, 
        entity: str
    ) -> Optional[Set[int]]:
        """
        Get candidate document IDs based on boolean query and metadata filters
        Returns None if no filters are active (implies all docs are candidates)
        """
        candidates = None
        
        # 1. Boolean Retrieval
        if self._is_boolean_query(query):
            boolean_docs = self.boolean_retriever.retrieve(query)
            candidates = boolean_docs
            
        # 2. Metadata Filtering
        filters_active = False
        meta_candidates = None
        
        # Category
        if category and category.lower() != "all" and category.lower() != "none":
            cat_docs = self.metadata_manager.filter_by_category(category)
            meta_candidates = cat_docs if meta_candidates is None else meta_candidates & cat_docs
            filters_active = True
            
        # Date Range
        if start_date or end_date:
            date_docs = self.metadata_manager.filter_by_date_range(start_date, end_date)
            meta_candidates = date_docs if meta_candidates is None else meta_candidates & date_docs
            filters_active = True
            
        # Entity
        if entity:
            ent_docs = self.metadata_manager.filter_by_entity(entity)
            meta_candidates = ent_docs if meta_candidates is None else meta_candidates & ent_docs
            filters_active = True
            
        # Combine Boolean and Metadata candidates
        if candidates is not None:
            if meta_candidates is not None:
                candidates = candidates & meta_candidates
        elif meta_candidates is not None:
            candidates = meta_candidates
            
        return candidates

    def _rank_candidates(self, candidate_ids: List[int], query: str, top_k: int) -> List[Dict]:
        """
        Rank specific candidate documents using Dense Retrieval
        """
        if not candidate_ids:
            return []
            
        # Get embeddings for candidates
        # This requires accessing the dense index directly
        # We can use the retriever's dense embeddings
        
        if self.retriever.base_retriever.dense_embeddings is None:
            # Fallback to just returning metadata if no dense index
            # Or use BM25 if possible (but hard to score specific docs without custom logic)
            # Let's just return them in ID order or arbitrary order if no dense index
            logger.warning("No dense embeddings for ranking candidates. Returning unranked.")
            metadata = self.metadata_manager.get_metadata(candidate_ids)
            results = []
            for doc_id in candidate_ids:
                if doc_id in metadata:
                    doc = metadata[doc_id]
                    doc['retrieval_score'] = 1.0
                    results.append(doc)
            return results[:top_k]
            
        # We have dense embeddings
        all_embeddings = self.retriever.base_retriever.dense_embeddings
        model = self.retriever.base_retriever.dense_model
        
        # Filter embeddings
        # Note: doc_id must match index in embeddings array
        # We assume doc_id corresponds to index in processed_articles which corresponds to embeddings
        
        valid_ids = [i for i in candidate_ids if 0 <= i < len(all_embeddings)]
        if not valid_ids:
            return []
            
        candidate_embeddings = all_embeddings[valid_ids]
        
        # Encode query
        query_embedding = model.encode([query], normalize_embeddings=True)[0]
        
        # Compute scores
        scores = np.dot(candidate_embeddings, query_embedding)
        
        # Sort
        # argsort gives indices into candidate_embeddings, which map to valid_ids
        sorted_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        metadata = self.metadata_manager.get_metadata([valid_ids[i] for i in sorted_indices])
        
        for idx in sorted_indices:
            doc_id = valid_ids[idx]
            score = float(scores[idx])
            
            if doc_id in metadata:
                doc = metadata[doc_id]
                doc['retrieval_score'] = score
                results.append(doc)
                
        return results

if __name__ == "__main__":
    # Test
    engine = HybridSearchEngine()
    
    # Test Boolean
    print("\nTesting Boolean Search:")
    results = engine.search_with_documents("economy AND growth", top_k=5)
    print(f"Found {results['total']} results")
    for r in results['results']:
        print(f"- {r['title']}")
        
    # Test Filter
    print("\nTesting Category Filter:")
    results = engine.search_with_documents("market", category="Business", top_k=5)
    print(f"Found {results['total']} results")
    for r in results['results']:
        print(f"- {r['title']} ({r['category']})")
