"""
Query Processing Module for Cortex IR System
Handles query classification, expansion, and spell correction
"""

import re
from typing import Dict, List, Tuple
from collections import Counter

import config
from utils import setup_logging
from preprocessing import QueryPreprocessor

logger = setup_logging(__name__)


class QueryClassifier:
    """
    Classifies queries into types: Breaking, Historical, Factual, Analytical
    """
    
    def __init__(self):
        """Initialize query classifier"""
        # Keywords for query type classification
        self.breaking_keywords = [
            'latest', 'recent', 'new', 'breaking', 'today', 'yesterday',
            'current', 'now', 'just', 'update', 'developing'
        ]
        
        self.historical_keywords = [
            'history', 'past', 'ago', 'when', 'before', 'after',
            'years', 'decade', 'century', 'historical', 'origin'
        ]
        
        self.factual_keywords = [
            'who', 'what', 'where', 'which', 'name', 'list',
            'winner', 'score', 'result', 'fact', 'detail'
        ]
        
        self.analytical_keywords = [
            'why', 'how', 'analyze', 'impact', 'effect', 'cause',
            'reason', 'explain', 'compare', 'trend', 'insight'
        ]
        
        logger.info("QueryClassifier initialized")
    
    def classify(self, query: str) -> str:
        """
        Classify query into one of four types
        
        Args:
            query: Query string
            
        Returns:
            Query type: 'breaking', 'historical', 'factual', or 'analytical'
        """
        query_lower = query.lower()
        
        # Count keyword matches for each type
        scores = {
            'breaking': sum(1 for kw in self.breaking_keywords if kw in query_lower),
            'historical': sum(1 for kw in self.historical_keywords if kw in query_lower),
            'factual': sum(1 for kw in self.factual_keywords if kw in query_lower),
            'analytical': sum(1 for kw in self.analytical_keywords if kw in query_lower)
        }
        
        # Return type with highest score
        if max(scores.values()) == 0:
            return 'analytical'  # Default
        
        return max(scores, key=scores.get)


class QueryExpander:
    """
    Expands queries using Pseudo-Relevance Feedback (PRF) and synonyms
    """
    
    def __init__(self):
        """Initialize query expander"""
        self.preprocessor = QueryPreprocessor()
        logger.info("QueryExpander initialized")
    
    def expand_with_prf(
        self, 
        query: str, 
        top_docs: List[Dict], 
        n_terms: int = 5
    ) -> str:
        """
        Expand query using Pseudo-Relevance Feedback
        
        Args:
            query: Original query
            top_docs: Top retrieved documents
            n_terms: Number of expansion terms to add
            
        Returns:
            Expanded query string
        """
        if not top_docs:
            return query
        
        # Process original query
        query_processed = self.preprocessor.preprocess_query(query)
        query_terms = set(query_processed['tokens'])
        
        # Collect terms from top documents
        doc_terms = []
        for doc in top_docs[:3]:  # Use top 3 documents
            if 'content_tokens' in doc:
                doc_terms.extend(doc['content_tokens'])
            elif 'all_tokens' in doc:
                doc_terms.extend(doc['all_tokens'])
        
        # Count term frequencies
        term_freq = Counter(doc_terms)
        
        # Remove query terms and get most common
        expansion_candidates = [
            term for term, freq in term_freq.most_common(n_terms * 3)
            if term not in query_terms and len(term) > 3
        ]
        
        # Take top n_terms
        expansion_terms = expansion_candidates[:n_terms]
        
        # Create expanded query
        expanded = query + " " + " ".join(expansion_terms)
        
        logger.debug(f"Expanded query: {query} -> {expanded}")
        
        return expanded
    
    def expand_with_entities(self, query: str, entities: List[str]) -> str:
        """
        Expand query with related entities
        
        Args:
            query: Original query
            entities: List of related entities
            
        Returns:
            Expanded query string
        """
        if not entities:
            return query
        
        # Add top entities
        expansion = " ".join(entities[:3])
        expanded = f"{query} {expansion}"
        
        logger.debug(f"Entity-expanded query: {query} -> {expanded}")
        
        return expanded


class SpellCorrector:
    """
    Simple spell correction for queries
    """
    
    def __init__(self):
        """Initialize spell corrector"""
        # Common corrections for sports/business terms
        self.corrections = {
            'bussiness': 'business',
            'companie': 'company',
            'companys': 'companies',
            'winned': 'won',
            'losed': 'lost',
            'playr': 'player',
            'teem': 'team',
            'recieve': 'receive',
            'achive': 'achieve'
        }
        
        logger.info("SpellCorrector initialized")
    
    def correct(self, query: str) -> str:
        """
        Correct spelling in query
        
        Args:
            query: Query string
            
        Returns:
            Corrected query string
        """
        words = query.split()
        corrected_words = []
        
        for word in words:
            word_lower = word.lower()
            if word_lower in self.corrections:
                corrected_words.append(self.corrections[word_lower])
                logger.debug(f"Corrected: {word} -> {self.corrections[word_lower]}")
            else:
                corrected_words.append(word)
        
        return " ".join(corrected_words)


class QueryProcessor:
    """
    Main query processing pipeline
    """
    
    def __init__(self):
        """Initialize query processor with all components"""
        self.preprocessor = QueryPreprocessor()
        self.classifier = QueryClassifier()
        self.expander = QueryExpander()
        self.spell_corrector = SpellCorrector()
        
        logger.info("QueryProcessor initialized")
    
    def process(
        self, 
        query: str, 
        expand: bool = False,
        top_docs: List[Dict] = None
    ) -> Dict:
        """
        Process query through complete pipeline
        
        Args:
            query: Raw query string
            expand: Whether to expand query
            top_docs: Top documents for PRF (if expand=True)
            
        Returns:
            Dictionary with processed query information
        """
        # Spell correction
        query_corrected = self.spell_corrector.correct(query)
        
        # Preprocess
        query_processed = self.preprocessor.preprocess_query(query_corrected)
        
        # Classify
        query_type = self.classifier.classify(query_corrected)
        
        # Expand if requested
        expanded_query = query_corrected
        if expand and query_type == 'analytical' and top_docs:
            expanded_query = self.expander.expand_with_prf(
                query_corrected, 
                top_docs
            )
        
        result = {
            'original': query,
            'corrected': query_corrected,
            'processed': query_processed,
            'type': query_type,
            'expanded': expanded_query,
            'tokens': query_processed['tokens'],
            'entities': query_processed['entities']
        }
        
        logger.info(f"Query processed: '{query}' -> type='{query_type}'")
        
        return result


def main():
    """Test query processing"""
    processor = QueryProcessor()
    
    test_queries = [
        "latest COVID updates",
        "causes of 2008 financial crisis",
        "who won the super bowl?",
        "impact of inflation on economy",
        "bussiness trends in teem sports"
    ]
    
    print("\n=== Query Processing Test ===\n")
    
    for query in test_queries:
        result = processor.process(query)
        print(f"Query: {result['original']}")
        print(f"  Type: {result['type']}")
        print(f"  Corrected: {result['corrected']}")
        print(f"  Tokens: {result['tokens']}")
        print(f"  Entities: {result['entities']}")
        print()


if __name__ == "__main__":
    main()
