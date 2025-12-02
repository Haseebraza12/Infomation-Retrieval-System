"""
Query Processing Module - Handles query analysis, correction, and expansion
"""

from typing import Dict, List, Set
import re
import spacy
from spellchecker import SpellChecker

from config import Config
from utils import logger, timer, load_pickle

config = Config()


class QueryProcessor:
    """
    Processes queries: spelling correction, entity extraction, classification
    """
    
    def __init__(self, config: Config = None):
        """Initialize query processor"""
        if config is None:
            config = Config()
        
        self.config = config
        
        # Load spaCy model
        logger.info(f"Loading spaCy model: {self.config.SPACY_MODEL}")
        self.nlp = spacy.load(self.config.SPACY_MODEL)
        
        # Initialize spell checker
        self.spell = SpellChecker()
        
        # Load vocabulary from processed data to enhance spell checker
        try:
            logger.info("Loading vocabulary for spell checker...")
            processed_path = self.config.PROCESSED_DATA_PATH
            if processed_path.exists():
                articles = load_pickle(processed_path)
                vocab = set()
                for article in articles:
                    # Add title and content tokens (assuming they are lists of strings)
                    # We use the raw tokens or lemmatized? 
                    # Spell checker usually works on raw words.
                    # But processed data might have 'all_tokens' which are lemmatized?
                    # Let's check what's available. 'title_tokens' and 'content_tokens' are usually lemmatized in preprocessing.py
                    # But we also have raw text. 
                    # If we want to correct to *index terms*, we should use the terms in the index.
                    # But users type raw words.
                    # If I type "runing", I want "running" (which might lemmatize to "run").
                    # If I type "sindh", and "sindh" is in index, I want to keep it.
                    # So adding lemmatized tokens to spell checker might be okay if we want to match index.
                    # But better to add raw words if possible. 
                    # Preprocessing stores 'title_tokens' and 'content_tokens' which are lemmatized.
                    # It doesn't seem to store raw tokens list easily accessible without re-tokenizing raw text.
                    # However, 'all_tokens' used for boolean index are lemmatized.
                    # Let's use the tokens we have. It's better than nothing.
                    vocab.update(article.get('title_tokens', []))
                    vocab.update(article.get('content_tokens', []))
                
                # Add to spell checker
                self.spell.word_frequency.load_words(list(vocab))
                logger.info(f"Loaded {len(vocab)} terms into spell checker")
        except Exception as e:
            logger.warning(f"Could not load vocabulary: {e}")
        
        # Query type patterns
        self.query_patterns = {
            'breaking': ['latest', 'breaking', 'recent', 'news', 'today', 'now'],
            'factual': ['who', 'what', 'when', 'where', 'which', 'define', 'explain'],
            'analytical': ['why', 'how', 'analyze', 'compare', 'impact', 'effect', 'cause'],
            'historical': ['history', 'past', 'previously', 'before', 'ago', 'since']
        }
        
        logger.info("Query processor initialized")
    
    def _correct_spelling(self, query: str) -> str:
        """Correct spelling errors in query"""
        words = query.split()
        corrected_words = []
        
        for word in words:
            # Check if word is in our index vocabulary (case insensitive)
            # self.spell.known([word]) checks if it's in the dictionary (English + Index)
            
            # If it's in the index (known), keep it.
            if self.spell.known([word]):
                 corrected_words.append(word)
            else:
                 # Not known. Is it capitalized? (Likely proper noun not in dict)
                 if word[0].isupper():
                     corrected_words.append(word)
                 else:
                     # Try to correct
                     corrected = self.spell.correction(word)
                     corrected_words.append(corrected if corrected else word)
        
        return ' '.join(corrected_words)
    
    def _classify_query(self, query: str) -> str:
        """
        Classify query type
        
        Args:
            query: Query string
            
        Returns:
            Query type (breaking, factual, analytical, historical, exploratory)
        """
        query_lower = query.lower()
        
        # Check patterns
        for qtype, patterns in self.query_patterns.items():
            for pattern in patterns:
                if pattern in query_lower:
                    return qtype
        
        # Default
        return 'exploratory'
    
    def _extract_entities(self, query: str) -> List[Dict]:
        """Extract named entities from query"""
        doc = self.nlp(query)
        
        entities = []
        for ent in doc.ents:
            entities.append({
                'text': ent.text,
                'label': ent.label_
            })
        
        return entities
    
    def _tokenize(self, query: str) -> List[str]:
        """Tokenize and lemmatize query"""
        doc = self.nlp(query.lower())
        
        tokens = [
            token.lemma_ 
            for token in doc 
            if not token.is_stop and not token.is_punct and token.is_alpha
        ]
        
        return tokens
    
    @timer
    def process(self, query: str) -> Dict:
        """
        Complete query processing pipeline
        
        Args:
            query: Raw query string
            
        Returns:
            Dictionary with processed query components
        """
        logger.info(f"Processing query: '{query}'")
        
        # Clean query
        query_clean = re.sub(r'\s+', ' ', query).strip()
        
        # Spelling correction
        query_corrected = self._correct_spelling(query_clean)
        
        # Classification
        query_type = self._classify_query(query_corrected)
        
        # Entity extraction
        entities = self._extract_entities(query_corrected)
        
        # Tokenization
        tokens = self._tokenize(query_corrected)
        
        result = {
            'original': query,
            'clean': query_clean,
            'corrected': query_corrected,
            'type': query_type,
            'entities': entities,
            'tokens': tokens
        }
        
        logger.info(f"Query processed: '{query}' -> type='{query_type}'")
        
        return result


def test_query_processor():
    """Test query processor"""
    processor = QueryProcessor()
    
    test_queries = [
        "latest sports news",
        "who won the championship?",
        "impact of inflation on economy",
        "history of artificial intelligence",
        "explin machine lerning"  # Spelling errors
    ]
    
    for query in test_queries:
        print(f"\n{'='*70}")
        print(f"Query: {query}")
        print('='*70)
        
        result = processor.process(query)
        
        print(f"Corrected: {result['corrected']}")
        print(f"Type: {result['type']}")
        print(f"Tokens: {result['tokens']}")
        print(f"Entities: {result['entities']}")


if __name__ == "__main__":
    test_query_processor()
