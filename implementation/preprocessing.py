"""
Preprocessing Module for Cortex IR System
Handles article preprocessing: tokenization, lemmatization, NER, and entity extraction
"""

import re
from typing import Dict, List, Tuple
from datetime import datetime
import pandas as pd
import spacy
from spacy.language import Language
from tqdm import tqdm

import config
from utils import setup_logging, timer, save_pickle, clean_text

logger = setup_logging(__name__)


class ArticlePreprocessor:
    """
    Preprocesses articles with tokenization, lemmatization, and NER
    """
    
    def __init__(self):
        """Initialize preprocessor with spaCy model"""
        logger.info(f"Loading spaCy model: {config.SPACY_MODEL}")
        try:
            self.nlp = spacy.load(config.SPACY_MODEL)
        except OSError:
            logger.warning(f"Model {config.SPACY_MODEL} not found. Downloading...")
            import os
            os.system(f"python -m spacy download {config.SPACY_MODEL}")
            self.nlp = spacy.load(config.SPACY_MODEL)
        
        # Add custom stopwords for query-adaptive removal
        self.query_stopwords = set([
            'who', 'what', 'when', 'where', 'why', 'how',
            'latest', 'news', 'update', 'information', 'article'
        ])
        
        logger.info("ArticlePreprocessor initialized")
    
    @timer
    def preprocess_article(self, article: Dict) -> Dict:
        """
        Preprocess a single article
        
        Args:
            article: Dictionary with keys: Title, Content, Date, Category
            
        Returns:
            Preprocessed article dictionary with additional fields
        """
        # Extract basic fields
        title = str(article.get('Title', '')).strip()
        content = str(article.get('Content', '')).strip()
        date = article.get('Date', '')
        category = article.get('Category', 'Unknown')
        
        # Clean text
        title_clean = clean_text(title)
        content_clean = clean_text(content)
        
        # Process with spaCy
        doc_title = self.nlp(title_clean)
        doc_content = self.nlp(content_clean[:1000000])  # Limit to 1M chars for spaCy
        
        # Tokenization and lemmatization
        title_tokens = [
            token.lemma_.lower() 
            for token in doc_title 
            if not token.is_stop and not token.is_punct and token.is_alpha
        ]
        
        content_tokens = [
            token.lemma_.lower() 
            for token in doc_content 
            if not token.is_stop and not token.is_punct and token.is_alpha
        ]
        
        # Named Entity Recognition with confidence filtering
        entities = self._extract_entities(doc_title, doc_content)
        
        # Date parsing
        parsed_date = self._parse_date(date)
        
        # Create preprocessed article
        preprocessed = {
            'id': article.get('id', hash(title + content)),
            'title': title,
            'content': content,
            'title_clean': title_clean,
            'content_clean': content_clean,
            'title_tokens': title_tokens,
            'content_tokens': content_tokens,
            'all_tokens': title_tokens + content_tokens,
            'entities': entities,
            'date': date,
            'parsed_date': parsed_date,
            'category': category,
            'doc_length': len(content_tokens),
            'title_length': len(title_tokens)
        }
        
        return preprocessed
    
    def _extract_entities(self, doc_title, doc_content) -> List[Dict]:
        """
        Extract named entities with confidence filtering
        
        Args:
            doc_title: spaCy Doc object for title
            doc_content: spaCy Doc object for content
            
        Returns:
            List of entity dictionaries with text, label, and confidence
        """
        entities = []
        seen_entities = set()
        
        # Process title entities (higher weight)
        for ent in doc_title.ents:
            entity_key = (ent.text.lower(), ent.label_)
            if entity_key not in seen_entities:
                entities.append({
                    'text': ent.text,
                    'label': ent.label_,
                    'confidence': 0.95,  # Title entities are more reliable
                    'in_title': True
                })
                seen_entities.add(entity_key)
        
        # Process content entities with confidence filtering
        for ent in doc_content.ents:
            entity_key = (ent.text.lower(), ent.label_)
            if entity_key not in seen_entities:
                # Simple confidence estimation based on entity properties
                confidence = self._estimate_entity_confidence(ent)
                
                if confidence >= config.ENTITY_CONFIDENCE_THRESHOLD:
                    entities.append({
                        'text': ent.text,
                        'label': ent.label_,
                        'confidence': confidence,
                        'in_title': False
                    })
                    seen_entities.add(entity_key)
        
        return entities
    
    def _estimate_entity_confidence(self, ent) -> float:
        """
        Estimate entity confidence based on heuristics
        
        Args:
            ent: spaCy entity
            
        Returns:
            Confidence score (0-1)
        """
        confidence = 0.85  # Base confidence
        
        # Boost for capitalized entities
        if ent.text[0].isupper():
            confidence += 0.05
        
        # Boost for longer entities
        if len(ent.text.split()) > 1:
            confidence += 0.05
        
        # Boost for specific entity types
        if ent.label_ in ['PERSON', 'ORG', 'GPE', 'DATE', 'MONEY']:
            confidence += 0.05
        
        return min(confidence, 1.0)
    
    def _parse_date(self, date_string: str):
        """
        Parse date string to datetime object
        
        Args:
            date_string: Date string
            
        Returns:
            datetime object or None
        """
        try:
            return pd.to_datetime(date_string)
        except:
            return None
    
    @timer
    def preprocess_corpus(self, df: pd.DataFrame) -> List[Dict]:
        """
        Preprocess entire corpus of articles
        
        Args:
            df: DataFrame with articles
            
        Returns:
            List of preprocessed article dictionaries
        """
        logger.info(f"Preprocessing {len(df)} articles...")
        
        processed_articles = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Preprocessing articles"):
            article = row.to_dict()
            article['id'] = idx  # Use DataFrame index as ID
            
            try:
                processed = self.preprocess_article(article)
                processed_articles.append(processed)
            except Exception as e:
                logger.error(f"Error preprocessing article {idx}: {e}")
                continue
        
        logger.info(f"Successfully preprocessed {len(processed_articles)} articles")
        
        return processed_articles
    
    @timer
    def save_processed_articles(self, processed_articles: List[Dict], filepath: str = None):
        """
        Save preprocessed articles to pickle file
        
        Args:
            processed_articles: List of preprocessed articles
            filepath: Output file path (default: from config)
        """
        if filepath is None:
            filepath = config.PROCESSED_DATA_PATH
        
        save_pickle(processed_articles, filepath)
        logger.info(f"Saved {len(processed_articles)} preprocessed articles to {filepath}")


class QueryPreprocessor:
    """
    Preprocesses queries for retrieval
    """
    
    def __init__(self, nlp: Language = None):
        """Initialize query preprocessor"""
        if nlp is None:
            self.nlp = spacy.load(config.SPACY_MODEL)
        else:
            self.nlp = nlp
        
        logger.info("QueryPreprocessor initialized")
    
    def preprocess_query(self, query: str, remove_stopwords: bool = False) -> Dict:
        """
        Preprocess a query
        
        Args:
            query: Raw query string
            remove_stopwords: Whether to remove stopwords
            
        Returns:
            Dictionary with processed query components
        """
        # Clean query
        query_clean = clean_text(query)
        
        # Process with spaCy
        doc = self.nlp(query_clean)
        
        # Tokenization and lemmatization
        if remove_stopwords:
            tokens = [
                token.lemma_.lower() 
                for token in doc 
                if not token.is_stop and not token.is_punct and token.is_alpha
            ]
        else:
            tokens = [
                token.lemma_.lower() 
                for token in doc 
                if not token.is_punct and token.is_alpha
            ]
        
        # Extract entities
        entities = [
            {'text': ent.text, 'label': ent.label_}
            for ent in doc.ents
        ]
        
        return {
            'original': query,
            'clean': query_clean,
            'tokens': tokens,
            'entities': entities,
            'token_string': ' '.join(tokens)
        }
    
    def expand_query_with_synonyms(self, query: str, synonyms: Dict[str, List[str]] = None) -> str:
        """
        Expand query with synonyms
        
        Args:
            query: Original query
            synonyms: Dictionary of word -> synonyms
            
        Returns:
            Expanded query string
        """
        if synonyms is None:
            # Default synonyms for common terms
            synonyms = {
                'win': ['victory', 'triumph', 'success'],
                'lose': ['defeat', 'loss', 'fail'],
                'company': ['corporation', 'business', 'firm'],
                'increase': ['rise', 'growth', 'surge'],
                'decrease': ['fall', 'decline', 'drop']
            }
        
        processed = self.preprocess_query(query)
        expanded_terms = []
        
        for token in processed['tokens']:
            expanded_terms.append(token)
            if token in synonyms:
                expanded_terms.extend(synonyms[token][:2])  # Add top 2 synonyms
        
        return ' '.join(expanded_terms)


def main():
    """Main function to run preprocessing pipeline"""
    logger.info("Starting preprocessing pipeline...")
    
    # Load dataset
    logger.info(f"Loading dataset from {config.DATA_PATH}")
    df = pd.read_csv(config.DATA_PATH)
    logger.info(f"Loaded {len(df)} articles")
    
    # Initialize preprocessor
    preprocessor = ArticlePreprocessor()
    
    # Preprocess corpus
    processed_articles = preprocessor.preprocess_corpus(df)
    
    # Save processed articles
    preprocessor.save_processed_articles(processed_articles)
    
    # Print statistics
    logger.info("\n=== Preprocessing Statistics ===")
    logger.info(f"Total articles: {len(processed_articles)}")
    logger.info(f"Average document length: {sum(a['doc_length'] for a in processed_articles) / len(processed_articles):.1f} tokens")
    
    categories = {}
    for article in processed_articles:
        cat = article['category']
        categories[cat] = categories.get(cat, 0) + 1
    logger.info(f"Categories: {categories}")
    
    total_entities = sum(len(a['entities']) for a in processed_articles)
    logger.info(f"Total entities extracted: {total_entities}")
    logger.info(f"Average entities per article: {total_entities / len(processed_articles):.1f}")
    
    logger.info("\nPreprocessing complete!")


if __name__ == "__main__":
    main()
