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

from config import Config
from utils import logger

# Initialize config
config = Config()


class ArticlePreprocessor:
    """
    Preprocesses articles with tokenization, lemmatization, and NER
    """
    
    def __init__(self, config: Config = None):
        """Initialize preprocessor with spaCy model"""
        if config is None:
            config = Config()
        self.config = config
        
        logger.info(f"Loading spaCy model: {self.config.SPACY_MODEL}")
        try:
            self.nlp = spacy.load(self.config.SPACY_MODEL)
        except OSError:
            logger.warning(f"Model {self.config.SPACY_MODEL} not found. Downloading...")
            import os
            os.system(f"python -m spacy download {self.config.SPACY_MODEL}")
            self.nlp = spacy.load(self.config.SPACY_MODEL)
        
        # Add custom stopwords for query-adaptive removal
        self.query_stopwords = set([
            'who', 'what', 'when', 'where', 'why', 'how',
            'latest', 'news', 'update', 'information', 'article'
        ])
        
        logger.info("ArticlePreprocessor initialized")
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not text or pd.isna(text):
            return ""
        
        # Convert to string
        text = str(text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)
        
        # Remove emails
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\-\']', '', text)
        
        return text.strip()
    
    def preprocess_article(self, article: Dict) -> Dict:
        """
        Preprocess a single article
        
        Args:
            article: Dictionary with keys: Title, Content, Date, Category
            
        Returns:
            Preprocessed article dictionary with additional fields
        """
        # Extract basic fields with safe handling
        # Handle both column naming conventions
        title = str(article.get('Heading', article.get('Title', ''))).strip() if pd.notna(article.get('Heading', article.get('Title'))) else ''
        content = str(article.get('Article', article.get('Content', ''))).strip() if pd.notna(article.get('Article', article.get('Content'))) else ''
        date = str(article.get('Date', '')) if pd.notna(article.get('Date')) else ''
        category = str(article.get('NewsType', article.get('Category', 'Unknown'))) if pd.notna(article.get('NewsType', article.get('Category'))) else 'Unknown'
        
        # Clean text
        title_clean = self.clean_text(title)
        content_clean = self.clean_text(content)
        
        # Skip empty articles
        if not title_clean and not content_clean:
            logger.warning(f"Skipping empty article: {article.get('id', 'unknown')}")
            return None
        
        # Process with spaCy (limit content length)
        try:
            # Disable some pipes for speed
            doc_title = self.nlp(title_clean[:1000]) if title_clean else self.nlp("")  
            doc_content = self.nlp(content_clean[:50000]) if content_clean else self.nlp("")  # Limit to 50k chars
        except Exception as e:
            logger.warning(f"Error processing article with spaCy: {e}")
            # Continue with simple tokenization
            title_tokens = title_clean.lower().split()
            content_tokens = content_clean.lower().split()
            doc_title = None
            doc_content = None
        
        # Tokenization and lemmatization
        if doc_title is not None and doc_content is not None:
            # Load NLTK stop words
            try:
                import nltk
                from nltk.corpus import stopwords
                try:
                    nltk.data.find('corpora/stopwords')
                except LookupError:
                    nltk.download('stopwords')
                nltk_stopwords = set(stopwords.words('english'))
            except Exception as e:
                logger.warning(f"Could not load NLTK stopwords: {e}. Using spaCy defaults.")
                nltk_stopwords = set()

            title_tokens = [
                token.lemma_.lower() 
                for token in doc_title 
                if (token.text == 'US' or (token.text.lower() not in nltk_stopwords and not token.is_stop and not token.is_punct and token.is_alpha))
            ]
            
            content_tokens = [
                token.lemma_.lower() 
                for token in doc_content 
                if (token.text == 'US' or (token.text.lower() not in nltk_stopwords and not token.is_stop and not token.is_punct and token.is_alpha))
            ]
            
            # Named Entity Recognition with confidence filtering
            entities = self._extract_entities(doc_title, doc_content)
        else:
            # Fallback to simple tokenization
            import re
            title_tokens = [w.lower() for w in re.findall(r'\b[a-z]+\b', title_clean.lower()) if len(w) > 2]
            content_tokens = [w.lower() for w in re.findall(r'\b[a-z]+\b', content_clean.lower()) if len(w) > 2]
            entities = []
        
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
            'all_tokens': (title_tokens * self.config.BM25_TITLE_BOOST) + content_tokens,
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
                
                if confidence >= self.config.ENTITY_CONFIDENCE_THRESHOLD:
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
        if not date_string or pd.isna(date_string):
            return None
            
        try:
            return pd.to_datetime(date_string)
        except:
            return None
    
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
        skipped_count = 0
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Preprocessing articles"):
            article = row.to_dict()
            article['id'] = idx  # Use DataFrame index as ID
            
            try:
                processed = self.preprocess_article(article)
                if processed is not None:
                    processed_articles.append(processed)
                else:
                    skipped_count += 1
            except Exception as e:
                logger.error(f"Error preprocessing article {idx}: {e}")
                skipped_count += 1
                continue
        
        logger.info(f"Successfully preprocessed {len(processed_articles)} articles")
        if skipped_count > 0:
            logger.warning(f"Skipped {skipped_count} articles due to errors")
        
        return processed_articles
    
    def save_processed_articles(self, processed_articles: List[Dict]):
        """
        Save preprocessed articles to pickle file
        
        Args:
            processed_articles: List of preprocessed articles
        """
        import pickle
        
        filepath = self.config.PROCESSED_DATA_PATH
        
        with open(filepath, 'wb') as f:
            pickle.dump(processed_articles, f)
        
        logger.info(f"Saved {len(processed_articles)} preprocessed articles to {filepath}")


class QueryPreprocessor:
    """
    Preprocesses queries for retrieval
    """
    
    def __init__(self, config: Config = None):
        """Initialize query preprocessor"""
        if config is None:
            config = Config()
        self.config = config
        self.nlp = spacy.load(self.config.SPACY_MODEL)
        
        logger.info("QueryPreprocessor initialized")
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not text or pd.isna(text):
            return ""
        
        text = str(text)
        text = re.sub(r'http\S+|www\S+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s\.\,\!\?\-\']', '', text)
        
        return text.strip()
    
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
        query_clean = self.clean_text(query)
        
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


def main():
    """Main function to run preprocessing pipeline"""
    logger.info("Starting preprocessing pipeline...")
    
    # Initialize config
    cfg = Config()
    
    # Load dataset with encoding handling
    logger.info(f"Loading dataset from {cfg.DATA_PATH}")
    
    # Try different encodings
    encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
    df = None
    
    for encoding in encodings:
        try:
            logger.info(f"Trying encoding: {encoding}")
            df = pd.read_csv(cfg.DATA_PATH, encoding=encoding, on_bad_lines='skip')
            logger.info(f"Successfully loaded with {encoding} encoding")
            break
        except UnicodeDecodeError:
            continue
        except Exception as e:
            logger.error(f"Error with {encoding}: {e}")
            continue
    
    if df is None:
        logger.error("Failed to load dataset with any encoding")
        return
    
    logger.info(f"Loaded {len(df)} articles")
    logger.info(f"Columns: {df.columns.tolist()}")
    
    # Initialize preprocessor
    preprocessor = ArticlePreprocessor(cfg)
    
    # Preprocess corpus
    processed_articles = preprocessor.preprocess_corpus(df)
    
    if not processed_articles:
        logger.error("No articles were successfully preprocessed!")
        return
    
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
