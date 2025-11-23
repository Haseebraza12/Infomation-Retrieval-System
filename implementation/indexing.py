"""
Indexing Module for Cortex IR System
Handles BM25, ColBERT, and SQLite metadata index construction
"""

import sqlite3
import json
from pathlib import Path
from typing import List, Dict
import numpy as np

import bm25s
from tqdm import tqdm

import config
from utils import setup_logging, timer, load_pickle, save_pickle

logger = setup_logging(__name__)


class BM25Indexer:
    """
    BM25 sparse index construction using bm25s library
    """
    
    def __init__(self, k1: float = None, b: float = None):
        """
        Initialize BM25 indexer
        
        Args:
            k1: BM25 k1 parameter (default from config)
            b: BM25 b parameter (default from config)
        """
        self.k1 = k1 if k1 is not None else config.BM25_K1
        self.b = b if b is not None else config.BM25_B
        self.retriever = None
        self.corpus = None
        
        logger.info(f"BM25Indexer initialized with k1={self.k1}, b={self.b}")
    
    @timer
    def build_index(self, processed_articles: List[Dict], save_path: str = None):
        """
        Build BM25 index from preprocessed articles
        
        Args:
            processed_articles: List of preprocessed article dictionaries
            save_path: Path to save index (default from config)
        """
        logger.info(f"Building BM25 index for {len(processed_articles)} articles...")
        
        # Store corpus reference
        self.corpus = processed_articles
        
        # Prepare corpus tokens with field boosting
        corpus_tokens = []
        for article in tqdm(processed_articles, desc="Preparing corpus"):
            # Combine title (weighted) and content tokens
            title_tokens = article['title_tokens'] * int(config.TITLE_BOOST)
            content_tokens = article['content_tokens']
            
            # Boost entity tokens
            entity_tokens = []
            for entity in article['entities']:
                if entity['confidence'] >= config.ENTITY_CONFIDENCE_THRESHOLD:
                    entity_tokens.extend(
                        entity['text'].lower().split() * int(config.ENTITY_BOOST)
                    )
            
            combined_tokens = title_tokens + content_tokens + entity_tokens
            corpus_tokens.append(combined_tokens)
        
        # Build BM25 index
        self.retriever = bm25s.BM25()
        
        # Tokenize and create index
        logger.info("Creating BM25 index...")
        corpus_tokens_array = bm25s.tokenize(corpus_tokens, stopwords=None)
        self.retriever.index(corpus_tokens_array)
        
        # Save index
        if save_path is None:
            save_path = config.BM25_INDEX_PATH
        
        self._save_index(save_path)
        
        logger.info(f"BM25 index built and saved to {save_path}")
    
    def _save_index(self, save_path: str):
        """Save BM25 index to disk"""
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save BM25 index
        self.retriever.save(save_path)
        
        # Save corpus reference
        corpus_path = Path(save_path) / "corpus_ref.pkl"
        save_pickle(self.corpus, str(corpus_path))
        
        logger.info(f"BM25 index saved to {save_path}")
    
    def load_index(self, load_path: str = None):
        """Load BM25 index from disk"""
        if load_path is None:
            load_path = config.BM25_INDEX_PATH
        
        logger.info(f"Loading BM25 index from {load_path}")
        
        # Load BM25 index
        self.retriever = bm25s.BM25.load(load_path, load_corpus=False)
        
        # Load corpus reference
        corpus_path = Path(load_path) / "corpus_ref.pkl"
        self.corpus = load_pickle(str(corpus_path))
        
        logger.info(f"BM25 index loaded successfully")


class ColBERTIndexer:
    """
    ColBERT dense index construction using RAGatouille
    """
    
    def __init__(self):
        """Initialize ColBERT indexer"""
        self.model = None
        self.index_name = "news_colbert_index"
        
        logger.info("ColBERTIndexer initialized")
    
    @timer
    def build_index(self, processed_articles: List[Dict], save_path: str = None):
        """
        Build ColBERT index from preprocessed articles
        
        Args:
            processed_articles: List of preprocessed article dictionaries
            save_path: Path to save index (default from config)
        """
        try:
            from ragatouille import RAGPretrainedModel
            
            logger.info(f"Building ColBERT index for {len(processed_articles)} articles...")
            logger.info("This may take 2-3 minutes...")
            
            # Initialize ColBERT model
            self.model = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")
            
            # Prepare documents for indexing
            documents = []
            doc_ids = []
            
            for article in tqdm(processed_articles, desc="Preparing documents"):
                # Combine title and content (truncate to 512 tokens)
                title = article['title']
                content = article['content'][:2000]  # Limit content length
                doc_text = f"{title}. {content}"
                
                documents.append(doc_text)
                doc_ids.append(str(article['id']))
            
            # Determine save path
            if save_path is None:
                save_path = config.COLBERT_INDEX_PATH
            
            # Index with ColBERT
            logger.info("Indexing with ColBERT (this takes time)...")
            self.model.index(
                collection=documents,
                document_ids=doc_ids,
                index_name=self.index_name,
                max_document_length=512,
                split_documents=False
            )
            
            logger.info(f"ColBERT index built successfully")
            
        except ImportError:
            logger.warning("RAGatouille not installed. Skipping ColBERT indexing.")
            logger.info("Install with: pip install ragatouille")
        except Exception as e:
            logger.error(f"Error building ColBERT index: {e}")
            logger.info("Continuing without ColBERT index...")
    
    def load_index(self, index_name: str = None):
        """Load ColBERT index"""
        try:
            from ragatouille import RAGPretrainedModel
            
            if index_name is None:
                index_name = self.index_name
            
            logger.info(f"Loading ColBERT index: {index_name}")
            self.model = RAGPretrainedModel.from_index(index_name)
            logger.info("ColBERT index loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading ColBERT index: {e}")
            self.model = None


class MetadataIndexer:
    """
    SQLite metadata store for fast filtering
    """
    
    def __init__(self, db_path: str = None):
        """
        Initialize metadata indexer
        
        Args:
            db_path: Path to SQLite database (default from config)
        """
        self.db_path = db_path if db_path is not None else config.METADATA_DB_PATH
        self.conn = None
        
        logger.info(f"MetadataIndexer initialized with db_path={self.db_path}")
    
    @timer
    def build_index(self, processed_articles: List[Dict]):
        """
        Build SQLite metadata index
        
        Args:
            processed_articles: List of preprocessed article dictionaries
        """
        logger.info(f"Building metadata index for {len(processed_articles)} articles...")
        
        # Create database directory
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Connect to database
        self.conn = sqlite3.connect(self.db_path)
        cursor = self.conn.cursor()
        
        # Create table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS articles (
            id INTEGER PRIMARY KEY,
            title TEXT NOT NULL,
            content TEXT NOT NULL,
            date TEXT,
            parsed_date TIMESTAMP,
            category TEXT,
            doc_length INTEGER,
            title_length INTEGER,
            entities TEXT,
            entity_count INTEGER,
            topic_id INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Insert articles
        for article in tqdm(processed_articles, desc="Building metadata store"):
            # Serialize entities as JSON
            entities_json = json.dumps(article['entities'])
            
            cursor.execute('''
            INSERT OR REPLACE INTO articles (
                id, title, content, date, parsed_date, category,
                doc_length, title_length, entities, entity_count, topic_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                article['id'],
                article['title'],
                article['content'],
                article['date'],
                article['parsed_date'],
                article['category'],
                article['doc_length'],
                article['title_length'],
                entities_json,
                len(article['entities']),
                article.get('topic_id', -1)
            ))
        
        # Create indices for fast filtering
        logger.info("Creating B-tree indices...")
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_category ON articles(category)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_date ON articles(date)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_parsed_date ON articles(parsed_date)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_doc_length ON articles(doc_length)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_topic_id ON articles(topic_id)')
        
        # Commit and close
        self.conn.commit()
        
        logger.info(f"Metadata index built successfully at {self.db_path}")
        
        # Print statistics
        cursor.execute("SELECT COUNT(*) FROM articles")
        count = cursor.fetchone()[0]
        logger.info(f"Total articles in metadata store: {count}")
        
        cursor.execute("SELECT category, COUNT(*) FROM articles GROUP BY category")
        categories = cursor.fetchall()
        logger.info(f"Articles by category: {dict(categories)}")
    
    def get_article_metadata(self, article_id: int) -> Dict:
        """Get metadata for a specific article"""
        if self.conn is None:
            self.conn = sqlite3.connect(self.db_path)
        
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM articles WHERE id = ?", (article_id,))
        row = cursor.fetchone()
        
        if row is None:
            return None
        
        columns = [desc[0] for desc in cursor.description]
        metadata = dict(zip(columns, row))
        
        # Parse entities JSON
        metadata['entities'] = json.loads(metadata['entities'])
        
        return metadata
    
    def filter_by_category(self, category: str) -> List[int]:
        """Get article IDs by category"""
        if self.conn is None:
            self.conn = sqlite3.connect(self.db_path)
        
        cursor = self.conn.cursor()
        cursor.execute("SELECT id FROM articles WHERE category = ?", (category,))
        return [row[0] for row in cursor.fetchall()]
    
    def filter_by_date_range(self, start_date: str, end_date: str) -> List[int]:
        """Get article IDs within date range"""
        if self.conn is None:
            self.conn = sqlite3.connect(self.db_path)
        
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT id FROM articles WHERE parsed_date BETWEEN ? AND ?",
            (start_date, end_date)
        )
        return [row[0] for row in cursor.fetchall()]
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            logger.info("Metadata database connection closed")


class TopicIndexer:
    """
    Topic modeling using BERTopic
    """
    
    def __init__(self):
        """Initialize topic indexer"""
        self.topic_model = None
        logger.info("TopicIndexer initialized")
    
    @timer
    def build_topic_model(self, processed_articles: List[Dict], save_path: str = None):
        """
        Build topic model using BERTopic
        
        Args:
            processed_articles: List of preprocessed article dictionaries
            save_path: Path to save model (default from config)
        """
        try:
            from bertopic import BERTopic
            from sklearn.feature_extraction.text import CountVectorizer
            
            logger.info(f"Building topic model for {len(processed_articles)} articles...")
            
            # Prepare documents for topic modeling
            documents = [
                article['content'][:1000] 
                for article in processed_articles
            ]
            
            # Initialize BERTopic with optimized settings
            vectorizer_model = CountVectorizer(
                stop_words='english',
                max_features=1000
            )
            
            self.topic_model = BERTopic(
                embedding_model=config.EMBEDDING_MODEL,
                vectorizer_model=vectorizer_model,
                min_topic_size=10,
                nr_topics='auto'
            )
            
            # Fit topic model
            logger.info("Fitting topic model (this may take a minute)...")
            topics, probs = self.topic_model.fit_transform(documents)
            
            # Add topics to processed articles
            for idx, (topic, prob) in enumerate(zip(topics, probs)):
                processed_articles[idx]['topic_id'] = int(topic)
                processed_articles[idx]['topic_prob'] = float(prob)
            
            # Save topic model
            if save_path is None:
                save_path = config.TOPIC_MODEL_PATH
            
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            self.topic_model.save(save_path)
            
            # Save updated processed articles
            save_pickle(processed_articles, config.PROCESSED_DATA_PATH)
            
            logger.info(f"Topic model built and saved to {save_path}")
            logger.info(f"Found {len(set(topics))} topics")
            
            # Print topic info
            topic_info = self.topic_model.get_topic_info()
            logger.info(f"\nTop 5 topics:\n{topic_info.head()}")
            
            return processed_articles
            
        except ImportError:
            logger.warning("BERTopic not installed. Skipping topic modeling.")
            logger.info("Install with: pip install bertopic")
            return processed_articles
        except Exception as e:
            logger.error(f"Error building topic model: {e}")
            return processed_articles
    
    def load_topic_model(self, load_path: str = None):
        """Load topic model from disk"""
        try:
            from bertopic import BERTopic
            
            if load_path is None:
                load_path = config.TOPIC_MODEL_PATH
            
            logger.info(f"Loading topic model from {load_path}")
            self.topic_model = BERTopic.load(load_path)
            logger.info("Topic model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading topic model: {e}")
            self.topic_model = None


def main():
    """Main function to run indexing pipeline"""
    logger.info("Starting indexing pipeline...")
    
    # Load preprocessed articles
    logger.info(f"Loading preprocessed articles from {config.PROCESSED_DATA_PATH}")
    processed_articles = load_pickle(config.PROCESSED_DATA_PATH)
    logger.info(f"Loaded {len(processed_articles)} preprocessed articles")
    
    # Build BM25 index
    logger.info("\n=== Building BM25 Index ===")
    bm25_indexer = BM25Indexer()
    bm25_indexer.build_index(processed_articles)
    
    # Build topic model first (adds topic_id to articles)
    logger.info("\n=== Building Topic Model ===")
    topic_indexer = TopicIndexer()
    processed_articles = topic_indexer.build_topic_model(processed_articles)
    
    # Build metadata index
    logger.info("\n=== Building Metadata Index ===")
    metadata_indexer = MetadataIndexer()
    metadata_indexer.build_index(processed_articles)
    metadata_indexer.close()
    
    # Build ColBERT index (optional, takes longer)
    logger.info("\n=== Building ColBERT Index ===")
    colbert_indexer = ColBERTIndexer()
    colbert_indexer.build_index(processed_articles)
    
    logger.info("\n=== Indexing Complete ===")
    logger.info("All indices have been built successfully!")


if __name__ == "__main__":
    main()
